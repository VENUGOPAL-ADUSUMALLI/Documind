"""All talking to Qdrant (the vector database) happens here.

Each chunk is stored as ONE point with TWO vectors:
  "dense"  numbers that capture the meaning of the text   (search by meaning)
  "bm25"   weights for the words in the text              (search by exact words)
"""

from qdrant_client import QdrantClient, models

from app import sparse
from app.config import settings

DENSE = "dense"
BM25 = "bm25"

# In hybrid mode, each of the two searches fetches at least this many results
# before they are merged, so the merge has enough to work with.
MIN_BRANCH_LIMIT = 20


def make_qdrant_client(url=None):
    """Connect to Qdrant. The special url ':memory:' gives a temporary one for tests."""
    url = url or settings.qdrant_url
    if url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=url)


def ensure_collection(client):
    """Create the collection (Qdrant's word for a table) if it doesn't exist yet."""
    if not client.collection_exists(settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                DENSE: models.VectorParams(
                    size=settings.embedding_dim,  # how many numbers in each vector
                    distance=models.Distance.COSINE,  # how "closeness" is measured
                )
            },
            sparse_vectors_config={
                # IDF = Qdrant gives rare words more weight than common words.
                BM25: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )


def only_this_document(document_id):
    """A Qdrant filter that matches only chunks belonging to one document."""
    return models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            )
        ]
    )


def upsert_chunks(client, points):
    """Save chunks. Each point is (chunk_id, text, dense_vector, payload).

    The keyword vector is computed here from the text.
    "Upsert" = insert, or overwrite if the id already exists.
    """
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            models.PointStruct(
                id=chunk_id,
                vector={DENSE: dense_vector, BM25: sparse.encode_document(text)},
                payload=payload,
            )
            for chunk_id, text, dense_vector, payload in points
        ],
    )


def delete_document_points(client, document_id):
    """Delete every vector that belongs to one document."""
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.FilterSelector(filter=only_this_document(document_id)),
    )


def search(client, dense_vector, query_text, limit=5, document_id=None, mode=None):
    """Find the best chunks for a question.

    mode "dense":  closest by meaning only
    mode "bm25":   best keyword match only
    mode "hybrid": run both, then merge the two ranked lists with
                   Reciprocal Rank Fusion (RRF): a chunk ranked high in BOTH
                   lists beats a chunk that is high in only one.

    If document_id is given, only that document's chunks are searched.
    """
    mode = mode or settings.retrieval_mode
    query_filter = only_this_document(document_id) if document_id else None
    branch_limit = max(limit, MIN_BRANCH_LIMIT)

    # Build the individual searches we need.
    searches = []
    if mode in ("dense", "hybrid"):
        searches.append(
            models.Prefetch(
                query=dense_vector, using=DENSE, limit=branch_limit, filter=query_filter
            )
        )
    keyword_query = sparse.encode_query(query_text)
    if mode in ("bm25", "hybrid") and keyword_query.indices:  # skip if no usable words
        searches.append(
            models.Prefetch(
                query=keyword_query, using=BM25, limit=branch_limit, filter=query_filter
            )
        )

    if not searches:
        return []

    # One search: run it directly. Two searches: run both and merge with RRF.
    if len(searches) == 1:
        result = client.query_points(
            collection_name=settings.qdrant_collection,
            query=searches[0].query,
            using=searches[0].using,
            limit=limit,
            query_filter=query_filter,
            with_payload=True,
        )
    else:
        result = client.query_points(
            collection_name=settings.qdrant_collection,
            prefetch=searches,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
    return result.points
