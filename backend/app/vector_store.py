from qdrant_client import QdrantClient, models

from app import sparse
from app.config import settings

DENSE = "dense"
BM25 = "bm25"

MIN_BRANCH_LIMIT = 20


def make_qdrant_client(url=None):
    url = url or settings.qdrant_url
    if url == ":memory:":
        return QdrantClient(":memory:")
    return QdrantClient(url=url)


def ensure_collection(client):
    if not client.collection_exists(settings.qdrant_collection):
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={
                DENSE: models.VectorParams(
                    size=settings.embedding_dim,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                BM25: models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )


def only_this_document(document_id):
    return models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            )
        ]
    )


def upsert_chunks(client, points):
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
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.FilterSelector(filter=only_this_document(document_id)),
    )


def search(client, dense_vector, query_text, limit=5, document_id=None, mode=None):
    mode = mode or settings.retrieval_mode
    query_filter = only_this_document(document_id) if document_id else None
    branch_limit = max(limit, MIN_BRANCH_LIMIT)

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
