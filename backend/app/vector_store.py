"""All talking to Qdrant (the vector database) happens here."""

from qdrant_client import QdrantClient, models

from app.config import settings


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
            vectors_config=models.VectorParams(
                size=settings.embedding_dim,  # how many numbers in each vector
                distance=models.Distance.COSINE,  # how "closeness" is measured
            ),
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
    """Save points. Each point is (chunk_id, vector, payload).

    "Upsert" = insert, or overwrite if the id already exists.
    """
    client.upsert(
        collection_name=settings.qdrant_collection,
        points=[
            models.PointStruct(id=chunk_id, vector=vector, payload=payload)
            for chunk_id, vector, payload in points
        ],
    )


def delete_document_points(client, document_id):
    """Delete every vector that belongs to one document."""
    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=models.FilterSelector(filter=only_this_document(document_id)),
    )


def search(client, query_vector, limit=5, document_id=None):
    """Find the chunks whose vectors are closest to the question's vector.

    If document_id is given, only that document's chunks are searched.
    """
    query_filter = only_this_document(document_id) if document_id else None
    result = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
        with_payload=True,
    )
    return result.points
