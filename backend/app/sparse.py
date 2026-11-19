
import re
import zlib
from collections import Counter

from qdrant_client import models

# Standard BM25 tuning numbers.
K1 = 1.2   # how quickly repeating a word stops adding weight
B = 0.75   # how much longer chunks are penalised
AVG_CHUNK_LENGTH = 150  # a typical chunk is about this many words

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "do", "does",
    "for", "from", "has", "have", "how", "i", "in", "is", "it", "its", "me", "my", "of",
    "on", "or", "s", "so", "that", "the", "their", "there", "these", "this", "to", "was",
    "we", "were", "what", "when", "where", "which", "who", "whom", "why", "will", "with",
    "you", "your",
}


def tokenize(text):
    words = re.findall(r"\w+", text.lower())
    return [word for word in words if word not in STOPWORDS]


def word_id(word):
    return zlib.crc32(word.encode())


def encode_document(text):
    words = tokenize(text)
    length = len(words)

    weights = {}
    for word, count in Counter(words).items():
        weight = count * (K1 + 1) / (count + K1 * (1 - B + B * length / AVG_CHUNK_LENGTH))
        weights[word_id(word)] = weights.get(word_id(word), 0) + weight

    return models.SparseVector(indices=list(weights), values=list(weights.values()))


def encode_query(text):
    ids = sorted({word_id(word) for word in tokenize(text)})
    return models.SparseVector(indices=ids, values=[1.0] * len(ids))
