"""Keyword (BM25) vectors, to complement the meaning-based (dense) vectors.

Idea: a sparse vector is a list of (word_id, weight) pairs, one pair per distinct
word in the text. Words the text doesn't contain are simply absent.
Searching compares the question's words with each chunk's words.

BM25 gives a word a high weight when it appears often in the chunk (but with
diminishing returns). The other half of BM25, "rare words matter more", is done
by Qdrant itself: the collection is created with the IDF modifier.
"""

import re
import zlib
from collections import Counter

from qdrant_client import models

# Standard BM25 tuning numbers.
K1 = 1.2   # how quickly repeating a word stops adding weight
B = 0.75   # how much longer chunks are penalised
AVG_CHUNK_LENGTH = 150  # a typical chunk is about this many words


# Very common words carry no meaning for keyword search ("what is the ..."),
# and they match almost every chunk, so we ignore them.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "do", "does",
    "for", "from", "has", "have", "how", "i", "in", "is", "it", "its", "me", "my", "of",
    "on", "or", "s", "so", "that", "the", "their", "there", "these", "this", "to", "was",
    "we", "were", "what", "when", "where", "which", "who", "whom", "why", "will", "with",
    "you", "your",
}


def tokenize(text):
    """Split text into lowercase words, dropping filler words.

    'Who is the Project Advisor?' -> ['project', 'advisor']
    """
    words = re.findall(r"\w+", text.lower())
    return [word for word in words if word not in STOPWORDS]


def word_id(word):
    """Turn a word into a number. crc32 is stable (the same word always gives the same id)."""
    return zlib.crc32(word.encode())


def encode_document(text):
    """BM25 weights for a chunk of text."""
    words = tokenize(text)
    length = len(words)

    weights = {}
    for word, count in Counter(words).items():
        weight = count * (K1 + 1) / (count + K1 * (1 - B + B * length / AVG_CHUNK_LENGTH))
        # Two different words could share an id (very rare), so add instead of overwrite.
        weights[word_id(word)] = weights.get(word_id(word), 0) + weight

    return models.SparseVector(indices=list(weights), values=list(weights.values()))


def encode_query(text):
    """A question is just 'these words matter': weight 1 for each distinct word."""
    ids = sorted({word_id(word) for word in tokenize(text)})
    return models.SparseVector(indices=ids, values=[1.0] * len(ids))
