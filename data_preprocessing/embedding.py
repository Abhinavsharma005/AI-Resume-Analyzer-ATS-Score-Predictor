"""
embedding.py
------------
Builds the embedding matrix used by the GRU model's Embedding layer.

By default this project uses a TRAINABLE embedding layer learned end-to-end
with the GRU (no external embedding download required -- keeps the project
fully self-contained and GPU-free).

Optionally, if a GloVe .txt file is available locally, `load_glove_embeddings`
can build a pretrained embedding matrix instead (set `use_pretrained=True` in
models/train.py). This is entirely optional and NOT required to run the
project -- BERT/Transformer embeddings are intentionally NOT used anywhere
in this project per the design constraints.
"""

import os
import numpy as np


def load_glove_embeddings(glove_path: str, word_index: dict, embedding_dim: int = 100,
                           vocab_size: int = 10000) -> np.ndarray:
    """
    Build an embedding matrix from a local GloVe file (optional, pretrained).

    Args:
        glove_path: path to a glove.6B.{dim}d.txt file on disk.
        word_index: tokenizer.tokenizer.word_index mapping word -> integer id.
        embedding_dim: dimensionality of the GloVe vectors being loaded.
        vocab_size: number of rows to allocate (must match Tokenizer.vocab_size).

    Returns:
        A (vocab_size, embedding_dim) numpy array suitable for
        Embedding(..., weights=[matrix], trainable=False/True).
    """
    if not os.path.exists(glove_path):
        raise FileNotFoundError(
            f"GloVe file not found at {glove_path}. "
            "Pretrained embeddings are optional -- omit `--use_pretrained` to "
            "train an embedding layer from scratch instead."
        )

    embeddings_index = {}
    with open(glove_path, encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings_index[word] = vector

    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, idx in word_index.items():
        if idx >= vocab_size:
            continue
        vector = embeddings_index.get(word)
        if vector is not None:
            embedding_matrix[idx] = vector

    return embedding_matrix


def build_trainable_embedding_config(vocab_size: int, embedding_dim: int = 128) -> dict:
    """
    Returns the config dict used by models/gru_model.py to construct a
    randomly-initialized, end-to-end trainable Embedding layer (the default
    path for this project).
    """
    return {"input_dim": vocab_size, "output_dim": embedding_dim, "trainable": True}
