"""
gru_model.py
------------
Defines the multi-output Bidirectional GRU architecture:

    Embedding -> Bidirectional GRU -> Bidirectional GRU -> shared Dense
        -> 3 output heads: category, experience_level, job_domain

This is the ONLY deep-learning component of the project. All ATS scoring,
skill extraction, and recommendation logic lives outside this model, in
ats_engine/ (rule-based + dictionary-based, fully explainable).

No BERT / Transformer / Hugging Face / LLM APIs are used anywhere here.
"""

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, GRU, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.optimizers import Adam


class ResumeGRUModel:
    """Builds and wraps the multi-output Bidirectional GRU classifier."""

    def __init__(self, vocab_size: int, max_len: int, num_categories: int,
                 num_experience_levels: int, num_job_domains: int,
                 embedding_dim: int = 128, gru_units_1: int = 128, gru_units_2: int = 64,
                 dropout_rate: float = 0.3, learning_rate: float = 1e-3):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_categories = num_categories
        self.num_experience_levels = num_experience_levels
        self.num_job_domains = num_job_domains
        self.embedding_dim = embedding_dim
        self.gru_units_1 = gru_units_1
        self.gru_units_2 = gru_units_2
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model: Model = self._build()

    def _build(self) -> Model:
        """Constructs the Keras functional-API multi-output model."""
        inputs = Input(shape=(self.max_len,), name="resume_tokens")

        x = Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embedding_dim,
            input_length=self.max_len,
            mask_zero=True,
            name="embedding",
        )(inputs)

        # Stacked Bidirectional GRU layers -- the core sequence-understanding component
        x = Bidirectional(GRU(self.gru_units_1, return_sequences=True), name="bigru_1")(x)
        x = Dropout(self.dropout_rate)(x)
        x = Bidirectional(GRU(self.gru_units_2), name="bigru_2")(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        # Shared dense representation before splitting into task-specific heads
        shared = Dense(64, activation="relu", name="shared_dense")(x)
        shared = Dropout(self.dropout_rate)(shared)

        # Three output heads (multi-task learning)
        category_out = Dense(32, activation="relu")(shared)
        category_out = Dense(self.num_categories, activation="softmax", name="category")(category_out)

        experience_out = Dense(32, activation="relu")(shared)
        experience_out = Dense(self.num_experience_levels, activation="softmax",
                                name="experience_level")(experience_out)

        domain_out = Dense(32, activation="relu")(shared)
        domain_out = Dense(self.num_job_domains, activation="softmax", name="job_domain")(domain_out)

        model = Model(inputs=inputs, outputs=[category_out, experience_out, domain_out],
                      name="ResumeIQ_BiGRU")

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss={
                "category": "sparse_categorical_crossentropy",
                "experience_level": "sparse_categorical_crossentropy",
                "job_domain": "sparse_categorical_crossentropy",
            },
            metrics={
                "category": ["accuracy"],
                "experience_level": ["accuracy"],
                "job_domain": ["accuracy"],
            },
        )
        return model

    def summary(self):
        return self.model.summary()

    def save(self, path: str):
        """Save the trained model in native Keras format (.keras)."""
        self.model.save(path)

    @staticmethod
    def load(path: str) -> Model:
        """Load a previously trained/saved model."""
        from tensorflow.keras.models import load_model
        return load_model(path)


if __name__ == "__main__":
    # Quick architecture sanity check
    demo = ResumeGRUModel(
        vocab_size=10000, max_len=300,
        num_categories=6, num_experience_levels=4, num_job_domains=6,
    )
    demo.summary()
