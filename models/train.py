"""
train.py
--------
End-to-end training script for the ResumeIQ Bidirectional GRU model.

Usage:
    python -m models.train
    python -m models.train --epochs 15 --batch_size 32

Outputs (saved to models/):
    saved_model.keras   - trained Keras model
    tokenizer.pkl        - fitted ResumeTokenizer
    label_encoders.pkl   - fitted LabelEncoderSet
    training_history.json
    evaluation_report.json   - accuracy/precision/recall/F1 + confusion matrices
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
)

from data_preprocessing.clean_text import TextCleaner
from data_preprocessing.tokenizer import ResumeTokenizer, LabelEncoderSet
from models.gru_model import ResumeGRUModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATASET_PATH = os.path.join(PROJECT_ROOT, "datasets", "resume_dataset.csv")

MODEL_PATH = os.path.join(BASE_DIR, "saved_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
LABEL_ENCODERS_PATH = os.path.join(BASE_DIR, "label_encoders.pkl")
HISTORY_PATH = os.path.join(BASE_DIR, "training_history.json")
EVAL_REPORT_PATH = os.path.join(BASE_DIR, "evaluation_report.json")


def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run "
            "`python datasets/generate_synthetic_dataset.py` first, or supply your "
            "own resume_dataset.csv with columns: resume_text, category, "
            "experience_level, job_domain."
        )
    return pd.read_csv(path)


def preprocess(df: pd.DataFrame, cleaner: TextCleaner) -> pd.Series:
    print("Cleaning text ...")
    return df["resume_text"].astype(str).apply(cleaner.clean)


def evaluate(model, X_test, y_test_dict, label_encoders: LabelEncoderSet) -> dict:
    """Computes accuracy/precision/recall/F1 + confusion matrix for each output head."""
    preds = model.predict(X_test, verbose=0)
    head_names = ["category", "experience_level", "job_domain"]

    report = {}
    for head_name, pred_probs in zip(head_names, preds):
        y_true = y_test_dict[head_name]
        y_pred = np.argmax(pred_probs, axis=1)

        acc = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred).tolist()
        class_names = label_encoders.classes(head_name)

        report[head_name] = {
            "accuracy": round(float(acc), 4),
            "precision_weighted": round(float(precision), 4),
            "recall_weighted": round(float(recall), 4),
            "f1_weighted": round(float(f1), 4),
            "confusion_matrix": cm,
            "class_names": class_names,
            "classification_report": classification_report(
                y_true, y_pred, target_names=class_names, zero_division=0, output_dict=True
            ),
        }
        print(f"\n[{head_name}] accuracy={acc:.4f}  precision={precision:.4f}  "
              f"recall={recall:.4f}  f1={f1:.4f}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Train the ResumeIQ Bidirectional GRU model.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--vocab_size", type=int, default=15000)
    parser.add_argument("--max_len", type=int, default=400)
    parser.add_argument("--test_size", type=float, default=0.2)
    args = parser.parse_args()

    # 1. Load data
    df = load_dataset(DATASET_PATH)
    print(f"Loaded {len(df)} resumes.")

    # 2. Clean text
    cleaner = TextCleaner()
    df["cleaned_text"] = preprocess(df, cleaner)

    # 3. Encode labels
    label_encoders = LabelEncoderSet()
    encoded_labels = label_encoders.fit_transform(df)

    # 4. Tokenize + pad
    tokenizer = ResumeTokenizer(vocab_size=args.vocab_size, max_len=args.max_len)
    X = tokenizer.fit_transform(df["cleaned_text"].tolist())

    # 5. Train/test split (stratify on category to keep classes balanced)
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices, test_size=args.test_size, random_state=42,
        stratify=encoded_labels["category"]
    )

    X_train, X_test = X[train_idx], X[test_idx]
    y_train = {k: v[train_idx] for k, v in encoded_labels.items()}
    y_test = {k: v[test_idx] for k, v in encoded_labels.items()}

    # 6. Build model
    gru = ResumeGRUModel(
        vocab_size=tokenizer.vocabulary_size,
        max_len=args.max_len,
        num_categories=len(label_encoders.classes("category")),
        num_experience_levels=len(label_encoders.classes("experience_level")),
        num_job_domains=len(label_encoders.classes("job_domain")),
    )
    gru.summary()

    # 7. Train
    from tensorflow.keras.callbacks import EarlyStopping

    history = gru.model.fit(
        X_train,
        {"category": y_train["category"], "experience_level": y_train["experience_level"],
         "job_domain": y_train["job_domain"]},
        validation_split=0.15,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)],
        verbose=2,
    )

    # 8. Evaluate
    eval_report = evaluate(gru.model, X_test, y_test, label_encoders)

    # 9. Save artifacts
    gru.save(MODEL_PATH)
    tokenizer.save(TOKENIZER_PATH)
    label_encoders.save(LABEL_ENCODERS_PATH)

    with open(HISTORY_PATH, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()}, f, indent=2)

    with open(EVAL_REPORT_PATH, "w") as f:
        json.dump(eval_report, f, indent=2)

    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved tokenizer -> {TOKENIZER_PATH}")
    print(f"Saved label encoders -> {LABEL_ENCODERS_PATH}")
    print(f"Saved evaluation report -> {EVAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
