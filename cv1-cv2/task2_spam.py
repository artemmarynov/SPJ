"""Task 2: a binary token matrix and logistic regression."""

import csv
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from task1_tokenization import tokenize


def main():
    path = Path(__file__).resolve().parent / "spam.csv"
    with path.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    # Remove duplicates so the same message cannot appear in both train and test.
    unique = {row["Message"]: row["Category"] for row in rows}
    messages = list(unique)
    labels = [1 if unique[message] == "spam" else 0 for message in messages]

    train_messages, test_messages, y_train, y_test = train_test_split(
        messages, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Each row is a message, each column is a unique token, each value is 0 or 1.
    vectorizer = CountVectorizer(
        tokenizer=tokenize, token_pattern=None, binary=True
    )
    # Learn the vocabulary on train only, then use the same vocabulary for test.
    x_train = vectorizer.fit_transform(train_messages)
    x_test = vectorizer.transform(test_messages)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(x_train, y_train)
    predicted = model.predict(x_test)

    print("=== Spam classification ===")
    print(f"CSV records: {len(rows)}")
    print(f"Duplicate messages removed: {len(rows) - len(unique)}")
    print(f"Unique messages: {len(messages)} (ham: {labels.count(0)}, spam: {labels.count(1)})")
    print(f"Train matrix (messages, tokens): {x_train.shape}")
    print(f"Test matrix  (messages, tokens): {x_test.shape}")

    print("\nTest metrics:")
    print(f"Accuracy:  {accuracy_score(y_test, predicted):.4f}")
    print(f"Precision: {precision_score(y_test, predicted, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predicted, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, predicted, zero_division=0):.4f}")

if __name__ == "__main__":
    main()
