"""Frequency matrices, five TF/IDF variants, search, and spam classification."""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


TF_METHODS = ("binary", "absolute", "relative", "logarithmic", "augmented")
IDF_METHODS = ("standard", "smoothed", "probabilistic", "maximum", "unit")
FOLDER = Path(__file__).resolve().parent


def tokenize(text):
    # Use the same tokenization rule as in exercises 1 and 2.
    return re.findall(r"\w+|[^\w\s]", text)


def compute_tf(counts, method):
    """Transform counts without assigning weights to absent terms."""
    tf = counts.astype(float)
    tf.eliminate_zeros()

    if method == "binary":
        tf.data[:] = 1
    elif method == "absolute":
        pass
    elif method in ("relative", "augmented"):
        if method == "relative":
            denominator = np.asarray(counts.sum(axis=1)).ravel()
        else:
            denominator = counts.max(axis=1).toarray().ravel()
        # An empty document remains a zero vector rather than dividing by zero.
        scale = np.divide(1.0, denominator, out=np.zeros_like(denominator, dtype=float),
                          where=denominator != 0)
        tf = tf.multiply(scale[:, None]).tocsr()
        if method == "augmented":
            tf.data = 0.5 + 0.5 * tf.data
    elif method == "logarithmic":
        tf.data = 1 + np.log(tf.data)
    else:
        raise ValueError(f"Unknown TF method: {method}")
    return tf


def compute_idf(counts, method):
    """Learn document frequencies from the reference document collection."""
    n_documents = counts.shape[0]
    df = np.asarray((counts > 0).sum(axis=0)).ravel().astype(float)

    if method == "standard":
        return np.log(n_documents / df)
    if method == "smoothed":
        return np.log((n_documents + 1) / (df + 1)) + 1
    if method == "probabilistic":
        # The slide's formula is undefined when df == N. Ignore those terms.
        idf = np.zeros_like(df)
        valid = df < n_documents
        idf[valid] = np.log((n_documents - df[valid]) / df[valid])
        return idf
    if method == "maximum":
        return np.log(df.max() / df)
    if method == "unit":
        return np.ones_like(df)
    raise ValueError(f"Unknown IDF method: {method}")


def weight(counts, tf_method, idf):
    # Each column is multiplied by the IDF of its term.
    return compute_tf(counts, tf_method).multiply(idf).tocsr()


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def search(messages, labels, args, output):
    # Search may use the whole collection; it does not evaluate a classifier.
    vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None)
    counts = vectorizer.fit_transform(messages)
    idf = compute_idf(counts, args.idf)
    documents = weight(counts, args.tf, idf)
    query = weight(vectorizer.transform([args.query]), args.tf, idf)

    similarities = cosine_similarity(query, documents).ravel()
    order = np.argsort(-similarities, kind="stable")
    rows = [
        {"rank": rank, "similarity": float(similarities[index]),
         "category": labels[index], "message": messages[index]}
        for rank, index in enumerate(order, start=1)
    ]
    save_csv(output / "search_results.csv", rows)

    print(f"\n=== Search: TF={args.tf}, IDF={args.idf} ===")
    print(f"Query: {args.query}")
    print(f"Frequency matrix (documents, terms): {counts.shape}")
    if query.nnz == 0 or not np.any(query.data):
        print("The query has no nonzero weights. All similarities are zero.")
    for row in rows[:5]:
        preview = row["message"].replace("\n", " ")[:160]
        print(f"{row['rank']}. {row['similarity']:.4f} [{row['category']}] {preview}")


def classify(messages, labels, args, output):
    train_messages, test_messages, y_train, y_test = train_test_split(
        messages, labels, test_size=0.2, random_state=42, stratify=labels
    )
    # Fit the vocabulary and IDF on train only to prevent test-data leakage.
    vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None)
    train_counts = vectorizer.fit_transform(train_messages)
    test_counts = vectorizer.transform(test_messages)
    new_counts = vectorizer.transform([args.message])

    results = []
    predictions = [
        {"actual": label, "message": message}
        for message, label in zip(test_messages, y_test)
    ]
    print("\n=== Logistic regression: all 25 TF/IDF combinations ===")
    print(f"Train matrix: {train_counts.shape}; test matrix: {test_counts.shape}")
    print(f"{'TF':<13} {'IDF':<14} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8} {'New':>5}")

    for tf_method in TF_METHODS:
        for idf_method in IDF_METHODS:
            idf = compute_idf(train_counts, idf_method)
            x_train = weight(train_counts, tf_method, idf)
            x_test = weight(test_counts, tf_method, idf)
            model = LogisticRegression(max_iter=2000, random_state=42)
            model.fit(x_train, y_train)
            predicted = model.predict(x_test)
            new_vector = weight(new_counts, tf_method, idf)
            new_prediction = model.predict(new_vector)[0]

            result = {
                "tf": tf_method, "idf": idf_method,
                "accuracy": accuracy_score(y_test, predicted),
                "precision": precision_score(y_test, predicted, pos_label="spam", zero_division=0),
                "recall": recall_score(y_test, predicted, pos_label="spam", zero_division=0),
                "f1": f1_score(y_test, predicted, pos_label="spam", zero_division=0),
                "new_prediction": new_prediction,
            }
            results.append(result)
            for row, prediction in zip(predictions, predicted):
                row[f"{tf_method}__{idf_method}"] = prediction
            print(f"{tf_method:<13} {idf_method:<14} {result['accuracy']:9.4f} "
                  f"{result['precision']:10.4f} {result['recall']:8.4f} "
                  f"{result['f1']:8.4f} {new_prediction:>5}", flush=True)

    save_csv(output / "comparison.csv", results)
    save_csv(output / "test_predictions.csv", predictions)
    selected = f"{args.tf}__{args.idf}"
    print(f"\nFirst 5 test predictions for TF={args.tf}, IDF={args.idf}:")
    for row in predictions[:5]:
        preview = row["message"].replace("\n", " ")[:110]
        print(f"actual={row['actual']}, predicted={row[selected]}: {preview}")
    chosen = next(row for row in results if row["tf"] == args.tf and row["idf"] == args.idf)
    print(f"\nNew message: {args.message}")
    print(f"Selected model prediction: {chosen['new_prediction']}")
    print("All test predictions are saved in test_predictions.csv.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tf", choices=TF_METHODS, default="absolute")
    parser.add_argument("--idf", choices=IDF_METHODS, default="smoothed")
    parser.add_argument("--query", default="free prize call now")
    parser.add_argument("--message", default="Congratulations! You won a free prize. Call now to claim it!")
    args = parser.parse_args()

    with (FOLDER / "spam.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    # As in exercise 2, remove exact duplicates before splitting the data.
    unique = {row["Message"]: row["Category"] for row in rows}
    messages, labels = list(unique), list(unique.values())
    output = FOLDER / "results"
    output.mkdir(exist_ok=True)
    print(f"CSV records: {len(rows)}; unique messages: {len(messages)}; duplicates removed: {len(rows) - len(messages)}")

    search(messages, labels, args, output)
    classify(messages, labels, args, output)
    print(f"\nResults saved to: {output}")


if __name__ == "__main__":
    main()
