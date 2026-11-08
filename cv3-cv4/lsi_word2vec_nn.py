"""Exercise 4: compare LSI, Word2Vec, CNN, RNN, and LSTM on spam messages."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from tfidf import compute_idf, save_csv, tokenize, weight
from neural_models import (
    average_embeddings, encode_sequences, predict_classifier,
    train_classifier, train_word2vec,
)


FOLDER = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topics", type=int, nargs="+", default=[20, 50, 100, 200])
    parser.add_argument("--dimension", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--word2vec-epochs", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=80)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=FOLDER / "results_advanced")
    parser.add_argument("--message", default="Congratulations! You won a free prize. Call now to claim it!")
    args = parser.parse_args()
    if min(args.dimension, args.epochs, args.word2vec_epochs, args.max_length, args.threads) < 1:
        parser.error("Dimensions, epoch counts, maximum length, and threads must be positive.")
    if any(topics < 1 for topics in args.topics):
        parser.error("Topic counts must be positive.")
    return args


def run(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with (FOLDER / "spam.csv").open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    unique = {row["Message"]: row["Category"] for row in rows}
    messages = np.array(list(unique))
    labels = np.array([int(category == "spam") for category in unique.values()])
    indices = np.arange(len(messages))
    train_ids, test_ids = train_test_split(indices, test_size=0.2, random_state=42, stratify=labels)
    train_messages, test_messages = messages[train_ids], messages[test_ids]
    y_train, y_test = labels[train_ids], labels[test_ids]
    train_tokens = [tokenize(text.lower()) for text in train_messages]
    test_tokens = [tokenize(text.lower()) for text in test_messages]
    new_tokens = [tokenize(args.message.lower())]

    save_csv(output / "split.csv", [
        {"document_id": int(index), "split": split,
         "category": "spam" if labels[index] else "ham", "message": messages[index]}
        for split, ids in [("train", train_ids), ("test", test_ids)] for index in ids
    ])
    print(f"Unique messages: {len(messages)}; train: {len(train_ids)}; test: {len(test_ids)}")
    print("All vocabularies, TF-IDF statistics, SVD components, and embeddings use train only.", flush=True)

    results = []
    predictions = [{"actual": "spam" if label else "ham", "message": message}
                   for message, label in zip(test_messages, y_test)]
    representations = {}

    def record(name, dimensions, predicted, new_prediction):
        result = {
            "model": name, "dimensions": dimensions,
            "accuracy": accuracy_score(y_test, predicted),
            "precision": precision_score(y_test, predicted, zero_division=0),
            "recall": recall_score(y_test, predicted, zero_division=0),
            "f1": f1_score(y_test, predicted, zero_division=0),
            "new_prediction": "spam" if new_prediction else "ham",
        }
        results.append(result)
        for row, prediction in zip(predictions, predicted):
            row[name] = "spam" if prediction else "ham"
        save_csv(output / "comparison.csv", results)
        save_csv(output / "test_predictions.csv", predictions)
        print(f"{name:<22} accuracy={result['accuracy']:.4f} precision={result['precision']:.4f} "
              f"recall={result['recall']:.4f} F1={result['f1']:.4f} new={result['new_prediction']}", flush=True)

    def linear_classifier(name, x_train, x_test, x_new):
        model = LogisticRegression(max_iter=2000, random_state=42)
        model.fit(x_train, y_train)
        record(name, x_train.shape[1], model.predict(x_test), model.predict(x_new)[0])

    # Recompute earlier baselines on exactly the same split instead of reading old scores.
    vectorizer = CountVectorizer(tokenizer=tokenize, token_pattern=None)
    train_counts = vectorizer.fit_transform(train_messages)
    test_counts = vectorizer.transform(test_messages)
    new_counts = vectorizer.transform([args.message])
    print(f"\n=== Previous approaches: vocabulary size {train_counts.shape[1]} ===", flush=True)
    unit_idf = compute_idf(train_counts, "unit")
    linear_classifier("Binary + LR", weight(train_counts, "binary", unit_idf),
                      weight(test_counts, "binary", unit_idf), weight(new_counts, "binary", unit_idf))
    idf = compute_idf(train_counts, "smoothed")
    x_train = weight(train_counts, "absolute", idf)
    x_test = weight(test_counts, "absolute", idf)
    x_new = weight(new_counts, "absolute", idf)
    linear_classifier("TF-IDF + LR", x_train, x_test, x_new)

    print("\n=== LSI: truncated SVD fitted on train only ===", flush=True)
    lsi_details = []
    for topics in dict.fromkeys(args.topics):
        if topics >= min(x_train.shape):
            raise ValueError(f"Topic count {topics} must be smaller than {min(x_train.shape)}.")
        svd = TruncatedSVD(n_components=topics, n_iter=7, random_state=42)
        train_lsi = svd.fit_transform(x_train)
        test_lsi = svd.transform(x_test)
        new_lsi = svd.transform(x_new)
        linear_classifier(f"LSI {topics} + LR", train_lsi, test_lsi, new_lsi)
        representations[f"lsi_{topics}_train"] = train_lsi
        representations[f"lsi_{topics}_test"] = test_lsi
        terms = vectorizer.get_feature_names_out()
        for index, component in enumerate(svd.components_[:5], start=1):
            # SVD axes have two poles, not probabilities of words in a topic.
            positive = terms[np.argsort(component)[-6:][::-1]].tolist()
            negative = terms[np.argsort(component)[:6]].tolist()
            lsi_details.append({"dimensions": topics, "component": index,
                                "positive_terms": " ".join(positive),
                                "negative_terms": " ".join(negative),
                                "explained_variance_total": float(svd.explained_variance_ratio_.sum())})
    save_csv(output / "lsi_topics.csv", lsi_details)

    print("\n=== Word2Vec: mean document embeddings + logistic regression ===", flush=True)
    for method in ("cbow", "skipgram"):
        vocabulary, vectors = train_word2vec(train_tokens, method, args.dimension, args.word2vec_epochs)
        train_mean = average_embeddings(train_tokens, vocabulary, vectors)
        test_mean = average_embeddings(test_tokens, vocabulary, vectors)
        new_mean = average_embeddings(new_tokens, vocabulary, vectors)
        linear_classifier(f"Word2Vec {method} + LR", train_mean, test_mean, new_mean)
        representations[f"word2vec_{method}_train"] = train_mean
        representations[f"word2vec_{method}_test"] = test_mean
        np.savez_compressed(output / f"word2vec_{method}.npz",
                            tokens=np.array(["<PAD>", "<UNK>"] + list(vocabulary)), vectors=vectors.numpy())
    # Reuse the last trained model (Skip-gram) for all three neural classifiers.
    train_sequences, train_lengths = encode_sequences(train_tokens, vocabulary, args.max_length)
    test_sequences, test_lengths = encode_sequences(test_tokens, vocabulary, args.max_length)
    new_sequences, new_lengths = encode_sequences(new_tokens, vocabulary, args.max_length)
    truncated_train = sum(len(tokens) > args.max_length for tokens in train_tokens)
    truncated_test = sum(len(tokens) > args.max_length for tokens in test_tokens)
    print(f"\nSequence length limit: {args.max_length}; truncated train/test: {truncated_train}/{truncated_test}", flush=True)
    for kind in ("cnn", "rnn", "lstm"):
        model = train_classifier(kind, vectors, train_sequences, train_lengths, y_train, args.epochs)
        predicted = predict_classifier(model, test_sequences, test_lengths)
        new_prediction = predict_classifier(model, new_sequences, new_lengths)[0]
        record(kind.upper(), args.dimension, predicted, new_prediction)
        torch.save({"kind": kind, "state_dict": model.state_dict(), "vocabulary": vocabulary,
                    "max_length": args.max_length, "dimension": args.dimension}, output / f"{kind}.pt")

    np.savez_compressed(output / "document_vectors.npz", train_ids=train_ids, test_ids=test_ids,
                        y_train=y_train, y_test=y_test, **representations)
    settings = {
        "topics": args.topics, "embedding_dimension": args.dimension,
        "word2vec_epochs": args.word2vec_epochs, "neural_epochs": args.epochs,
        "max_length": args.max_length, "threads": args.threads, "seed": 42,
        "tf": "absolute", "idf": "smoothed", "new_message": args.message,
        "python_torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__,
        "truncated_train": truncated_train, "truncated_test": truncated_test,
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"\nNew message: {args.message}")
    print(f"All results saved to: {output.resolve()}")


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    # Small text datasets can be slower when BLAS creates too many CPU threads.
    with threadpool_limits(limits=args.threads):
        run(args)


if __name__ == "__main__":
    main()
