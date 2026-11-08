"""Compare a word's contextual and fixed token vectors across sentences."""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer


FOLDER = Path(__file__).resolve().parent
MODEL = "distilbert/distilbert-base-uncased"
REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"


def find_word(sentence, word):
    """Require one complete occurrence so the target word is unambiguous."""
    matches = list(re.finditer(r"(?<!\w)" + re.escape(word) + r"(?!\w)", sentence, re.IGNORECASE))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one occurrence of {word!r}: {sentence!r}")
    return matches[0].span()


def target_positions(offsets, span):
    """Select every subtoken overlapping the target's character span."""
    start, end = span
    positions = [index for index, (left, right) in enumerate(offsets)
                 if left < end and right > start]
    if not positions or offsets[positions[0]][0] > start or offsets[positions[-1]][1] < end:
        raise ValueError("The target word is missing or truncated. Use a shorter sentence.")
    return positions


def cosine_matrix(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    return np.clip(normalized @ normalized.T, -1, 1)


def save_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_matrix(name, matrix):
    print(f"\n{name}")
    print("     " + " ".join(f"{'S' + str(index + 1):>6}" for index in range(len(matrix))))
    for index, row in enumerate(matrix, start=1):
        print(f"S{index:<3} " + " ".join(f"{value:6.3f}" for value in row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, default=FOLDER / "examples.json")
    parser.add_argument("--output", type=Path, default=FOLDER / "results")
    parser.add_argument("--offline", action="store_true", help="Use already downloaded model files only.")
    args = parser.parse_args()
    data = json.loads(args.examples.read_text(encoding="utf-8"))
    word, examples = data["word"], data["examples"]
    if not word.strip() or len(examples) < 2:
        raise ValueError("Provide a target word and at least two example sentences.")
    sentences = [row["sentence"] for row in examples]
    spans = [find_word(sentence, word) for sentence in sentences]
    torch.set_num_threads(2)

    print(f"Model: {MODEL}\nTarget word: {word}\nLoading pretrained weights...", flush=True)
    options = {"revision": REVISION, "cache_dir": str(FOLDER / ".model_cache"),
               "local_files_only": args.offline}
    tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True, **options)
    model = AutoModel.from_pretrained(MODEL, use_safetensors=True, **options).eval()
    encoded = tokenizer(sentences, padding=True, truncation=True, max_length=128,
                        return_offsets_mapping=True, return_tensors="pt")
    offsets = encoded.pop("offset_mapping").tolist()
    positions = [target_positions(row, span) for row, span in zip(offsets, spans)]

    with torch.inference_mode():
        # These lookup vectors do not yet include positions or sentence context.
        input_vectors = model.get_input_embeddings()(encoded["input_ids"])
        # The final encoder layer incorporates the entire unmasked sentence.
        contextual = model(**encoded).last_hidden_state
        fixed_vectors = np.stack([input_vectors[index, selected].mean(0).numpy()
                                  for index, selected in enumerate(positions)])
        word_vectors = np.stack([contextual[index, selected].mean(0).numpy()
                                 for index, selected in enumerate(positions)])

    fixed_cosine = cosine_matrix(fixed_vectors)
    contextual_cosine = cosine_matrix(word_vectors)
    args.output.mkdir(parents=True, exist_ok=True)
    details = []
    print(f"Vector size: {word_vectors.shape[1]}")
    for index, (example, selected) in enumerate(zip(examples, positions)):
        ids = encoded["input_ids"][index, selected].tolist()
        subtokens = tokenizer.convert_ids_to_tokens(ids)
        details.append({"id": f"S{index + 1}", "sense": example["sense"],
                        "sentence": example["sentence"], "subtokens": " ".join(subtokens),
                        "token_positions": " ".join(map(str, selected))})
        print(f"S{index + 1} [{example['sense']}] {example['sentence']}")
        print(f"   Target subtokens: {subtokens}")
    save_csv(args.output / "sentences.csv", details)

    for name, matrix in [("fixed_cosine", fixed_cosine), ("contextual_cosine", contextual_cosine)]:
        rows = [{"id": f"S{index + 1}", **{f"S{column + 1}": float(value)
                 for column, value in enumerate(row)}} for index, row in enumerate(matrix)]
        save_csv(args.output / f"{name}.csv", rows)
        print_matrix(name, matrix)

    pairs = []
    for left in range(len(examples)):
        for right in range(left + 1, len(examples)):
            pairs.append({"left": f"S{left + 1}", "right": f"S{right + 1}",
                          "same_sense": examples[left]["sense"] == examples[right]["sense"],
                          "fixed_cosine": float(fixed_cosine[left, right]),
                          "contextual_cosine": float(contextual_cosine[left, right])})
    save_csv(args.output / "pairs.csv", pairs)
    print("\nMean pairwise cosine (self-comparisons excluded):")
    for same, label in [(True, "Same sense"), (False, "Different senses")]:
        selected = [row["contextual_cosine"] for row in pairs if row["same_sense"] == same]
        if selected:
            print(f"{label}: {np.mean(selected):.4f} ({len(selected)} pairs)")

    np.savez_compressed(args.output / "word_vectors.npz", fixed=fixed_vectors,
                        contextual=word_vectors, sentences=np.array(sentences))
    metadata = {"model": MODEL, "revision": REVISION, "word": word,
                "dimensions": word_vectors.shape[1], "examples": examples,
                "layer": "last_hidden_state", "pooling": "mean over target word subtokens",
                "torch": torch.__version__, "transformers": transformers.__version__,
                "numpy": np.__version__}
    (args.output / "settings.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
