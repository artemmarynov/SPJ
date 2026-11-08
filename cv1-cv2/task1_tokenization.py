"""Task 1: CoNLL-U -> text -> custom tokens -> comparison."""

import re
from pathlib import Path


# A word (including Slovak letters and numbers) or one punctuation mark.
TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]")


def tokenize(text):
    return TOKEN_PATTERN.findall(text)


def read_conllu(path):
    """Read surface tokens as (FORM, MISC) pairs for each sentence."""
    sentences = []
    tokens = []
    original_text = None
    covered_until = 0

    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\r\n")
            if not line:
                if tokens:
                    sentences.append((tokens, original_text))
                tokens, original_text, covered_until = [], None, 0
            elif line.startswith("# text = "):
                # Use this comment only to check the reconstructed text.
                original_text = line[len("# text = "):]
            elif not line.startswith("#"):
                columns = line.split("\t")
                token_id, form, misc = columns[0], columns[1], columns[9]
                if "." in token_id:
                    continue  # For example, 13.1: an empty node absent from the text.
                if "-" in token_id:
                    # For example, 10-11 = aby: use the whole token, not its parts.
                    covered_until = int(token_id.split("-")[1])
                elif int(token_id) <= covered_until:
                    continue
                tokens.append((form, misc))

    if tokens:  # Keep the last sentence even without a trailing blank line.
        sentences.append((tokens, original_text))
    return sentences


def reconstruct(tokens):
    """Restore spaces and record the original token boundaries."""
    text = ""
    spans = set()
    for index, (form, misc) in enumerate(tokens):
        start = len(text)
        text += form
        spans.add((start, len(text)))
        if index < len(tokens) - 1 and "SpaceAfter=No" not in misc.split("|"):
            text += " "
    return text, spans


def main(path):
    sentences = read_conllu(path)

    texts = []
    total_gold = total_predicted = correct = checked_texts = correct_texts = 0

    for tokens, original_text in sentences:
        text, gold_spans = reconstruct(tokens)
        predicted_spans = {match.span() for match in TOKEN_PATTERN.finditer(text)}
        matches = len(gold_spans & predicted_spans)
        total_gold += len(gold_spans)
        total_predicted += len(predicted_spans)
        correct += matches
        texts.append(text)

        if original_text is not None:
            checked_texts += 1
            correct_texts += text == original_text

    output = path.with_name(path.stem + "_reconstructed.txt")
    # Join sentences with one space; preserve spaces within each sentence.
    output.write_text(" ".join(texts), encoding="utf-8")

    print(f"\n=== {path.name} ===")
    print(f"Sentences: {len(sentences)}")
    print(f"Reconstruction matches # text: {correct_texts}/{checked_texts}")
    print(f"Original surface tokens: {total_gold}")
    print(f"Our tokens: {total_predicted}")
    print(f"Correct tokens (same character boundaries): {correct}")
    print(f"Correct / original: {correct / total_gold:.2%}")
    print(f"Reconstructed text saved to: {output.name}")

if __name__ == "__main__":
    folder = Path(__file__).resolve().parent
    for filename in ("sk_snk-ud-dev.conllu", "sk_snk-ud-test.conllu"):
        main(folder / filename)
