"""Word2Vec with negative sampling and small CNN/RNN/LSTM classifiers."""

from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, TensorDataset


class Word2Vec(nn.Module):
    """Standard CBOW/Skip-gram objective with negative sampling."""

    def __init__(self, vocabulary_size, dimension):
        super().__init__()
        self.input = nn.Embedding(vocabulary_size, dimension, padding_idx=0)
        self.output = nn.Embedding(vocabulary_size, dimension)
        nn.init.uniform_(self.input.weight, -0.5 / dimension, 0.5 / dimension)
        nn.init.zeros_(self.output.weight)
        with torch.no_grad():
            self.input.weight[0].zero_()

    def forward(self, contexts, targets, negatives):
        # CBOW averages context words; Skip-gram supplies a single center word.
        mask = contexts.ne(0).unsqueeze(-1)
        hidden = (self.input(contexts) * mask).sum(1) / mask.sum(1).clamp(min=1)
        positive = (hidden * self.output(targets)).sum(1)
        negative = (self.output(negatives) * hidden.unsqueeze(1)).sum(2)
        # A sampled true target is not treated as a negative example.
        negative_loss = F.logsigmoid(-negative).masked_fill(negatives.eq(targets[:, None]), 0)
        return -(F.logsigmoid(positive) + negative_loss.sum(1)).mean()


def word2vec_examples(documents, vocabulary, method, window=2):
    """Build training examples within documents, never across their boundaries."""
    contexts, targets = [], []
    for document in documents:
        ids = [vocabulary[token] for token in document if token in vocabulary]
        for position, center in enumerate(ids):
            neighbors = ids[max(0, position - window):position] + ids[position + 1:position + window + 1]
            if not neighbors:
                continue
            if method == "cbow":
                contexts.append(neighbors + [0] * (2 * window - len(neighbors)))
                targets.append(center)
            elif method == "skipgram":
                for neighbor in neighbors:
                    contexts.append([center])
                    targets.append(neighbor)
            else:
                raise ValueError(f"Unknown Word2Vec method: {method}")
    if not targets:
        raise ValueError("No Word2Vec context pairs. Use more training text.")
    return torch.tensor(contexts, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def train_word2vec(documents, method, dimension=64, epochs=5, seed=42):
    torch.manual_seed(seed)
    counts = Counter(token for document in documents for token in document)
    # Reserve 0 for padding and 1 for unknown words; learn words seen at least twice.
    vocabulary = {token: index + 2 for index, (token, count) in
                  enumerate((item for item in counts.items() if item[1] >= 2))}
    contexts, targets = word2vec_examples(documents, vocabulary, method)
    size = len(vocabulary) + 2
    distribution = torch.zeros(size)
    for token, index in vocabulary.items():
        distribution[index] = counts[token] ** 0.75
    distribution /= distribution.sum()

    model = Word2Vec(size, dimension)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    batch_size = 1024
    print(f"\nWord2Vec {method}: {len(vocabulary)} words, {len(targets)} examples", flush=True)
    for epoch in range(epochs):
        total_loss = 0.0
        order = torch.randperm(len(targets))
        for start in range(0, len(order), batch_size):
            indices = order[start:start + batch_size]
            negatives = torch.multinomial(distribution, len(indices) * 5, replacement=True).reshape(-1, 5)
            loss = model(contexts[indices], targets[indices], negatives)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(indices)
        print(f"  epoch {epoch + 1}/{epochs}: loss={total_loss / len(targets):.4f}", flush=True)

    vectors = model.input.weight.detach().clone()
    # Give unseen words a neutral starting vector for the supervised networks.
    vectors[1] = vectors[2:].mean(0)
    return vocabulary, vectors


def average_embeddings(documents, vocabulary, vectors):
    """Represent each document by the mean of its known word vectors."""
    matrix = vectors.numpy()
    result = np.zeros((len(documents), matrix.shape[1]), dtype=np.float32)
    for index, document in enumerate(documents):
        ids = [vocabulary[token] for token in document if token in vocabulary]
        if ids:
            result[index] = matrix[ids].mean(0)
    return result


def encode_sequences(documents, vocabulary, max_length):
    """Preserve word order, truncate long texts, and pad short texts on the right."""
    sequences = torch.zeros((len(documents), max_length), dtype=torch.long)
    lengths = torch.zeros(len(documents), dtype=torch.long)
    for index, document in enumerate(documents):
        ids = [vocabulary.get(token, 1) for token in document[:max_length]] or [1]
        sequences[index, :len(ids)] = torch.tensor(ids)
        lengths[index] = len(ids)
    return sequences, lengths


class TextClassifier(nn.Module):
    def __init__(self, kind, vectors, hidden_size=64):
        super().__init__()
        self.kind = kind
        # Start from trained Word2Vec vectors, then fine-tune on labeled train data.
        self.embedding = nn.Embedding.from_pretrained(vectors.clone(), freeze=False, padding_idx=0)
        dimension = vectors.shape[1]
        if kind == "cnn":
            self.encoder = nn.Conv1d(dimension, hidden_size, kernel_size=3, padding=1)
        elif kind == "rnn":
            self.encoder = nn.RNN(dimension, hidden_size, batch_first=True)
        elif kind == "lstm":
            self.encoder = nn.LSTM(dimension, hidden_size, batch_first=True)
        else:
            raise ValueError(f"Unknown classifier: {kind}")
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, sequences, lengths):
        embedded = self.embedding(sequences)
        if self.kind == "cnn":
            features = F.relu(self.encoder(embedded.transpose(1, 2)))
            # Padding must not win global max pooling.
            features = features.masked_fill(sequences.eq(0).unsqueeze(1), -1e9)
            hidden = features.amax(dim=2)
        else:
            # Packing prevents padding from changing the final recurrent state.
            packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, hidden = self.encoder(packed)
            if self.kind == "lstm":
                hidden = hidden[0]  # LSTM returns both hidden and cell states.
            hidden = hidden[-1]
        return self.output(self.dropout(hidden)).squeeze(1)


def train_classifier(kind, vectors, sequences, lengths, labels, epochs=8, seed=42):
    torch.manual_seed(seed)
    model = TextClassifier(kind, vectors)
    dataset = TensorDataset(sequences, lengths, torch.tensor(labels, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    print(f"\nTraining {kind.upper()}", flush=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch, batch_lengths, target in loader:
            loss = criterion(model(batch, batch_lengths), target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()
            total_loss += loss.item() * len(target)
        print(f"  epoch {epoch + 1}/{epochs}: loss={total_loss / len(dataset):.4f}", flush=True)
    return model


def predict_classifier(model, sequences, lengths):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(sequences), 256):
            logits = model(sequences[start:start + 256], lengths[start:start + 256])
            predictions.extend(logits.ge(0).int().tolist())
    return np.array(predictions)
