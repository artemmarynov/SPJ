# Natural Language Processing Exercises

Python coursework for SPJ at TUKE. Each exercise has its own dependency files and usage instructions.

| Directory | Topics |
| --- | --- |
| [cv1-cv2](cv1-cv2/) | CoNLL-U reconstruction, regex tokenization, binary features, spam classification |
| [cv3-cv4](cv3-cv4/) | TF-IDF, retrieval, LSI, Word2Vec, CNN, RNN, LSTM |
| [cv5-cv6](cv5-cv6/) | Contextual word embeddings with pretrained DistilBERT |
| [cv7-cv8](cv7-cv8/) | Local RAG with Ollama and Slovak Wikipedia |

Use Python 3.14 and create a local environment:

```powershell
python -m venv .venv
```

Install the requirements for the exercise you want to run. See its README for commands and required inputs. RAG requires a running Ollama instance and downloads its datasets automatically.

Datasets, model weights, indexes, generated results, and Python caches are excluded from Git. Source code, dependency files, documentation, and tests are versioned.
