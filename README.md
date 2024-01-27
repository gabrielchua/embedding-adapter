# Embedding Adapter 💬 📐

[![PyPI version](https://img.shields.io/pypi/v/embedding-adapter.svg)](https://pypi.org/project/embedding-adapter/)

Finetune embedding models in just 4 lines of code.

# Quick Start ⚡
**Installation**
```bash
pip install embedding_adapter
```
**Usage**
```python
from embedding_adapter import EmbeddingAdapter
adapter = EmbeddingAdapter()
EmbeddingAdapter.fit(query_embeddings, document_embeddings, labels)
EmbeddingAdapter.transform(new_embeddings)
```

Once you've trained the adapter, you can use patch your pre-trained embedding model.

```python
patch = EmbeddingAdapter.patch()
adapted_embeddings = patch(original_embedding_fn("SAMPLE_TEXT"))
```

# Synthetic Label Generation 🧪
No user feedback to use as labels? 🤔 Create synthetic label the `LabelGenerator` util

```python
from embedding_adapter.utils import LabelGenerator
generator = LabelGenerator()
generator.run()
```

**Note:** This requires an OpenAI API key saved as an `OPENAI_API_KEY` env var.

# License 📄

This project is licensed under the MIT License.
