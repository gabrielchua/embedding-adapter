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
adapter.fit(query_embeddings, document_embeddings, labels)
adapter.transform(new_embeddings)
```

Once you've trained the adapter, you can use patch your pre-trained embedding model.

```python
patch = adapter.patch()
adapted_embeddings = patch(original_embedding_fn("SAMPLE_TEXT"))
```

# Use Cases/Why do I need to tune my embeddings ❓

Embeddings are predominantly utilized for Retrieval Augmented Generation (RAG) or semantic search applications. However, their effectiveness can significantly vary depending on the context. This is where the need for tuning comes into play.

Consider training an adaptor for your pre-trained embedding model, such as OpenAI's text-embedding-3-small or the open-source gte-large. This customization enables your model to interpret tokens accurately within the specific context of your application. For example, the word "Pandas" 🐼 could refer to the animal or the widely used Python library for data manipulation. Without tuning, your model may not distinguish between these vastly different contexts.

Moreover, tuning your embeddings is crucial if you aim to utilize a smaller model—perhaps due to hardware constraints like the absence of GPUs for inference. In such cases, an adaptor can enhance retrieval performance, ensuring efficiency without compromising on accuracy. 


# Synthetic Label Generation 🧪
No user feedback to use as labels? 🤔 Create synthetic labels with the `LabelGenerator` util

```python
from embedding_adapter.utils import LabelGenerator
generator = LabelGenerator()
generator.run()
```

**Note:** This requires an OpenAI API key saved as an `OPENAI_API_KEY` env var.

# License 📄

This project is licensed under the MIT License.
