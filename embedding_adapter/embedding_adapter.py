"""
embedding_adapter.py

This module provides a class for training an embedding adapter to fine-tune a pre-trained embedding model to
a specific document and retrieval task.
"""
from typing import Optional, Any

from pydantic import BaseModel
import numpy as np
import torch
from tqdm import tqdm

class EmbeddingAdapter(BaseModel):
    """
    A class for adapting embeddings to improve query-document matching.

    Attributes:
        _adapter (Optional[np.ndarray]): The adapter matrix.
        _embedding_len (Optional[int]): The length of the embeddings the adapter is trained on.
        _dataset (Optional[TensorDataset]): The dataset used for training the adapter.
    """
    _adapter: Optional[Any] = None
    _embedding_len: Optional[int] = None
    _dataset: Optional[Any] = None

    def _model(self, query_embedding, document_embedding, adapter_matrix):
        """
        The model function that applies the adapter matrix to the query embedding.

        Args:
            query_embedding (torch.Tensor): The query embedding.
            document_embedding (torch.Tensor): The document embedding.
            adapter_matrix (torch.Tensor): The adapter matrix.

        Returns:
            torch.Tensor: The cosine similarity between the updated query embedding and the document embedding.
        """
        updated_query_embedding = torch.matmul(adapter_matrix, query_embedding)
        return torch.cosine_similarity(updated_query_embedding, document_embedding, dim=0)

    def _mse_loss(self, query_embedding, document_embedding, adapter_matrix, label):
        """
        Calculates the mean squared error loss for the adapter training.

        Args:
            query_embedding (torch.Tensor): The query embedding.
            document_embedding (torch.Tensor): The document embedding.
            adapter_matrix (torch.Tensor): The adapter matrix.
            label (torch.Tensor): The target label.

        Returns:
            torch.Tensor: The mean squared error loss.
        """
        return torch.nn.MSELoss()(self._model(query_embedding, document_embedding, adapter_matrix), label)

    def fit(self, query_embedding, document_embedding, label):
        """
        Trains the adapter matrix using the provided query embeddings, document embeddings, and labels.

        Args:
            query_embeddings (np.ndarray): An array of query embeddings.
            document_embeddings (np.ndarray): An array of document embeddings.
            labels (np.ndarray): An array of labels.

        Raises:
            ValueError: If the input arrays have mismatched lengths.
        """
        # Prepare the dataset
        adapter_query_embeddings = torch.Tensor(np.array(query_embedding))
        adapter_doc_embeddings = torch.Tensor(np.array(document_embedding))
        adapter_labels = torch.Tensor(np.expand_dims(np.array(label),1))
        self._dataset = torch.utils.data.TensorDataset(adapter_query_embeddings, adapter_doc_embeddings, adapter_labels)

        # Initialize the training parameters
        min_loss = float('inf')
        best_matrix = None

        # Initialize the adapter matrix
        self._embedding_len = len(adapter_query_embeddings[0])
        adapter_matrix = torch.randn(self._embedding_len, self._embedding_len, requires_grad=True)

        # Train the adapter matrix
        for epoch in tqdm(range(100)):
            for query_embedding, document_embedding, label in self._dataset:
                loss = self._mse_loss(query_embedding, document_embedding, adapter_matrix, label)

                if loss < min_loss:
                    min_loss = loss
                    best_matrix = adapter_matrix.clone().detach().numpy()

                loss.backward()
                with torch.no_grad():
                    adapter_matrix -= 0.01 * adapter_matrix.grad
                    adapter_matrix.grad.zero_()

        print(f"Best loss: {min_loss.detach().numpy()}")

        self._adapter = best_matrix

    def transform(self, transform_embedding):
        """
        Transforms the given query embeddings using the trained adapter matrix.

        Args:
            transform_embedding (np.ndarray): The query embeddings to be transformed.

        Returns:
            np.ndarray: The transformed query embeddings.

        Raises:
            RuntimeError: If the adapter is not trained.
            ValueError: If the length of the embedding to transform does not match the adapter.
        """
        if self._adapter is None:
            raise RuntimeError("Adapter not trained")
        input_len = len(transform_embedding)
        if input_len != self._embedding_len:
            raise ValueError(f"Embedding to transform is of length {input_len}. Adapter was trained for embeddings of length {self._embedding_len}.")
        return np.matmul(self._adapter, np.array(transform_embedding).T).T

    def patch(self):
        """
        Returns a function that wraps around the original embedding function to apply the transformation.

        Returns:
            Any: A function that transforms embeddings using the trained adapter.
        """
        return self.transform
