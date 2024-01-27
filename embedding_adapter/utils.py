"""
label_generator.py

This module provides a LabelGenerator class that creates synthetic labels 
to train an embedding adaptor.
"""
import os
from typing import List, Optional, Any

from pydantic import BaseModel
from tqdm import tqdm 
from openai import OpenAI

LABEL_GENERATOR_SYS_PROMPT = "You are a helpful expert researcher and discussion faciliator. "\
                             "Your goal is to classify if the given statement is a possible answer to the given query. "\
                             "If yes, reply with 1. If no, reply with 0."

class LabelGenerator(BaseModel):
    """
    A class to generate synthetic labels using an LLM client.

    Attributes:
        sys_message (str): A system prompt for the LLM client.
        _llm_client (Optional[Any]): An instance of an LLM client.
        _synthetic_labels (Optional[List[int]]): A list of synthetic labels.
    """
    sys_message: str = LABEL_GENERATOR_SYS_PROMPT
    _llm_client: Optional[OpenAI] = None
    _synthetic_labels: Optional[Any] = None

    def __init__(self, **data) -> None:
        """Initializes the LabelGenerator and sets up the label generator client."""
        super().__init__(**data)
        self._set_up_label_generator()

    def _set_up_label_generator(self) -> None:
        """Sets up the label generator client by initializing the LLM client."""
        if "OPENAI_API_KEY" not in os.environ:
            raise OSError("OPENAI_API_KEY is not set")
        self._llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _evaluate(self, model: str, query: str, chunk: str) -> int:
        """
        Evaluates a given query and chunk to generate a synthetic label.

        Args:
            model (str): The name of the LLM model to use.
            query (str): The query string.
            chunk (str): The document chunk to evaluate.

        Returns:
            int: The synthetic label (-1 or 1).
        """
        prompt = f"<QUERY>{query}</QUERY> <STATEMENT>{chunk}</STATEMENT>"
        response = self._llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": self.sys_message},
                      {"role": "user", "content": prompt}],
            temperature=0,
            seed=1,
            max_tokens=1,
            logit_bias={"15": 100, # token id for 0
                        "16": 100  # token id for 1
                        }
        )
        evaluation = int(response.choices[0].message.content)
        # Map back to -1 vs 1
        if evaluation == 0:
            evaluation = -1
        return evaluation

    def run(self, queries: List[str], document_chunks: List[str], model: str = "gpt-3.5-turbo") -> List[int]:
        """
        Creates synthetic labels for a list of queries and document chunks.

        Args:
            queries (List[str]): A list of query strings.
            document_chunks (List[str]): A list of document chunk strings.
            model (str): The name of the LLM model to use.

        Returns:
            List[int]: A list of synthetic labels.

        Raises:
            ValueError: If the length of queries and document chunks does not match.
        """
        q_len = len(queries)
        c_len = len(document_chunks)
        if len(queries) != len(queries):
            raise ValueError(f"The length of the queries ({q_len}) must be the same as the length of the document chunks ({c_len})")
        self._synthetic_labels = []
        for example in tqdm(zip(queries, document_chunks), total=q_len):
            query = example[0]
            chunk = example[1]
            synthetic_label = self._evaluate(model, query, chunk)
            self._synthetic_labels.append(synthetic_label)
        return self._synthetic_labels
