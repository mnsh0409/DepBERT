import pandas as pd
import spacy
import numpy as np
from transformers import BertTokenizer

class Preprocessor:
    """
    Handles data loading, dependency parsing, and feature creation.
    """
    def __init__(self, model_name='bert-base-uncased', max_len=128):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.nlp = spacy.load("en_core_web_sm")
        self.max_len = max_len
        self.dep_tags = self._get_dep_tags()

    def _get_dep_tags(self):
        """Gets all possible dependency tags from SpaCy."""
        return list(self.nlp.get_pipe("parser").labels)

    def load_data(self, file_path):
        """Loads data from a CSV file."""
        return pd.read_csv(file_path)

    def get_dependency_features(self, text):
        """
        Extracts dependency features from a given text.
        Returns a co-occurrence matrix and a label encoding matrix.
        """
        doc = self.nlp(text)
        dep_map_co_occurrence = np.zeros((self.max_len, self.max_len), dtype=int)
        dep_map_label = np.zeros((self.max_len, self.max_len), dtype=int)
        
        dep_tag_to_id = {tag: i + 1 for i, tag in enumerate(self.dep_tags)}

        for token in doc:
            if token.i < self.max_len and token.head.i < self.max_len:
                # Co-occurrence
                dep_map_co_occurrence[token.i, token.head.i] = 1
                dep_map_co_occurrence[token.head.i, token.i] = 1
                
                # Label Encoding
                dep_id = dep_tag_to_id.get(token.dep_, 0)
                dep_map_label[token.i, token.head.i] = dep_id

        return dep_map_co_occurrence, dep_map_label

    def encode_text(self, text):
        """Encodes text using BERT tokenizer."""
        return self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
