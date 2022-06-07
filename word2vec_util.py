import numpy as np
from collections import Counter
from typing import Tuple, Any, Optional
from nltk.corpus import stopwords


class TextCorpusProcess():
    def __init__(self, punct_dict: dict, prod_descs: list[str]) -> None:
        self.punct_dict = punct_dict
        self.prod_descs = prod_descs
        self.prod_word_lists = self.create_prod_word_lists()
        self.word_list = self.create_word_list()
        self.sorted_vocab = self.create_sorted_vocab()
        self.word_to_idx, self.idx_to_word = self.create_lookup_tables()
        self.word_idx_list = self.create_word_idx_list()
        self.prod_word_idx_lists = self.create_prod_word_idx_lists()
        self.freq_dict = self.create_freq_dict()
        self.vocab_size = len(self.freq_dict)
    
    def replace_punctuations(self, text: str) -> str:
        """Replaces punctuations in the given text with correspodning tags

        Args:
            text (str): given text

        Returns:
            str: process text
        """
        for punct, tag in self.punct_dict.items():
            text = text.replace(punct, ' ' + tag + ' ')
        return text

    def preprocess(self, text: str) -> str:
        """Processes text for word2vec training

        Args:
            text (str): given text

        Returns:
            str: process text
        """
        text = text.lower()
        text = self.replace_punctuations(text)
        words = text.split()
        return words

    def create_prod_word_lists(self) -> list[list[str]]:
        """Creates a list of word lists, each word list corresponds to
        a product description.

        Returns:
            list[list[str]]: list of prod description word lists
        """
        prod_word_lists = []
        for prod_desc in self.prod_descs:
            prod_words = self.preprocess(prod_desc)
            prod_words = [word for word in prod_words if not word
                     in stopwords.words('english')]
            prod_word_lists.append(prod_words)
        return prod_word_lists
    
    def create_word_list(self) -> list[str]:
        """Create a list of all words in the corpus

        Returns:
            list[str]: list of words
        """
        word_list = []
        for words in self.prod_word_lists:
            word_list.extend(words)
        return word_list

    def create_sorted_vocab(self) -> list[str]:
        """Creates a vocab list sorted based on frequency
        """
        word_counts = Counter(self.word_list)
        # sorting the words from most to least frequent in text occurrence
        return sorted(word_counts, key=word_counts.get, reverse=True)
    
        
    def create_lookup_tables(self) -> Tuple[dict, dict]:
        """Creates word2idx and idx2word lookup dicts

        Returns:
            Tuple[dict, dict]: word2idx, idx2word
        """
        # create idx2word dict
        idx_to_word = {ii: word for ii, word in enumerate(self.sorted_vocab)}
        word_to_idx = {word: ii for ii, word in idx_to_word.items()}

        return word_to_idx, idx_to_word

    def create_word_idx_list(self) -> list[int]:
        """Creates a list word indices from the list of actual words"""
        return [self.word_to_idx[word] for word in self.word_list]

    def create_freq_dict(self) -> dict:
        """Creates a dict that contains word frequecy"""
        word_idx_counts = Counter(self.word_idx_list)
        total_count = len(self.word_idx_list)
        freqs = {word: count/total_count for word,
                 count in word_idx_counts.items()}
        return freqs

    def create_prod_word_idx_lists(self) -> list[list[str]]:
        """Converts the prod list of word lists to its corresponding word indices

        Returns:
            list[list[str]]: list of prod description word index lists
        """
        prod_word_idx_lists = []
        for wordlist in self.prod_word_lists:
            prod_word_idx_lists.append(
                [self.word_to_idx[word] for word in wordlist])
        return prod_word_idx_lists
