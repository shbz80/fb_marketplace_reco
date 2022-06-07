import numpy as np
from collections import Counter
from typing import Tuple, Any, Optional, List
from nltk.corpus import stopwords


class TextCorpusProcess():
    def __init__(self, punct_dict: dict, prod_descs: List[str]) -> None:
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

    def create_prod_word_lists(self) -> List[List[str]]:
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
    
    def create_word_list(self) -> List[str]:
        """Create a list of all words in the corpus

        Returns:
            list[str]: list of words
        """
        word_list = []
        for words in self.prod_word_lists:
            word_list.extend(words)
        return word_list

    def create_sorted_vocab(self) -> List[str]:
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

    def create_word_idx_list(self) -> List[int]:
        """Creates a list word indices from the list of actual words"""
        return [self.word_to_idx[word] for word in self.word_list]

    def create_freq_dict(self) -> dict:
        """Creates a dict that contains word frequecy"""
        word_idx_counts = Counter(self.word_idx_list)
        total_count = len(self.word_idx_list)
        freqs = {word: count/total_count for word,
                 count in word_idx_counts.items()}
        return freqs

    def create_prod_word_idx_lists(self) -> List[List[str]]:
        """Converts the prod list of word lists to its corresponding word indices

        Returns:
            list[list[str]]: list of prod description word index lists
        """
        prod_word_idx_lists = []
        for wordlist in self.prod_word_lists:
            prod_word_idx_lists.append(
                [self.word_to_idx[word] for word in wordlist])
        return prod_word_idx_lists


class Word2VecDataloader():
    """Creates a dataloader generator."""
    
    def __init__(
        self, 
        prod_word_idx_lists: List[List[str]],
        batch_size: int = 64, 
        window_size: int = 5) -> None:
        """
        Args:
            prod_word_idx_lists (list[list[str]]): list of word index lists
            batch_size (int, optional): number of input words in batch.
            Defaults to 64.
            window_size (int, optional): window size for context words.
            Defaults to 5.
        """
        self.prod_word_idx_lists = prod_word_idx_lists
        self.batch_size = batch_size
        self.window_size = window_size

    def generate_batch(self):
        """Generates batches

        Yields:
            input, target: the input target pair of a batch
        """
        sample_count = 0
        inputs = []
        targets = []
        # loop through each product description
        for words_idx in self.prod_word_idx_lists:
            words_len = len(words_idx)
            start_idx = 0
            # loops through the current product description in batches
            while start_idx < words_len:
                if start_idx + self.batch_size > words_len:
                    end_idx = words_len
                else:
                    end_idx = start_idx + self.batch_size
                # extract the current batch of input words
                words_batch = words_idx[start_idx:end_idx]
                # iterate through all words in the batch
                for i, word in enumerate(words_batch):
                    # get context words for the current input word
                    # and set them as targets
                    word_targets = self.get_target(words_batch, i)
                    targets.extend(word_targets)
                    inputs.extend([word]*len(word_targets))
                sample_count += len(words_batch)
                # yield if the sample count in the current batch staisfies
                # the batch size. If not, go to the next batch in the current
                # prod desc. If the current prod desc is too small for the batch,
                # go to the next prod desc.
                if sample_count >= self.batch_size:
                    yield inputs, targets
                    inputs = []
                    targets = []
                    sample_count = 0
                start_idx = end_idx

    def get_target(self, words: List[str], idx: int) -> List[str]:
        """Creates a list of target context words within a given word list

        Args:
            words (list[str]): the given wordlist
            idx (int): index of the chosen input word

        Returns:
            list[str]: a list of target context words
        """
        w = np.random.randint(1, self.window_size + 1)
        start = idx - w if (idx - w) > 0 else 0
        stop = idx + w if (idx + w) < len(words) else len(words)
        target_words = words[start:idx] + words[idx+1:stop+1]

        return list(target_words)

