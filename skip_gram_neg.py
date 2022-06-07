import torch
from torch import Tensor, nn
from typing import List

class SkipGramNeg(nn.Module):
    """The Skip-Gram negative samping model."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 100,
        noise_dist: List[float] = None,
        neg_sample_size: int = 5,
        batch_size: int = 64,
        ) -> None:
        """
        Args:
            vocab_size (int): size of vocabulary
            embed_dim (int, optional): embedding dimension. 
            Defaults to 100.
            noise_dist (list[float], optional): noise distribution.
            Defaults to None.
            neg_sample_size (int, optional): Number of neg samples per
            input sample. Defaults to 5.
            batch_size (int, optional): batch size. Defaults to 64.
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.noise_dist = noise_dist
        self.neg_sample_size = neg_sample_size
        self.batch_size = batch_size

        # define embedding layers for input and output words
        # initialize both embedding tables with uniform distribution
        self.in_embed = nn.Embedding(vocab_size, embed_dim)
        self.out_embed = nn.Embedding(vocab_size, embed_dim)

    def forward_input(self, input_words: List[int]) -> Tensor:
        # return input vector embeddings
        input_vectors = self.in_embed(input_words)
        return input_vectors

    def forward_output(self, output_words: List[int]) -> Tensor:
        # return output vector embeddings
        output_vectors = self.out_embed(output_words)
        return output_vectors

    def generate_neg_samples(
                             self, 
                             extended_batch_size,
                             exclude_words: List[int] = None, 
                             device: str = "cpu",
                             ) -> Tensor:
        """Generate noise vectors with shape (batch_size, neg_sample_size,
        embed_dim)

        Args:
            extended_batch_size (int): the total batch size
            exclude_words (list[int], optional): list of words to exclude.
            Defaults to None.
            device (str, optional): Defaults to "cpu".

        Returns:
            Tensor: negative sample vectors
        """
        if self.noise_dist is None:
            # sample words uniformly
            noise_dist = torch.ones(self.vocab_size)
        else:
            assert(self.vocab_size==len(self.noise_dist))
            noise_dist = self.noise_dist

        # sample words from the noise distribution
        # TODO: implement exclude list
        if exclude_words is None:
            neg_words = torch.multinomial(
                noise_dist,
                extended_batch_size * self.neg_sample_size,
                replacement=True)
        else:
            neg_samples = []
            num_samples = extended_batch_size * self.neg_sample_size
            while num_samples > 0:
                samples = torch.multinomial(
                    noise_dist,
                    num_samples,
                    replacement=True).tolist()
                samples = [word for word in samples if not word in exclude_words]
                neg_samples.extend(samples)
                num_samples -= len(samples)
            neg_words = torch.tensor(neg_samples)

        neg_words = neg_words.to(device)

        # reshape the embeddings so shape 
        # (batch_size, neg_sample_size, embed_dim)
        neg_vectors = self.out_embed(neg_words).view(
            extended_batch_size, self.neg_sample_size, self.embed_dim)

        return neg_vectors

class NegativeSamplingLoss(nn.Module):
    """A custom loss for the skip-gram negative sampling model"""
    def __init__(self):
        super().__init__()

    def forward(self, 
                input_vectors: Tensor, 
                output_vectors: Tensor, 
                neg_vectors: Tensor,
                ):

        batch_size, embed_dim = input_vectors.shape

        # input vectors as a batch of column vectors
        input_vectors = input_vectors.view(batch_size, embed_dim, 1)

        # output vectors as a batch of row vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_dim)

        # bmm = batch matrix multiplication
        # positive sample loss
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()

        # negative sample loss
        neg_loss = torch.bmm(neg_vectors.neg(),
                               input_vectors).sigmoid().log()
        # sum the losses over the sample of noise vectors
        neg_loss = neg_loss.squeeze().sum(1)

        # return average batch loss
        return -(out_loss + neg_loss).mean()
