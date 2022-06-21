import torch.nn as nn

class TextClassification(nn.Module):
    def __init__(self, word_embd_dim, num_classes, word_kernel_size=2):
        super().__init__()
        self.word_embd_dim = word_embd_dim
        self.word_kernel_size = word_kernel_size
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=1,
                      out_channels=48,
                      kernel_size=self.word_kernel_size * self.word_embd_dim,
                      stride=self.word_embd_dim,
                      padding=(self.word_kernel_size - 1) * self.word_embd_dim),
            # nn.ReLU(),
            nn.Tanh(),
            # nn.MaxPool1d(3, 3),
            # nn.Conv1d(in_channels=32,
            #           out_channels=8,
            #           kernel_size=2,
            #           stride=1,
            #           padding=1),
            # nn.ReLU(),
            # nn.MaxPool1d(3, 3),
            nn.Flatten(),
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(in_features=63408, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, inputs):
        x = self.cnn_layers(inputs)
        return self.linear_layer(x)
