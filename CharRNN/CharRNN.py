import torch.nn as nn
import string


class CharRNN(nn.Module):
    all_characters = string.printable
    n_characters = len(all_characters)

    def __init__(self, input_size=n_characters, hidden_size=1024, output_size=n_characters, n_layers=2):
        super(CharRNN, self).__init__()
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=.3)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden

    def embed(self, input, num_hidden_to_keep):
        encoded = self.encoder(input).transpose(0, 1)
        outputs, _ = self.rnn(encoded)
        seq_len = outputs.shape[0]
        return outputs[:num_hidden_to_keep].cpu().numpy()
