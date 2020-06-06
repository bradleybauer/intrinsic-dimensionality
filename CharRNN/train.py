import torch
import torch.nn as nn
from torch.autograd import Variable
import argparse
import os

from helpers import *
from model import *
from generate import *

# Parse command line arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str)
argparser.add_argument('--model', type=str, default="lstm")
argparser.add_argument('--n_epochs', type=int, default=40000)
argparser.add_argument('--print_every', type=int, default=100)
argparser.add_argument('--hidden_size', type=int, default=1024)
argparser.add_argument('--n_layers', type=int, default=3)
argparser.add_argument('--learning_rate', type=float, default=0.01)
argparser.add_argument('--chunk_len', type=int, default=200)
argparser.add_argument('--batch_size', type=int, default=100)
argparser.add_argument('--shuffle', action='store_true')
argparser.add_argument('--load', type=str, default='')
args = argparser.parse_args()


def random_training_set(chunk_len, batch_size, file, file_len):
    inp = torch.LongTensor(batch_size, chunk_len)
    target = torch.LongTensor(batch_size, chunk_len)
    for bi in range(batch_size):
        start_index = random.randint(0, file_len - chunk_len)
        end_index = start_index + chunk_len + 1
        chunk = file[start_index:end_index]
        inp[bi] = charTensor(chunk[:-1])
        target[bi] = charTensor(chunk[1:])
    inp = Variable(inp)
    target = Variable(target)
    inp = inp.cuda()
    target = target.cuda()
    return inp, target


def train(inp, target):
    # a,b = decoder.init_hidden(args.batch_size)
    # a = a.cuda()
    # b = b.cuda()
    # hidden = (a,b)
    hidden = None
    decoder.zero_grad()
    loss = 0

    for c in range(args.chunk_len):
        output, hidden = decoder(inp[:, c], hidden)
        loss += criterion(output.view(args.batch_size, -1), target[:, c])

    loss.backward()
    decoder_optimizer.step()

    return float(loss) / args.chunk_len


def save():
    save_filename = os.path.splitext(os.path.basename(args.filename))[0] + '.pt'
    torch.save([decoder, decoder_optimizer], save_filename)
    print('Saved as %s' % save_filename)


# Initialize models and start training

decoder = CharRNN(
    n_characters,
    args.hidden_size,
    n_characters,
    model=args.model,
    n_layers=args.n_layers,
)
print(decoder)


def countParams(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print('Number of trainable parameters:', countParams(decoder))
print('Trainable parameters MB:', countParams(decoder) * 4 / (1024 ** 2))
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.learning_rate)
if args.load:
    print("Loading Model")
    [decoder, decoder_optimizer] = torch.load(args.load)
criterion = nn.CrossEntropyLoss()

decoder.cuda()

start = time.time()
all_losses = []
loss_avg = 0

print("Loading Data")
file, file_len = readFile(args.filename)

print("Training for %d epochs..." % args.n_epochs)
epoch = 1
while True:
    loss = train(*random_training_set(args.chunk_len, args.batch_size, file, file_len))
    loss_avg += loss

    if epoch % args.print_every == 0:
        print('[', time_since(start), ',', epoch, ',', loss, ']')
        print(generate(decoder, 'Wh', 100), '\n')
        print('Loss Average:', loss_avg / epoch)
        save()
    epoch += 1
