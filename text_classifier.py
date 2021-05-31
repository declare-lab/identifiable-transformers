import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

import torchtext
from torchtext.data.utils import get_tokenizer

import random
from collections import Counter
import pandas as pd


'''
    user-specifications
'''
import argparse
parser = argparse.ArgumentParser(description='Short sample app')

parser.add_argument('-dataset', action="store", type=str, default="ag_news")
parser.add_argument('-kdim', action="store", type=int, default=16)
parser.add_argument('-nhead', action="store", type=int, default=4)
parser.add_argument('-concat', action="store", type=bool, default=True)
parser.add_argument('-embedim', action="store", type=int, default=32)
parser.add_argument('-batch', action="store", type=int, default=64)
parser.add_argument('-epochs', action="store", type=int, default=10)
parser.add_argument('-lr', action="store", type=float, default=0.001)
parser.add_argument('-vocab_size', action="store", type=int, default=100000)
parser.add_argument('-max_text_len', action="store", type=int, default=512)
parser.add_argument('-valid_frac', action="store", type=float, default=0.3)

config = parser.parse_args()
print("\n\n->Configurations are:")
[print(k,": ",v) for (k,v) in vars(config).items()]


#GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cuda':
    print("\n*--running on GPU!--*")
else:
    print("\n*--can not find GPU, running on CPU!--*")

#decide main factors
BATCH_SIZE = config.batch
N_HEAD = config.nhead
KDIM = config.kdim
CONCAT_HEADS = config.concat
EMBEDDING_DIM = config.embedim

if CONCAT_HEADS:
    VDIM = EMBEDDING_DIM // N_HEAD
else:
    VDIM = EMBEDDING_DIM


''' 
    load data
'''
if config.dataset == "ag_news":
    train_dataset, test_dataset = torchtext.datasets.AG_NEWS(root='.data', split=('train', 'test'))
elif config.dataset == "imdb":
    train_dataset, test_dataset = torchtext.datasets.IMDB(root='.data', split=('train', 'test'))
elif config.dataset == "sogou":
    train_dataset, test_dataset = torchtext.datasets.SogouNews(root='.data', split=('train', 'test'))
elif config.dataset == "yelp_p":
    train_dataset, test_dataset = torchtext.datasets.YelpReviewPolarity(root='.data', split=('train', 'test'))
elif config.dataset == "yelp_f":
    train_dataset, test_dataset = torchtext.datasets.YelpReviewFull(root='.data', split=('train', 'test'))
elif config.dataset == "amazon_p":
    train_dataset, test_dataset = torchtext.datasets.AmazonReviewPolarity(root='.data', split=('train', 'test'))
elif config.dataset == "amazon_f":
    train_dataset, test_dataset = torchtext.datasets.AmazonReviewFull(root='.data', split=('train', 'test'))
elif config.dataset == "yahoo":
    train_dataset, test_dataset = torchtext.datasets.YahooAnswers(root='.data', split=('train', 'test'))
elif config.dataset == "dbpedia":
    train_dataset, test_dataset = torchtext.datasets.DBpedia(root='.data', split=('train', 'test'))
else:
    data = pd.read_csv(config.dataset)

    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=0.3)
    train_text = train_data['text'].values.tolist()
    train_labels = train_data['label'].values.tolist()
    train_dataset = [([train_labels[i], train_text[i]]) for i in range(len(train_text))]
    test_text = test_data['text'].values.tolist()
    test_labels = test_data['label'].values.tolist()
    test_dataset = [([test_labels[i], test_text[i]]) for i in range(len(test_text))]


'''
    process data
'''

print("\n->BATCH_SIZE: ", BATCH_SIZE)

# since above retuns iterators
if type(train_dataset) != list:
    train_dataset = list(train_dataset)

if type(test_dataset) != list:
    test_dataset = list(test_dataset)



# mapping labels to integers
OUTPUT_LABELS = set([label for (label, text) in train_dataset])

lab2int = {}
count = 0
for lab in OUTPUT_LABELS:
    lab2int[lab] = count
    count += 1

print("\n->Labels are: {}".format(OUTPUT_LABELS))
print("\nlabel to id mapping: {}".format(lab2int))

train_dataset = [(lab2int[label], text) for (label, text) in train_dataset]
random.shuffle(train_dataset)

test_dataset = [(lab2int[label], text) for (label, text) in test_dataset]
random.shuffle(test_dataset)


# split train-valid-test
valid_frac = config.valid_frac
train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset, 
                                                                [int(round(len(train_dataset)*(1-valid_frac),0)), 
                                                                int(round(len(train_dataset)*valid_frac,0))])

print("\n->Train set: {}, Valid set: {}, Test set: {}".format(len(train_dataset), len(valid_dataset), len(test_dataset)))


# tokenizer type
tokenizer = get_tokenizer("basic_english")

# vocab
counter = Counter()
for (label, line) in train_dataset:
    counter.update(tokenizer(line))

# vocab length is max_size + 2 for special tokens
max_vocab_size = config.vocab_size
vocab = torchtext.vocab.Vocab(counter, max_size=max_vocab_size, specials=('<unk>', '<pad>', '<cls>', '<sep>'), specials_first=True)

# example: text_pipeline('here is the an example') --> [0, 22, 3, 31, 0]
text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]

# padding function
padding_pipeline = lambda x: x[0] + [vocab.stoi['<pad>'] for p in range(x[1] - len(x[0]))]

# special token append
special_token_pipeline = lambda x: [[vocab.stoi['<cls>']] + p + [vocab.stoi['<sep>']] for p in x]

MAX_LEN = config.max_text_len

# dataloader
def collate_batch(batch):
    label_list, text_lists, offsets, text_lenghts = [], [], [0], []
    for (_label, _text) in batch:
         label_list.append(_label)
         processed_text = text_pipeline(_text)
         text_lists.append(processed_text)
    text_lists = special_token_pipeline(text_lists)
    text_lenghts = [len(token_list) for token_list in text_lists]
    max_text_len = max(text_lenghts + [MAX_LEN-2])
    text_lists = [padding_pipeline((token_list, max_text_len)) for token_list in text_lists]
    text_lenghts = torch.tensor(text_lenghts, dtype=torch.int64)
    text_lists = torch.tensor(text_lists, dtype=torch.int64)
    label_list = torch.tensor(label_list, dtype=torch.int64)
    return label_list.to(device), text_lists.to(device), text_lenghts.to(device)


'''
    Model parameters

    1) KDIM smaller value => can be more identifiable

    2) CONCAT_HEADS:
        True => regular transformer, 
        False => identifiable upto length of vdim.
'''


DIM_FEEDFORWARD_TRX = 256
DROPOUT = 0.5

INPUT_DIM = len(vocab)
OUTPUT_DIM = len(OUTPUT_LABELS)

print("\nvocab length: {}".format(INPUT_DIM))

PAD_IDX = vocab.unk_index   #stoi: dict<-string_to_index; PAD_IDX=1

print("\nModel configuration::\n BATCH_SIZE: {}\n N_HEAD: {}\n CONCAT_HEADS: {}\n KDIM: {}\n VDIM: {}\n EMBEDDING_DIM: {}\n MAX_LEN: {}"
        .format(BATCH_SIZE, N_HEAD, CONCAT_HEADS, KDIM, VDIM, EMBEDDING_DIM, MAX_LEN)
        )


'''
    initialise model
'''
import model_identifiable as M

model = M.Transformer(
            vocab_size=INPUT_DIM, 
            embedding_dim=EMBEDDING_DIM,
            n_head=N_HEAD,
            concat_heads=CONCAT_HEADS,
            kdim=KDIM,
            vdim=VDIM,
            max_len=MAX_LEN,
            dim_feedforward=DIM_FEEDFORWARD_TRX,
            output_dim=OUTPUT_DIM, 
            dropout=DROPOUT,
            device = device
            )

model = model.to(device)


'''
    Training and evaluation
'''
def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()
    #
    for idx, (label, text, text_lenghts) in enumerate(dataloader):
        optimizer.zero_grad()
        predited_label = model(text, text_lenghts)
        loss = criterion(predited_label, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()

def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0
    #
    with torch.no_grad():
        for idx, (label, text, text_lenghts) in enumerate(dataloader):
            predited_label = model(text, text_lenghts)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count

from torch.utils.data.dataset import random_split

# Hyperparameters
EPOCHS = config.epochs
LR = config.lr 

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
total_accu = None

import time

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, collate_fn=collate_batch)

# training
for epoch in range(1, EPOCHS + 1):
    epoch_start_time = time.time()
    train(train_dataloader)
    accu_val = evaluate(valid_dataloader)
    if total_accu is not None and total_accu > accu_val:
      scheduler.step()
    else:
       total_accu = accu_val
    print('-' * 59)
    print('| end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f} '.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))
    print('-' * 59)


print('Checking the results of test dataset.')
accu_test = evaluate(test_dataloader)
print('test accuracy {:8.3f}'.format(accu_test))


