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
parser = argparse.ArgumentParser(description='identifiable transformer')

parser.add_argument('-dataset', action="store", type=str, default="ag_news")
parser.add_argument('-kdim', action="store", type=int, default=16)
parser.add_argument('-nhead', action="store", type=int, default=4)
parser.add_argument('-embedim', action="store", type=int, default=32)
parser.add_argument('-batch', action="store", type=int, default=64)
parser.add_argument('-epochs', action="store", type=int, default=10)
parser.add_argument('-lr', action="store", type=float, default=0.001)
parser.add_argument('-dropout', action="store", type=float, default=0.1)
parser.add_argument('-vocab_size', action="store", type=int, default=100000)
parser.add_argument('-max_text_len', action="store", type=int, default=512)
parser.add_argument('-valid_frac', action="store", type=float, default=0.3)
parser.add_argument('-add_heads', action="store_true", default=False)
parser.add_argument('-pos_emb', action="store_true", default=False)
parser.add_argument('-return_attn', action="store_true", default=False)

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
CONCAT_HEADS = not config.add_heads
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

MAX_LEN = config.max_text_len

# vocab length is max_size + 2 for special tokens
max_vocab_size = config.vocab_size
vocab = torchtext.vocab.Vocab(counter, max_size=max_vocab_size, specials=('<pad>', '<unk>', '<cls>', '<sep>'), specials_first=True)

# example: text_pipeline('here is the an example') --> [0, 22, 3, 31, 0]
tokenize_clip_pipeline = lambda sentence: [vocab[word] for word in tokenizer(sentence)][:MAX_LEN-2]

# special token append
special_token_pipeline = lambda token_list: [vocab.stoi['<cls>']] + token_list + [vocab.stoi['<sep>']]

# padding function
padding_pipeline = lambda token_list: token_list + [vocab.stoi['<pad>'] for p in range(MAX_LEN-len(token_list))]

# dataloader
def collate_batch(batch):
    label_lists, mask_lists, text_lists = [], [], []
    for (_label, _text) in batch:

        #batch list of labels
        label_lists.append(_label)

        #tokenize text
        process_text = tokenize_clip_pipeline(_text)
        process_text = special_token_pipeline(process_text)

        #define mask
        mask_lists.append([1 for m in range(len(process_text))] + [0 for m in range(MAX_LEN - len(process_text))])

        #pad the token list
        process_text = padding_pipeline(process_text)
        text_lists.append(process_text)

    #convert lists into tensors
    label_lists = torch.tensor(label_lists, dtype=torch.int64)
    mask_lists = torch.tensor(mask_lists, dtype=torch.int64)
    text_lists = torch.tensor(text_lists, dtype=torch.int64)

    #put the batch on device and return tensors
    return label_lists.to(device), mask_lists.to(device), text_lists.to(device)


'''
    Model parameters

    1) KDIM smaller value => can be more identifiable

    2) CONCAT_HEADS:
        True => regular transformer, 
        False => identifiable upto length of vdim.
'''


DIM_FEEDFORWARD_TRX = 256
DROPOUT = config.dropout

INPUT_DIM = len(vocab)
OUTPUT_DIM = len(OUTPUT_LABELS)

POS_EMB = config.pos_emb

RETURN_ATTN = config.return_attn

print("\nModel configuration::\n \
        BATCH_SIZE: {}\n \
        N_HEAD: {}\n \
        KDIM: {}\n \
        VDIM: {}\n \
        EMBEDDING_DIM: {}\n \
        MAX_LEN: {}\n \
        VOCAB_LEN: {}\n \
        DROPOUT:{}\n \
        CONCAT_HEADS: {}\n \
        POS_EMB: {}\n \
        Return Attentions: {}\n\n"
        .format(BATCH_SIZE, N_HEAD, KDIM, VDIM, EMBEDDING_DIM, MAX_LEN, max_vocab_size, DROPOUT, CONCAT_HEADS, POS_EMB, RETURN_ATTN)
        )

PAD_IDX = vocab.stoi['<pad>']   #PAD_IDX=0

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
            pos_emb = POS_EMB,
            device = device,
            pad_id = PAD_IDX,
            return_attn_weights=RETURN_ATTN
            )

model = model.to(device)


'''
    Training and evaluation
'''

#define training module
def train(dataloader):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500
    start_time = time.time()
    #
    for idx, (label, mask, text) in enumerate(dataloader):

        #flush gradients
        optimizer.zero_grad()

        #feed inputs to the model
        predited_label, attn_weights = model(mask, text)

        #calculate the loss
        loss = criterion(predited_label, label)

        #compute gradients
        loss.backward()

        #gradient step
        optimizer.step()

        #compute metric score 
        total_acc += (predited_label.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches '
                  '| train accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                              total_acc/total_count))
            total_acc, total_count = 0, 0
            start_time = time.time()


#define evaluation module
def evaluate(dataloader):
    model.eval()
    total_acc, total_count = 0, 0

    #disable graph building
    with torch.no_grad():

        for (label, mask, text) in dataloader:

            #feed input to the model
            predited_label, attn_weights = model(mask, text)

            #compute loss
            loss = criterion(predited_label, label)

            #compute metric score
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc/total_count


# Hyperparameters
EPOCHS = config.epochs
LR = config.lr 

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_accu = None


'''
Start training iterations
'''

#load data in batch
import time

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, collate_fn=collate_batch)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, collate_fn=collate_batch)

#training
best_val = 0.0
best_test = 0.0
best_epoch = 0
for epoch in range(1, EPOCHS + 1):
    print('-' * 59)

    epoch_start_time = time.time()

    train(train_dataloader)

    accu_val = evaluate(valid_dataloader)

    print('\n+end of epoch {:3d} | time: {:5.2f}s | '
          'valid accuracy {:8.3f}]'.format(epoch,
                                           time.time() - epoch_start_time,
                                           accu_val))

    accu_test = evaluate(test_dataloader)

    print('+test accuracy {:8.3f}'.format(accu_test))

    if accu_val > best_val:
        best_val = accu_val
        best_test = accu_test
        best_epoch = epoch

print('-' * 59)
print('Best valid accuracy is {:8.3f} at epoch {} at which test accuracy is {:8.3f}'.format(best_val, best_epoch, best_test))


