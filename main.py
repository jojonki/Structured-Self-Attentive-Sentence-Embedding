import random
import torch
import torch.nn as nn
from utils import load_task, make_word_vector, to_var, load_glove_weights, frobenius
from utils import save_pickle, load_pickle
from models import SelfAttentiveNet

data = load_task('./dataset/review.json')
# data = load_pickle('data.pickle')
# save_pickle(data, 'data.pickle')

vocab = set()
for review, _ in data:
    vocab |= set(review)

vocab = ['<PAD>'] + list(sorted(vocab))
w2i = dict((w, i) for i, w in enumerate(vocab, 0))
i2w = dict((i, w) for i, w in enumerate(vocab, 0))
print('vocab size', len(vocab))

n_dev = 2000
split_id = len(data) - n_dev
train_data = data[:split_id]
dev_data = data[split_id:]

n_epoch = 4
batch_size = 64
embd_size = 100
attn_hops = 60

use_penalization = True
penalization_coeff = 1.0
I = to_var(torch.zeros(batch_size, attn_hops, attn_hops))

# pre_embd = None
pre_embd = torch.from_numpy(load_glove_weights('./dataset', embd_size, len(vocab), w2i)).type(torch.FloatTensor)
# save_pickle(args.pre_embd, './pickles/glove_embd.pickle')


def test(model, data, batch_size):
    correct = 0
    count = 0
    losses = []
    penals = []
    for i in range(0, len(data)-batch_size, batch_size): # TODO use last elms
        batch = data[i:i+batch_size]
        x = [d[0] for d in batch]
        sent_len = max([len(xx) for xx in x])
        x = make_word_vector(x, w2i, sent_len) # (bs, n)
        labels = to_var(torch.LongTensor([d[1] for d in batch])) # (bs,)

        preds, attentions = model(x) # (bs, n_classes), (bs, r, n)
        loss = loss_fn(preds, labels)
        if use_penalization:
            attentions_T = attentions.transpose(2, 1).contiguous() # (bs, n, r)
            extra_loss = frobenius(torch.bmm(attentions, attentions_T) - I) # (bs, n)
            penals.append(extra_loss)
            loss = loss + penalization_coeff * extra_loss

        losses.append(loss.data[0])
        _, pred_ids = torch.max(preds, 1)
        correct += torch.sum(pred_ids == labels).data[0]
        count += batch_size
    print('Loss:', sum(losses) / count)
    print('Penalty:', sum(penals) / count)
    print('Accuracy:', correct / count)


def train(model, data, optimizer, n_epoch, batch_size, dev_data=None):
    model.train()
    for epoch in range(1, n_epoch+1):
        print('---Epoch {}'.format(epoch))
        random.shuffle(data)
        for i in range(0, len(data)-batch_size, batch_size): # TODO use last elms
            batch = data[i:i+batch_size]
            x = [d[0] for d in batch]
            sent_len = max([len(xx) for xx in x])
            x = make_word_vector(x, w2i, sent_len) # (bs, n)
            labels = to_var(torch.LongTensor([d[1] for d in batch])) # (bs,)

            preds, attentions = model(x) # (bs, n_classes)
            loss = loss_fn(preds, labels)
            if i % (batch_size * 20) == 0:
                print('Epoch', epoch, ', Loss', loss.data[0])
            model.zero_grad()
            loss.backward()
            optimizer.step()
        # evaluate model for  every epoch
        model.eval()
        test(model, dev_data, batch_size)
        model.train()


model = SelfAttentiveNet(len(vocab), pre_embd, embd_size=embd_size, attn_hops=attn_hops)
if torch.cuda.is_available():
    model.cuda()

loss_fn = loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

train(model, train_data, optimizer, n_epoch, batch_size, dev_data)
