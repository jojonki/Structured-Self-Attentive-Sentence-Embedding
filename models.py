import torch
import torch.nn as nn
import torch.nn.functional as F


class WordEmbedding(nn.Module):
    """
    In : (N, sentence_len)
    Out: (N, sentence_len, embd_size)
    """
    def __init__(self, vocab_size, embd_size, pre_embd=None, is_train_embd=False):
        super(WordEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embd_size)
        if pre_embd is not None:
            print('Set pretrained embedding weights')
            self.embedding.weight = nn.Parameter(pre_embd, requires_grad=is_train_embd)

    def forward(self, x):
        return F.relu(self.embedding(x))


class SelfAttentiveNet(nn.Module):
    def __init__(self,
                 vocab_size,
                 pre_embd=None,
                 embd_size=100,
                 hidden_size=100,
                 attn_hops=30,
                 mlp_d=350,
                 mlp_hidden=512,
                 n_classes=5):
        super(SelfAttentiveNet, self).__init__()
        self.embd_size   = embd_size
        self.hidden_size = hidden_size # u
        r = attn_hops
        d = mlp_d

        self.word_emb   = WordEmbedding(vocab_size, embd_size, pre_embd)
        self.encoder    = nn.GRU(self.embd_size,
                                 self.hidden_size,
                                 batch_first=True,
                                 bidirectional=True)
        self.fc1 = nn.Linear(r*2*self.hidden_size, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, n_classes)

        initrange = 0.1
        self.Ws1 = nn.Parameter(torch.Tensor(1, d, 2*hidden_size).uniform_(-initrange, initrange))
        self.Ws2 = nn.Parameter(torch.Tensor(1, r, d).uniform_(-initrange, initrange))

    def forward(self, x):
        bs = x.size(0) # batch size
        n  = x.size(1) # sentence length

        x    = self.word_emb(x) # (bs, n, embd_size)

        H, _ = self.encoder(x) # (bs, n, 2u)
        H_T  = torch.transpose(H, 2, 1).contiguous() # (bs, 2u, n)

        A    = F.tanh(torch.bmm(self.Ws1.repeat(bs, 1, 1), H_T)) # (bs, d, n)
        A    = torch.bmm(self.Ws2.repeat(bs, 1, 1), A) # (bs, r, n)
        A    = F.softmax(A.view(-1, n)).view(bs, -1, n) # (bs, r, n)

        M    = torch.bmm(A, H) # (bs, r, 2u)

        out = F.relu(self.fc1(M.view(bs, -1))) # (bs, mlp_hidden)
        out = F.log_softmax(self.fc2(out)) # (bs, mlp_hidden)

        return out, A
