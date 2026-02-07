import torch
import torch.nn as nn

class DLRM(nn.Module):
    def __init__(self, dense_size=8, vocab_sizes=[4, 3, 2], hidden_dim=2):
        super(DLRM, self).__init__()

        self.dense_size = dense_size
        self.vocab_sizes = vocab_sizes
        self.hidden_dim = hidden_dim


        # Bottom MLP - Process dense features
        self.bot_l = nn.Sequential(
            nn.Linear(self.dense_size, 3), 
            nn.ReLU(),                     
            nn.Linear(3, self.hidden_dim),
            nn.ReLU(),                     
        )
        # Embedding tables - Process sparse features
        self.emb_l = nn.ModuleList()
        for vocab_size in self.vocab_sizes:
            self.emb_l.append(nn.Embedding(vocab_size, self.hidden_dim))


        num_top_inp = self.hidden_dim * (1 + len(self.vocab_sizes)) # 1 from dense, len(vocab_sizes) from sparse
        # Top MLP - Process interactions
        self.top_l = nn.Sequential(
            nn.Linear(num_top_inp, 4), 
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            # nn.Sigmoid()
        )

    def forward(self, x, lS_i):
        x = self.bot_l(x)
        ly = []
        for i, idx in enumerate(lS_i):
            ly.append(self.emb_l[i](idx))
        x = torch.cat([x] + ly, dim=1)
        x = self.top_l(x)
        return x


if __name__ == "__main__":
    torch.manual_seed(0)
    dense_size = 4
    vocab_sizes=[4, 3, 2]
    hidden_dim = 2
    dlrm = DLRM(dense_size=dense_size, vocab_sizes=vocab_sizes, hidden_dim=hidden_dim)
    dlrm.load_state_dict(torch.load("dlrm.pth"))

    dense = torch.randn(1, dense_size)
    sparse = [torch.randint(0, vocab_size, (1,)) for vocab_size in vocab_sizes]
    y = dlrm(dense, sparse)
    print(y)
