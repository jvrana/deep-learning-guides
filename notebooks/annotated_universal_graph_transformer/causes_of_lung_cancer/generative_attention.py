from torch.nn import functional as F
from tqdm.auto import tqdm
from IPython import display
from torch import nn
import torch
from abc import ABC, abstractmethod
from torch import nn
from typing import *
import torch
from copy import deepcopy
import matplotlib as mpl
from matplotlib import cm
from dgl import ops as Fops
from torch.nn import functional as F
import typing
import typing
from copy import deepcopy
from torch.utils.data import DataLoader, Dataset
import dgl
import random
from matplotlib import pylab as plt


class Compose(typing.Callable):

    def __init__(self, *transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x


class AddIncomingEdges(typing.Callable):

    def __init__(self, embed_key):
        self.embed_key = embed_key

    def __call__(self, g):
        n = g.number_of_nodes()
        g.add_nodes(n)
        g.ndata[self.embed_key][n:] = torch.arange(n, 2 * n)
        edges = torch.stack([
            torch.arange(n, 2 * n),
            torch.arange(0, n),
        ])
        g.add_edges(edges[0], edges[1])
        return g

class AddIncomingEdges2(typing.Callable):

    def __init__(self, embed_key, length):
        self.embed_key = embed_key
        self.length = length

    def __call__(self, g):
        n = g.number_of_nodes()
        g.add_nodes(n)
        g.ndata[self.embed_key][n:] = g.ndata[self.embed_key][:n] + self.length
        edges = torch.stack([
            torch.arange(n, 2 * n),
            torch.arange(0, n),
        ])
        g.ndata['y'][:] = 1
        g.ndata['y_hat'][n:] = 1
        g.add_edges(edges[0], edges[1])
        return g

class Deepcopy(typing.Callable):

    def __call__(self, x):
        return deepcopy(x)

class ToDGLFullyConnected(typing.Callable):

    def __init__(self, keys, feature_key='feat', target_key='targ'):
        self.keys = keys
        self.key_to_idx = {k: i for i, k in enumerate(keys)}
        self.feature_key = feature_key
        self.target_key = target_key

    @staticmethod
    def leading_zero(x):
        yield torch.zeros_like(x[0])
        yield from x

    @classmethod
    def get_complete_edges(cls, g):
        x = torch.cumsum(g.batch_num_nodes(), 0)
        all_edges = []
        for a, b in zip(cls.leading_zero(x), x):
            edges = torch.combinations(torch.arange(a, b))
            all_edges.append(edges)
        return torch.cat(all_edges, 0).T

    @classmethod
    def add_complete_edges(cls, g):
        n1 = g.number_of_nodes()
        edges = cls.get_complete_edges(g)
        g.add_edges(edges[0], edges[1])
        g.add_edges(edges[1], edges[0])
        n2 = g.number_of_nodes()
        assert n1 == n2
        return g

    def __call__(self, row):
        g = dgl.graph(([], []))
        g.add_nodes(len(row))
        idx = [self.key_to_idx[k] for k in row.keys()]
        g.ndata[self.feature_key] = torch.tensor(idx).long()
        self.add_complete_edges(g)
        g.ndata[self.target_key] = torch.tensor(list(row.values())).unsqueeze(1).float()
        return g


class CloneNodeData(typing.Callable):

    def __init__(self, from_key: str, to_key: str):
        self.from_key = from_key
        self.to_key = to_key

    def __call__(self, g):
        g.ndata[self.to_key] = g.ndata[self.from_key].detach().clone()
        return g


class MaskNodeData(typing.Callable):

    def __init__(self, index: int, key: str, mask_key='m', mask_value=0):
        self.index = index
        self.key = key
        self.mask_key = mask_key
        self.mask_value = mask_value

    def __call__(self, g):
        mask = torch.zeros_like(g.ndata[self.key]).long()
        mask[self.index] = 1
        mask = mask.bool()
        g.ndata[self.mask_key] = mask
        g.ndata[self.key][mask] = self.mask_value
        return g


class RandomMaskNodeData(typing.Callable):

    def __init__(self, key: str, mask_key='m', mask_value=0):
        self.key = key
        self.mask_key = mask_key
        self.mask_value = mask_value

    def __call__(self, g):
        mask = torch.zeros_like(g.ndata[self.key]).long()
        i = torch.randint(0, len(mask), (1,))
        mask[i] = 1
        mask = mask.bool()
        g.ndata[self.mask_key] = mask
        g.ndata[self.key][mask] = self.mask_value
        return g


# t = Compose(
#     ToDGLFullyConnected(list(df.columns), feature_key='x', target_key='y'),
#     CloneNodeData('y', 'y_hat'),
#     MaskNodeData(-1, key='y', mask_key='mask', mask_value=-1))
# t(row.to_dict())


def clones(net, N):
    return [deepcopy(net) for _ in range(N)]


class SizedModule(ABC):

    @abstractmethod
    def get_size(self) -> int:
        ...


class AddNorm(nn.Module):

    def __init__(self, size: Optional[int] = None, dropout: float = 0.1, layer: Optional[SizedModule] = None):
        super().__init__()
        if size is None and layer is None:
            return ValueError("Either size or layer must be provided")
        self.size = size or layer.get_size()
        self.layer = layer
        self.norm = nn.LayerNorm(self.size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, args=None, kwargs=None, layer: Optional[SizedModule] = None):
        kwargs = kwargs or dict()
        if args is None:
            args = (x,)
        layer = layer or self.layer
        return self.norm(x + self.dropout(layer(*args, **kwargs)))


class MultiHeadAttention(nn.Module):

    def __init__(self, dim_model, h):
        super().__init__()
        assert dim_model % h == 0
        self.h = h
        self.dim_model = dim_model
        self.d_k = dim_model // h
        self.linears = clones(nn.Linear(dim_model, dim_model), 4)
        self.attn = None

    def _view_head(self, x):
        return x.view(x.size(0), -1, self.h, self.d_k).transpose(1, 2)

    def forward(self, g, query, key, value):
        q = self._view_head(self.linears[0](query))
        k = self._view_head(self.linears[1](key))
        v = self._view_head(self.linears[2](value))
        x = Fops.v_dot_u(g, q, k) / self.d_k ** 0.5
        score = Fops.edge_softmax(g, x)
        #         score = Fops.v_dot_u(g, q, k) / self.d_k**0.5
        #         score = F.leaky_relu(Fops.v_dot_u(g, q, k) / self.d_k**0.5)
        out = Fops.u_mul_e_sum(g, v, score)
        out = out.transpose(1, 2).view(g.number_of_nodes(), self.h * self.d_k)
        score = score.view(score.size(0), self.h, -1)
        self.attn = score
        out = self.linears[3](out)
        return out


class Network(nn.Module):

    def __init__(self, d_model, h=16, n_heads=4, dropout=0.2):
        super().__init__()
        self.src_embedding = nn.Sequential(
            nn.Embedding(d_model, h),
            nn.Linear(h, h),
            nn.LeakyReLU(),
            nn.Linear(h, h),
            nn.LeakyReLU()
        )
        self.dst_embedding = nn.Sequential(
            nn.Embedding(d_model, h),
            nn.Linear(h, h),
            nn.LeakyReLU(),
            nn.Linear(h, h),
            nn.LeakyReLU()
        )
        self.encode = nn.Sequential(
            nn.Linear(1, h),
            nn.LeakyReLU(),
        )
        self.attn = AddNorm(h, layer=MultiHeadAttention(h, n_heads), dropout=dropout)
        self.core = nn.Sequential(
            nn.Linear(h, h),
            nn.LeakyReLU(),
        )
        self.decode = nn.Sequential(
            nn.Linear(h, h),
            nn.LeakyReLU(),
            nn.Linear(h, 1)
        )

    def forward(self, g, x, y, n_loops=5):
        with g.local_scope():
            g.ndata['a'] = self.src_embedding(x.flatten().long())
            g.ndata['b'] = self.dst_embedding(x.flatten().long())
            g.ndata['h'] = self.encode(y.float())
            out_arr = []
            for i in range(n_loops):
                g.ndata['h'] = self.attn(g.ndata['h'], args=(g, g.ndata['a'], g.ndata['b'], g.ndata['h']))
                g.ndata['h'] = self.core(g.ndata['h'])
                out = nn.Sigmoid()(self.decode(g.ndata['h']))
                out_arr.append(out)

            return out_arr


class GumbelSoftMaxSampler(nn.Module):

    def __init__(self, hard=False):
        super().__init__()
        self.hard = hard

    def forward(self, logits):
        return F.gumbel_softmax(logits=logits, hard=self.hard)


class Parallel(nn.Module):

    def __init__(self, *modules):
        super().__init__()
        self.mods = nn.ModuleList(modules)
        self.n_modules = len(modules)

    def forward(self, x):
        assert x.shape[-1] == len(self.mods)
        out = []
        for i, mod in enumerate(self.mods):
            _x = x[..., i]
            try:
                result = mod(_x)
            except Exception as e:
                msg = "Error found in forward prop of module {}, {}\n".format(i, str(mod)[:1000])
                msg = msg + str(e)
                raise e.__class__(msg)
            out.append(result)
        return torch.cat(out, 1)


class ParallelEmbedding(Parallel):

    def __init__(self, *dims):
        mods = []
        for a, b in dims:
            mods.append(nn.Embedding(a, b))
        super().__init__(*mods)


class Dense(nn.Module):

    def __init__(self, *dims, dropout=0.2):
        super().__init__()
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.LeakyReLU(), nn.Dropout(dropout)]
        self.layers = nn.Sequential(*layers)

    def __call__(self, x):
        return self.layers(x)


x = torch.randn(10, 12)
encoder = Dense(12, 16, 16, 2)
sampler = GumbelSoftMaxSampler()
decoder = Dense(2, 16, 16, 12)

x = torch.randn(10, 12)
logits = encoder(x)
y = sampler(logits)
x = decoder(y)
print(x.shape)


class GenerativeNetwork(nn.Module):

    def __init__(self, d_model, h=32, n_heads=4, dropout=0.2):
        super().__init__()

        # embedding for src nodes
        self.src_embed = nn.Sequential(
            nn.Embedding(d_model, h),
            Dense(h, h, dropout=dropout)
        )

        # embedding for dst nodes
        self.dst_embed = nn.Sequential(
            nn.Embedding(d_model, h),
            Dense(h, h, dropout=dropout)
        )

        # embedding to parameterize the "combining" function for inputs
        self.embed = nn.Sequential(
            ParallelEmbedding((d_model, h), (2, 1)),
            Dense(h + 1, h, dropout=dropout)
        )

        # encode incoming data
        self.encoder = nn.Sequential(
            Dense(h + 2, h, dropout=dropout)
        )

        # attention to determine interactions
        self.attn = AddNorm(h, layer=MultiHeadAttention(h, n_heads), dropout=dropout)

        # decode to logits
        self.decoder = nn.Sequential(
            Dense(h, h),
            nn.Linear(h, 2)
        )

        # sample from logits
        self.sampler = GumbelSoftMaxSampler(hard=True)

    def forward(self, g, feature, target, feature2, n_loops=3):
        with g.local_scope():
            q = self.src_embed(feature)
            k = self.dst_embed(feature)

            m = torch.stack([feature, feature2], -1)
            m = self.embed(m)
            x = target
            out = [x]
            for i in range(n_loops):
                h = torch.cat([x, m], 1)
                v = self.encoder(h)
                h = self.attn(v, args=(g, q, k, v))
                logits = self.decoder(h)
                #                 x = self.sampler(logits)
                x = nn.Sigmoid()(logits)
                out.append(x)
            return out


def one_hot(x: torch.Tensor, num_classes: int, device=None, dtype=torch.long):
    to_shape = None
    if len(x.shape) > 1:
        to_shape = tuple(x.shape) + (num_classes,)
        x = x.flatten()
    b = torch.zeros(x.shape[0], num_classes, device=device, dtype=dtype)
    b[torch.arange(x.shape[0], device=device), x.to(device)] = 1
    if to_shape:
        b = b.view(to_shape)
    return b


class OneHotNode(typing.Callable):

    def __init__(self, key, num_classes, out_key=None, dtype=torch.long):
        self.num_classes = num_classes
        self.key = key
        if out_key is None:
            self.out_key = self.key
        self.dtype = dtype

    def __call__(self, g):
        g.ndata[self.out_key] = one_hot(g.ndata[self.key].long(), self.num_classes, dtype=self.dtype)
        return g


class NodeSqueeze(typing.Callable):

    def __init__(self, key, dim):
        self.dim = dim
        self.key = key

    def __call__(self, g):
        g.ndata[self.key] = g.ndata[self.key].squeeze(self.dim)
        return g


class PdDataset(Dataset):

    def __init__(self, df, transforms=None):
        self.df = df
        self.transforms = transforms
        self.transformed = {}

    def __getitem__(self, idx):
        x = self.df.iloc[idx].to_dict()
        if self.transforms:
            if isinstance(self.transforms, (list, tuple)):
                for t in self.transforms:
                    x = t(x)
            elif callable(self.transforms):
                x = self.transforms(x)
            else:
                raise TypeError("Transforms must be callable or an list or tuple of callables")
        return x

    def __len__(self):
        return len(self.df)

    def split(self, *splits):
        x = torch.tensor(splits)
        x = torch.cumsum(x, 0) / x.sum()

        idx = len(self) * x
        idx = [0] + idx.long().tolist()
        idx[-1] = None
        idx[0] = None
        datasets = []
        for i, j in zip(idx[:-1], idx[1:]):
            datasets.append(self.__class__(self.df.iloc[i:j], transforms=self.transforms))
        return datasets


class CachedDataset(Dataset):

    def __init__(self, dataset):
        self.cache = {}
        self.dataset = dataset

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        else:
            x = self.dataset[idx]
            self.cache[idx] = x
            return x

    def __len__(self):
        return len(self.dataset)

    def split(self, *splits):
        return [self.__class__(s) for s in self.dataset.split(*splits)]


Transforms = Compose(
    Deepcopy(),
    ToDGLFullyConnected(list(df.columns), feature_key='x', target_key='y'),
    CloneNodeData('y', 'y_hat'),
    RandomMaskNodeData(key='y', mask_key='mask', mask_value=1),
    AddIncomingEdges('x'),
    OneHotNode('y', 2, dtype=torch.float),
    OneHotNode('y_hat', 2, dtype=torch.float),
    NodeSqueeze('y', 1),
    NodeSqueeze('y_hat', 1)
)
dataset = PdDataset(df=pd.concat([df], ignore_index=True), transforms=Transforms)
# dataset = CachedDataset(dataset)
train_dataset, eval_dataset = dataset.split(0.8, 0.2)
datasets = {
    'train': train_dataset,
    'eval': eval_dataset,
    'full': dataset
}

loaders = {
    'train': DataLoader(datasets['train'], batch_size=128, collate_fn=dgl.batch, shuffle=True),
    'eval': DataLoader(datasets['eval'], batch_size=len(datasets['eval']), collate_fn=dgl.batch)
}

net = gennet = GenerativeNetwork(len(df.columns) * 2)
optim = torch.optim.AdamW(net.parameters(), lr=1e-3)
lossfn = torch.nn.BCELoss()

for g in loaders['eval']:
    net(g, g.ndata['x'].flatten(), g.ndata['y'], g.ndata['mask'].long().flatten())


def compute_loss(net, g):
    m = g.ndata['mask'].flatten()
    y_arr = net(g, g.ndata['x'].view(-1), g.ndata['y'], g.ndata['mask'].long().view(-1), n_loops=10)
    y_hat = g.ndata['y_hat']
    losses = []
    for y in y_arr:
        losses.append(lossfn(y[m], y_hat[m]))
    loss = torch.stack(losses)
    return loss.mean()

def attn_to_sparse(g: dgl.DGLGraph, attn: torch.Tensor):
    n = g.number_of_nodes()
    i = torch.stack(g.edges())
    v = attn
    x = torch.sparse_coo_tensor(i, v.flatten(), (n, n))
    return x

def compute_train_loss(net, g):
    net.train()
    return compute_loss(net, g)


def compute_eval_loss(net, g):
    net.eval()
    with torch.no_grad():
        return compute_loss(net, g)


n_epochs = 1000
train_losses = []
eval_losses = []
device = 'cpu'
net.to(device)
for epoch in tqdm(range(n_epochs)):

    # evaluate
    if epoch % 1 == 0:
        epoch_eval_loss = []
        for g in loaders['eval']:
            g = g.to(device)
            eval_loss = compute_eval_loss(net, g)
            epoch_eval_loss.append(eval_loss.detach().item())
        epoch_eval_loss = torch.tensor(epoch_eval_loss)
        eval_losses.append((epoch, epoch_eval_loss.mean()))

    # train
    epoch_train_loss = []
    for g in loaders['train']:
        g = g.to(device)
        train_loss = compute_train_loss(net, g)
        optim.zero_grad()
        train_loss.backward()
        optim.step()
        epoch_train_loss.append(train_loss.detach().item())
    epoch_train_loss = torch.tensor(epoch_train_loss)
    train_losses.append((epoch, epoch_train_loss.mean()))

    # plot
    display.clear_output(wait=True)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    epoch, loss = zip(*train_losses)
    axes[0].plot(epoch, loss)

    epoch, loss = zip(*eval_losses)
    axes[1].plot(epoch, loss)

    g = loaders['eval'].dataset[2]
    out = net(g, g.ndata['x'].flatten(), g.ndata['y'], g.ndata['mask'].flatten().long())
    attn = net.attn.layer.attn.squeeze(-1).detach()
    attn = attn_to_sparse(g, attn).to_dense()
    x = attn.mean(-1)[:, :12]

    sns.heatmap(x, cmap='binary', xticklabels=list(df.columns), yticklabels=list(df.columns), ax=axes[2])
    plt.show()