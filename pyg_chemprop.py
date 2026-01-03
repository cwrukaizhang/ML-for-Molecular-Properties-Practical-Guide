import torch
import torch.nn as nn
import torch.utils.data
from torch_geometric.data import Data, Dataset
from torch_geometric.data.data import size_repr
# from torch_geometric.nn import global_add_pool, global_mean_pool
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_scatter import scatter_sum
from tqdm import tqdm


class RevIndexedData(Data):
    def __init__(self, orig):
        super(RevIndexedData, self).__init__()
        if orig:
            for key in orig.keys():
                self[key] = orig[key]
            
            flip = torch.flip(self["edge_index"], (0,))
            self["revedge_index"] = torch.where((self["edge_index"].t() == flip.t()[:, None]).all(-1))[1]

    def __inc__(self, key, value, *args, **kwargs):
        if key == "revedge_index":
            return self.revedge_index.max().item() + 1
        else:
            return super().__inc__(key, value)

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return "{}({})".format(cls, ", ".join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return "{}(\n{}\n)".format(cls, ",\n".join(info))


class RevIndexedDataset(Dataset):
    def __init__(self, orig):
        super(RevIndexedDataset, self).__init__()
        self.dataset = [RevIndexedData(data) for data in tqdm(orig)]

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)


def directed_mp(message, edge_index, revedge_index):
    m = scatter_sum(message, edge_index[1], dim=0)
    m_all = m[edge_index[0]]
    m_rev = message[revedge_index]
    return m_all - m_rev


def aggregate_at_nodes(num_nodes, message, edge_index):
    m = scatter_sum(message, edge_index[1], dim=0, dim_size=num_nodes)
    return m[torch.arange(num_nodes)]


class DMPNNEncoder(nn.Module):
    def __init__(self, hidden_size, node_fdim, edge_fdim, depth=3):
        super(DMPNNEncoder, self).__init__()
        self.act_func = nn.ReLU()
        self.W1 = nn.Linear(node_fdim + edge_fdim, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W3 = nn.Linear(node_fdim + hidden_size, hidden_size, bias=True)
        self.depth = depth

    def forward(self, data):
        x, edge_index, revedge_index, edge_attr, num_nodes, batch = (
            data.x,
            data.edge_index,
            data.revedge_index,
            data.edge_attr,
            data.num_nodes,
            data.batch,
        )

        # initialize messages on edges
        init_msg = torch.cat([x[edge_index[0]], edge_attr], dim=1).float()
        h0 = self.act_func(self.W1(init_msg))

        # directed message passing over edges
        h = h0
        for _ in range(self.depth - 1):
            m = directed_mp(h, edge_index, revedge_index)
            h = self.act_func(h0 + self.W2(m))

        # aggregate in-edge messages at nodes
        v_msg = aggregate_at_nodes(num_nodes, h, edge_index)

        z = torch.cat([x, v_msg], dim=1)
        node_attr = self.act_func(self.W3(z))

        # readout: pyg global pooling
        return global_mean_pool(node_attr, batch)


class GCNEncoder(nn.Module):
    def __init__(self, hidden_size, node_fdim, depth=3, dropout=0.0):
        super(GCNEncoder, self).__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1")

        self.act_func = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_fdim, hidden_size))
        for _ in range(depth - 1):
            self.convs.append(GCNConv(hidden_size, hidden_size))

    def forward(self, data):
        """Encodes a molecular graph into a graph-level embedding.

        Accepts either a PyG `Data` object with attributes `x`, `edge_index`,
        and optionally `batch`, or a tuple of tensors `(x, edge_index[, batch])`.

        This flexibility makes it easier to wrap for attribution methods.
        """

        # Support both `Data` and tuple inputs
        if hasattr(data, "x") and hasattr(data, "edge_index"):
            x = data.x
            edge_index = data.edge_index
            batch = getattr(data, "batch", None)
        else:
            if not isinstance(data, (tuple, list)) or len(data) not in (2, 3):
                raise TypeError(
                    "GCNEncoder.forward expects a PyG Data object or (x, edge_index[, batch])"
                )
            x = data[0]
            edge_index = data[1]
            batch = data[2] if len(data) == 3 else None

        h = x.float()
        for conv in self.convs:
            h = conv(h, edge_index)
            h = self.act_func(h)
            h = self.dropout(h)

        # If batch is missing, treat as a single graph
        if batch is None:
            batch = torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        else:
            batch = batch.to(device=h.device, dtype=torch.long)

        # readout: pyg global pooling
        return global_mean_pool(h, batch)
    
def train(config, loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    criterion = config["loss"]
    model = config["model"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]

    model = model.to(device)
    model.train()
    for batch in tqdm(loader, total=len(loader)):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out.squeeze(), batch.y.float())
        loss.backward()
        optimizer.step()
        scheduler.step()
def make_prediction(config, loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    model = config["model"]

    model = model.to(device)
    model.eval()
    y_pred = []
    y_true = []
    for batch in tqdm(loader, total=len(loader)):
        batch = batch.to(device)
        with torch.no_grad():
            batch_preds = model(batch)
        y_pred.extend(batch_preds)
        y_true.extend(batch.y)
    return torch.stack(y_pred).cpu(), torch.stack(y_true).cpu()

# Define a silent train function to avoid tqdm clutter during optimization
def train_silent(config, loader, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    criterion = config["loss"]
    model = config["model"]
    optimizer = config["optimizer"]
    scheduler = config["scheduler"]

    model = model.to(device)
    model.train()
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = criterion(out.squeeze(), batch.y.float())
        loss.backward()
        optimizer.step()
        scheduler.step()