import torch
import torch.nn as nn
import torch.nn.functional as F

# GCN and GraphSage
class GraphSageLayer(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, agg_type: str):
        super(GraphSageLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.agg_type = agg_type
        self.act = nn.ReLU()

        if self.agg_type == 'gcn':
            self.weight = nn.Linear(self.dim_in, self.dim_out, bias=False)
            self.bias = nn.Linear(self.dim_in, self.dim_out, bias=False)

        elif self.agg_type == 'mean':
            self.weight = nn.Linear(2 * self.dim_in, self.dim_out, bias=False)

        elif self.agg_type == 'maxpool':
            self.linear_pool = nn.Linear(self.dim_in, self.dim_in, bias=True)
            self.weight = nn.Linear(2 * self.dim_in, self.dim_out, bias=False)

        else:
            raise RuntimeError(f"Unknown aggregation type: {self.agg_type}")

    def forward(self, feat: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        if self.agg_type == 'gcn':
            feat_h = feat[edge[0]]
            idx_t = edge[1]
            agg_neighbor = torch.zeros(feat.size(0), feat.size(1), dtype=torch.float32).to(feat.device)
            agg_neighbor = agg_neighbor.index_add_(0, idx_t, feat_h)
            degree = torch.bincount(idx_t, minlength=feat.size(0)).unsqueeze(1).to(feat.device)
            inv_degree = torch.where(degree == 0.0, 1.0, 1.0 / degree)
            feat_agg = agg_neighbor * inv_degree
            out = F.normalize(self.act(self.weight(feat_agg) + self.bias(feat)), 2, -1)

        elif self.agg_type == 'mean':
            feat_h = feat[edge[0]]
            idx_t = edge[1]
            agg_neighbor = torch.zeros(feat.size(0), feat.size(1), dtype=torch.float32).to(feat.device)
            agg_neighbor = agg_neighbor.index_add_(0, idx_t, feat_h)
            degree = torch.bincount(idx_t, minlength=feat.size(0)).unsqueeze(1).to(feat.device)
            inv_degree = torch.where(degree == 0.0, 1.0, 1.0 / degree)
            feat_agg = agg_neighbor * inv_degree
            out = F.normalize(self.act(self.weight(torch.cat((feat_agg, feat), 1))), 2, -1)

        elif self.agg_type == 'maxpool':
            feat = self.act(self.linear_pool(feat))
            feat_h = feat[edge[0]]
            idx_t = edge[1]
            scatter_idx = idx_t.unsqueeze(-1).repeat(1, feat.size(1))
            feat_agg = torch.zeros(feat.size(0), feat.size(1), dtype=torch.float32).to(feat.device)
            feat_agg = feat_agg.scatter_reduce(0, scatter_idx, feat_h, reduce='amax', include_self=False)
            out = F.normalize(self.act(self.weight(torch.cat((feat_agg, feat), 1))), 2, -1)

        else:
            raise RuntimeError(f"Unknown aggregation type: {self.agg_type}")

        return out

class GraphSage(nn.Module):
    def __init__(self, num_layers: int, dim_in: int, dim_hidden: int, dim_out: int, agg_type: str):
        super(GraphSage, self).__init__()
        self.num_layers = num_layers
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.agg_type = agg_type

        self.layers = nn.ModuleList()
        for l in range(num_layers):
            self.layers.append(GraphSageLayer(self.dim_in if l == 0 else self.dim_hidden, self.dim_hidden, agg_type))

        self.classifier = nn.Sequential(
            nn.Linear(2 * self.dim_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, feat: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        x_in = feat
        for layer in self.layers:
            x_out = layer(x_in, edge)
            x_in = x_out
        return x_out

    def predict(self, head: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        head_tail = torch.cat([head, tail], dim=-1)
        score = self.classifier(head_tail)
        return score

# GAT
class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.5, alpha: float = 0.2) -> None:
        super(GATLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Parameter(torch.empty(size=(in_dim, out_dim)))
        nn.init.xavier_uniform_(self.W.data)
        self.a = nn.Parameter(torch.empty(size=(2 * out_dim, 1)))
        nn.init.xavier_uniform_(self.a.data)

        self.leakyrelu = nn.LeakyReLU(negative_slope=self.alpha)
        self.batch_norm = nn.BatchNorm1d(out_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, feat: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        message = feat @ self.W
        attn_src = message @ self.a[:self.out_dim, :]
        attn_dst = message @ self.a[self.out_dim:, :]

        src, dst = edges
        attn_scores = self.leakyrelu(attn_src[src] + attn_dst[dst])
        attn_scores = attn_scores - attn_scores.max()  # for stabilization of softmax

        # Edge softmax
        exp_attn_scores = attn_scores.exp()
        exp_sum = torch.zeros((feat.shape[0], 1), device=feat.device).scatter_add_(
            dim=0,
            index=dst.unsqueeze(1),
            src=exp_attn_scores
        ) + 1e-10  # Prevent division by zero

        attn_coeffs = exp_attn_scores / exp_sum[dst]
        attn_coeffs = self.dropout_layer(attn_coeffs)

        # Weighted aggregation
        out = torch.zeros_like(message, device=feat.device).scatter_add_(
            dim=0,
            index=dst.unsqueeze(1).expand(-1, self.out_dim),
            src=message[src] * attn_coeffs
        )
        out = self.batch_norm(out)
        return out

class GAT(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int, dropout: float = 0.5, alpha: float = 0.2, num_heads: int = 8) -> None:
        super(GAT, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.dropout = dropout
        self.num_heads = num_heads

        self.attn_heads = nn.ModuleList()
        for _ in range(num_heads):
            self.attn_heads.append(
                GATLayer(self.dim_in, self.dim_hidden, dropout=dropout, alpha=alpha)
            )

        self.output_layer = GATLayer(self.dim_hidden * num_heads, self.dim_out, dropout=dropout, alpha=alpha)
        self.residual = nn.Linear(self.dim_in, self.dim_out)

        self.classifier = nn.Sequential(
            nn.Linear(2 * self.dim_out, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, feat: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        x_in = feat
        multi_head_out = []
        for attn_head in self.attn_heads:
            multi_head_out.append(attn_head(x_in, edge))
        x_out = torch.cat(multi_head_out, dim=-1)
        x_out = self.output_layer(x_out, edge)
        x_out += self.residual(x_in)  # Residual connection
        return x_out

    def predict(self, head: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
        head_tail = torch.cat([head, tail], dim=-1)
        score = self.classifier(head_tail)
        return score