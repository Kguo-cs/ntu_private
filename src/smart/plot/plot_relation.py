# gat_explainability.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# PyG & Captum
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GNNExplainer
from captum.attr import IntegratedGradients

# ---- 示例 GAT 模型（节点分类） ----
class GATModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=1):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim, heads=heads, concat=True)
        self.gat2 = GATConv(hidden_dim * heads, out_dim, heads=1, concat=False)

    def forward(self, x, edge_index, return_attn=False):
        # return_attn: if True, return attention weights from last layer
        x = F.elu(self.gat1(x, edge_index))
        # For second layer, request attention weights:
        if return_attn:
            out, (edge_idx, attn_weights) = self.gat2(x, edge_index, return_attention_weights=True)
            return out, (edge_idx, attn_weights)  # logits, attn info
        else:
            out = self.gat2(x, edge_index)
            return out

# ---- utilities: load model & data (填充/替换成你自己的) ----
def load_demo_graph():
    # replace with your data loader. return PyG Data object and labels
    # Example dummy graph:
    num_nodes = 10
    T = 5  # history length
    feat_dim = T * 2  # e.g., x,y per timestep flattened
    x = torch.randn(num_nodes, feat_dim, requires_grad=True)
    # fully connected small graph for demo
    row = []
    col = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                row.append(i); col.append(j)
    edge_index = torch.tensor([row, col], dtype=torch.long)
    y = torch.randint(0, 2, (num_nodes,))  # binary node labels
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

# ---- 1) Gradient-based sensitivity ----
def gradient_sensitivity(model, data, node_idx, target_class=None):
    model.eval()
    x = data.x.clone().detach().requires_grad_(True)
    logits = model(x, data.edge_index)
    if target_class is None:
        # pick predicted class for that node
        pred = logits[node_idx].argmax().item()
        target = pred
    else:
        target = target_class

    score = logits[node_idx, target]
    model.zero_grad()
    score.backward(retain_graph=True)
    grads = x.grad[node_idx].detach().cpu().numpy()  # gradient w.r.t features of node
    return grads  # shape (feat_dim,)

# ---- 2) Integrated Gradients (via Captum). Need a wrapper that returns scalar for target node ----
class ModelWrapperForCaptum(torch.nn.Module):
    def __init__(self, model, edge_index, node_idx, target):
        super().__init__()
        self.model = model
        self.edge_index = edge_index
        self.node_idx = node_idx
        self.target = target

    def forward(self, x):
        logits = self.model(x, self.edge_index)
        # return scalar logit for the target node & class
        return logits[self.node_idx, self.target].unsqueeze(0)

def integrated_gradients_attr(model, data, node_idx, target_class=None, baseline=None, steps=50):
    model.eval()
    # Determine target class if not provided
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
    if target_class is None:
        target = int(logits[node_idx].argmax().item())
    else:
        target = int(target_class)

    # baseline: zeros or mean
    if baseline is None:
        baseline = torch.zeros_like(data.x)
    wrapper = ModelWrapperForCaptum(model, data.edge_index, node_idx, target)
    ig = IntegratedGradients(wrapper)
    attributions, delta = ig.attribute(inputs=data.x,
                                       baselines=baseline,
                                       target=None,  # wrapper already selects
                                       return_convergence_delta=True,
                                       n_steps=steps)
    # attributions shape == data.x shape; take node slice
    return attributions[node_idx].detach().cpu().numpy(), float(delta)

# ---- 3) Perturbation importance (masking single variable / feature dimension) ----
def perturbation_importance(model, data, node_idx, target_class=None, mode='zero'):
    model.eval()
    x = data.x.clone()
    baseline_logits = model(x, data.edge_index)
    if target_class is None:
        target = int(baseline_logits[node_idx].argmax().item())
    else:
        target = int(target_class)
    base_score = float(baseline_logits[node_idx, target].cpu().item())

    feat_dim = x.size(1)
    impacts = np.zeros(feat_dim)
    for i in range(feat_dim):
        x_pert = x.clone()
        if mode == 'zero':
            x_pert[node_idx, i] = 0.0
        elif mode == 'noise':
            x_pert[node_idx, i] = torch.randn(1) * x_pert[:, i].std()
        out = model(x_pert, data.edge_index)
        score = float(out[node_idx, target].cpu().item())
        impacts[i] = base_score - score  # positive => that feature reduces score when masked
    return impacts  # shape (feat_dim,)

# ---- 4) GAT attention extraction & visualization ----
def extract_gat_attention(model, data):
    # forward with return_attn True on last layer
    model.eval()
    with torch.no_grad():
        logits, attn_info = model(data.x, data.edge_index, return_attn=True)
    edge_idx, attn_weights = attn_info  # attn_weights shape: [num_edges, heads] or [num_edges]
    # convert to numpy mapping
    return edge_idx.cpu().numpy(), attn_weights.detach().cpu().numpy()

def plot_attention_heatmap(num_nodes, edge_index_np, attn_weights_np, head=0, figsize=(6,6)):
    # convert attention per-edge to adjacency matrix for given head
    A = np.zeros((num_nodes, num_nodes))
    rows = edge_index_np[0]
    cols = edge_index_np[1]
    # attn_weights shape might be (E,) or (E, heads)
    if attn_weights_np.ndim == 2:
        attn = attn_weights_np[:, head]
    else:
        attn = attn_weights_np
    for r, c, a in zip(rows, cols, attn):
        A[int(r), int(c)] = a
    plt.figure(figsize=figsize)
    plt.imshow(A, aspect='auto')
    plt.title(f'GAT attention heatmap (head {head})')
    plt.xlabel('source node')
    plt.ylabel('target node')
    plt.colorbar()
    plt.show()

# ---- 5) GNNExplainer (node-level explanation) ----
def run_gnnexplainer(model, data, node_idx, epochs=100):
    explainer = GNNExplainer(model, epochs=epochs)
    model.eval()
    # returns subgraph node_mask and edge_mask, and feature_mask
    node_feat_mask, edge_mask = explainer.explain_node(node_idx, data.x, data.edge_index)
    # node_feat_mask shape = (feat_dim,)
    return node_feat_mask.detach().cpu().numpy(), edge_mask.detach().cpu().numpy()

# ---- 6) Pairwise interaction (simple perturbation-based matrix) ----
def pairwise_interaction_matrix(model, data, node_idx, mode='zero'):
    # impact_ij = effect of masking both i and j minus sum of individual mask effects
    base_imp = perturbation_importance(model, data, node_idx, mode=mode)
    feat_dim = base_imp.shape[0]
    inter = np.zeros((feat_dim, feat_dim))
    x = data.x.clone()
    with torch.no_grad():
        logits = model(x, data.edge_index)
    if torch.is_tensor(logits):
        target = int(logits[node_idx].argmax().item())
    else:
        target = 0
    base_score = float(logits[node_idx, target].cpu().item())

    for i in range(feat_dim):
        for j in range(i+1, feat_dim):
            x_pert = x.clone()
            if mode == 'zero':
                x_pert[node_idx, i] = 0.0
                x_pert[node_idx, j] = 0.0
            out = model(x_pert, data.edge_index)
            score = float(out[node_idx, target].cpu().item())
            combined_imp = base_score - score
            inter[i, j] = combined_imp - (base_imp[i] + base_imp[j])
            inter[j, i] = inter[i, j]
    return inter

# ---- Example run ----
if __name__ == "__main__":
    # Load data & model
    data = load_demo_graph()
    in_dim = data.x.size(1)
    model = GATModel(in_dim=in_dim, hidden_dim=16, out_dim=2, heads=2)
    # optionally load your model weights:
    # model.load_state_dict(torch.load('gat_model.pth'))

    node_idx = 0  # target node to explain

    # 1) gradient sensitivity
    grads = gradient_sensitivity(model, data, node_idx)
    print("Gradient sensitivity (per feature):", grads)

    # 2) Integrated Gradients
    ig_attr, delta = integrated_gradients_attr(model, data, node_idx)
    print("IG attributions:", ig_attr, "delta:", delta)

    # 3) perturbation importance
    pert_imp = perturbation_importance(model, data, node_idx)
    print("Perturbation impacts:", pert_imp)

    # 4) GAT attention
    edge_idx_np, attn_np = extract_gat_attention(model, data)
    plot_attention_heatmap(num_nodes=data.num_nodes, edge_index_np=edge_idx_np, attn_weights_np=attn_np, head=0)

    # 5) GNNExplainer
    feat_mask, edge_mask = run_gnnexplainer(model, data, node_idx)
    print("GNNExplainer feature mask:", feat_mask[:10], "edge mask (len):", len(edge_mask))

    # 6) pairwise interaction
    inter_mat = pairwise_interaction_matrix(model, data, node_idx)
    print("Pairwise interaction matrix shape:", inter_mat.shape)
    plt.figure(figsize=(6,6))
    plt.imshow(inter_mat)
    plt.title("Pairwise Interaction (perturbation-based)")
    plt.colorbar()
    plt.show()
