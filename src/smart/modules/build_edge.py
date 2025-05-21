from torch_geometric.nn.pool import knn_graph,knn
import torch.nn.functional as F







def radiusGraphNearest(x, batch, r, loop, max_num_neighbors):
    edge_index = knn_graph(x, k=max_num_neighbors, batch=batch, loop=loop)
    row, col = edge_index
    distances = (x[col] - x[row]).norm(dim=1)
    mask = distances <= r
    final_edge_index = edge_index[:, mask]

    return final_edge_index

def radiusGraphNearest2(x,y,r, batch_x,batch_y,  max_num_neighbors):
    edge_index = knn(x, y, max_num_neighbors, batch_x=batch_x, batch_y=batch_y)
    row, col = edge_index
    distances = (x[col] - y[row]).norm(dim=1)
    mask = distances <= r
    final_edge_index = edge_index[:, mask]

    return final_edge_index