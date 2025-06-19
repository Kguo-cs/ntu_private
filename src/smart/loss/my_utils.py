def scene_centric(self, pos, heading, centering_pos, centering_heading, batch):
    heading = heading - centering_heading[batch]

    pos = pos - centering_pos[batch]

    cos_a = torch.cos(centering_heading)[batch]
    sin_a = torch.sin(centering_heading)[batch]

    x, y = pos[..., 0], pos[..., 1]
    x_rot = cos_a * x + sin_a * y
    y_rot = -sin_a * x + cos_a * y

    pos = torch.stack([x_rot, y_rot], dim=-1)

    return pos, heading


def preprocess(self, tokenized_map, tokenized_agent):
    batch = tokenized_map["batch"]

    pos = tokenized_map["position"]

    heading = tokenized_map["orientation"]

    centering_pos = scatter_mean(pos, batch, dim=0)

    centering_heading = scatter_mean(heading, batch, dim=0)

    pos, heading = self.scene_centric(pos, heading, centering_pos, centering_heading, batch)

    tokenized_map["position"] = pos
    tokenized_map["heading"] = heading

    pos = tokenized_agent["sampled_pos"]
    heading = tokenized_agent["sampled_heading"]
    batch = tokenized_agent["batch"]

    pos, heading = self.scene_centric(pos, heading, centering_pos[:, None], centering_heading[:, None], batch)

    tokenized_agent["sampled_pos"] = pos

    tokenized_agent["sampled_heading"] = heading

    tokenized_agent["centering_pos"] = centering_pos
    tokenized_agent["centering_heading"] = centering_heading

    return tokenized_map, tokenized_agent


def filter_map(self, tokenized_map, tokenized_agent):
    pos_a = tokenized_agent["sampled_pos"]
    n_step = pos_a.shape[1]

    pos_pl = tokenized_map["position"]
    mask = tokenized_agent["valid_mask"]
    batch_s = torch.cat(
        [
            tokenized_agent["batch"] + tokenized_agent["num_graphs"] * t
            for t in range(n_step)
        ],
        dim=0,
    )  # [n_agent*n_step]

    batch_pl = torch.cat(
        [
            tokenized_map["batch"] + tokenized_agent["num_graphs"] * t
            for t in range(n_step)
        ],
        dim=0,
    )  # [n_pl*n_step]

    mask_pl2a = mask.transpose(0, 1).reshape(-1)
    pos_s = pos_a.transpose(0, 1).flatten(0, 1)
    map_point_num = len(pos_pl)
    pos_pt = pos_pl.repeat(n_step, 1)
    edge_index_pl2a = radiusGraphNearest2(x=pos_s[:, :2],
                                          y=pos_pt[:, :2],
                                          r=self.pl2a_radius,
                                          batch_x=batch_s,
                                          batch_y=batch_pl,
                                          max_num_neighbors=self.pt2a_neighbor)
    edge_index_pl2a = edge_index_pl2a[:, mask_pl2a[edge_index_pl2a[1]]]
    used_point = torch.unique(edge_index_pl2a[0] % map_point_num)

    # edge_index_pl2pl = radiusGraphNearest2(x=pos_pl[used_point],
    #                                       y=pos_pl,
    #                                       r=20,
    #                                       batch_x=tokenized_map["batch"][used_point],
    #                                       batch_y=tokenized_map["batch"],
    #                                       max_num_neighbors=10)
    #
    # used_point=torch.unique(edge_index_pl2pl[0])

    used_mask = torch.isin(torch.arange(map_point_num, device=pos_s.device), used_point)

    for key in tokenized_map.keys():
        if key != 'token_traj_src':
            tokenized_map[key] = tokenized_map[key][used_mask]
    return tokenized_map


