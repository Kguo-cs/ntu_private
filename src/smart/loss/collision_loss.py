def get_collision_loss(self, tokenized_agent, tokenized_map, dis_col_pred, train_mask, all_valid, key):
    # col_pred = self.encoder.col_head(tokenized_agent["feat_a_nodetach"][all_valid])
    #
    # if train_mask is not None:
    #     col_pred=col_pred[train_mask]
    # else:
    #     col_pred = col_pred.reshape(-1,col_pred.shape[-1])

    # if self.encoder.pred_dis_aux:
    #     dis_col_pred = self.encoder.dis_col_head(dis_feature)

    if self.encoder.agent_encoder.use_sign_dist:
        sign_dist = signed_distance_boxes_sat_fast(tokenized_agent["sampled_pos"][:, 2:],
                                                   tokenized_agent["sampled_heading"][:, 2:],
                                                   tokenized_agent["shape"][:, :2],
                                                   tokenized_agent["batch"])

        col_flag = sign_dist < 0
        hist = {"min_val": -5.0, "max_val": 10.0, "num_bins": 3}

        target = value_to_hist_class(sign_dist, **hist)
        col_loss = F.cross_entropy(col_pred, target[train_mask])
        dis_loss = F.cross_entropy(dis_col_pred.reshape(-1, col_pred.shape[-1]), target[all_valid].reshape(-1))
    else:
        col_flag = oriented_box_collision(tokenized_agent["sampled_pos"][:, 2:],
                                          tokenized_agent["sampled_heading"][:, 2:],
                                          tokenized_agent["shape"][:, :2],
                                          tokenized_agent["batch"])[0].float()[all_valid]

        if train_mask is not None:
            col_flag = col_flag[train_mask]
        else:
            col_flag = col_flag.reshape(-1)

        # col_loss = self.bce_loss(col_pred[:,0], col_flag)
        col_loss = 0
        # self.log('train/' + key + '_col_loss', col_loss.item(), on_step=True, batch_size=1)

        if self.encoder.pred_dis_aux:
            dis_loss = self.bce_loss(dis_col_pred.reshape(-1), col_flag)
            self.log('train/' + key + '_dis_col_loss', dis_loss.item(), on_step=True, batch_size=1)
        else:
            dis_loss = 0

    self.log('train/' + key + '_col_rate', col_flag.float().mean().item(), on_step=True, batch_size=1)

    if self.encoder.map_encoder.pred_offroad:

        near_dist = corners_offroad_signed_distance_per_batch(tokenized_agent["sampled_pos"][:, 2:][all_valid],
                                                              tokenized_agent["sampled_heading"][:, 2:][all_valid],
                                                              tokenized_agent["shape"][:, :2][all_valid],
                                                              tokenized_agent["batch"][all_valid],
                                                              tokenized_map["global_edge"],
                                                              tokenized_map["batch_edge"],
                                                              )[1]
        offroad_flag = (near_dist < 0).float()

        if train_mask is not None:
            valid_off_flag = offroad_flag[train_mask]
        else:
            valid_off_flag = offroad_flag.reshape(-1)

        off_road_loss = self.bce_loss(col_pred[:, 1], valid_off_flag)

        if self.encoder.pred_dis_aux:
            dis_off_road_loss = self.bce_loss(dis_col_pred[:, :, 1], offroad_flag[all_valid])
            self.log('train/' + key + '_dis_off_loss', dis_off_road_loss.item(), on_step=True, batch_size=1)
        else:
            dis_off_road_loss = 0

        self.log('train/' + key + '_off_road_loss', off_road_loss.item(), on_step=True, batch_size=1)
        self.log('train/' + key + '_offroad_rate', valid_off_flag.mean().item(), on_step=True, batch_size=1)

        dis_loss = dis_loss + dis_off_road_loss
        col_loss = col_loss + off_road_loss

    return 0.1 * col_loss + dis_loss
