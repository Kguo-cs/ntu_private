else:
# agent_contour = cal_polygon_contour(tokenized_agent_rollout["sampled_pos"][all_valid][:, 2:],
#                                  tokenized_agent_rollout["sampled_heading"][all_valid][:, 2:],
#                                  tokenized_agent_rollout["token_agent_shape"][all_valid][:, None])
#
# agent_reward= -torch.linalg.norm(agent_contour-gt_contour,dim=-1).mean(-1)
#
# agent_rewards= (agent_reward-agent_reward.mean())/(agent_reward.std()+1e-4)

with torch.no_grad():
    error_pred = self.encoder.discriminator.predict_agent(tokenized_agent_rollout["sampled_idx"],
                                                          tokenized_agent_rollout["goal_idx"],
                                                          tokenized_agent_rollout["valid_mask"],
                                                          tokenized_agent_rollout["sampled_pos"],
                                                          tokenized_agent_rollout["sampled_heading"],
                                                          tokenized_agent_rollout,
                                                          tokenized_agent_rollout["detach_map_feature"],
                                                          tokenized_agent_rollout["light_idx"],
                                                          None)[0]

    pos_error = torch.linalg.norm(error_pred[:, :, :2], dim=-1).mean()
    heading_error = (error_pred[:, :, 2]).abs().mean()

    self.log("train/agent_pos_error", pos_error.item(), on_step=True, batch_size=1)
    self.log("train/agent_heading_error", heading_error.item(), on_step=True, batch_size=1)

    agent_contour = cal_polygon_contour(tokenized_agent_rollout["sampled_pos"][all_valid][:, 2:],
                                        tokenized_agent_rollout["sampled_heading"][all_valid][:, 2:], token_agent_shape)

    pos_global, head_global = transform_to_global(error_pred[:, :, :2].reshape(-1, 1, 2),
                                                  error_pred[:, :, 2].reshape(-1, 1),
                                                  tokenized_agent_rollout["sampled_pos"][all_valid][:, 2:].reshape(-1,
                                                                                                                   2),
                                                  tokenized_agent_rollout["sampled_heading"][all_valid][:, 2:].reshape(
                                                      -1))

    pred_pos = pos_global.reshape(error_pred.shape[0], error_pred.shape[1], 2)

    pred_heading = head_global.reshape(error_pred.shape[0], error_pred.shape[1])

    pred_contour = cal_polygon_contour(pred_pos, pred_heading, token_agent_shape)

    agent_rewards = -torch.linalg.norm(agent_contour - pred_contour, dim=-1).mean(
        -1)  # torch.linalg.norm(error_pred,ord=1,dim=-1)

    # agent_rewards = (agent_rewards - agent_rewards.mean()) / (agent_rewards.std() + 1e-4)
