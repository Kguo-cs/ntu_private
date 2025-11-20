import torch

def plot_reward(pred,data,tokenized_agent,discriminator,map_feature):
    # first set (top row)
    pred = torch.load("./waymo_data/pred.pt")
    scenario_path_A = data["tfrecord_path"][0]
    sampled_pos = pred["sampled_pos"]  # torch.round(tokenized_agent["sampled_pos"]*10)/10##
    sampled_heading = pred[ "sampled_heading"]  # torch.round(wrap_angle(tokenized_agent["sampled_heading"])/np.pi*30)*np.pi/30#

    disc_out = discriminator.predict_agent(tokenized_agent["sampled_idx"],
                                                        None,
                                                        tokenized_agent["valid_mask"],  # expert_
                                                        sampled_pos,
                                                        sampled_heading,
                                                        tokenized_agent,
                                                        map_feature,
                                                        [],
                                                        None,
                                                        # latent_z=tokenized_agent["latent_z"]
                                                        )  # [0]#Metrics-Guided Adversarial Training
    ego_rewards_A, nei_sum_rewards_A = disc_out[2]  # .detach()

    disc_val_A = (ego_rewards_A + nei_sum_rewards_A).detach().cpu().numpy()  # shape [N, K]

    # second set (bottom row)
    tokenized_agent_B, pred_B, data_B, map_feature_B = torch.load("./waymo_data/pred2.pt")
    scenario_path_B = data_B["tfrecord_path"][0]

    sampled_pos = pred_B["sampled_pos"]  # torch.round(tokenized_agent["sampled_pos"]*10)/10##
    sampled_heading = pred_B[
        "sampled_heading"]  # torch.round(wrap_angle(tokenized_agent["sampled_heading"])/np.pi*30)*np.pi/30#

    disc_out = discriminator.predict_agent(tokenized_agent_B["sampled_idx"],
                                                        None,
                                                        tokenized_agent_B["valid_mask"],  # expert_
                                                        sampled_pos,
                                                        sampled_heading,
                                                        tokenized_agent_B,
                                                        map_feature_B,
                                                        [],
                                                        None,
                                                        # latent_z=tokenized_agent["latent_z"]
                                                        )  # [0]#Metrics-Guided Adversarial Training
    ego_rewards_B, nei_sum_rewards_B = disc_out[2]  # .detach()

    disc_val_B = (ego_rewards_B + nei_sum_rewards_B).detach().cpu().numpy()

    torch.save((tokenized_agent, scenario_path_A, disc_val_A, pred,tokenized_agent_B, scenario_path_B, disc_val_B, pred_B),"pred_all.pt")

    plot_rollout_frames_pair(
        tokenized_agent, scenario_path_A, disc_val_A, pred,
        tokenized_agent_B, scenario_path_B, disc_val_B, pred_B,
        frames=(30, 50, 70, 90),
        radius_m=45.0,
        vmin=0.0, vmax=2.0,  # shared color scale
        cmap_name="RdYlGn"
    )
