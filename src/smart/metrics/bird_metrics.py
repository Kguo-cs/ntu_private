import torch


# "linear_speed_likelihood",
# "linear_acceleration_likelihood",
# "angular_speed_likelihood",
# "angular_acceleration_likelihood",
# "distance_to_nearest_object_likelihood",
# "time_to_collision_likelihood"
# collision_indication_likelihood


def compute_kinematic_features(traj,heading,fps=29.97):
    velocity = (traj[:,:, 1:] - traj[:, :, :-1]) * fps

    acceleration = (velocity[:, :, 1:] - velocity[:, :, :-1]) * fps

    speed = torch.linalg.norm(velocity,dim=-1)

    acc=torch.linalg.norm(acceleration,dim=-1)

    dh_step = _wrap_angle(central_diff(heading, pad_value=np.nan) * 2) / 2
    dh = dh_step / seconds_per_step
    d2h_step = _wrap_angle(central_diff(dh_step, pad_value=np.nan) * 2) / 2
    d2h = d2h_step / (seconds_per_step ** 2)

    return speed, acc


def compute_bird_metrics(pred_traj,gt_traj,gt_mask,fps=29.97):

    pred_mask=pred_traj==0

    speed,acc,=compute_kinematic_features(pred_traj,pred_mask,fps=fps)

    gt_speed,gt_acc=compute_kinematic_features(gt_traj[:,None],gt_mask[:,None],fps=fps)

    gt_heading=1


    return 1

