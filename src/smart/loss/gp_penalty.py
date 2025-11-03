# if "a2a_entropy" in tokenized_agent.keys():
#     a2a_entropy=disc_out[2].mean()#tokenized_agent["a2a_entropy"].mean()
#     self.log("train/" + key + "_a2a_entropy", a2a_entropy.item(), on_step=True, batch_size=1)

# bce_loss=bce_loss+0.01*a2a_entropy
#
# if  key == "expert":
# expert_pos=tokenized_agent["sampled_pos"]#tokenized_agent["expert_sampled_pos"]#
# expert_sampled_heading=tokenized_agent["sampled_heading"]#tokenized_agent["expert_sampled_heading"]#
# expert_valid_mask=tokenized_agent["valid_mask"]#tokenized_agent["expert_valid_mask"]#
# pos=tokenized_agent["sampled_pos"]
# heading=tokenized_agent["sampled_heading"]
# valid_mask=tokenized_agent["valid_mask"]
#
# # batch_idx = tokenized_agent['batch']
# # alpha = torch.rand(size=(max(batch_idx) + 1, 1), device=batch_idx.device)
# #
# # alpha=alpha[batch_idx]
#
# #alpha= torch.rand((pos.size(0), pos.size(1)), device=pos.device)
# # interpolate_pos = pos.clone()#alpha[:,:,None] * expert_pos + (1 - alpha[:,:,None]) * pos
# # interpolate_heading =heading.clone() #alpha * expert_sampled_heading + (1 - alpha) * heading
# #
# # interpolates_pos=torch.cat((interpolate_pos, interpolate_heading[:,:,None]), dim=-1)
# #
# # interpolates=interpolates_pos[valid_mask]#[train_mask,2:]
# #
# # interpolates.requires_grad_(True)  # IMPORTANT
# #
# # interpolates_pos[valid_mask]=interpolates
# perturbed_pos=pos+0.01*torch.randn_like(pos)
# perturbed_heading=heading+0.01*torch.randn_like(heading)
#
# scores= self.encoder.discriminator.predict_agent(tokenized_agent["sampled_idx"],
#                                                 None,
#                                                 valid_mask,
#                                                 perturbed_pos,
#                                                 perturbed_heading ,
#                                                 tokenized_agent,
#                                                 tokenized_agent["detach_map_feature"],
#                                                 tokenized_agent["light_idx"],
#                                                 None)[0]
# # score_sum = scores.view(-1).sum()
# #
# # gradients = torch.autograd.grad(
# #     outputs=score_sum,
# #     inputs=interpolates,
# #     create_graph=True,
# #     retain_graph=True,
# #     only_inputs=True,
# # )[0]  # shape: [B, T, 3]
# gradients = (scores - logit) / 0.01
#
# gp=gradients.pow(2).mean()
#
# self.log("train/"+key+"_gp", gp, on_step=True, batch_size=1)
