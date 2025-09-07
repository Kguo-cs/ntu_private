if self.encoder.agent_encoder.use_infogail:

    logits = self.encoder.RecognitionQ.predict_agent(tokenized_agent_rollout["sampled_idx"],
                                                     tokenized_agent_rollout["goal_idx"],
                                                     tokenized_agent_rollout["valid_mask"],
                                                     tokenized_agent_rollout["sampled_pos"],
                                                     tokenized_agent_rollout["sampled_heading"],
                                                     tokenized_agent_rollout,
                                                     tokenized_agent_rollout["detach_map_feature"],
                                                     tokenized_agent_rollout["light_idx"],
                                                     None)[0]  # [all_valid]

    latent_z = tokenized_agent_rollout["latent_z"][all_valid]

    if logits.shape[-1] == self.encoder.agent_encoder.k_dim:
        # index = tokenized_agent["batch"][all_valid]

        # logits=logits[index]
        log_q = F.log_softmax(logits, dim=-1)
        action = latent_z[:, :, None].repeat(1, log_q.shape[1], 1)  # [:,1:-1,None]#
        z_logp = torch.gather(log_q, dim=-1, index=action).squeeze(-1)  # larger z likelihood # [B, Tm1, T_a]
        kl_prior = 0
    else:
        mu = logits[:, :, :self.encoder.agent_encoder.k_dim]
        logvar = logits[:, :, self.encoder.agent_encoder.k_dim:]

        z = latent_z.expand_as(mu)  # [B, T, k_dim]
        std = torch.exp(0.5 * logvar)

        base = Normal(loc=mu, scale=std)
        dist = Independent(base, reinterpreted_batch_ndims=1)  # event dim = last

        z_logp = dist.log_prob(z)  # shape: [...]

        mu_p = torch.zeros_like(mu)
        logvar_p = torch.zeros_like(logvar)

        var_q = logvar.exp()
        var_p = logvar_p.exp()

        kl_prior = 0.5 * (logvar_p - logvar + (var_q + (mu - mu_p).pow(2)) / var_p - 1).sum(-1).mean()
        self.log("train/mu", mu.mean().item(), on_step=True, batch_size=1)
        self.log("train/std", std.mean().item(), on_step=True, batch_size=1)

    loss_q = -z_logp.mean()  # increase the z likelihood
    expert_nll = expert_nll + loss_q + kl_prior

    self.log("train/loss_q", loss_q.item(), on_step=True, batch_size=1)
    mi_beta = 0.1
    agent_rewards = agent_rewards + mi_beta * z_logp.detach()

    # # Optional entropy regularizer on P to avoid overconfidence
    # p_probs = log_p.exp()
    # H_p = -(p_probs * log_p).sum(-1).mean()
    #
    # tau=0.01
    #
    # loss_prior = kl_qp - tau * H_p  # tau ~ 0.01–0.1

    # @self.log("train/loss_prior", loss_prior.item(), on_step=True, batch_size=1)
