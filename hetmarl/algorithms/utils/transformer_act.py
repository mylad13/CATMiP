import torch
from torch.distributions import Normal
from torch.distributions import Categorical
from torch.nn import functional as F

def discrete_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv, 
                                available_actions=None, rnn_states=None, deterministic=False, active_agents=None):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    output_action = torch.zeros((batch_size, n_agent, 1), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)
    returned_rnn_states = torch.zeros_like(rnn_states)
    for i in range(n_agent):
        logit, all_rnn_states = decoder(shifted_action, obs_rep, obs, rnn_states)
        logit = logit[:, i, :]
        returned_rnn_states[:, i, :] = all_rnn_states[:, i, :]
        if available_actions is not None:
            logit[available_actions[:, i, :] == 0] = -1e10

        distri = Categorical(logits=logit)
        action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action.unsqueeze(-1)
        output_action_log[:, i, :] = action_log.unsqueeze(-1)
        if i + 1 < n_agent:
            shifted_action[:, i + 1, 1:] = F.one_hot(action, num_classes=action_dim)
    return output_action, output_action_log, returned_rnn_states

def multidiscrete_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                available_actions=None, deterministic=False, active_agents=None):
    
    # shifted_action = torch.zeros((batch_size, n_agent, action_dim[0] + 1, action_dim[1] + 1)).to(**tpdv)
    # shifted_action[:, 0, 0, 0] = 1
    # print("batch size is: ", batch_size)
    shifted_action = torch.zeros((batch_size, n_agent, 1 + action_dim[0] + action_dim[1])).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    
    output_action = torch.zeros((batch_size, n_agent, len(action_dim)), dtype=torch.long)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        logits = decoder(shifted_action, obs_rep, obs)

        cnt1 = 0
        # agent_action=[]
        for logit in logits:
            logit_i = logit[:, i, :]
            if available_actions is not None:
                if cnt1 == 0: #for the row action
                    mask = torch.full((batch_size, action_dim[0]), True)
                    for b in range(batch_size):
                        for r in range(action_dim[0]):
                            mask[b,r] = torch.any(available_actions[b,i,r]) # rows that have no available columns set mask to false
                    logit_i[mask == False] = -1e10
                    pass
                elif cnt1 == 1: #for the column action
                    mask = torch.full((batch_size, action_dim[1]), True)
                    for b in range(batch_size):
                        for c in range(action_dim[1]):
                            mask[b,c] = available_actions[b, i, output_action[b, i, 0], c]==1 # columns that have no available actions set mask to false
                    logit_i[mask == False] = -1e10
            distri = Categorical(logits=logit_i)
            action = distri.probs.argmax(dim=-1) if deterministic else distri.sample()
            action_log = distri.log_prob(action)
            output_action[:, i, cnt1] = action.unsqueeze(-2)
            output_action_log[:, i, cnt1] = action_log.unsqueeze(-2)
            cnt1 += 1
        if i + 1 < n_agent:
            out_action = output_action[:, i, :].numpy()
            cnt2 = 0 #counting over batches
            for act in out_action:
                # shifted_action[cnt2, i + 1, :, :] = 0 # was previously shifted_action[cnt2, i + 1, 1:, 1:] = 0
                # shifted_action[cnt2, i + 1, 1+act[0],1+act[1]] = 1 #one-hot encoding the action
                shifted_action[cnt2, i + 1, :] = 0 # was previously shifted_action[cnt2, i + 1, 1:, 1:] = 0
                shifted_action[cnt2, i + 1, 1 + act[0]] = 1 #first one-hot encoding (row)
                shifted_action[cnt2, i + 1, 1 + action_dim[0] + act[1]] = 1 #second one-hot encoding (column)
                # shifted_action[cnt2, i + 1, 1 + action_dim[0] + action_dim[1] + act[2]] = 1 #third one-hot encoding (interaction)

                cnt2 += 1
    return output_action, output_action_log

def discrete_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None, rnn_states=None, active_agents=None):
    one_hot_action = F.one_hot(action.squeeze(-1), num_classes=action_dim)  # (batch, n_agent, action_dim)
    shifted_action = torch.zeros((batch_size, n_agent, action_dim + 1)).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    if active_agents is not None:
        shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :] * active_agents[:, :-1, :]
    else:
        shifted_action[:, 1:, 1:] = one_hot_action[:, :-1, :]
    logit, _ = decoder(shifted_action, obs_rep, obs, rnn_states)
    if available_actions is not None:
        logit[available_actions == 0] = -1e10

    distri = Categorical(logits=logit)
    action_log = distri.log_prob(action.squeeze(-1)).unsqueeze(-1)
    entropy = distri.entropy().unsqueeze(-1)
    if active_agents is not None:
        entropy = (entropy*active_agents).sum()/active_agents.sum() #added active agents mask for asynch
    else:
        entropy = entropy.mean()
    return action_log, entropy

def multidiscrete_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv,
                          available_actions=None, active_agents=None):
    # print("action shape is: ", action.shape) # (n_rollout_threads*episode_length, n_agent, n_action) for centralized and synchronous training
    # shifted_action = torch.zeros((batch_size, n_agent, action_dim[0] + 1, action_dim[1] + 1)).to(**tpdv)
    # shifted_action[:, 0, 0, 0] = 1

    shifted_action = torch.zeros((batch_size, n_agent, 1 + action_dim[0] + action_dim[1])).to(**tpdv)
    shifted_action[:, 0, 0] = 1
    cnt2 = 0
    for act in action: # iterates in the batch
        act = act.cpu().numpy()
        for n in range(n_agent-1):
            if active_agents is not None:
                if not active_agents[cnt2, n]:
                    shifted_action[cnt2, 1+n, 0] = 1 # action of inactive agent
                else:
                    shifted_action[cnt2, 1+n, 1+act[n,0]] = 1 #first one-hot encoding (row)
                    shifted_action[cnt2, 1+n, 1 + action_dim[0] + act[n,1]] = 1 #second one-hot encoding (column)
            else:    
                shifted_action[cnt2, 1+n, 1+act[n,0]] = 1 #first one-hot encoding (row)
                shifted_action[cnt2, 1+n, 1 + action_dim[0] + act[n,1]] = 1 #second one-hot encoding (column)
            # shifted_action[cnt2, 1+n, 1 + action_dim[0] + action_dim[1] + act[n,2]] = 1 #third one-hot encoding (interaction)
        cnt2 += 1
    logits = decoder(shifted_action, obs_rep, obs)
    action_logs = []
    entropies = []
    cnt1 = 0
    for logit in logits:
        if available_actions is not None:
            # print("going through action masking")
            if cnt1 == 0: #for the row action
                mask = torch.full((batch_size, n_agent, action_dim[0]), True)
                for b in range(batch_size):
                    for a in range(n_agent):
                        for r in range(action_dim[0]):
                            mask[b,a,r] = torch.any(available_actions[b,a,r]) # rows that have no available columns set mask to false
                logit[mask == False] = -1e10
            elif cnt1 == 1: #for the column action
                mask = torch.full((batch_size, n_agent, action_dim[1]), True)
                for b in range(batch_size):
                    for a in range(n_agent):
                        for c in range(action_dim[1]):
                            mask[b,a,c] = available_actions[b, a, action[b, a, 0], c]==1 # columns that have no available actions set mask to false
                logit[mask == False] = -1e10
        distri = Categorical(logits=logit)
        action_log = distri.log_prob(action[:,:,cnt1].squeeze(-1)).unsqueeze(-1)
        action_logs.append(action_log)
        entropy = distri.entropy().unsqueeze(-1)

        if active_agents is not None:
            entropies.append((entropy*active_agents).sum()/active_agents.sum()) #added active agents mask for asynch
        else:
            entropies.append(entropy.mean())
        cnt1 += 1
    action_logs = torch.cat(action_logs, -1) # not sure
    entropies = torch.tensor(entropies).mean()
    
    return action_logs, entropies


def continuous_autoregreesive_act(decoder, obs_rep, obs, batch_size, n_agent, action_dim, tpdv,
                                  deterministic=False):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    output_action = torch.zeros((batch_size, n_agent, action_dim), dtype=torch.float32)
    output_action_log = torch.zeros_like(output_action, dtype=torch.float32)

    for i in range(n_agent):
        act_mean = decoder(shifted_action, obs_rep, obs)[:, i, :]
        action_std = torch.sigmoid(decoder.log_std) * 0.5

        # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
        # distri = Normal(act_mean, log_std.exp())
        distri = Normal(act_mean, action_std)
        action = act_mean if deterministic else distri.sample()
        action_log = distri.log_prob(action)

        output_action[:, i, :] = action
        output_action_log[:, i, :] = action_log
        if i + 1 < n_agent:
            shifted_action[:, i + 1, :] = action

        # print("act_mean: ", act_mean)
        # print("action: ", action)

    return output_action, output_action_log


def continuous_parallel_act(decoder, obs_rep, obs, action, batch_size, n_agent, action_dim, tpdv):
    shifted_action = torch.zeros((batch_size, n_agent, action_dim)).to(**tpdv)
    shifted_action[:, 1:, :] = action[:, :-1, :]

    act_mean = decoder(shifted_action, obs_rep, obs)
    action_std = torch.sigmoid(decoder.log_std) * 0.5
    distri = Normal(act_mean, action_std)

    # log_std = torch.zeros_like(act_mean).to(**tpdv) + decoder.log_std
    # distri = Normal(act_mean, log_std.exp())

    action_log = distri.log_prob(action)
    entropy = distri.entropy()
    return action_log, entropy
