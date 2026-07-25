#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
import time

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# In[2]:


env_id = "CartPole-v1"
num_envs = 1
total_timesteps = 100_000
learning_rate = 2e-3
buffer_size = 10_000
gamma = 0.99
tau = 1.0
target_network_frequency = 50
batch_size = 32
start_eps = 1
end_eps = 0.01
exploration_duration = int(total_timesteps * 0.1)
num_steps_before_training = 5_000
train_frequency = 4
seed = None
video_path = "dqn_cartpole_videos"


# In[3]:


def make_env(env_id, capture_video, seed):
    if capture_video:
        env = gym.make(env_id, render_mode="rgb_array")
        env = gym.wrappers.RecordVideo(env, video_path, episode_trigger=lambda episode: True)
    else:
        env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if seed is not None:
        env.action_space.seed(seed)

    return env


# In[4]:


class QNetwork(nn.Module):
    def __init__(self, envs, n_hidden=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(envs.single_observation_space.shape[0], n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, envs.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x)


# In[5]:


def linear_schedule(start_eps, end_eps, duration, t):
    slope = (end_eps - start_eps) / duration
    return max(slope * t + start_eps, end_eps)


# In[6]:


envs = gym.vector.SyncVectorEnv([lambda: make_env(env_id, False, seed if seed is None else seed+i) for i in range(num_envs)])


# In[7]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[8]:


q_network = QNetwork(envs).to(device)
optimizer = optim.Adam(q_network.parameters(), lr=learning_rate)
target_network = QNetwork(envs).to(device)
target_network.load_state_dict(q_network.state_dict())


# In[9]:


class ReplayBuffer:
    def __init__(self, obs_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros(size, dtype=np.uint8)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.uint8)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32): 
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(s=self.obs1_buf[idxs],
                    s2=self.obs2_buf[idxs],
                    a=self.acts_buf[idxs],
                    r=self.rews_buf[idxs],
                    d=self.done_buf[idxs])


# In[10]:


rb = ReplayBuffer(envs.single_observation_space.shape[0], buffer_size)


# In[11]:


def np2torch(a, to_float=True):
    if to_float:
        dtype = torch.float32
    else:
        dtype = torch.int64
    return torch.as_tensor(a, dtype=dtype, device=device)


# In[12]:


# Training Loop
episode_returns = []
losses = []
start_time = time.time()
obs, _ = envs.reset(seed=seed)
autoreset = np.zeros(num_envs, dtype=bool)

for global_step in range(total_timesteps):

    epsilon = linear_schedule(start_eps, end_eps, exploration_duration, global_step)

    if random.random() < epsilon:
        actions = np.array([envs.single_action_space.sample() for _ in range(num_envs)])
    else:
        q_values = q_network(np2torch(obs))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()

    next_obs, rewards, dones, truncateds, infos = envs.step(actions)

    # Vectorized environments store terminal info in infos["final_info"]
    if "final_info" in infos:
        for i, final_info in enumerate(infos["final_info"]):
            # final_info will be None for envs that didn't terminate this step
            if final_info is not None and "episode" in final_info:
                # .item() extracts the float from the numpy array (e.g. array([22.]) -> 22.0)
                ret = final_info["episode"]["r"].item()
                episode_returns.append(ret)
                print(f"globl_step={global_step}, episode={len(episode_returns)}, return={ret:.2f}")

    for i in range(num_envs):
        if not autoreset[i]:
            rb.store(obs[i], actions[i], rewards[i], next_obs[i], dones[i])

    obs = next_obs

    autoreset = np.logical_or(truncateds, dones)

    if global_step > num_steps_before_training:
        if global_step % train_frequency == 0:
            batch = rb.sample_batch(batch_size)


            with torch.no_grad():
                target_max, _ = target_network(np2torch(batch["s2"])).max(dim=1)
                td_target = np2torch(batch["r"].flatten()) + gamma * np2torch(1 - batch["d"].flatten()) * target_max

            a = np2torch(batch["a"], to_float=False)
            a = a.reshape(-1, 1)
            s = np2torch(batch["s"])
            qsa = q_network(s)

            pred = qsa.gather(1, a).squeeze()

            loss = F.mse_loss(pred, td_target)
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    if (global_step + 1) % 1000 == 0:
        print(f"steps per second: {(global_step + 1) / (time.time() - start_time):.2f}")

    if global_step % target_network_frequency == 0:
        for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
            target_network_param.data.copy_(tau * q_network_param.data + (1.0 - tau) * target_network_param.data)

    if len(episode_returns) > 10 and np.all(np.equal(np.array(episode_returns[-10:]), 500)):
        print(f"Solved in {global_step} steps!")
        break


# In[13]:


envs.close()


# In[14]:


def smooth(x, a=0.1):
    y = [x[0]]
    for xi in x[1:]:
        yi = a * xi + (1 - a) * y[-1]
        y.append(yi)
    return y


# In[15]:


plt.plot(episode_returns, alpha=0.2, label="raw")
plt.plot(smooth(episode_returns), label="smoothed")
plt.title("Episode Returns");
plt.show();


# In[16]:


plt.plot(losses)
plt.title("Losses");
plt.show();


# In[17]:


model_path = "dqn_cartpole_model.pt"
torch.save(q_network.state_dict(), model_path)


# In[18]:


envs_eval = gym.vector.SyncVectorEnv([lambda: make_env(env_id, True, seed if seed is None else seed+i) for i in range(num_envs)])
model = QNetwork(envs_eval).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()


# In[19]:


n_episodes_eval = 10
eval_returns = np.zeros(n_episodes_eval)
obs, _ = envs_eval.reset(seed=seed)
for i in range(n_episodes_eval):
    episode_done = False
    while not episode_done:
        q_values = model(np2torch(obs))
        actions = torch.argmax(q_values, dim=1).cpu().numpy()
        obs, rewards, dones, truncateds, infos = envs_eval.step(actions)

        if dones[0] or truncateds[0]:
            episode_done = True
            if "final_info" in infos:
                final_info = infos["final_info"][0]
                if final_info is not None and "episode" in final_info:
                    ret = final_info["episode"]["r"].item()
                    eval_returns[i] = ret
                    print(f"eval episode={i}, return={ret:.2f}")

envs_eval.close()


# In[20]:


plt.hist(eval_returns, bins=10)
plt.title("Evaluation Returns");
plt.show();


# In[ ]:




