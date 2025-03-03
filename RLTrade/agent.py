import math
import random

from itertools import count

import numpy as np
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from RLTrade.model import DQN, ReplayMemory, Transition

class DQNAgent:
    def __init__(
            self,
            n_observations, 
            n_actions,
            memory_capacity=500000
        ):
        self.BATCH_SIZE = 128
        self.GAMMA = 0.99
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 1000
        self.TAU = 0.005
        self.LR = 1e-4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.n_observations = n_observations
        self.n_actions = n_actions

        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.LR)
        self.memory = ReplayMemory(memory_capacity)
        self.steps_done = 0

    def greedy(self, state):
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return self.policy_net(state).max(1).indices.view(1, 1)
        
    def epsilon_greedy(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            return self.greedy(state)
        else:
            return torch.tensor([[random.choice(range(self.n_actions))]], device=self.device, dtype=torch.long)
        
    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def train(self, env, num_episodes=20):
        history_metrics = []
        for i_episode in range(num_episodes):
            # action_stats = {0: 0, 1: 0, 2: 0}
            # Initialize the environment and get its state
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

            episode_reward = 0
            for t in count():
                action = self.epsilon_greedy(state)
                # action_stats[action.item()] += 1
                observation, reward, terminated, truncated, _ = env.step(action.item())
                # print(observation)
                reward = torch.tensor([reward], device=self.device)
                episode_reward += reward.item()
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    metric = env.unwrapped.get_metrics()
                    metric["episodic_reward"] = episode_reward
                    history_metrics.append(metric)
                    break
        return history_metrics, env
    
    def eval(self, env):
        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        done = False
        while not done:
            action = self.greedy(state)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
            done = terminated or truncated
        return env
    
class XGBoostAgent:
    def __init__(
            self, 
            n_observations, 
            n_actions,
            memory_capacity=500000
        ):
        self.n_observations = n_observations
        self.n_actions = n_actions
        self.memory = ReplayMemory(memory_capacity)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.0
        self.epsilon_decay = 0.5
        self.model = None

        self.DISCOUNT = 0.95

    def getQvalues(self, state):
        """Get Q-values for a given state."""
        if self.model is None:
            return np.zeros(self.n_actions)

        np_state = np.reshape(state, [1, self.n_observations])
        actions = np.arange(self.n_actions).reshape(-1, 1)
        input_data = np.hstack((np.tile(np_state, (self.n_actions, 1)), actions))
        input_xgb = xgb.DMatrix(input_data)
        action_Qs = self.model.predict(input_xgb)
        return action_Qs

    def getMaxQ(self, state):
        """Get maximum Q-value for a given state."""
        return np.max(self.getQvalues(state))

    def act(self, state, epilson_greedy=True):
        """Choose an action based on the current policy."""
        if epilson_greedy and (np.random.rand() < self.epsilon or self.model is None):
            return random.randrange(self.n_actions)
        return np.argmax(self.getQvalues(state))

    def replay(self):
        """Train the model using the memory."""
        max_depth = 6
        num_round = 3

        gamma = 0.05
        eta = 0.75

        params = {
            'max_depth': max_depth, 
            'eta': eta, 
            'gamma': gamma, 
            'objective': 'reg:linear',
        }

        mem = self.memory.get_all()
        mem = Transition(*zip(*mem))
        states = np.array(mem.state)
        actions = np.array(mem.action).reshape(-1, 1)
        rewards = np.array(mem.reward)

        input_training = np.hstack((states, actions))
        training_data = xgb.DMatrix(input_training, label=rewards)

        self.model = xgb.train(params, training_data, num_boost_round=num_round)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def n_step_return(self, t, temp_memory, n=100):
        """Compute the n-step return."""
        target = 0.

        for i in range(n):
            if t + i >= len(temp_memory):
                break
            target += (self.DISCOUNT ** i) * temp_memory[t+i].reward

        final_state = temp_memory[t+n].next_state if t + n < len(temp_memory) else None
        if final_state is not None:
            target += (self.DISCOUNT ** n) * self.getMaxQ(final_state)

        return target

    def train(self, env, num_episodes=20, replay_every=2):
        history_metrics = []
        for i_episode in range(num_episodes):
            temp_memory = []
            state, info = env.reset()
            done = False
            episode_reward = 0
            for t in count():
                action = self.act(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward

                done = terminated or truncated
                if terminated:
                    next_state = None
                temp_memory.append(Transition(state, action, next_state, reward))
                state = next_state

                if done:
                    metric = env.unwrapped.get_metrics()
                    metric["episodic_reward"] = episode_reward
                    history_metrics.append(metric)
                    break
    
            for t in range(len(temp_memory)):
                G_t = self.n_step_return(t, temp_memory)
                self.memory.push(
                    temp_memory[t].state.copy(), 
                    temp_memory[t].action, 
                    temp_memory[t].next_state.copy(), 
                    G_t
                )

            if (i_episode+1) % replay_every == 0:
                self.replay()

        return history_metrics, env
    
    def eval(self, env):
        state, info = env.reset()
        done = False
        while not done:
            action = self.act(state, epilson_greedy=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = next_state
            done = terminated or truncated
        return env