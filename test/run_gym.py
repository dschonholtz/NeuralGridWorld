import gym_examples
import gym
from model import CNNLSTM
import torch
import numpy as np


def run_nn():
    num_agents = 5
    models = [CNNLSTM(4) for _ in range(num_agents)]

    env = gym.make('gym_examples/GridWorld-v0', render_mode='human', size=75, num_agents=num_agents, num_targets=70)
    observation, info = env.reset()

    criterion = torch.nn.CrossEntropyLoss()  # mean-squared error for regression
    for epoch in range(1000):
        actions = []
        outputs = None
        for i in range(num_agents):
            model = models[i]
            # add a dim to the observation to pass it to the network
            obs = torch.from_numpy(np.expand_dims(observation[f"agent{i}_obs"],  axis=0)).float()
            outputs = model.forward(obs)  # forward pass
            # map the highest output to action
            action = torch.argmax(outputs)
            actions.append(action)
        observation, rewards, terminated, truncated, info = env.step(actions)
        for i in range(num_agents):
            model = models[i]
            action = actions[i]
            reward = rewards[i]
            result_reward = torch.tensor([-2, -2, -2, -2])
            if reward > 0:
                result_reward[action] = reward * 10
            loss = criterion(outputs, result_reward.float()[None, :])
            loss.backward(retain_graph=True)  # calculates the loss of the loss function
            optimizer = torch.optim.Adam(model.parameters(), lr=.005)
            optimizer.zero_grad()
            optimizer.step()  # improve from loss, i.e backprop
            if epoch % 100 == 0:
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


def main():
    env = gym.make('gym_examples/GridWorld-v0', render_mode='human', size=100, num_agents=5, num_targets=30)
    observation, info = env.reset()

    for _ in range(1000):
        actions = [env.action_space.sample() for _ in range(env.num_agents)]
        observation, reward, terminated, truncated, info = env.step(actions)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == '__main__':
    run_nn()
