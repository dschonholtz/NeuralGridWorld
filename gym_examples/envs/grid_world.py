import gym
from gym import spaces
import pygame
import numpy as np


class GridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, render_mode=None, size=5, num_agents=1, num_targets=1, agent_observation_radius=7):
        self.size = size  # The size of the square grid
        self.window_size = 1024  # The size of the PyGame window
        self.num_targets = num_targets

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        temp_space = {}
        for i in range(num_agents):
            temp_space["agent" + str(i)] = spaces.Box(0, size - 1, shape=(2,), dtype=int)
            temp_space["agent" + str(i) + "_obs"] = spaces.Box(
                0, size - 1, shape=(2 * agent_observation_radius + 1, 2 * agent_observation_radius + 1), dtype=int)
        for i in range(num_targets):
            temp_space["target" + str(i)] = spaces.Box(0, size - 1, shape=(2,), dtype=int)
        self.observation_space = spaces.Dict(temp_space)
        self._target_locations_map = {}
        self._target_locations = []

        # We have 4 actions, corresponding to "right", "up", "left", "down", "right"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space` to 
        the direction we will walk in if that action is taken.
        I.e. 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        self.num_agents = num_agents
        self._agent_locations = [self.np_random.integers(0, self.size, size=2, dtype=int) for _ in range(self.num_agents)]

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None

    def _get_agent_obs(self, agent_location, agent_observation_radius):
        obs = np.zeros((2 * agent_observation_radius + 1, 2 * agent_observation_radius + 1), dtype=int)
        for i in range(-agent_observation_radius, agent_observation_radius + 1):
            for j in range(-agent_observation_radius, agent_observation_radius + 1):
                if str(agent_location[0] + i) + "," + str(agent_location[1] + j) in self._target_locations_map:
                    obs[i + agent_observation_radius, j + agent_observation_radius] = 1
        return obs

    def _get_obs(self):
        obs = {}
        for i in range(self.num_agents):
            obs["agent" + str(i)] = self._agent_locations[i]
            obs["agent" + str(i) + "_obs"] = self._get_agent_obs(self._agent_locations[i], 3)
        for i in range(self.num_targets):
            obs["target" + str(i)] = self._target_locations[i]
        return obs

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self._agent_locations = [self.np_random.integers(0, self.size, size=2, dtype=int) for _ in range(
            self.num_agents)]

        # We will sample the target's location randomly until it does not coincide with the agent's location
        for i in range(self.num_targets):
            target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            while str(target_location[0]) + "," + str(target_location[1]) in self._target_locations_map:
                target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            self._target_locations.append(target_location)
            self._target_locations_map[str(target_location[0]) + "," + str(target_location[1])] = i

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, actions):
        rewards = np.array([0.0 for _ in range(self.num_agents)])
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        directions = [self._action_to_direction[int(action)] for action in actions]
        # We use `np.clip` to make sure we don't leave the grid
        for i in range(self.num_agents):
            self._agent_locations[i] = np.clip(
                self._agent_locations[i] + directions[i], 0, self.size - 1
            )
            if str(self._agent_locations[i][0]) + "," + str(self._agent_locations[i][1]) in self._target_locations_map:
                rewards[i] = 1
                new_target = self.np_random.integers(
                    0, self.size, size=2, dtype=int
                )
                self._target_locations[
                    self._target_locations_map[
                        str(self._agent_locations[i][0]) + "," + str(self._agent_locations[i][1])]
                ] = new_target
                del self._target_locations_map[
                        str(self._agent_locations[i][0]) + "," + str(self._agent_locations[i][1])]
                self._target_locations_map[str(new_target[0]) + "," + str(new_target[1])] = i
            else:
                rewards[i] = 0
        observation = self._get_obs()

        if self.render_mode == "human":
            self._render_frame()

        info = self._get_info()

        return observation, rewards, False, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the targets
        for target_location in self._target_locations:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * target_location,
                    (pix_square_size, pix_square_size),
                ),
            )
        # Now we draw the agent
        for i in range(self.num_agents):
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (self._agent_locations[i] + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
