from gymnasium import spaces, utils
from gymnasium.core import ObsType
import numpy as np
from typing import Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
import cv2

from miniworld.entity import Box, COLOR_NAMES
from miniworld.miniworld import MiniWorldEnv
from miniworld.params import DEFAULT_PARAMS


class OneRoom(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Environment in which the goal is to go to a red box placed randomly in one big room.
    The `OneRoom` environment has two variants. The `OneRoomS6` environment gives you
    a room with size 6 (the `OneRoom` environment has size 10). The `OneRoomS6Fast`
    environment also is using a room with size 6, but the turning and moving motion
    is larger.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing a RGB image of what the agents sees.

    ## Rewards:

    +(1 - 0.2 * (step_count / max_episode_steps)) when red box reached and zero otherwise.

    ## Arguments

    ```python
    env = gym.make("MiniWorld-OneRoom-v0")
    # or
    env = gym.make("MiniWorld-OneRoomS6-v0")
    # or
    env = gym.make("MiniWorld-OneRoomS6Fast-v0")
    ```

    """

    def __init__(self, size=10, max_episode_steps=180, **kwargs):
        assert size >= 2
        self.size = size

        MiniWorldEnv.__init__(self, max_episode_steps=max_episode_steps, **kwargs)
        utils.EzPickle.__init__(
            self, size=size, max_episode_steps=max_episode_steps, **kwargs
        )

        # Allow only movement actions (left/right/forward)
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)

        self.box = self.place_entity(Box(color="red"))
        self.place_agent()

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        if self.near(self.box):
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info


class OneRoomS6(OneRoom):
    def __init__(self, size=6, max_episode_steps=100, **kwargs):
        super().__init__(size=size, max_episode_steps=max_episode_steps, **kwargs)


# Parameters for larger movement steps, fast stepping
default_params = DEFAULT_PARAMS.no_random()
default_params.set("forward_step", 0.7)
default_params.set("turn_step", 45)


class OneRoomS6Fast(OneRoomS6):
    def __init__(
        self, max_episode_steps=50, params=default_params, domain_rand=False, **kwargs
    ):
        super().__init__(
            max_episode_steps=max_episode_steps,
            params=params,
            domain_rand=domain_rand,
            **kwargs
        )


default_params = DEFAULT_PARAMS.no_random()
default_params.set("forward_step", 0.7)
default_params.set("turn_step", 15)


class OneRoomS6FastMulti(OneRoomS6):
    _COLOR_NAMES = ['blue', 'red', 'green', 'yellow']

    def __init__(
        self,
        size=6,
        max_episode_steps=50,
        random_init=False,
        num_boxes=2,
        single_task=False,
        box_color_idx=0,
        params=default_params,
        domain_rand=False,
        **kwargs,
    ):
        self.single_task = single_task
        self.box_color_idx = box_color_idx
        self.random_init = random_init
        # print("Random init: ", self.random_init)
        self.num_boxes = num_boxes
        self.set_task(0)
        self.size = size
        super().__init__(
            size=size,
            max_episode_steps=max_episode_steps,
            params=params,
            domain_rand=domain_rand,
            **kwargs
        )
        # Add one more action for the "no-op" action
        self.action_space = spaces.Discrete(self.action_space.n + 1)

    def set_task(self, env_id):
        print(f"Setting the environment env id: {env_id}")
        self.env_id = env_id
        self.box_color = None
        rng = np.random.RandomState(seed=env_id)
        self.np_random = rng
        self.indices = list(range(self.num_boxes))
        self.color_names = self._COLOR_NAMES[:self.num_boxes]
        self.num_distractors = self.num_boxes - 1

    def _gen_world(self):
        self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)

        if self.box_color is None and self.random_init:
            raise NotImplementedError("Random init not implemented")
            # Place the boxes once then fix their poses
            if self.single_task:
                self.box_color = self.color_names[0]    # Fix the box color to be blue
            else:
                self.box_color = self.color_names[self.np_random.randint(len(self.color_names))]
            self.color_names.remove(self.box_color)

            self.box = self.place_entity(Box(color=self.box_color))
            self.box_position = self.box.pos.copy()
            self.box_direction = self.box.dir

            self.distractor_colors = []
            self.distractor_positions = []
            self.distractor_directions = []

            for _ in range(self.num_distractors):
                color = self.color_names[self.np_random.randint(0, len(self.color_names))]
                self.color_names.remove(color)

                distractor = Box(color=color, ghost=True)
                self.place_entity(distractor)

                self.distractor_colors.append(color)
                self.distractor_positions.append(distractor.pos.copy())
                self.distractor_directions.append(distractor.dir)

            self.agent = self.place_agent()
            self.initial_agent_position = self.agent.pos.copy()
            self.initial_agent_direction = self.agent.dir
        
        elif self.box_color is None:
            # Place the boxes in the corners
            # print("Placing the boxes in the corners")
            if self.single_task:
                idx = self.box_color_idx
            else:
                idx = self.np_random.choice(self.indices)
            self.indices.remove(idx)

            self.box_color = self.color_names[idx]
            rad = Box(color=self.box_color).radius

            # Let each color always be in the same corner
            corners = [0, 1, 2, 3]
            # # Shuffle the color-to-corner assignments
            # self.np_random.shuffle(corners)
            init_range = 0.0
            corner_ranges = {
                0: (rad, rad, rad + init_range, rad + init_range),
                1: (self.size - rad - init_range, rad, self.size - rad, rad + init_range),
                2: (self.size - rad - init_range, self.size - rad - init_range, self.size - rad, self.size - rad),
                3: (rad, self.size - rad - init_range, rad + init_range, self.size - rad),
            }
            box_corner = corners[idx]

            self.box = self.place_entity(
                Box(color=self.box_color),
                min_x=corner_ranges[box_corner][0],
                min_z=corner_ranges[box_corner][1],
                max_x=corner_ranges[box_corner][2],
                max_z=corner_ranges[box_corner][3],
                dir=0.,
            )
            self.box_position = self.box.pos.copy()
            self.box_direction = self.box.dir

            # print(f'{self.box_color} is in corner {box_corner}')

            self.distractor_colors = []
            self.distractor_positions = []
            self.distractor_directions = []

            for i in range(self.num_distractors):
                idx = self.np_random.choice(self.indices)
                self.indices.remove(idx)
                color = self.color_names[idx]

                distractor = Box(color=color, ghost=True)
                distractor_corner = corners[idx]
                self.place_entity(
                    distractor,
                    min_x=corner_ranges[distractor_corner][0],
                    min_z=corner_ranges[distractor_corner][1],
                    max_x=corner_ranges[distractor_corner][2],
                    max_z=corner_ranges[distractor_corner][3],
                    dir=0.,
                )

                self.distractor_colors.append(color)
                self.distractor_positions.append(distractor.pos.copy())
                self.distractor_directions.append(distractor.dir)

                # print(f'{color} is in corner {distractor_corner}')

            self.agent = self.place_agent(
                min_x=self.size // 2,
                min_z=self.size // 2,
                max_x=self.size // 2,
                max_z=self.size // 2,
                dir=0.,
            )
            self.initial_agent_position = self.agent.pos.copy()
            self.initial_agent_direction = self.agent.dir
        else:
            self.box = self.place_entity(
                Box(color=self.box_color),
                pos=self.box_position,
                dir=self.box_direction)

            self.distractors = []
            for i in range(self.num_distractors):
                distractor = self.place_entity(
                    Box(color=self.distractor_colors[i], ghost=True),
                    pos=self.distractor_positions[i],
                    dir=self.distractor_directions[i])
                self.distractors.append(distractor)

            self.place_agent(pos=self.initial_agent_position, dir=self.initial_agent_direction)

    def step(self, action):
        if action == self.action_space.n - 1:
            self.step_count += 1
            obs = self.render_obs()
            truncation = (self.step_count >= self.max_episode_steps)
            info = {}
        else:
            obs, _, _, truncation, info = super().step(action)

        if self.near_and_face_box():
            reward = 1.0
        else:
            reward = 0.0
        termination = self.step_count >= self.max_episode_steps

        return obs, reward, termination, truncation, info

    def render(self, goal_text=False, action=0):
        img = super().render()
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        if goal_text:
            text = f"{self.box_color}, {int(self.near_and_face_box())}"
            text_color = (255, 255, 255)
            position = (1, 1)
            img = Image.fromarray(img)
            ImageDraw.Draw(img).text(position, text, fill=text_color)
            if action == 0:
                verbose_action = "left"
            elif action == 1:
                verbose_action = "right"
            elif action == 2:
                verbose_action = "forward"
            else:
                verbose_action = "no-op"
            text = f"{verbose_action}"
            text_color = (255, 255, 255)
            position = (1, 20)
            ImageDraw.Draw(img).text(position, text, fill=text_color)
            img = np.array(img)

        return img

    def opt_a(self, state, pose, dir):
        del state
        del pose
        del dir

        dir = self.agent.dir_vec[[0, -1]]
        agent_pos = self.agent.pos[[0, -1]]
        box_pos = self.box.pos[[0, -1]]

        diff_vec = box_pos - agent_pos

        # compute angle between diff_vec and dir (in degrees)
        angle = np.arccos(np.dot(diff_vec, dir) / (np.linalg.norm(diff_vec) * np.linalg.norm(dir))) * 180 / np.pi
        cross_product = np.cross(diff_vec, dir)
        if cross_product < 0:
            angle = 360 - angle

        if angle > 30 and angle < (360 - 30):
            if angle < 180:
                return 0
            else:
                return 1
        else:
            if self.near_enough(self.box):
                # Return the no-op
                return 3
            else:
                # Move towards the goal
                return 2

    def near_enough(self, ent0, ent1=None):
        if ent1 is None:
            ent1 = self.agent

        dist = np.linalg.norm(ent0.pos - ent1.pos)
        return dist < ent0.radius + ent1.radius + 1.5 * self.max_forward_step

    def near_and_face_box(self):
        if not self.near_enough(self.box):
            return False

        dir = self.agent.dir_vec[[0, -1]]
        agent_pos = self.agent.pos[[0, -1]]
        box_pos = self.box.pos[[0, -1]]

        diff_vec = box_pos - agent_pos

        # compute angle between diff_vec and dir (in degrees)
        angle = np.arccos(np.dot(diff_vec, dir) / (np.linalg.norm(diff_vec) * np.linalg.norm(dir))) * 180 / np.pi
        cross_product = np.cross(diff_vec, dir)
        if cross_product < 0:
            angle = 360 - angle

        if angle > 30 and angle < (360 - 30):
            return False
        else:
            return True
