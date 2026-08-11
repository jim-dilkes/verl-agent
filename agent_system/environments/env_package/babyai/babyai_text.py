# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BabyAI text environment.

Copied from the verl-latest GiGPO recipe (``recipe/gigpo/babyai_text_env.py``)
so that both training stacks share byte-identical observation text, action
names, prompts, and action parsing. Do not edit the logic here without
updating the other side.

Requires only ``gymnasium`` and ``minigrid``. The text observation is generated
from the symbolic partial view (no babyai-text / balrog dependency). This module
has no verl imports so its logic can be unit-tested standalone.
"""

from typing import Optional

BABYAI_ACTION_SPACE = [
    "turn left",
    "turn right",
    "go forward",
    "pick up",
    "drop",
    "toggle",
]

DEFAULT_ACTION = "go forward"

ACTION_DESCRIPTIONS = {
    "turn left": "turn to the left",
    "turn right": "turn to the right",
    "go forward": "take one step forward",
    "pick up": "pick up the object one step in front of you",
    "drop": "drop the object that you are holding",
    "toggle": "manipulate the object one step in front of you",
}

ACTION_VARIANTS = {
    "turnleft": "turn left",
    "turnright": "turn right",
    "goforward": "go forward",
    "pickup": "pick up",
    "move forward": "go forward",
}

SYSTEM_PROMPT_TEMPLATE = """You are an agent playing a grid navigation game.

Your goal is to: {mission}

Available actions:
{action_list}

Rules:
- You cannot "go forward" if a wall or object blocks you.
- Use "toggle" to open doors or interact with the object in front of you.

At each turn you receive an observation of what you see. Reply with your
reasoning (optional, keep it short), then output exactly one action in the
format: <action>your action</action>"""


def build_system_prompt(mission: str) -> str:
    action_list = "\n".join(f'- "{action}": {desc}' for action, desc in ACTION_DESCRIPTIONS.items())
    return SYSTEM_PROMPT_TEMPLATE.format(mission=mission, action_list=action_list)


def parse_action(text: str) -> tuple[Optional[str], str, bool]:
    """Parse an LLM response into a BabyAI action.

    Args:
        text: full LLM response text.

    Returns:
        tuple: (extracted or None, action to execute, is_valid).
    """
    extracted = None
    open_tag, close_tag = "<action>", "</action>"
    if open_tag in text and close_tag in text:
        start = text.index(open_tag) + len(open_tag)
        end = text.find(close_tag, start)
        if end != -1:
            extracted = text[start:end].strip().lower().replace("_", " ")
            extracted = ACTION_VARIANTS.get(extracted, extracted)

    is_valid = extracted in BABYAI_ACTION_SPACE
    action = extracted if is_valid else DEFAULT_ACTION
    return extracted, action, is_valid


def describe_observation(image) -> str:
    """Build a text description from a minigrid agent-centric symbolic view.

    The view is a (W, H, 3) array of (object_idx, color_idx, state); the agent
    sits at (W // 2, H - 1) facing "up" (decreasing row). Style follows
    BabyAI-Text: one line per visible object with relative steps.

    Args:
        image: symbolic partial view from ``obs["image"]``.

    Returns:
        str: newline-joined description, e.g. "a red ball 2 steps ahead and 1 step to the left".
    """
    from minigrid.core.constants import IDX_TO_COLOR, IDX_TO_OBJECT, STATE_TO_IDX

    idx_to_state = {v: k for k, v in STATE_TO_IDX.items()}
    width, height, _ = image.shape
    agent_col, agent_row = width // 2, height - 1

    lines = []
    carrying = None
    nearest_wall = {"ahead": None, "left": None, "right": None}
    for col in range(width):
        for row in range(height):
            obj_idx, color_idx, state_idx = (int(v) for v in image[col, row])
            obj = IDX_TO_OBJECT.get(obj_idx, "unseen")
            if obj in ("unseen", "empty", "floor"):
                continue

            forward = agent_row - row
            lateral = col - agent_col
            if forward == 0 and lateral == 0:
                carrying = f"{IDX_TO_COLOR.get(color_idx, '')} {obj}".strip()
                continue

            if obj == "wall":
                # Only report the nearest wall straight ahead / left / right.
                if lateral == 0 and forward > 0:
                    if nearest_wall["ahead"] is None or forward < nearest_wall["ahead"]:
                        nearest_wall["ahead"] = forward
                elif forward == 0 and lateral < 0:
                    if nearest_wall["left"] is None or -lateral < nearest_wall["left"]:
                        nearest_wall["left"] = -lateral
                elif forward == 0 and lateral > 0:
                    if nearest_wall["right"] is None or lateral < nearest_wall["right"]:
                        nearest_wall["right"] = lateral
                continue

            color = IDX_TO_COLOR.get(color_idx, "")
            name = f"{color} {obj}".strip()
            if obj == "door":
                state = idx_to_state.get(state_idx, "")
                if state:
                    name = f"{state} {name}"
            lines.append(f"a {name} {_relative_position(forward, lateral)}")

    for direction, dist in nearest_wall.items():
        if dist is not None:
            steps = "1 step" if dist == 1 else f"{dist} steps"
            lines.append(f"a wall {steps} {direction if direction == 'ahead' else 'to the ' + direction}")

    if carrying is not None:
        lines.append(f"You carry a {carrying}")

    if not lines:
        return "You see nothing notable."
    return "You see:\n" + "\n".join(f"- {line}" for line in lines)


def _relative_position(forward: int, lateral: int) -> str:
    parts = []
    if forward > 0:
        parts.append("1 step ahead" if forward == 1 else f"{forward} steps ahead")
    if lateral != 0:
        side = "right" if lateral > 0 else "left"
        dist = abs(lateral)
        parts.append(f"1 step to the {side}" if dist == 1 else f"{dist} steps to the {side}")
    return " and ".join(parts) if parts else "right here"


class BabyAITextEnv:
    """Minigrid BabyAI env with text observations and text actions.

    Collapses the gymnasium 5-tuple into (obs_text, reward, done, info).
    """

    def __init__(self, env_id: str = "BabyAI-GoToRedBallGrey-v0", invalid_action_penalty: float = 0.0):
        import gymnasium as gym
        import minigrid  # noqa: F401  (registers BabyAI envs)

        self.env = gym.make(env_id)
        self.invalid_action_penalty = invalid_action_penalty
        self.mission: str = ""

    @property
    def max_steps(self) -> int:
        return self.env.unwrapped.max_steps

    def reset(self, seed: Optional[int] = None) -> tuple[str, str]:
        obs, _info = self.env.reset(seed=seed)
        self.mission = obs["mission"]
        return describe_observation(obs["image"]), self.mission

    def step(self, action_text: str) -> tuple[str, float, bool, dict]:
        _extracted, action, is_valid = parse_action(action_text)
        action_idx = BABYAI_ACTION_SPACE.index(action)
        obs, reward, terminated, truncated, info = self.env.step(action_idx)
        reward = float(reward)
        if not is_valid:
            reward -= self.invalid_action_penalty
        info = dict(info or {})
        info.update({"is_valid_action": is_valid, "action": action, "success": terminated and reward > 0})
        return describe_observation(obs["image"]), reward, terminated or truncated, info

    def close(self):
        self.env.close()
