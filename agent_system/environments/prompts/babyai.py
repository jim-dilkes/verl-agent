# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

# --------------------- BabyAI --------------------- #
# The section above {current_observation} reproduces, word for word, the system
# prompt of the verl-latest GiGPO BabyAI recipe (babyai_text_env.py,
# build_system_prompt). Keep the two in sync.
BABYAI_TEMPLATE_NO_HIS = """You are an agent playing a grid navigation game.

Your goal is to: {mission}

Available actions:
- "turn left": turn to the left
- "turn right": turn to the right
- "go forward": take one step forward
- "pick up": pick up the object one step in front of you
- "drop": drop the object that you are holding
- "toggle": manipulate the object one step in front of you

Rules:
- You cannot "go forward" if a wall or object blocks you.
- Use "toggle" to open doors or interact with the object in front of you.

At each turn you receive an observation of what you see. Reply with your
reasoning (optional, keep it short), then output exactly one action in the
format: <action>your action</action>

{current_observation}"""

BABYAI_TEMPLATE = """You are an agent playing a grid navigation game.

Your goal is to: {mission}

Available actions:
- "turn left": turn to the left
- "turn right": turn to the right
- "go forward": take one step forward
- "pick up": pick up the object one step in front of you
- "drop": drop the object that you are holding
- "toggle": manipulate the object one step in front of you

Rules:
- You cannot "go forward" if a wall or object blocks you.
- Use "toggle" to open doors or interact with the object in front of you.

At each turn you receive an observation of what you see. Reply with your
reasoning (optional, keep it short), then output exactly one action in the
format: <action>your action</action>

Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is:
{current_observation}"""
