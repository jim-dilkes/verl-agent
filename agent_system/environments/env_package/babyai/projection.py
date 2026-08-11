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

from typing import List

from agent_system.environments.env_package.babyai.babyai_text import parse_action


def babyai_projection(actions: List[str]):
    """
    A function to process the actions.
    actions: the list of actions to be processed, it is a list of strings.
    Expected format:
        ...optional reasoning...<action>turn left</action>

    The raw text is passed through unchanged; BabyAITextEnv parses it with the
    same parse_action, so both this projection and the env agree on validity
    and on the executed action (invalid -> "go forward").
    """
    valids = [0] * len(actions)

    for i in range(len(actions)):
        _extracted, _action, is_valid = parse_action(actions[i])
        if is_valid:
            valids[i] = 1

    return actions, valids
