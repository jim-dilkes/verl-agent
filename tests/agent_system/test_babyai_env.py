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
"""CPU tests for the BabyAI text env package.

Only ``gymnasium``, ``minigrid``, and ``pytest`` are required. The modules
under test are loaded by file path so that the heavy verl / ray / torch
imports in the parent packages are never triggered.

Set ``BABYAI_REFERENCE_MODULE`` to the path of the verl-latest recipe module
(``recipe/gigpo/babyai_text_env.py``) to run the cross-arm parity tests.
"""

import importlib.util
import os
import sys
import types

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
BABYAI_PKG = "agent_system.environments.env_package.babyai"
REFERENCE_MODULE_PATH = os.environ.get(
    "BABYAI_REFERENCE_MODULE",
    os.path.expanduser("~/Projects/verl-latest/recipe/gigpo/babyai_text_env.py"),
)
ENV_ID = "BabyAI-GoToRedBallGrey-v0"


def _load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    for i in range(1, name.count(".") + 1):
        parent = name.rsplit(".", i)[0]
        if parent not in sys.modules:
            sys.modules[parent] = types.ModuleType(parent)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


babyai_text = _load_module(
    f"{BABYAI_PKG}.babyai_text",
    os.path.join(REPO_ROOT, "agent_system/environments/env_package/babyai/babyai_text.py"),
)
projection = _load_module(
    f"{BABYAI_PKG}.projection",
    os.path.join(REPO_ROOT, "agent_system/environments/env_package/babyai/projection.py"),
)
prompts = _load_module(
    "agent_system.environments.prompts.babyai",
    os.path.join(REPO_ROOT, "agent_system/environments/prompts/babyai.py"),
)


@pytest.mark.parametrize(
    "text,expected_action,expected_valid",
    [
        ("I should explore. <action>turn left</action>", "turn left", True),
        ("<action>Go Forward</action>", "go forward", True),
        ("<action>pickup</action>", "pick up", True),
        ("<action>pick_up</action>", "pick up", True),
        ("<action>fly</action>", "go forward", False),
        ("no tags at all", "go forward", False),
        ("<action>turn right", "go forward", False),
        ("<action>toggle</action> and then <action>drop</action>", "toggle", True),
    ],
)
def test_parse_action(text, expected_action, expected_valid):
    _extracted, action, is_valid = babyai_text.parse_action(text)
    assert action == expected_action
    assert is_valid == expected_valid
    assert action in babyai_text.BABYAI_ACTION_SPACE


def test_projection_passes_raw_text_and_flags_validity():
    texts = [
        "<action>turn left</action>",
        "<action>fly</action>",
        "no tags at all",
        "reasoning first <action>toggle</action>",
    ]
    actions, valids = projection.babyai_projection(list(texts))
    assert actions == texts
    assert valids == [1, 0, 0, 1]


def test_reset_is_deterministic():
    env_a = babyai_text.BabyAITextEnv(ENV_ID)
    env_b = babyai_text.BabyAITextEnv(ENV_ID)
    obs_a, mission_a = env_a.reset(seed=7)
    obs_b, mission_b = env_b.reset(seed=7)
    assert obs_a == obs_b
    assert mission_a == mission_b
    assert isinstance(obs_a, str) and len(obs_a) > 0
    env_a.close()
    env_b.close()


def test_step_contract_and_invalid_action_handling():
    env = babyai_text.BabyAITextEnv(ENV_ID)
    env.reset(seed=3)
    next_obs, reward, done, info = env.step("<action>turn left</action>")
    assert isinstance(next_obs, str)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert info["is_valid_action"] is True
    assert info["action"] == "turn left"

    _obs, _reward, _done, info = env.step("gibberish")
    assert info["is_valid_action"] is False
    assert info["action"] == "go forward"
    env.close()


def test_anchor_obs_is_state_stable():
    """The same state reached twice yields the same text obs (GiGPO anchor property)."""
    env = babyai_text.BabyAITextEnv(ENV_ID)
    first_obs, _ = env.reset(seed=11)
    for _ in range(4):
        env.step("<action>turn left</action>")
    # Four left turns restore the initial orientation; "drop" with empty hands is a no-op.
    obs_after_loop, _reward, _done, _info = env.step("<action>drop</action>")
    assert obs_after_loop == first_obs
    env.close()


def test_no_his_template_matches_reference_system_prompt():
    """NO_HIS prompt == reference system prompt + blank line + raw observation."""
    mission = "go to the red ball"
    obs = "You see:\n- a red ball 2 steps ahead"
    rendered = prompts.BABYAI_TEMPLATE_NO_HIS.format(mission=mission, current_observation=obs)
    expected = babyai_text.build_system_prompt(mission) + "\n\n" + obs
    assert rendered == expected


@pytest.mark.skipif(not os.path.exists(REFERENCE_MODULE_PATH), reason="verl-latest reference module not available")
class TestParityWithReferenceArm:
    @classmethod
    def setup_class(cls):
        cls.reference = _load_module("babyai_reference_module", REFERENCE_MODULE_PATH)

    def test_system_prompt_identical(self):
        mission = "go to the red ball"
        assert babyai_text.build_system_prompt(mission) == self.reference.build_system_prompt(mission)

    @pytest.mark.parametrize("seed", [0, 7, 123, 65535, 1_000_000])
    def test_rollout_byte_identical(self, seed):
        env_ours = babyai_text.BabyAITextEnv(ENV_ID)
        env_ref = self.reference.BabyAITextEnv(ENV_ID)

        obs_ours, mission_ours = env_ours.reset(seed=seed)
        obs_ref, mission_ref = env_ref.reset(seed=seed)
        assert obs_ours == obs_ref
        assert mission_ours == mission_ref

        script = [
            "<action>turn left</action>",
            "<action>go forward</action>",
            "not an action",
            "<action>turn right</action>",
            "<action>go forward</action>",
            "<action>pick up</action>",
            "<action>toggle</action>",
            "<action>go forward</action>",
        ]
        for action_text in script:
            obs_ours, r_ours, d_ours, i_ours = env_ours.step(action_text)
            obs_ref, r_ref, d_ref, i_ref = env_ref.step(action_text)
            assert obs_ours == obs_ref
            assert r_ours == r_ref
            assert d_ours == d_ref
            assert i_ours["action"] == i_ref["action"]
            assert i_ours["is_valid_action"] == i_ref["is_valid_action"]
            if d_ours:
                break
        env_ours.close()
        env_ref.close()

    @pytest.mark.parametrize(
        "text",
        [
            "<action>turn left</action>",
            "<action>pick_up</action>",
            "<action>Move Forward</action>",
            "garbage",
            "<action>fly</action>",
        ],
    )
    def test_parse_action_identical(self, text):
        assert babyai_text.parse_action(text) == self.reference.parse_action(text)
