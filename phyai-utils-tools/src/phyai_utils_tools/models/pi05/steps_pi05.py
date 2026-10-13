"""pi0.5 prompt-prep step — state discretization + prompt assembly.

:class:`StateTokenizerPrepareStep` is pi0.5-specific (the openpi/PaliGemma state
binning + prompt template), so it lives under ``models/pi05`` rather than the
generic ``processing/steps`` tree. Importing this module registers the step
under lerobot's name ``pi05_prepare_state_tokenizer_processor_step`` so the
lerobot ``policy_preprocessor.json`` loads. The generic tokenization that
consumes the prompt it produces is
:class:`~phyai_utils_tools.processing.steps.text_steps.TokenizerStep`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from phyai_utils_tools.models.pi05.ops_pi05 import (
    STATE_NUM_BINS,
    build_prompts,
    discretize_state,
)
from phyai_utils_tools.processing.pipeline import (
    ProcessorStep,
    ProcessorStepRegistry,
)
from phyai_utils_tools.processing.transition import PROMPT, STATE, TASK, Transition

logger = logging.getLogger(__name__)


def _clean_task(task: str) -> str:
    return task.strip().replace("_", " ").replace("\n", " ")


@ProcessorStepRegistry.register("pi05_prepare_state_tokenizer_processor_step")
@dataclass
class StateTokenizerPrepareStep(ProcessorStep):
    """Build the pi0.5 language prompt.

    ``discrete_state_input=True`` (lerobot / default pi0.5) discretizes
    ``STATE`` and writes ``"Task: <task>, State: <bins>;\\nAction: "``.

    ``discrete_state_input=False`` matches OpenPI ``PaligemmaTokenizer`` when
    state is not part of the prompt: the cleaned task plus a separate
    newline, with no ``Task:`` / ``State:`` wrapper. The CALVIN ABC→D SFT
    checkpoint was trained that way (``Pi0Config.discrete_state_input`` is
    false), so a state-bin prompt never matches its language tokens.
    """

    num_bins: int = STATE_NUM_BINS
    max_state_dim: int = 32
    discrete_state_input: bool = True

    def __call__(self, transition: Transition) -> Transition:
        tasks = transition.get(TASK)
        if tasks is None:
            raise ValueError("StateTokenizerPrepareStep requires a TASK entry.")
        if isinstance(tasks, str):
            tasks = [tasks]

        out = transition.copy()
        if not self.discrete_state_input:
            # OpenPI encodes the task and "\\n" separately; for this
            # tokenizer those token ids match a single string that ends
            # in a newline.
            out[PROMPT] = [f"{_clean_task(task)}\n" for task in tasks]
            if not getattr(self, "_logged_pi0_prompt", False):
                logger.info(
                    "pi05 prompt (discrete_state_input=false): %r",
                    out[PROMPT][0][:180],
                )
                self._logged_pi0_prompt = True
            return out

        state = transition.get(STATE)
        if state is not None:
            discretized = discretize_state(state, num_bins=self.num_bins)
            out[PROMPT] = build_prompts(list(tasks), discretized)
        else:
            out[PROMPT] = [
                f"Task: {_clean_task(task)};\nAction: " for task in tasks
            ]
        return out

    def get_config(self) -> dict[str, Any]:
        # lerobot persists no config for this step; keep parity (empty {}).
        return {}


__all__ = ["StateTokenizerPrepareStep"]
