# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Shell scripts must add ``verl_omni`` to a preset ``VERL_USE_EXTERNAL_MODULES``.

verl imports these modules before its ``verl.plugins`` entry points, so on a
platform registered by a hardware plugin the user sets the plugin module, for
example ``verl_hardware_plugin``, and the scripts append ``verl_omni`` after it.
A script that assigns a fixed value discards that setting. See
docs/start/install_xpu.md.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

EXPORT_LINE = "export VERL_USE_EXTERNAL_MODULES=${VERL_USE_EXTERNAL_MODULES:+${VERL_USE_EXTERNAL_MODULES},}verl_omni"
HYDRA_VALUE = "${oc.env:VERL_USE_EXTERNAL_MODULES,verl_omni}"

_ASSIGNMENT = re.compile(r"VERL_USE_EXTERNAL_MODULES=")


def _allowed(line: str) -> bool:
    stripped = line.strip()
    if stripped == EXPORT_LINE:
        return True
    # Hydra runtime_env override, which reads the value the script exported.
    return f"runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES={HYDRA_VALUE}'" in stripped


def test_scripts_add_verl_omni_to_preset_external_modules():
    offenders = []
    for path in sorted(ROOT.glob("**/*.sh")):
        if any(part.startswith(".") for part in path.relative_to(ROOT).parts):
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("#") or not _ASSIGNMENT.search(line):
                continue
            if not _allowed(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, f"Use `{EXPORT_LINE}` in shell, and `{HYDRA_VALUE}` in Hydra overrides:\n" + "\n".join(
        offenders
    )


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        (None, "verl_omni"),
        ("", "verl_omni"),
        ("verl_hardware_plugin", "verl_hardware_plugin,verl_omni"),
        # verl imports each module once, so a repeated verl_omni is harmless.
        ("verl_hardware_plugin,verl_omni", "verl_hardware_plugin,verl_omni,verl_omni"),
    ],
)
def test_export_line_puts_preset_modules_first(preset, expected):
    env = {k: v for k, v in os.environ.items() if k != "VERL_USE_EXTERNAL_MODULES"}
    if preset is not None:
        env["VERL_USE_EXTERNAL_MODULES"] = preset
    result = subprocess.run(
        ["bash", "-c", f'set -u; {EXPORT_LINE}; printf %s "$VERL_USE_EXTERNAL_MODULES"'],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert result.stdout == expected
