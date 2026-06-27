# Copyright 2025 Enactic, Inc.
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

"""Run the OpenArm ⇄ SSR Agent event-bus bridge inside Isaac Sim.

Launches Isaac Sim, builds the manipulation env (``Isaac-Manip-OpenArm-v0`` — IK
end-effector control + binary gripper + TiledCamera + apple & orange), connects
to the SSR Agent bus server and serves ``arm.action.execute`` requests, replying
with ``arm.grasp.completed`` / ``arm.action.completed`` events (result + scene +
camera frame). It also advertises the arm's capabilities on
``arm.capabilities`` so the agent discovers the supported action types/skills at
runtime. This wakes the SSR agent's bus-handler turns, which verify each step and
re-plan or advance.

Requires ``ssr-agent`` (bus + shared protocol) and ``ssr-robotics`` (env runner)
installed in the Isaac Lab Python environment.

Example::

    # brain machine: an SSR process auto-starts an embedded bus server, or run:
    ssr bus serve --host 0.0.0.0 --port 8765
    # GPU machine (this script):
    ./isaaclab.sh -p scripts/ssr_bridge/run_openarm_bridge.py \
        --bus ws://<brain-host>:8765 --task Isaac-Manip-OpenArm-v0 --headless
    # brain machine: drive a natural-language instruction
    ssr arm do "把苹果放到橘子上" --bus-url ws://<brain-host>:8765
"""

import argparse
import time

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="OpenArm ⇄ SSR Agent bus bridge.")
parser.add_argument("--bus", type=str, required=True, help="ws:// URL of the SSR bus server")
parser.add_argument("--api-key", type=str, default=None, help="bus API key, if required")
parser.add_argument("--task", type=str, default="Isaac-Manip-OpenArm-v0", help="gym id")
parser.add_argument("--num_envs", type=int, default=1, help="number of environments")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# TiledCamera requires the cameras pipeline.
args_cli.enable_cameras = True

# launch omniverse app FIRST (before importing isaaclab env modules)
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

from ssr_robotics.env_runner import EnvRunner, connect_remote  # noqa: E402
from ssr_robotics.isaac_env import IsaacOpenArmEnv  # noqa: E402


def main() -> None:
    env = IsaacOpenArmEnv(task=args_cli.task, num_envs=args_cli.num_envs)
    client = connect_remote(args_cli.bus, source="openarm-env", api_key=args_cli.api_key)
    runner = EnvRunner(client, env).start()
    print(f"[ssr_bridge] connected to {args_cli.bus}; task={args_cli.task}. "
          "Serving arm.action.execute … (Ctrl-C to stop)")
    try:
        while simulation_app.is_running():
            time.sleep(0.1)
    finally:
        runner.stop()
        try:
            client.close()
        except Exception:
            pass
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
