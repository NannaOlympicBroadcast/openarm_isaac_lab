# SSR Agent bridge for OpenArm

`run_openarm_bridge.py` connects the OpenArm Isaac Lab environment to an
[SSR Agent](https://github.com/NannaOlympicBroadcast/ssr-agent) event bus so the
agent can carry out **natural-language instructions** (e.g. "把苹果放到橘子上")
using an asynchronous, multi-agent paradigm: the agent discovers the arm's
capabilities, invokes a skill → suspends its session → a bus event wakes a checker
agent → it verifies the step from the camera → re-plans or advances.

## What was added to this repo

* **`Isaac-Manip-OpenArm-v0`** (and `…-Play-v0`) — a manipulation variant of the
  unimanual lift task
  (`unimanual/lift/config/manip_env_cfg.py`). It uses **end-effector pose
  control** (`DifferentialInverseKinematicsActionCfg`, body `openarm_hand`) plus
  the existing binary gripper, adds a 320×240 RGB `TiledCamera`
  (`scene["tiled_camera"]`), and two named objects — `object` (apple) and
  `orange`. The lift task already declared the arm action as
  `JointPositionActionCfg | DifferentialInverseKinematicsActionCfg`, so this only
  selects the IK variant.
* **`scripts/ssr_bridge/run_openarm_bridge.py`** — the launcher below.

The bridge advertises the arm's action space (introspected at runtime) and skills
(`pick`, `place_on`, `place_at`, `move_above`, `raw`) to the agent; the agent
plans only with what is advertised.

## Prerequisites (Isaac Lab python env)

```bash
python -m pip install -e /path/to/ssr-agent
python -m pip install -e /path/to/ssr-robotics
python -m pip install -e source/openarm   # this repo's package, if not already
```

## Run

```bash
# brain machine: SSR auto-starts an embedded bus server, or run a standalone one
ssr bus serve --host 0.0.0.0 --port 8765

# GPU machine: launch the bridge (cameras are enabled automatically)
./isaaclab.sh -p scripts/ssr_bridge/run_openarm_bridge.py \
    --bus ws://<brain-host>:8765 --task Isaac-Manip-OpenArm-v0 --headless

# brain machine: drive a natural-language instruction (needs GEMINI_API_KEY)
ssr arm do "把苹果放到橘子上" --bus-url ws://<brain-host>:8765
```
