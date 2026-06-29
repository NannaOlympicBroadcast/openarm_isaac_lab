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

The script launches Isaac Sim itself (via `AppLauncher`) and enables the camera
pipeline automatically, so just run it with the Python that has Isaac Lab.

### pip-installed Isaac Lab (Windows / Linux) — no `isaaclab.sh`

```powershell
# 1) bus server (any machine; can be the same box)
ssr bus serve --host 127.0.0.1 --port 8765

# 2) the bridge — plain python in the Isaac Lab pip env
python scripts\ssr_bridge\run_openarm_bridge.py --bus ws://127.0.0.1:8765 --task Isaac-Manip-OpenArm-v0
#   add --headless to run without a viewer window

# 3) drive a natural-language instruction (needs GEMINI_API_KEY in ~/.ssr/.env)
ssr arm do "把苹果放到橘子上" --bus-url ws://127.0.0.1:8765
```

### Source (git) Isaac Lab — via the launcher

```bash
ssr bus serve --host 0.0.0.0 --port 8765
./isaaclab.sh -p scripts/ssr_bridge/run_openarm_bridge.py \
    --bus ws://<brain-host>:8765 --task Isaac-Manip-OpenArm-v0 --headless
ssr arm do "把苹果放到橘子上" --bus-url ws://<brain-host>:8765
```

Start order matters: **bus server → bridge → `ssr arm do`**, so the bridge has
advertised its capabilities before the agent plans.
