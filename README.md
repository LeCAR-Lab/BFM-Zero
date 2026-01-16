<h1 align="center"> BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning </h1>

<div align="center">

[[Website]](https://lecar-lab.github.io/BFM-Zero/)
<!-- [[Arxiv]](https://lecar-lab.github.io/SoFTA/) -->
<!-- [[Video]](https://www.youtube.com/) -->

<img src="static/images/ip.png" style="height:50px;" />
<img src="static/images/meta.png" style="height:50px;" />

</div>

# BFM-Zero Deployment Stack

> **Deployment stack for BFM-Zero on the Unitree G1 (Jetson Orin)**

This repository provides a complete deployment solution for running BFM-Zero policies on the Unitree G1 robot. It includes everything you need to test in simulation and deploy to the real robot.

## 📦 What's Included

- **ONNX Policy Runner** (`rl_policy/bfm_zero.py`, exposing `BFMZeroPolicy`)
- **Robot & Policy Configuration** files under `config/`
- **Experiment Configs** for tracking, reward inference, and goal reaching (`config/exp/`)

---

## 🚀 Quick Start

### Environment Setup

1. **Create and activate the runtime environment** (Python 3.10 recommended):
   ```bash
   conda create -n bfm0real python=3.10 -y
   conda activate bfm0real
   ```

2. **Install Python dependencies**:
   ```bash
   cd motivo_isaac_deploy
   pip install -r requirements.txt
   ```

### Downloading the Motivo / BFM-Zero ONNX Model

After downloading the model, your directory structure should look like this:

```
model/
├── exported/
│   └── model.onnx              # ONNX policy model
├── tracking_inference/
│   └── *.pkl                   # Latent variables for tracking tasks
├── reward_inference/
│   └── *.pkl                   # Latent variables for reward inference tasks
└── goal_inference/
    └── *.pkl                   # Latent variables for goal reaching tasks
```

**Directory Structure:**
- **`exported/model.onnx`** - The main ONNX policy model file
- **`tracking_inference/*.pkl`** - Pre-computed latent variables for motion tracking (shape: `[seq_length, 256]`)
- **`reward_inference/*.pkl`** - Pre-computed latent variables for reward inference tasks
- **`goal_inference/*.pkl`** - Pre-computed latent variables for goal reaching tasks
## 🧪 Sim-to-Sim Test Workflow

> **⚠️ Highly Recommended Before Real Robot Deployment**  
> Not required on Jetson if you only plan to run on the real robot, but highly recommended to validate every motion first.

### Step 1: Launch the Simulation (MuJoCo)

**Standard (Linux/Windows):**
```bash
python -m sim_env.base_sim \
  --robot_config=./config/robot/g1.yaml \
  --scene_config=./config/scene/g1_29dof.yaml
```

**macOS (with MuJoCo's `mjpython`):**
```bash
mjpython -m sim_env.base_sim \
  --robot_config=./config/robot/g1.yaml \
  --scene_config=./config/scene/g1_29dof.yaml
```

You should now see a MuJoCo environment:

<video src="examples/mujoco_env.mov" controls width="600"></video>

<details>
<summary><b>🎮 Interactive Simulation Commands</b></summary>

- **`7`** - Lift the robostack
- **`8`** - Lower down the robostack
- **`9`** - Release the robostack

</details>

### Step 2: Start the Policy Process

In a **separate terminal**, activate the environment and navigate to the project:

```bash
conda activate bfm0real
cd motivo_isaac_deploy
```

**Run the policy:**

```bash
python rl_policy/bfm_zero.py \
  --robot_config config/robot/g1.yaml \
  --policy_config ${POLICY_CONFIG} \
  --model_path ${MODEL_ONNX_PATH} \
  --task ${TASK}
```

We provide three types of tasks: `tracking`, `reward inference`, and `goal reaching`.

<details>
<summary><b>⌨️ Interactive Terminal Commands</b></summary>

| Key | Action | Task Type |
|-----|--------|-----------|
| **`]`** | Start policy action (stops at `stop` frame) | All |
| **`[`** | Start tracking | Tracking |
| **`p`** | Stop at the `stop` frame | Tracking |
| **`n`** | Switch to next reward/goal | Reward/Goal |
| **`i`** | Reset robot to initial position | All |
| **`o`** | Emergency stop (release all control) | All |

</details>

---

## 📋 Task-Specific Instructions

### 🎯 Tracking

```bash
./rl_policy/tracking.sh
```

**Configuration:** `config/exp/tracking/<your-motion>.yaml`

| Parameter | Description | Example |
|-----------|-------------|---------|
| `ctx_path` | `.pkl` file containing `seq_length` latent variables. Shape: `[seq_length, 256]` | `../tracking_inference/zs_walking.pkl` |
| `start` | Start frame index in the context sequence | `0` |
| `end` | End frame index (set to `None` for continuous tracking) | `2000` or `None` |
| `gamma` | Discount factor for context averaging. Controls weight of recent vs. older frames | `0.8` |
| `window_size` | Number of frames to average over. Larger = smoother but more delayed | `3` |

> **💡 Tip:** If you generate latent z with discounted window, set `window_size` to `1`.

**Example Result:**

<video src="examples/tracking.mov" controls width="600"></video>

---

### 🎁 Reward Inference

```bash
./rl_policy/reward.sh
```

**Configuration:** `config/exp/reward/<your-rewards>.yaml`

| Parameter | Description |
|-----------|-------------|
| `ctx_path` | Path to the reward inference `.pkl` file containing reward-specific latent variables |
| `selected_rewards_filter_z` | List of dictionaries specifying which rewards and z indices to use |

**Example Configuration:**
```yaml
selected_rewards_filter_z:
  - reward: "raisearms-m-l"
    z_ids: [3]
  - reward: "sitonground"
    z_ids: [3, 4]
```

Each entry selects specific z latent variables (different subsampled buffers are used to infer different zs) for a given reward.

**Example Result:**

<video src="examples/rewards.mov" controls width="600"></video>

---

### 🎯 Goal Reaching

```bash
./rl_policy/goal.sh
```

**Configuration:** `config/exp/goal/<your-goal-states>.yaml`

| Parameter | Description |
|-----------|-------------|
| `ctx_path` | Path to the goal inference `.pkl` file containing goal-specific latent variables |
| `selected_goals` | List of goal names to select from the context dictionary |

**Example Configuration:**
```yaml
selected_goals: [
  'walk3_subject3_9044',    # on your knees
  'fight1_subject3_5559',   # hands on hip
  'dance1_subject3_4024',   # raise arm high
]
```

**Example Result:**

<video src="examples/goal.mov" controls width="600"></video>

---

## 🤖 On-Robot Deployment (Jetson Orin, Unitree G1)

### Required Setup
- Target platform: onboard Orin Jetson of the Unitree G1 (ssh into the robot and copy this codebase).
- Install the Unitree C++ SDK Python binding from https://github.com/EGalahad/unitree_sdk2 to enable 50 Hz control. Update the import path in `rl_policy/base_policy.py` after building the binding.


### Running on the Real Robot
- Replace `"<your-unitree-sdk2-path>"` in `./rl_policy/bfm_zero.py`
- Use the real-robot config: `config/robot/g1_real.yaml`. 

  (i.e. `python rl_policy/bfm_zero.py \
    --robot_config config/robot/g1.yaml \
    --policy_config ${POLICY_CONFIG} \
    --model_path ${MODEL_ONNX_PATH} \
    --task  ${TASK}`)

---


## 👥 Citation

If you find this project useful in your research, please consider citing:

```bibtext
@article{li2025bfmzero,
  title   = {BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning},
  author  = {Yitang Li and Zhengyi Luo and Tonghe Zhang and Cunxi Dai and Anssi Kanervisto and Andrea Tirinzoni and Haoyang Weng and Kris Kitani and Mateusz Guzek and Ahmed Touati and Alessandro Lazaric and Matteo Pirotta and Guanya Shi},
  journal = {arXiv preprint arXiv:2505.06776},
  year    = {2025}
}
```

This sim-to-real repo is built upon:

```bibtex
@misc{weng2025hdmilearninginteractivehumanoid,
      title={HDMI: Learning Interactive Humanoid Whole-Body Control from Human Videos}, 
      author={Haoyang Weng and Yitang Li and Nikhil Sobanbabu and Zihan Wang and Zhengyi Luo and Tairan He and Deva Ramanan and Guanya Shi},
      year={2025},
      eprint={2509.16757},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2509.16757}, 
}

@article{he2025asap,
  title={ASAP: Aligning Simulation and Real-World Physics for Learning Agile Humanoid Whole-Body Skills},
  author={He, Tairan and Gao, Jiawei and Xiao, Wenli and Zhang, Yuanhang and Wang, Zi and Wang, Jiashun and Luo, Zhengyi and He, Guanqi and Sobanbabu, Nikhil and Pan, Chaoyi and Yi, Zeji and Qu, Guannan and Kitani, Kris and Hodgins, Jessica and Fan, Linxi "Jim" and Zhu, Yuke and Liu, Changliu and Shi, Guanya},
  journal={arXiv preprint arXiv:2502.01143},
  year={2025}
}
```

## License

BFM-Zero is licensed under the CC BY-NC 4.0 license. See [LICENSE](LICENSE) for details.