<h1 align="center"> BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning </h1>

<div align="center">

[[arXiv]](https://arxiv.org/abs/2511.04131)
[[Paper]](https://lecar-lab.github.io/BFM-Zero/resources/paper.pdf)
[[Website]](https://lecar-lab.github.io/BFM-Zero/)

<!-- [[Arxiv]](https://lecar-lab.github.io/SoFTA/) -->
<!-- [[Video]](https://www.youtube.com/) -->

<img src="static/images/ip.png" style="height:50px;" />
<img src="static/images/meta.png" style="height:50px;" />
</div>

## Code

Code will be released in stages:

- [x] **Pretrained checkpoints + sim-to-sim / sim-to-real deployment**  
  → [`deploy`](https://github.com/LeCAR-Lab/BFM-Zero/tree/deploy) branch

- [x] **Minimal inference code + tutorial**  
  → [`minimal_inference`](https://github.com/LeCAR-Lab/BFM-Zero/tree/minimal_inference) branch

- [x] **Full training and evaluation pipelines**

- [ ] **Minimal training code (RTX 4090 support)**

# BFM-Zero Training

Humanoidverse training for BFM-Zero with Isaac Sim or MuJoCo.

## Requirements

- Python 3.10
- CUDA-capable GPU
- Isaac Sim (Linux) or MuJoCo for simulation

## Installation

### 1. Clone and fetch large files (Git LFS)

This repo uses [Git LFS](https://git-lfs.github.com/) for motion data and model. After cloning, install LFS and pull the large files:

```bash
git clone https://github.com/LeCAR-Lab/BFM-Zero.git
cd BFM-Zero
git lfs install
git lfs pull
```
Note: If the repository exceeded its LFS budget,  you can access the data here: https://huggingface.co/LeCAR-Lab/BFM-Zero/tree/main/data

### 2. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Or with pip: `pip install uv`

### 3. Install dependencies

From this directory (BFM-Zero):

```bash
uv sync
```

## Data

- **Motion data**: Included via Git LFS in `humanoidverse/data/` after `git lfs pull`. `lafan_29dof.pkl` is for evaluation; `lafan_29dof_10s-clipped.pkl` is for training. 


## Training

### Launch

```bash
uv run python -m humanoidverse.train
```

### Main parameters

Training is driven by `humanoidverse.train` (see `train_bfm_zero()` in `train.py`). Key options:

| Area | Parameters |
|------|------------|
| **Scale** | `num_env_steps`, `online_parallel_envs`, `buffer_size`, `checkpoint_every_steps` |
| **Paths** | `work_dir`, env’s `lafan_tail_path` (expert motion data) |
| **Run** | `seed`, `use_wandb`, `wandb_pname` / `wandb_gname` / `wandb_ename` |
| **Policy / optim** | `update_agent_every`, `num_agent_updates`, `num_seed_steps`; agent config (e.g. `batch_size`, `lr_actor`, `lr_critic`, `discount`) |
| **Robot / env** | Overridden via `hydra_overrides` (e.g. `robot=...`, `robot.control.action_scale=...`, `env.config.lie_down_init=...`) |

Override from code by passing a custom `TrainConfig`, or extend the CLI to accept Hydra/tyro overrides.

Tips: After 50-100 M steps training, eval/emd should lower than 0.75.


<div align="center">
<img src="static/images/training_curve.png" style="height:300px;" />
</div>

---

## Inference

After training, three scripts handle inference and export:

| Script | Purpose |
|--------|---------|
| **`humanoidverse.tracking_inference`** | Motion tracking → extract latent \(z\), export ONNX |
| **`humanoidverse.goal_inference`** | Goal-reaching → compute \(z\) for different goals |
| **`humanoidverse.reward_inference`** | Reward-based tasks → compute \(z\) and evaluate performance |

### Example outputs (videos)

Videos from `BFM-Zero/model` after running each inference script with `--save_mp4`:

**1. Tracking inference** — expert (left) vs policy (right):

<img src="https://github.com/LeCAR-Lab/BFM-Zero/blob/main/model/tracking_inference/tracking.gif" controls width="400">


**2. Goal inference** — goal-reaching rollout:

<img src="https://github.com/LeCAR-Lab/BFM-Zero/blob/main/model/goal_inference/goal.gif" controls width="400">

**3. Reward inference** — example task (e.g. move-ego):

<img src="https://github.com/LeCAR-Lab/BFM-Zero/blob/main/model/reward_inference/move-ego-low0.6-0-0.7.gif" controls width="400"></img>

---

All scripts use **tyro** for the CLI. General usage:

```bash
uv run python -m humanoidverse.tracking_inference --help
uv run python -m humanoidverse.goal_inference --help
uv run python -m humanoidverse.reward_inference --help
```

**Common arguments:**

- `--model_folder`: Path to the trained model directory (must contain `checkpoint/` and `config.json`).
- `--data_path` (optional): Override the default LaFan data path.
- `--headless` (default: `True`): Run without GUI; use `--no-headless` to show the viewer.
- `--save_mp4`: Save rendered videos.

**Output:** All inference scripts export the policy to ONNX (`{model_name}.onnx`) in their respective output subdirectories under `exported/`.

---

### Tracking inference

Runs motion tracking, exports ONNX, and optionally saves a comparison video (expert vs policy).

```bash
uv run python -m humanoidverse.tracking_inference \
    --model_folder /path/to/model \
    --data_path humanoidverse/data/lafan_29dof.pkl \
    --no-headless \
    --save_mp4
```

- `--model_folder` should point to the **outer** model directory (the one that contains `checkpoint/`).
- You can set motion IDs and visualization length inside the script if exposed.

**Outputs** (under `model_folder/tracking_inference/`):

- `zs_{MOTION_ID}.pkl`: Latent \(z\) for each motion.
- `tracking.mp4`: Expert vs policy comparison (when `--save_mp4` is set).

---

### Goal inference

Computes \(z\) for predefined goals and optionally renders goal-reaching videos.

```bash
uv run python -m humanoidverse.goal_inference \
    --model_folder /path/to/model \
    --data_path humanoidverse/data/lafan_29dof.pkl \
    --save_mp4
```

- Iterates over predefined goals and computes the corresponding \(z\).
- Requires `goal_frames_lafan29dof.json` (the script searches for it in several locations).

**Outputs** (under `model_folder/goal_inference/`):

- `goal_reaching.pkl`: Dictionary `{goal_name -> z}`.
- `videos/*.mp4`: Per-goal videos (if `--save_mp4 True`).

---

### Reward inference

Runs reward-based task inference: computes \(z\) and optionally runs rollouts for evaluation.

```bash
uv run python -m humanoidverse.reward_inference \
    --model_folder /path/to/model \
    --save_mp4 
```

**Key arguments:**

| Argument | Description |
|----------|-------------|
| `--num_samples` | Number of samples in the buffer per inference run. |
| `--n_inferences` | Number of inference latents per reward task. |
| `--episode_length` | Steps per rollout. |
| `--skip_rollouts` | If `True`, only compute \(z\); do not run visualization rollouts. |

**Outputs** (under `model_folder/reward_inference/`):

- `reward_locomotion.pkl`: Dictionary `{task_name -> z}`.
- `videos/*.mp4`: Per-task videos (when `--save_mp4` is set).

---



## License

BFM-Zero is licensed under the CC BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

## Citation

If you find this project useful in your research, please consider citing:

<!--
```bibtex
@article{li2025bfmzero,
  title   = {BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning},
  author  = {Yitang Li and Zhengyi Luo and Tonghe Zhang and Cunxi Dai and Anssi Kanervisto and Andrea Tirinzoni and Haoyang Weng and Kris Kitani and Mateusz Guzek and Ahmed Touati and Alessandro Lazaric and Matteo Pirotta and Guanya Shi},
  journal = {arXiv preprint arXiv:2505.06776},
  year    = {2025}
}
```
Wrong arXiv id here!!
--> 

```bibtex
@misc{li2025bfmzeropromptablebehavioralfoundation,
      title={BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning}, 
      author={Yitang Li and Zhengyi Luo and Tonghe Zhang and Cunxi Dai and Anssi Kanervisto and Andrea Tirinzoni and Haoyang Weng and Kris Kitani and Mateusz Guzek and Ahmed Touati and Alessandro Lazaric and Matteo Pirotta and Guanya Shi},
      year={2025},
      eprint={2511.04131},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2511.04131}, 
}
```

## Contact

If you have any problems, please contact [liyitang475@gmail.com](mailto:liyitang475@gmail.com).

