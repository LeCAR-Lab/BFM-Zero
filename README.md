<h1 align="center"> BFM-Zero: A Promptable Behavioral Foundation Model for Humanoid Control Using Unsupervised Reinforcement Learning </h1>

<div align="center">

[[arXiv]](https://arxiv.org/abs/2511.04131)
[[Paper]](https://lecar-lab.github.io/BFM-Zero/resources/paper.pdf)
[[Website]](https://lecar-lab.github.io/BFM-Zero/)

<img src="static/images/ip.png" style="height:50px;" />
<img src="static/images/meta.png" style="height:50px;" />

</div>



# 🤖 BFM-Zero Minimal Inference Code

A minimal, self-contained implementation for running BFM-Zero (Behavioral Foundation Model) inference on humanoid robots in MuJoCo. This repository provides tools for three types of latent `z` inference: reward inference, goal inference, and tracking inference.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.3.7-green.svg)](https://mujoco.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1+-orange.svg)](https://pytorch.org/)

## ✨ Features

- **🎁 Reward Inference**: Infer latent `z` from reward-weighted observations to optimize for specific behaviors (locomotion, rotation, arm movements)
- **🎯 Goal Inference**: Infer latent `z` for goal-reaching tasks (e.g., T-pose, specific joint configurations)
- **📹 Tracking Inference**: Infer latent `z` sequences for motion tracking (dance, walking, complex movements)
- **🤖 MuJoCo Simulation**: Full physics simulation environment for G1 humanoid robot
- **📤 ONNX Export**: Export trained models to ONNX format for deployment
- **📊 Jupyter Tutorial**: Comprehensive notebook with examples and explanations

## 🚀 Installation

### Prerequisites

- Python 3.10 

```bash
# Create a new conda environment
conda create -n bfm0 python=3.10 -y
conda activate bfm0

# Install dependencies
pip install -r requirements.txt
```


## 🎯 Quick Start

1. **Activate the environment**:
   ```bash
   conda activate bfm0
   ```

2. **Download the model**:
      run
   ```bash
   python download_hf_model.py --token <YOUR_HF_TOKEN>
   ```

   You can either pass your Hugging Face token as a flag (as above), or set it via the `HF_TOKEN` environment variable:

   ```bash
   export HF_TOKEN=your_token_here
   python download_hf_model.py
   ```
   After downloading the model, your directory structure should look like this:

   ```
   model/
   ├── checkpoint/
       └── model/   
       └── buffer/             
   ```

3. **Follow the notebook `inference_tutorial.ipynb`** to learn about:
   - Loading BFM-Zero models
   - Understanding observations and latent `z`
   - Performing reward, goal, and tracking inference
   - Exporting models to ONNX

## 📁 Project Structure

```
minimal_model_inference_code/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── inference_tutorial.ipynb           # Main tutorial notebook
├── env.py                             # MuJoCo environment wrapper
├── common.py                          # Configuration constants
├── sample_data.npz                    # Sample motion data
├── example_motion/
|   ├── dance1_subject2_50_jpos.npz    # example 50Hz motions
|   ├── <your motions here>
├── bfm_zero_inference_code/           # Core inference code
│   ├── fb_cpr_aux/                    # Forward-backward CPR auxiliary model
│   │   └── model.py                   # FBcprAuxModel implementation
│   ├── inference/                     # Inference utilities
│   │   ├── rewards.py                 # Reward functions (locomotion, rotation, etc.)
│   │   └── utils.py                   # Helper functions
│   ├── nn_models.py                   # Neural network architectures
│   ├── normalizers.py                 # Observation normalizers
│   └── g1_for_reward_inference.xml    # MuJoCo robot model
├── model/                             # Pre-trained model checkpoints
│   └── checkpoint/                    # Model weights and configs
|        └── model/                 
|           └── model.safetensors
|           └── config.json
|           └── init_kwargs.json
└── videos/                            # Generated simulation videos
```

## 🤝 Note

This is a minimal inference codebase. For full inference script, please refer to the full training code & script (coming soon).


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

## License

BFM-Zero is licensed under the CC BY-NC 4.0 license. See [LICENSE](LICENSE) for details.


**Happy Inferencing! 🚀**
