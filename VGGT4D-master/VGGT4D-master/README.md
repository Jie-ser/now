<div align="center">

<h2>VGGT4D: Mining Motion Cues in Visual Geometry Transformers for 4D Scene Reconstruction</h2>

<!-- Badges -->
<p>
  <a href="https://3dagentworld.github.io/vggt4d/">
    <img src="https://img.shields.io/badge/Project-Page-blue?logo=web&logoColor=white" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2511.19971">
    <img src="https://img.shields.io/badge/arXiv-2511.19971-B31B1B?logo=arxiv&logoColor=white" alt="arXiv">
  </a>
  <a href="https://github.com/3DAgentWorld/VGGT4D">
    <img src="https://img.shields.io/badge/Code-GitHub-black?logo=github" alt="Code">
  </a>
</p>

<!-- Authors -->
<p>
  <strong>Yu Hu</strong><sup>1</sup>* &nbsp;&nbsp;
  <strong>Chong Cheng</strong><sup>1,2</sup>* &nbsp;&nbsp;
  <strong>Sicheng Yu</strong><sup>1</sup>*
</p>

<p>
  <strong>Xiaoyang Guo</strong><sup>2</sup> &nbsp;&nbsp;
  <strong>Hao Wang</strong><sup>1</sup>†
</p>

<!-- Affiliations -->
<p>
  <sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)<br>
  <sup>2</sup>Horizon Robotics
</p>

<p>
  * Equal contribution. &nbsp;&nbsp; † Corresponding author.
</p>

</div>


## Quick Start

This section will guide you through setting up the environment and running VGGT4D on your own data.

### 1. Environment Setup

We **recommend using `pyenv` together with `virtualenv`** to ensure a clean and reproducible Python environment.

```bash
# Select Python version
pyenv shell 3.12

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118

# Install remaining project requirements
pip install -r requirements.txt
```

### 2. Download Pre-trained Checkpoint

Download the pre-trained model checkpoint:

```bash
mkdir -p ckpts/
wget -c "https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt?download=true" -O ckpts/model_tracker_fixed_e20.pt
```

### 3. Run the Demo

Run the VGGT4D demo script to process your scene data:

```bash
python demo_vggt4d.py --input_dir <path_to_input_dir> --output_dir <path_to_output_dir>
```

**Input Directory Structure:**

The input directory should follow this structure:
```
input_dir/
├── scene1/
│   ├── image001.jpg
│   ├── image002.jpg
│   └── ...
└── scene2/
    ├── image001.png
    ├── image002.png
    └── ...
```

Each scene subdirectory should contain image files in `.jpg` or `.png` format.

**Example Usage:**

```bash
python demo_vggt4d.py --input_dir ./datasets/input_dir --output_dir ./outputs
```

**Output Files:**

The script processes each scene and generates the following outputs in the output directory:
- Depth maps (`frame_%04d.npy` format)
- Depth confidence maps (`conf_%04d.npy` format)
- Camera intrinsics (`pred_intrinsics.txt`)
- Camera poses in TUM format (`pred_traj.txt`)
- Refined dynamic masks (`dynamic_mask_%04d.png` format)
- RGB images (`frame_%04d.png` format)

## TODO

- [x] Release code
- [ ] Data preprocess scripts
- [ ] Evaluation scripts
- [ ] Visualization scripts
- [ ] Long sequence implementation

## Acknowledgements

We thank the authors of [VGGT](https://github.com/facebookresearch/vggt), [DUSt3R](https://github.com/naver/dust3r), and [Easi3R](https://github.com/Inception3D/Easi3R) for releasing their models and code. Their contributions to geometric learning and dynamic reconstruction provided essential foundations for this work, along with many other inspiring works in the community.

## License


This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this software for both academic and commercial purposes, provided that proper attribution is given.

See the [LICENSE](LICENSE) file for details.

## Citation

If you find **VGGT4D** useful for your research, please cite our paper:

```bibtex
@misc{hu2025vggt4d,
      title={VGGT4D: Mining Motion Cues in Visual Geometry Transformers for 4D Scene Reconstruction}, 
      author={Yu Hu and Chong Cheng and Sicheng Yu and Xiaoyang Guo and Hao Wang},
      year={2025},
      eprint={2511.19971},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.19971}, 
}
```
