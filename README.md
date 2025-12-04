# Interpretability-Guided-Defense (Confidence-Masking Fork)

> **Note:** This is **not** the official IG-Defense repository.
> It is a **fork** of the original ECCV 2024 codebase, used for experiments on
> confidence-based masking and reproduction of the main CIFAR-10 results.

Original upstream repo: [https://github.com/Trustworthy-ML-Lab/Interpretability-Guided-Defense](https://github.com/Trustworthy-ML-Lab/Interpretability-Guided-Defense)

---

## 1. What’s in this fork?

This fork keeps the original IG-Defense code and adds a few things on top:

* ✅ Reproduction of **RN18 + CIFAR-10** results (baseline, LO-IR masking).
* ✅ Additional **confidence-masking variants** (using `--mask-which conf`) for comparison with LO-IR.
* ✅ Support for **TRADES-AWP WideResNet-34-10 CIFAR-10** via the checkpoint
  `TRADES-AWP_cifar10_linf_wrn34-10.pt`.
* ✅ Extra analysis / result artifacts:

  * `Baseline_rn18_cifar10_results.txt`
  * `LOIR_rn18_cifar10_results.txt`
  * `Analysis_FinalSummary.png`
  * `eval_nomasking.png`
  * threshold logs like `0.8.txt`, `0.85.txt`, `0.9.txt`, `0.95.txt`

The core research idea and most of the implementation still come from:

> A. Kulkarni and T.-W. Weng, **Interpretability-Guided Test-Time Adversarial Defense**, ECCV 2024.

If you just want the original IG-Defense implementation, use the upstream repo instead.

---

## 2. Original IG-Defense (upstream summary)

* Proposes a neuron-interpretability-guided test-time defense (**IG-Defense**) that uses neuron importance ranking to improve adversarial robustness.
* IG-Defense is **training-free**, **efficient**, and **effective**.
* Provides robustness gains on standard **CIFAR-10**, **CIFAR-100**, and **ImageNet-1k** benchmarks.
* Shows up to **+3.4%**, **+3.8%**, and **+1.5%** robustness improvements against a wide range of white-box, black-box, and adaptive attacks, while being ~4× faster than many existing test-time defenses.

Project page (original): [https://lilywenglab.github.io/Interpretability-Guided-Defense/](https://lilywenglab.github.io/Interpretability-Guided-Defense/)

<p align="center">
<img src="https://github.com/user-attachments/assets/5cd73bf7-c8c7-4707-8828-a6be5ad21c64" width="900">
</p>

---

## 3. Environment & Requirements

Tested with:

* Python **3.11**
* CUDA 12 (e.g., NVIDIA Tesla V100 on Purdue Scholar)
* Recent PyTorch (`torch>=2.0` works fine)

### Basic setup

```bash
# (optional, example with conda)
conda create -n igdefense python=3.11 -y
conda activate igdefense

# clone this fork
git clone https://github.com/weiC29/interpretability-guided-defense-conf.git
cd interpretability-guided-defense-conf

# core dependencies
pip install -r requirements.txt
pip install git+https://github.com/RobustBench/robustbench.git

# additional packages used in this fork
pip install ftfy regex tqdm matplotlib
pip install git+https://github.com/fra31/auto-attack.git
```

If you are using a specific GPU/driver stack, install the matching PyTorch + CUDA build from
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) and then install the remaining packages.

---

## 4. Pretrained Weights

Place the following checkpoints under `checkpoints/`:

* **DAJAT ResNet18 (CIFAR-10)**
  `Addepalli2022Efficient_RN18.pt`
  Download: [https://drive.google.com/uc?id=1m5vhdzIUUKhDbsZdOG9z76Eyp6f4xe_f](https://drive.google.com/uc?id=1m5vhdzIUUKhDbsZdOG9z76Eyp6f4xe_f)

* **TRADES-AWP WideResNet-34-10 (CIFAR-10)**
  `TRADES-AWP_cifar10_linf_wrn34-10.pt`
  Download: [https://drive.google.com/uc?id=1hlVTLZkveYGWpE9-46Wp5NVZt1slz-1T](https://drive.google.com/uc?id=1hlVTLZkveYGWpE9-46Wp5NVZt1slz-1T)

* **(Optional) FAT ResNet50 (ImageNet-1k)**
  `FAT_ResNet50_ImageNet.pt`
  Download: [https://drive.google.com/uc?id=1UrNEtLWs-fjlM2GPb1JpBGtpffDuHH_4](https://drive.google.com/uc?id=1UrNEtLWs-fjlM2GPb1JpBGtpffDuHH_4)

Other robust models can be imported from the [RobustBench model zoo](https://github.com/RobustBench/robustbench/tree/master/robustbench/model_zoo).
If you add new models, place their architectures under `models/` and follow the existing patterns.

---

## 5. Neuron Importance Ranking (LO-IR / CLIP-Dissect)

The original IG-Defense ranking scripts still work in this fork.

### LO-IR rankings

```bash
# DAJAT RN18 CIFAR-10 (default)
bash scripts/get_loir_rankings.sh
```

This generates LO-IR rankings saved under:

```text
saved_loir_rankings/<checkpoint-name>/<layer-name>/unit*.npy
```

For WideResNet-34-10 experiments we use:

* checkpoint: `TRADES-AWP_cifar10_linf_wrn34-10.pt`
* layer name: `block3`

### CLIP-Dissect (CD-IR) rankings (optional)

```bash
bash scripts/get_cdir_rankings.sh
```

For ImageNet experiments, update:

* `utils.py` (around L130–L131)
* `clip-dissect/utils.py` (around L19)

with the path to your ImageNet subset. The authors used a random 10% train subset created with
[this helper repo](https://github.com/BenediktAlkin/ImageNetSubsetGenerator).

---

## 6. Analysis Experiments (including confidence masking)

The `analysis.py` script can run ablations over different masking strategies.
Please generate LO-IR rankings first with `get_loir_rankings.sh`.

### Example: RN18 + CIFAR-10

```bash
# Baseline analysis (no masking)
python analysis.py \
  --arch rn18_val \
  --dataset cifar10 \
  --load-model Addepalli2022Efficient_RN18.pt \
  --layer-name layer4 \
  --important-dim 50 \
  --mask-which none

# LO-IR masking
python analysis.py \
  --arch rn18_val \
  --dataset cifar10 \
  --load-model Addepalli2022Efficient_RN18.pt \
  --layer-name layer4 \
  --important-dim 50 \
  --mask-which loir

# Confidence-based masking (this fork)
python analysis.py \
  --arch rn18_val \
  --dataset cifar10 \
  --load-model Addepalli2022Efficient_RN18.pt \
  --layer-name layer4 \
  --important-dim 50 \
  --mask-which conf
```

Result summaries from our runs are stored in:

* `Baseline_rn18_cifar10_results.txt`
* `LOIR_rn18_cifar10_results.txt`
* `Analysis_FinalSummary.png`

---

## 7. AutoAttack Evaluation

The `eval.py` script evaluates:

* the **base model** (no masking),
* **LO-IR-defended** model, and
* **confidence-masked** model (this fork)

against AutoAttack on CIFAR-10.

### Example: RN18 + CIFAR-10

```bash
# Baseline (no masking)
python eval.py \
  --arch rn18_val \
  --dataset cifar10 \
  --load-model Addepalli2022Efficient_RN18.pt \
  --layer-name layer4 \
  --important-dim 50 \
  --mask-which none

# LO-IR masking
python eval.py \
  --arch rn18_val \
  --dataset cifar10 \
  --load-model Addepalli2022Efficient_RN18.pt \
  --layer-name layer4 \
  --important-dim 50 \
  --mask-which loir

# Confidence masking
python eval.py \
  --arch rn18_val \
  --dataset cifar10 \
  --load-model Addepalli2022Efficient_RN18.pt \
  --layer-name layer4 \
  --important-dim 50 \
  --mask-which conf
```

The exact CLI options and defaults are documented in `eval.py`.
Plots like `eval_nomasking.png` come from these evaluations.

> **Adaptive attacks:** not modified in this fork; for details, refer to the original paper and repo.

---

## 8. Notes for Purdue Scholar (internal use)

These are **informal notes** from running the code on Purdue’s **Scholar** cluster with a Tesla V100 (16GB):

### Environment

```bash
module load anaconda/2024.02-py311
conda create -n igdefense python=3.11 -y
conda activate igdefense
cd ~/Interpretability-Guided-Defense
# then install dependencies as described above
```

### Example SLURM script

```bash
#!/bin/bash
#SBATCH -A gpu
#SBATCH -p scholar-gpu
#SBATCH -J wrn34_c10_loir
#SBATCH -N 1
#SBATCH -c 3
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH -o logs/wrn34_c10_loir_%j.out
#SBATCH -e logs/wrn34_c10_loir_%j.err

module load anaconda/2024.02-py311
source activate igdefense
cd ~/Interpretability-Guided-Defense

python get_loir_rankings.py \
  --dataset cifar10 \
  --arch wrn34_10 \
  --load-model checkpoints/TRADES-AWP_cifar10_linf_wrn34-10.pt \
  --layer-name block3 \
  --start-dim 0 \
  --end-dim 640
```

Adjust `--start-dim` / `--end-dim`, batch size, and time limit as needed (large models can be close to the 4h wall).

---

## 9. Sources

* IG-Defense project page: [https://lilywenglab.github.io/Interpretability-Guided-Defense/](https://lilywenglab.github.io/Interpretability-Guided-Defense/)
* Original repo: [https://github.com/Trustworthy-ML-Lab/Interpretability-Guided-Defense](https://github.com/Trustworthy-ML-Lab/Interpretability-Guided-Defense)
* CLIP-Dissect: [https://github.com/Trustworthy-ML-Lab/CLIP-dissect](https://github.com/Trustworthy-ML-Lab/CLIP-dissect)
* RobustBench: [https://robustbench.github.io/](https://robustbench.github.io/)

---

## 10. Citation

If you use this code (or this fork) in academic work, please cite the **original** paper:

```bibtex
@inproceedings{kulkarni2024igdefense,
    title     = {Interpretability-Guided Test-Time Adversarial Defense},
    author    = {Kulkarni, Akshay and Weng, Tsui-Wei},
    booktitle = {European Conference on Computer Vision},
    year      = {2024}
}
```
