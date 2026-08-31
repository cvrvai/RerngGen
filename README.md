# RerngGen — Open-Source Generative Model Lab

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/Tests-Passing%20(11%2F11)-brightgreen.svg)](research/01_dit/tests)

> **"Train small, prove the system, then scale."**  
> RerngGen is an open-source, docs-first generative media research and engineering stack demonstrating the complete path from **data engineering → model architecture → single-GPU training → evaluation → distributed scaling → specialist adapters → image-to-video (I2V)**.

---

## 🌟 Architecture Lineage & Research Labs

To avoid premature architectural lock-in, RerngGen uses isolated research laboratories to evaluate and audit generative backbone paradigms before adopting them into the core production engine:

```plain text
research/
├── 01_dit/       # [CURRENT] Baseline DiT (adaLN-Zero, pooled text conditioning, 2D sin/cos)
├── 02_mmdit/     # Multimodal DiT (Joint image/text attention streams)
├── 03_flux/      # FLUX-style Transformer (Double-stream -> single-stream blocks)
├── 04_sana/      # SANA-style Efficiency (DC-AE latent compression & linear attention)
└── 05_lumina/    # Lumina / Next-DiT (RoPE, unified token spaces, video pathway)
```

---

## 🪜 Model Scaling Ladder

| Scale Tier | Parameters | Primary Purpose | Target Compute |
| :--- | :--- | :--- | :--- |
| **Debug** | $10\text{–}30\text{ M}$ | Tensor contracts, shape asserts, zero-init identity, fast smoke tests | CPU / 1x RTX 3090 |
| **Tiny** | $\sim 100\text{ M}$ | Single-batch & tiny-dataset overfit checks, loss convergence | 1x RTX 3090 |
| **Small** | $\sim 300\text{–}400\text{ M}$ | Scaling baseline comparisons, memory benchmarking | 1x RTX 3090 |
| **Base** | $\sim 700\text{ M+}$ | Primary local research prototype (latents + Flow Matching) | 1x RTX 3090 (24 GB) |
| **Large / XL** | $1.3\text{ B} \to 2\text{ B+}$ | Scaling study & production weights | Multi-GPU Cloud (4–8x A100/H100) |
| **Video** | Temporal extension | Image-to-Video (I2V) & Text-to-Video (T2V) | Cloud Multi-GPU |

---

## 📂 Repository Structure

```plain text
RerngGen/
├── docs/                      # PRD, Architecture Specs, ADRs, Governance
├── research/                  # Isolated Research Laboratories
│   ├── 01_dit/                # Lab 01: Baseline DiT implementation & audits
│   │   ├── configs/           # debug.yaml, tiny.yaml
│   │   ├── src/               # patch_embed.py, unpatchify.py, ...
│   │   ├── tests/             # unit tests per component
│   │   └── audit/             # Step-by-step audit checklists and findings
├── src/rernggen/              # Adopted production core architecture
├── configs/                   # model/, train/, hardware/, data/ configs
├── scripts/                   # smoke_train.py, overfit_tiny.py, sample.py
├── tests/                     # unit/ and integration/ test suites
├── LICENSE                    # Apache 2.0 License
├── pyproject.toml             # Package metadata and dependencies
└── pytest.ini                 # Pytest configuration
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/cvrvai/RerngGen.git
cd RerngGen

# Create virtual environment and install dependencies
uv venv --python 3.10 .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

# Install PyTorch with CUDA 12.4
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install pytest pyyaml einops
```

### 2. Run Lab 01 Tests
```bash
pytest -v
```

---

## ⚖️ Ethics, Governance & Artist Rights

- **Data Provenance First:** The base foundation model is trained strictly on open, fully licensed datasets with auditable metadata manifests.
- **Specialist Adapters:** Specific artist styles or culturally unique artistic traditions (e.g. Khmer children's book style) are trained exclusively as lightweight **LoRA adapters** governed by explicit bilateral licensing agreements.

---

## 📄 License
This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.
