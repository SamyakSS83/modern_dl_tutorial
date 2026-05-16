# PyTorch Deep Learning Curriculum

A self-study curriculum for building deep learning implementation skills from the ground up using PyTorch. Covers CNNs, transfer learning, contrastive learning, object detection, transformers, language models, vision transformers, and parameter-efficient fine-tuning — ending in a multimodal capstone.

Designed for someone with theoretical background in machine learning who wants to build strong practical and implementation intuition.

---

## Credits

Curriculum designed and authored by **Samyak Sanghvi**.

Assignments draw on concepts and problem settings from **COL780: Computer Vision** and **COL772: Natural Language Processing** (IIT Delhi). The datasets, architectural targets, and evaluation protocols are inspired by coursework from both courses, adapted for self-study.

---

## Structure

```
pytorch-assignments/
├── README.md               ← you are here
├── requirements.txt        ← all dependencies
├── RESOURCES.md            ← annotated reading list for all assignments
│
├── A01/  PyTorch training loop from scratch (MNIST)
├── A02/  Transfer learning and ResNet fine-tuning (EuroSAT)
├── A03/  SE blocks, Focal Loss, Grad-CAM (EuroSAT)
├── A04/  Contrastive learning — SimCLR (EuroSAT)
├── A05/  Object detection — YOLOv8 with SE head (VisDrone)
├── A06/  Attention mechanism from scratch
├── A07/  BPE tokenizer + decoder language model (Hindi CC-100)
├── A08/  DeiT-3 fine-tuning + attention maps (EuroSAT)
├── A09/  Transformers without normalisation — DyT (EuroSAT)
├── A10/  BERT — classification and NER (SST-2, CoNLL-2003)
├── A11/  LoRA and parameter-efficient fine-tuning (Alpaca)
├── A12/  Capstone — multimodal classification pipeline (EuroSAT)
├── A13/  RL for LLMs — GRPO + Unsloth Sudoku solver
└── A14/  Agentic AI — tool-using agents with Gemini API
```

Each assignment folder contains a `README.md` with full theory, mathematical derivations, reading material, task description, deliverables, and skeleton starter code.

---

## Assignment Map

| # | Assignment | Key Concepts | Dataset | VRAM |
|---|-----------|-------------|---------|------|
| A01 | Training loop from scratch | CNN, backprop, Adam, cross-entropy | MNIST | <1 GB |
| A02 | Transfer learning | ResNet-18, fine-tuning, BatchNorm | EuroSAT | ~2 GB |
| A03 | Architectural modifications | SE blocks, Focal Loss, Grad-CAM | EuroSAT | ~3 GB |
| A04 | Contrastive learning | SimCLR, NT-Xent, linear probe, UMAP | EuroSAT | ~7 GB |
| A05 | Object detection | YOLOv8, FPN, mAP, head surgery | VisDrone | ~6 GB |
| A06 | Attention from scratch | Scaled dot-product, MHA, causal masking | — | <1 GB |
| A07 | Tokenizer + LM | BPE, decoder transformer, perplexity | Hindi CC-100 | ~4 GB |
| A08 | Vision Transformer | DeiT-3, patch embeddings, CLS attention | EuroSAT | ~5 GB |
| A09 | Dynamic Tanh | LayerNorm replacement, DyT, α analysis | EuroSAT | ~5 GB |
| A10 | BERT | MLM, bidirectional attention, NER | SST-2, CoNLL-2003 | ~6 GB |
| A11 | LoRA | Low-rank adaptation, QLoRA, Unsloth | Alpaca | ~16 GB |
| A12 | Multimodal capstone | Late fusion, ViT + BERT, JEPA context | EuroSAT | ~10 GB |
| A13 | RL for LLMs — Sudoku | GRPO, RLVR, reward shaping, Unsloth | Generated | ~8 GB |
| A14 | Agentic AI with Gemini | Tool use, ReAct, multi-agent orchestration | — | CPU |

---

## Prerequisites

- Python 3.10+
- PyTorch 2.2+ (with CUDA if available)
- Comfortable with Python, NumPy, and basic ML concepts
- Theoretical background in linear algebra and calculus

---


## How to use this curriculum

1. Read `RESOURCES.md` briefly to get an overview of the supplementary material available.
2. Start at A01. Do not skip it even if it feels basic — the training loop infrastructure you build there is reused in every subsequent assignment.
3. Each `README.md` tells you exactly which prior assignments it depends on. Do not start an assignment until its dependencies are complete.
4. Run the `download_data.py` script in each folder before anything else. Data downloads can be slow — start them before reading the theory.
5. The RESOURCES.md entry for each assignment lists what to read and when. Use it as a reading schedule alongside the implementation work.



## Datasets used

| Dataset | Assignments | Size | Notes |
|---------|------------|------|-------|
| MNIST | A01 | ~11 MB | Auto-downloaded via torchvision |
| EuroSAT RGB | A02, A03, A04, A08, A09, A12 | ~90 MB | Download script in A02 |
| VisDrone 2019 | A05 | ~1.5 GB | Download script in A05 |
| Hindi CC-100 | A07 | ~50 MB (subset) | Download script in A07 |
| SST-2 | A10 | ~3 MB | Via HuggingFace `datasets` |
| CoNLL-2003 | A10 | ~1 MB | Via HuggingFace `datasets` |
| Alpaca-cleaned | A11 | ~25 MB | Via HuggingFace `datasets` |

---

## Lectures

| Lecture | Link |
|---------|------|
| Trees by Shourya | https://drive.google.com/file/d/1BgQ5BIHGi34kpsq-fPZ1GAYtVkmUjsFf/view |

---

## License

Educational use only. All dataset usage is subject to the original dataset licenses. Model weights (ResNet, DeiT-3, BERT, LLaMA) are subject to their respective model licenses.