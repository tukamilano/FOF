# FOF (First-Order Formula) - Transformer-based Theorem Prover

A system that automates propositional logic theorem proving using Transformer models. Combined with [pyprover](https://github.com/kaicho8636/pyprover), it provides a consistent workflow from formula generation → training → inference → self-improvement.

## 🚀 Key Features

- **Hierarchical Classification Architecture**: Independently predicts tactic types and arguments
- **Inference Evaluation Suite**: Compare and verify various inference methods
- **Large-scale Data Operations**: Efficiency through GCS integration and deduplication
- **Parallel Data Collection/Training**: Multi-process, multi-GPU, and AMP support
- **Experiment Tracking**: Detailed logging and visualization with wandb

## 🔰 Quick Start (Inference Only)

Quickly test inference with a pretrained model:

```bash
python validation/inference_hierarchical.py \
  --model_path models/pretrained_model.pth \
  --count 100 \
  --max_steps 30 \
  --verbose
```

- See `validation/pretrained_model_validation.txt` for additional benchmarks

## Environment Setup

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python 3.8+ (recommended: 3.9-3.11)
- PyTorch
- [pyprover](https://github.com/kaicho8636/pyprover)
- wandb (optional)
- `google-cloud-storage` if using GCS

## Project Structure (Overview)

```
FOF/
├── automation/                   # Automation scripts
│   ├── create_temperature_mixture.sh
│   ├── run_self_improvement.sh
│   ├── run_train_simple_loop.sh
│   └── README.md
├── src/
│   ├── core/                     # Transformer/encoder/parameters
│   ├── data_generation/          # Generation/collection (with parallelization)
│   ├── interaction/              # Self-improvement data collection
│   ├── training/                 # Training/analysis/deduplication
│   └── compression/              # Compression utilities
├── validation/                   # Inference/comparison
├── tests/                        # Test suite
├── models/                       # Pretrained/checkpoints
└── pyprover/                     # Theorem prover
```

## Usage

### 1) Data Generation

```bash
# Parallel data collection (local storage)
python src/data_generation/auto_data_parallel_collector.py \
  --count 1000 \
  --workers 4 \
  --examples_per_file 100

# Save directly to GCS
python src/data_generation/auto_data_parallel_collector.py \
  --count 10000 \
  --workers 8 \
  --gcs_bucket your-bucket \
  --gcs_prefix generated_data/
```

### 2) Deduplication and Analysis

```bash
python src/training/deduplicate_generated_data.py \
  --input_dir generated_data \
  --output_dir deduplicated_data

python src/training/analyze_generated_data.py
```

### 3) Training (Simple)

```bash
python src/training/train_simple.py \
  --data_dir deduplicated_data \
  --batch_size 32 \
  --learning_rate 3e-4 \
  --num_epochs 10

# Track with wandb
python src/training/train_simple.py --use_wandb --wandb_project fof-training
```

For more detailed workflows and two-stage deduplication, see `src/training/README.md`.

### 4) Inference and Comparison

```bash
# Hierarchical classification inference
python validation/inference_hierarchical.py \
  --model_path models/pretrained_model.pth \
  --count 100 \
  --max_steps 30

# Beam search and other comparisons
python validation/inference_beam_search.py --help
python validation/compare_inference_methods.py --help
```

## Parallel Training and Optimization Options

- Supports DataLoader parallelization, multiple GPUs (DataParallel), AMP, and gradient accumulation
- See `src/training/PARALLEL_TRAINING.md` for examples and recommended settings

## Automation Scripts (automation/)

See `automation/README.md` for a quick guide. Grant execution permissions before running:

```bash
chmod +x automation/*.sh
```

Examples:

```bash
# Temperature mixture generation
./automation/create_temperature_mixture.sh RL3

# Training loop (e.g., RL1→RL2)
./automation/run_train_simple_loop.sh RL1 RL2 your-gcs-bucket-prefix

# Self-improvement data collection
./automation/run_self_improvement.sh RL3
```

## Models/Checkpoints

- `models/pretrained_model.pth`: Pretrained model
- `models/RL*_*.pth`: Models obtained from SFT cycles (temperature, beam search, top_k, etc.)

## Recommended Workflow (Summary)

```bash
# 1. Generation
python src/data_generation/auto_data_parallel_collector.py --count 1000 --workers 4

# 2. Deduplication
python src/training/deduplicate_generated_data.py --input_dir generated_data --output_dir deduplicated_data

# 3. Training
python src/training/train_simple.py --data_dir deduplicated_data --use_wandb

# 4. Inference
python validation/inference_hierarchical.py --verbose
```

## Acknowledgments

This project uses the following OSS:

- [pyprover](https://github.com/kaicho8636/pyprover)
- PyTorch
- wandb
