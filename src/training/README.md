# Generated Data Training

This directory contains scripts for training models using data from the `generated_data` directory.

## File Structure

### Core Functionality
- `train_with_generated_data.py`: Main training script (for `generated_data`)
- `run_training.py`: Wrapper script for easy training execution
- `inference_hierarchical.py`: Hierarchical classification model inference script

### Analysis & Utilities
- `analyze_generated_data.py`: Script to analyze generated data content
- `check_duplicates.py`: Script to check for data duplicates

### Documentation
- `README.md`: This file

## Usage

### 1. Data Analysis

First, check the content of generated data:

```bash
python src/training/analyze_generated_data.py
```

To check duplicate details:

```bash
python src/training/check_duplicates.py
```

### 2. Training Execution

#### Simple execution (recommended)
```bash
python src/training/run_training.py
```

#### Execution with detailed settings
```bash
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 16 \
    --learning_rate 3e-4 \
    --num_epochs 5 \
    --save_path models/hierarchical_model_generated.pth \
    --eval_split 0.2 \
    --max_seq_len 256 \
    --use_wandb \
    --wandb_project fof-training
```

### 3. Inference Execution

Run inference using the trained model:

```bash
python src/training/inference_hierarchical.py \
    --model_path models/hierarchical_model_generated.pth \
    --count 10 \
    --max_steps 20 \
    --temperature 1.0 \
    --verbose \
    --use_wandb \
    --wandb_project fof-inference
```

## Parameter Description

- `--data_dir`: Directory containing generated data (default: `generated_data`)
- `--batch_size`: Batch size (default: 16)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--num_epochs`: Number of epochs (default: 5)
- `--save_path`: Model save path (default: `models/hierarchical_model_generated.pth`)
- `--eval_split`: Evaluation data ratio (default: 0.2)
- `--max_seq_len`: Maximum sequence length (default: 256)
- `--remove_duplicates`: Remove duplicate examples with same state_hash (default: enabled)
- `--keep_duplicates`: Keep duplicate examples (disables `--remove_duplicates`)
- `--use_wandb`: Record logs with wandb
- `--wandb_project`: wandb project name (default: `fof-training`/`fof-inference`)
- `--wandb_run_name`: wandb run name (default: auto-generated)

## Using wandb

### 1. Install wandb
```bash
pip install wandb
```

### 2. Login to wandb
```bash
wandb login
```

### 3. Log during training
```bash
python src/training/train_with_generated_data.py --use_wandb --wandb_project fof-training
```

### 4. Log during inference
```bash
python src/training/inference_hierarchical.py --use_wandb --wandb_project fof-inference
```

### Recorded Information
- **During training**: Loss, accuracy, learning rate per epoch
- **During inference**: Success/failure per example, step count, confidence, tactic usage frequency

## Data Format

The `generated_data` directory must contain JSON files in the following format:

```json
[
  {
    "example_hash": "unique_hash",
    "meta": {
      "goal_original": "original_goal",
      "is_proved": true
    },
    "steps": [
      {
        "step_index": 0,
        "premises": ["premise1", "premise2"],
        "goal": "current_goal",
        "tactic": {
          "main": "tactic_name",
          "arg1": "argument1",
          "arg2": "argument2"
        },
        "tactic_apply": true,
        "state_hash": "state_hash"
      }
    ]
  }
]
```

## Two-Stage Deduplication Workflow

To streamline deduplication during training, we provide a two-stage workflow:

### Stage 1: Pre-Deduplication (Recommended)

Remove duplicates before training and generate deduplicated data:

```bash
# Execute deduplication
python src/training/deduplicate_generated_data.py \
    --input_dir generated_data \
    --output_dir deduplicated_data \
    --report_file deduplication_report.json \
    --verbose
```

**Output Format:**
After deduplication, proof continuity is lost, so data is saved as a simple collection of steps:

```json
[
  {
    "step_index": 0,
    "premises": [],
    "goal": "((((b ∧ a) ∧ ((b ∨ c) → False)) ∧ (c → (b → c))) → b)",
    "tactic": {
      "main": "intro",
      "arg1": null,
      "arg2": null
    },
    "tactic_apply": true,
    "state_hash": "29326faf43695967bc47255fc73a580c"
  }
]
```

**Benefits:**
- Deduplication and training are separated, improving efficiency
- Deduplicated data can be reused multiple times
- Generates detailed deduplication statistics report
- Reduces memory usage during training
- Improved memory efficiency with simple step collection format

### Stage 2: Training Execution

Train using deduplicated data:

```bash
# Use deduplicated data (recommended)
python src/training/train_with_generated_data.py \
    --use_deduplicated_data \
    --data_dir deduplicated_data \
    --batch_size 32 \
    --learning_rate 3e-4

# Or traditional method (deduplication during training)
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --learning_rate 3e-4
```

### About Traditional Deduplication

`generated_data` may contain duplicate examples with the same state (`state_hash`). The traditional method automatically removes duplicates during training:

- **With deduplication** (default): Only the first example with the same `state_hash` is kept
- **Keep duplicates**: Use `--keep_duplicates` option to retain duplicates

Deduplication improves training data quality and enables more efficient training.

## Notes

1. **Data Directory**: Ensure the `generated_data` directory exists and contains JSON files
2. **Deduplicated Data**: When using `--use_deduplicated_data`, run `deduplicate_generated_data.py` first
3. **Memory Usage**: Set `batch_size` and `max_seq_len` appropriately
4. **Training Time**: Training may take time. GPU will be used automatically if available
5. **Model Saving**: Models are saved to the `models` directory (automatically created if it doesn't exist)
6. **Deduplication Effect**: Deduplication may reduce training data count (typically 7-10%)
7. **Workflow Recommendation**: For efficient training, we recommend the two-stage workflow (pre-deduplication → training)
