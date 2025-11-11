# Parallel Training Guide

This document explains how to use parallel training in the FOF project.

## Available Parallelization Methods

### 1. DataLoader Parallelization
Parallelizes data loading to speed up data preprocessing.

```bash
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --num_workers 8 \
    --use_wandb
```

### 2. GPU Parallelization (DataParallel)
Parallelizes training using multiple GPUs on a single machine.

```bash
# Use all GPUs
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --use_data_parallel \
    --use_wandb

# Use specific GPUs only
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --use_data_parallel \
    --gpu_ids "0,1,2" \
    --use_wandb

# Use only one specific GPU
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --use_data_parallel \
    --gpu_ids "0" \
    --use_wandb
```

### 3. Mixed Precision Training (AMP)
Reduces memory usage and improves training speed.

```bash
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --use_amp \
    --use_wandb
```

### 4. Gradient Accumulation
Simulates larger batch sizes to improve memory efficiency.

```bash
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --use_wandb
```

## Combination Examples

### High Performance Configuration
```bash
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --num_workers 8 \
    --use_data_parallel \
    --use_amp \
    --gradient_accumulation_steps 2 \
    --use_wandb \
    --wandb_project fof-high-performance
```

### Memory Efficient Configuration
```bash
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 8 \
    --gradient_accumulation_steps 8 \
    --use_amp \
    --use_wandb
```

### Maximum Performance Configuration
```bash
python src/training/train_with_generated_data.py \
    --data_dir generated_data \
    --batch_size 32 \
    --num_workers 8 \
    --use_data_parallel \
    --use_amp \
    --gradient_accumulation_steps 2 \
    --use_wandb \
    --wandb_project fof-max-performance
```

## Parameter Descriptions

### Parallelization-Related Parameters

- `--num_workers`: Number of data loading workers (default: 4)
- `--use_data_parallel`: Use DataParallel (multiple GPUs)
- `--gpu_ids`: GPU IDs to use (e.g., "0,1,2" or "all")
- `--use_amp`: Use mixed precision training
- `--gradient_accumulation_steps`: Number of gradient accumulation steps (default: 1)
- `--validation_frequency`: Validation execution frequency (default: every 10000 data points)

## Notes

1. **Memory Usage**: Parallelization may increase memory usage
2. **Batch Size Adjustment**: Adjusting batch size when parallelizing is recommended
3. **Gradient Accumulation**: Use gradient accumulation when large batch sizes are needed
4. **GPU Conflicts**: `--use_data_parallel` automatically uses all GPUs, which may conflict with other processes
5. **Explicit GPU Specification**: In production environments, explicitly specifying GPUs with `--gpu_ids` is recommended

## Troubleshooting

### Out of Memory Errors
- Reduce batch size
- Use gradient accumulation
- Use mixed precision training

### Slow Data Loading
- Increase `num_workers`
- Verify that `pin_memory=True` is enabled

### GPU Conflict Errors
- Explicitly specify GPUs with `--gpu_ids`
- Check GPU usage with `nvidia-smi`
- Verify that other processes are not using GPUs
