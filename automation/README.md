# Automation Scripts

## Commands

### Create mixture datasets
```bash
./create_temperature_mixture.sh RL3
```

### Train models
```bash
./run_train_simple_loop.sh RL1 RL2 fof-data-20251010-milano
```

### Collect self-improvement data
```bash
./run_self_improvement.sh RL3
```

## Complete Cycle Example

### RL1 → RL2
```bash
./create_temperature_mixture.sh RL1
./run_train_simple_loop.sh RL1 RL2 fof-data-20251010-milano
./create_temperature_mixture.sh RL2
./run_self_improvement.sh RL2
```

### RL2 → RL3
```bash
./run_train_simple_loop.sh RL2 RL3 fof-data-20251010-milano
./create_temperature_mixture.sh RL3
./run_self_improvement.sh RL3
```