# Swept Volume Neural Network Estimation (GPU Architecture Analysis - 591)

**Name:** [Your Name]
**Course Number:** 591
**Assignment Choice:** GPU: Network Architecture Analysis

**Python and PyTorch versions used:**
- Python 3.10+
- PyTorch 2.0+ (CUDA 11.8/12.1)

### Execution Commands

**Exact command to run one experimental configuration and seed:**
```bash
python3 neural_network.py --train-size 100000 --hidden-layers 2 --neurons 64 --lr 1e-3 --batch-size 10000 --seed 0 --epochs 1000 --device cuda