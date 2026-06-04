#!/bin/bash
# Install Stage 3 (training) Python dependencies.
#
# Usage: bash stage3/setup_stage3_env.sh

set -euo pipefail

echo "=== Stage 3 environment setup ==="

module load python/3.13.5

# Core training stack
PACKAGES=(
    "torch>=2.4"
    "transformers>=4.55,<5.0"   # stable v4 for VLM compatibility
    "peft>=0.13"
    "accelerate>=1.0"
    "bitsandbytes>=0.44"
    "datasets>=3.0"
    "sentencepiece"
    "tiktoken"
    "einops"
    "timm"
    "av"                       # video reading (optional, frames are pre-extracted)
    "qwen-vl-utils"            # Qwen3-VL helpers
    "tensorboard"
    "pyyaml"
)

for pkg in "${PACKAGES[@]}"; do
    echo "  pip install --user '${pkg}' ..."
    pip install --user "${pkg}"
done

echo ""
echo "Verifying imports ..."
python3 -c "
import torch, transformers, peft, accelerate
print(f'  torch        {torch.__version__}')
print(f'  transformers {transformers.__version__}')
print(f'  peft         {peft.__version__}')
print(f'  accelerate   {accelerate.__version__}')
print(f'  cuda avail   {torch.cuda.is_available()}')
"

echo ""
echo "Stage 3 environment ready."
