#!/usr/bin/env bash
# Generate synthetic scRNA-seq data for CAMDA 2026 Track II.
#
# Usage (from project root):
#   ./scripts/run_scrna.sh [experiment_name]
#
# Examples:
#   ./scripts/run_scrna.sh                  # experiment label: eps7_k4
#   ./scripts/run_scrna.sh eps7_k4_n150
#
# Output: results/scrna/synthetic_{experiment_name}.h5ad
#
# To change ε, K, n_1way, n_2way, subsampling: edit configs/onek1k.yaml

set -euo pipefail

EXPERIMENT="${1:-eps7_k4}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$PROJECT_ROOT/configs/onek1k.yaml"

echo "Config    : $CONFIG"
echo "Experiment: $EXPERIMENT"
echo ""

python "$PROJECT_ROOT/src/scrna_runner.py" "$CONFIG" --experiment "$EXPERIMENT"
