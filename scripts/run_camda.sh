#!/usr/bin/env bash
# Generate synthetic data for CAMDA 2026 Track I (blue team).
#
# Run from the project root:
#   ./scripts/run_camda.sh BRCA eps7_k8
#   ./scripts/run_camda.sh COMBINED eps7_k8
#
# To change ε, k, S, R, Q, P: edit configs/BRCA.yaml or configs/COMBINED.yaml

set -euo pipefail

DATASET="${1:-BRCA}"
EXPERIMENT="${2:-eps7_k8}"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="$PROJECT_ROOT/configs/${DATASET}.yaml"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: config not found: $CONFIG"
    echo "DATASET must be BRCA or COMBINED"
    exit 1
fi

echo "Dataset   : TCGA-${DATASET}"
echo "Experiment: ${EXPERIMENT}"
echo "Config    : $CONFIG"
echo ""

python "$PROJECT_ROOT/src/camda_runner.py" "$CONFIG" --split all --experiment "$EXPERIMENT"
