param([string]$Device='cpu')
$ErrorActionPreference='Stop'
python -m at2hjo.cli train --config configs/revision4/smoke.json --seed 2000 --device $Device --output runs/quick_joint
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
