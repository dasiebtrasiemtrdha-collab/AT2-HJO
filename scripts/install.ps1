param([ValidateSet('cpu','cuda')][string]$Backend='cuda')
$ErrorActionPreference='Stop'
python -m venv .venv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$Lock=if ($Backend -eq 'cuda') {'requirements/windows-cuda126-py312.lock'} else {'requirements/windows-cpu-py312.lock'}
& .venv/Scripts/python -m pip install -r $Lock
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .venv/Scripts/python -m pip install --no-deps -e .
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& .venv/Scripts/python -m pip check
exit $LASTEXITCODE
