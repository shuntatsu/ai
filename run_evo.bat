@echo off
chcp 65001
echo Starting Evolution Training...
python scripts/run_evolution.py --generations 3 --population 5 --steps-per-gen 2000 --output-dir outputs/evolution_test
if %ERRORLEVEL% NEQ 0 (
    echo Error occurred with code %ERRORLEVEL% >> evolution_log.txt
) else (
    echo Training complete >> evolution_log.txt
)
