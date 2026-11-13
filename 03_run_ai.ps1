# Run WolfGPT (Phase 3) from Windows -- no need to remember WSL paths.
#
#   .\03_run_ai.ps1                            interactive: type a prompt, Enter, get text
#   .\03_run_ai.ps1 "The best way to learn"    one completion, 200 tokens
#   .\03_run_ai.ps1 "Once upon a time" 500      one completion, 500 tokens
#   .\03_run_ai.ps1 "The sky is" 100 0          greedy: always the most likely token, stays on topic
#
# It only starts Ubuntu, activates the venv and runs phase3/generate.py; the model itself
# lives in phase3/runs/main.pt. Any other sampling flag needs bash directly:
#   wsl -d Ubuntu -e bash -lc "cd /mnt/c/Users/USER/Projetos/WolfGPT/phase3 && ~/wolfgpt-env/bin/python generate.py --top_k 20"
param(
    [string]$Prompt,
    [int]$Tokens = 200,
    [double]$Temperature = 0.9,
    [int]$TopK = 50
)

# C:\Users\... -> /mnt/c/Users/... so the same path means the same thing inside Ubuntu
$here = "/mnt/" + $PSScriptRoot.Substring(0, 1).ToLower() + $PSScriptRoot.Substring(2).Replace('\', '/')

# The prompt is passed as separate arguments and forwarded with "$@": putting it inside the
# bash -lc string would lose its quotes on the way through Windows and arrive word-split.
$run = 'cd "$0/phase3" && exec ~/wolfgpt-env/bin/python generate.py "$@"'
$flags = @("--tokens", $Tokens, "--temperature", $Temperature, "--top_k", $TopK)
if ($Prompt) { $flags = @("--prompt", $Prompt) + $flags }

Write-Host "Starting WolfGPT..." -ForegroundColor Green
wsl -d Ubuntu -e bash -lc $run bash $here @flags
