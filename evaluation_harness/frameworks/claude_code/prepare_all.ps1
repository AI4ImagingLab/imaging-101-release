# =================================================================
# Prepare all Claude Code (tmp_L1) sandboxes for 17 tasks
# Creates: tmp_L1/<task_name>/ with .prompt.md ready to paste
# Usage: .\prepare_all.ps1 [-Level L1]
# =================================================================

param(
    [ValidateSet("L1","L2","L3")]
    [string]$Level = "L1"
)

$RepoRoot = (Resolve-Path "$PSScriptRoot\..\..\..")
$OutBase = Join-Path $RepoRoot "tmp_$Level"

$AllTasks = @(
    "conventional_ptychography"
    "eht_black_hole_dynamic"
    "eht_black_hole_feature_extraction_dynamic"
    "eht_black_hole_original"
    "eht_black_hole_tomography"
    "eht_black_hole_UQ"
    "fourier_ptychography"
    "hessian_sim"
    "insar_phase_unwrapping"
    "light_field_microscope"
    "reflection_ODT"
    "seismic_FWI_original"
    "single_molecule_light_field"
    "SSNP_ODT"
)

Write-Host "========================================================"
Write-Host "  Preparing Claude Code sandboxes"
Write-Host "  Level: $Level"
Write-Host "  Output: $OutBase"
Write-Host "  Tasks: $($AllTasks.Count)"
Write-Host "========================================================"

New-Item -ItemType Directory -Force -Path $OutBase | Out-Null

$count = 0
foreach ($task in $AllTasks) {
    $count++
    Write-Host "[$count/$($AllTasks.Count)] Preparing $task..."
    
    $workspaceDir = Join-Path $OutBase $task
    
    try {
        python -m evaluation_harness prepare `
            --task $task `
            --level $Level `
            --workspace-dir $workspaceDir 2>$null
    } catch {
        Write-Host "  Warning: Failed to prepare $task" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "========================================================"
Write-Host "  DONE: $count sandboxes prepared in $OutBase"
Write-Host ""
Write-Host "  For each task:"
Write-Host "    1. Open the folder in Claude Code"
Write-Host "    2. Paste .prompt.md into the agent"
Write-Host "    3. Let it produce output/reconstruction.npy"
Write-Host "    4. Collect with:"
Write-Host "       python -m evaluation_harness collect ``"
Write-Host "           --task <task> --workspace-dir $OutBase\<task> ``"
Write-Host "           --level $Level --agent-name claude_code"
Write-Host "========================================================"
