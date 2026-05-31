# Fetches the latest ovrtx Python wheel from the official GitHub releases page
# and installs it into the current Python environment.
#
# Usage:
#   .\scripts\install_ovrtx.ps1
#
# This avoids pinning to a specific version that might be out of date; the
# script picks the highest semver wheel matching the host Python's cp tag.

$ErrorActionPreference = "Stop"

$pyTag = & python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')"
Write-Host "Detected Python tag: $pyTag"

$releaseApi = "https://api.github.com/repos/NVIDIA-Omniverse/ovrtx/releases/latest"
$release = Invoke-RestMethod -Uri $releaseApi -Headers @{ "User-Agent" = "ppp-installer" }

$asset = $release.assets | Where-Object {
    $_.name -like "*${pyTag}*win_amd64.whl"
} | Select-Object -First 1

if (-not $asset) {
    Write-Error "Could not find an ovrtx wheel for $pyTag in release $($release.tag_name)."
    exit 1
}

$dest = Join-Path $env:TEMP $asset.name
Write-Host "Downloading $($asset.name) -> $dest"
Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $dest

Write-Host "Installing $dest"
& python -m pip install --upgrade $dest

Write-Host "Done."
