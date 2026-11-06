# Update CAST on a Windows host: pull the latest code, rebuild, restart.
# Uses Docker's classic builder, which works from an SSH session (BuildKit asks
# Windows for stored credentials, which SSH sessions can't provide).
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
git pull -q
$env:DOCKER_BUILDKIT = "0"
docker build -q -f docker/Dockerfile -t castlive:local .
docker build -q -f docker/Dockerfile.worker -t castlive-worker:local .
Set-Location docker
docker compose up -d
docker compose ps --format "{{.Service}} {{.State}} {{.CreatedAt}}"
