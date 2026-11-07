# Update CAST Live on a Windows host: pull the latest code and restart.
# The code is mounted into the containers (compose.yml), so nothing is rebuilt unless you ask:
#   powershell -File docker\update.ps1          # code changes: pull + restart (seconds)
#   powershell -File docker\update.ps1 -Build   # Dockerfiles or dependencies changed
# Builds use Docker's classic builder, which works from an SSH session (BuildKit asks Windows
# for stored credentials, which SSH sessions can't provide). A restart stops a model that is
# being made; it is retried from its last finished stage.
param([switch]$Build)
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
git pull -q
if ($Build) {
    $env:DOCKER_BUILDKIT = "0"
    docker build -q -f docker/Dockerfile -t castlive:local .
    docker build -q -f docker/Dockerfile.worker -t castlive-worker:local .
    docker build -q -f docker/Dockerfile.live -t castlive-live:local .
    docker image prune -f  # the replaced images' layers
}
Set-Location docker
docker compose up -d
if (-not $Build) { docker compose restart capture studio worker reconstructor live }
docker compose ps --format "{{.Service}} {{.State}} {{.CreatedAt}}"
