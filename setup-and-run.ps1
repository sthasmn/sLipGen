#requires -RunAsAdministrator
<#
.SYNOPSIS
    Automates the setup and execution of the sLipGen project on Windows.
.DESCRIPTION
    This script performs the following actions:
    1. Checks for and installs WSL (Windows Subsystem for Linux).
    2. Checks if Docker Desktop is installed and guides the user if it's not.
    3. Checks if Git is installed and guides the user if it's not.
    4. Clones the sLipGen repository from GitHub.
    5. Builds the project's Docker image.
    6. Runs the processing pipeline on a user-specified directory of videos.
.PARAMETER VideosPath
    The absolute path to the directory containing the video files you want to process.
.EXAMPLE
    .\setup-and-run.ps1 -VideosPath "C:\Users\YourUser\Documents\MyVideos"
#>
param(
    [Parameter(Mandatory=$true)]
    [string]$VideosPath
)

# --- Function to check for and handle dependencies ---
function Check-Dependency {
    param (
        [string]$CommandName,
        [string]$DownloadUrl,
        [string]$AppName
    )

    if (-not (Get-Command $CommandName -ErrorAction SilentlyContinue)) {
        Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
        Write-Host "$AppName is not found in your PATH." -ForegroundColor Yellow
        Write-Host "The script will now open the official download page." -ForegroundColor Yellow
        Write-Host "Please install $AppName, ensuring you allow it to be added to the system PATH." -ForegroundColor Yellow
        Write-Host "IMPORTANT: After installation, please RESTART this PowerShell terminal and run the script again." -ForegroundColor Yellow
        Write-Host "------------------------------------------------------------" -ForegroundColor Yellow
        Start-Process $DownloadUrl
        Read-Host "Press Enter to exit the script. Please restart PowerShell after installation."
        exit
    } else {
        Write-Host "$AppName is already installed." -ForegroundColor Green
    }
}

# --- Main Script ---

# 1. Check if the video path exists
if (-not (Test-Path -Path $VideosPath -PathType Container)) {
    Write-Host "Error: The specified video path does not exist: $VideosPath" -ForegroundColor Red
    exit
}
Write-Host "Video directory found at: $VideosPath" -ForegroundColor Green

# 2. Check for WSL and install if necessary
Write-Host "--- Checking for WSL (Windows Subsystem for Linux) ---" -ForegroundColor Cyan
$wslStatus = wsl --status 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "WSL is not installed. Attempting to install..." -ForegroundColor Yellow
    wsl --install --no-distribution
    Write-Host "--------------------------------------------------------------------------" -ForegroundColor Green
    Write-Host "WSL has been installed. A system RESTART is required to complete the setup." -ForegroundColor Green
    Write-Host "Please restart your computer and then run this script again." -ForegroundColor Green
    Write-Host "--------------------------------------------------------------------------"
    Read-Host "Press Enter to exit."
    exit
} else {
    Write-Host "WSL is already installed." -ForegroundColor Green
}

# 3. Check for Docker and Git
Write-Host "--- Checking for Dependencies (Docker & Git) ---" -ForegroundColor Cyan
Check-Dependency -CommandName "docker" -AppName "Docker Desktop" -DownloadUrl "https://docs.docker.com/desktop/install/windows-install/"
Check-Dependency -CommandName "git" -AppName "Git" -DownloadUrl "https://git-scm.com/download/win"

# 4. Clone the repository
$repoUrl = "https://github.com/sthasmn/sLipGen.git"
$repoDir = "sLipGen"
Write-Host "--- Checking for sLipGen Repository ---" -ForegroundColor Cyan
if (-not (Test-Path -Path $repoDir)) {
    Write-Host "Cloning repository from $repoUrl..." -ForegroundColor Yellow
    git clone $repoUrl
    cd $repoDir
} else {
    Write-Host "Repository directory already exists." -ForegroundColor Green
    cd $repoDir
}

# 5. Build the Docker image
$imageName = "slipgen"
Write-Host "--- Building Docker Image: $imageName ---" -ForegroundColor Cyan
Write-Host "This may take several minutes..." -ForegroundColor Yellow
docker build -t $imageName .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: Docker build failed. Please check the Dockerfile and logs." -ForegroundColor Red
    exit
}
Write-Host "Docker image built successfully." -ForegroundColor Green

# 6. Run the processing pipeline
Write-Host "--- Starting the sLipGen Processing Pipeline ---" -ForegroundColor Cyan
Write-Host "Mounting your video directory: $VideosPath" -ForegroundColor Yellow
Write-Host "Press CTRL+C to stop the process at any time." -ForegroundColor Yellow

docker run --gpus all --rm -it -v "$VideosPath`:/videos" $imageName python main.py /videos/

Write-Host "--- Processing complete. ---" -ForegroundColor Green
