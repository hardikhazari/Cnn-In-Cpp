# Set build directory
$BuildDir = "build"

# Create build directory if it doesn't exist
if (!(Test-Path $BuildDir)) {
    New-Item -ItemType Directory -Path $BuildDir
}

# Configure with CMake using Clang and Ninja
cmake -S . -B $BuildDir -G "Ninja" -DCMAKE_CXX_COMPILER=clang++

# Build
cmake --build $BuildDir

# Run
if ($LASTEXITCODE -eq 0) {
    Write-Host "`n--- Running CNN In C++ ---" -ForegroundColor Green
    & "$BuildDir/CnnInCpp.exe"
} else {
    Write-Host "`nBuild failed." -ForegroundColor Red
}
