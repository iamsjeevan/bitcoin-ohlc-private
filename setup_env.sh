#!/bin/bash

# --- Configuration ---
PYTHON_VERSION="3.11"
PROJECT_DIR="." # Current directory
DATA_RAW_DIR="data/raw"
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
DOWNLOAD_SCRIPT="download_data.py"

echo "--- Starting Environment Setup for BTC Prediction Pipeline (macOS) ---"

# --- 1. Check for Homebrew ---
if ! command -v brew &> /dev/null; then
    echo "Error: Homebrew not found. Please install Homebrew first:"
    echo "/bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    echo "After installation, you might need to restart your terminal."
    exit 1
fi
echo "Homebrew is installed."

# --- 2. Install Python via Homebrew ---
# We install a specific version to ensure consistency and avoid conflicts with system Python.
if ! brew list --formula | grep -q "python@${PYTHON_VERSION}"; then
    echo "Python ${PYTHON_VERSION} not found. Installing via Homebrew..."
    brew install "python@${PYTHON_VERSION}"
else
    echo "Python ${PYTHON_VERSION} is already installed."
fi

# Set Homebrew Python 3.11 as the default for this session for venv creation
# This is CRUCIAL to ensure the correct python is used to create the venv
export PATH="$(brew --prefix python@${PYTHON_VERSION})/bin:$PATH"

# Verify the python3 that will be used for venv creation
echo "Verified python3 for venv: $(which python3)"
if [[ ! "$(which python3)" =~ "python@${PYTHON_VERSION}" ]]; then
    echo "Warning: python3 in PATH is not the expected Homebrew Python ${PYTHON_VERSION}. Please ensure your shell's PATH is correctly set up for Homebrew."
    echo "You might need to add 'export PATH=\"/opt/homebrew/opt/python@${PYTHON_VERSION}/bin:\$PATH\"' to your ~/.zshrc or ~/.bash_profile."
    echo "Attempting to continue, but be aware of potential issues."
fi


# --- 3. Install TA-Lib C Library ---
if ! brew list --formula | grep -q "ta-lib"; then
    echo "TA-Lib C library not found. Installing via Homebrew..."
    brew install ta-lib
else
    echo "TA-Lib C library is already installed."
fi

# --- 4. Create Virtual Environment ---
echo "Creating virtual environment '${VENV_DIR}' using python3..."
python3 -m venv "${VENV_DIR}"
if [ $? -ne 0 ]; then
    echo "Error: Failed to create virtual environment."
    exit 1
fi
echo "Virtual environment created."

# --- 5. Activate Virtual Environment ---
echo "Activating virtual environment..."
source "${VENV_DIR}/bin/activate"
if [ $? -ne 0 ]; then
    echo "Error: Failed to activate virtual environment."
    exit 1
fi
echo "Virtual environment activated: $(which python)" # Confirm activated python version

# --- 6. Upgrade pip and setuptools within the venv ---
# This prevents build issues with certain packages like TA-Lib
echo "Upgrading pip, setuptools, and wheel in virtual environment..."
pip install --upgrade pip setuptools wheel
if [ $? -ne 0 ]; then
    echo "Warning: Failed to upgrade pip/setuptools/wheel. This might cause issues with package installation."
fi

# --- 7. Create requirements.txt ---
echo "Creating ${REQUIREMENTS_FILE}..."
cat <<EOF > "${REQUIREMENTS_FILE}"
pandas
pyarrow
python-binance
TA-Lib
tqdm
EOF
echo "Dependencies listed in ${REQUIREMENTS_FILE}."

# --- 8. Install Python Libraries ---
echo "Installing Python packages from ${REQUIREMENTS_FILE} into virtual environment..."
# Using specific TA-Lib version often helps with compilation issues
pip install --no-cache-dir "TA-Lib==0.4.28" # Explicitly install a known stable version
pip install --no-cache-dir -r <(grep -v "TA-Lib" "${REQUIREMENTS_FILE}") # Install others, excluding TA-Lib
if [ $? -ne 0 ]; then
    echo "Error installing Python packages. Please check the output above."
    echo "You might need to manually run 'pip install -r requirements.txt' or debug TA-Lib installation."
    deactivate # Exit venv on error
    exit 1
fi
echo "Python packages installed successfully."

# --- 9. Create Data Directories ---
echo "Creating data directories: ${DATA_RAW_DIR}/"
mkdir -p "${DATA_RAW_DIR}"
echo "Directory structure created."

# --- 10. Create empty download_data.py file ---
echo "Creating empty ${DOWNLOAD_SCRIPT} file..."
touch "${DOWNLOAD_SCRIPT}"
echo "${DOWNLOAD_SCRIPT} created. You can now add the Python code to it."

echo "--- Environment Setup Complete! ---"
echo "You are now in the virtual environment."
echo "To exit, type 'deactivate'."
echo "To reactivate later, navigate to your project directory and run: 'source ${VENV_DIR}/bin/activate'"