#!/bin/bash

# Exit immediately if any command fails
set -e

echo "Updating system..."
sudo apt update && sudo apt upgrade -y

echo "Adding CRAN GPG key and repository..."
sudo apt install -y dirmngr gnupg apt-transport-https ca-certificates software-properties-common

sudo gpg --keyserver keyserver.ubuntu.com --recv-key 'E298A3A825C0D65DFD57CBB651716619E084DAB9'
sudo gpg --export 'E298A3A825C0D65DFD57CBB651716619E084DAB9' | sudo tee /etc/apt/trusted.gpg.d/cran.asc > /dev/null

# Add CRAN repo for your Ubuntu version
UBUNTU_CODENAME=$(lsb_release -cs)
sudo add-apt-repository "deb https://cloud.r-project.org/bin/linux/ubuntu $UBUNTU_CODENAME-cran40/"

echo "Installing R..."
sudo apt update
sudo apt install -y r-base

echo "Installing R packages..."
sudo Rscript -e 'install.packages(c("lme4", "lmerTest", "Matrix", "methods", "report", "tibble",
"broom.mixed"), repos="https://cloud.r-project.org")'

echo "Creating conda environment 'pymer-env'..."
conda create -y -n pymer-env python=3.10
conda activate pymer-env

echo "Installing Python packages..."
conda install -y -c conda-forge rpy2
pip install pymer4

echo "Setup complete. To activate your environment, run:"
echo "conda activate pymer-env"
