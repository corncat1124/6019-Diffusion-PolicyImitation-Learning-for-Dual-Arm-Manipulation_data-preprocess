# Data Preprocessing for Dual-Arm Manipulation

This repository contains preprocessing scripts and datasets for the project **Diffusion Policy Imitation Learning for Dual-Arm Manipulation**.

## 📂 Repository Contents
- `datapreprocess.py` – Python script for preprocessing raw data.
- `liftpot_actions.npy` – NumPy array containing recorded actions.
- `liftpot_images.npy` – NumPy array containing image data (tracked with Git LFS).
- `stats.json` – JSON file with statistical metadata.

## ⚠️ Important: Large Files
Some files (e.g., `liftpot_images.npy`, `rawdata.zip`) exceed GitHub’s 100MB limit.  
These files are tracked using **Git LFS (Large File Storage)**.

### How to Download Large Files
To ensure you can access `.npy` and `.zip` files correctly:
1. Install Git LFS:
   ```bash
   git lfs install
