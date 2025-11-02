# Installation

This document contains detailed instructions for installing dependencies for GLAD. The code is tested on an Ubuntu 18.04 system
with an NVIDIA GeForce RTX 3070 GPU.

## Tested software stack

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.7.13 (Conda) | Matches the environment used by the authors |
| CUDA Toolkit | 11.1 | Required by the supplied TensorRT engines |
| cuDNN | 8.0.5 | Bundled with the CUDA 11.1 runtime from NVIDIA |
| PyTorch | 1.8.1 | Build compiled against CUDA 11.1 |
| TorchVision | 0.9.1 | Compatible with PyTorch 1.8.1 |
| NumPy | 1.20.3 | Stable with Python 3.7 and PyTorch 1.8.x |
| OpenCV-Python | 4.4.0.46 | Matches the version used to export the TensorRT engines |
| TensorRT | 7.2.2.3 | Required to deserialize the bundled `.engine` files |
| PyCUDA | 2021.1 | Built against CUDA 11.0/11.1; provides CUDA context helpers |
| Pillow | 8.4.0 | Pulled in by TorchVision transforms |

If you must deviate from these versions (e.g., newer GPU drivers), rebuild the TensorRT engines with the
matching toolchain and keep the PyTorch/TorchVision version pair aligned. The rest of the Python packages remain compatible as
long as you stay within the same major/minor ranges listed above.

## Create and activate the environment

```bash
conda create --name glad python=3.7.13
conda activate glad
```

> **Tip:** If you already have a CUDA 11.1 capable driver installed, you do not need to install the full CUDA toolkit system-wide.
> The `cudatoolkit` package that ships with the PyTorch build below is sufficient.

## Install Python dependencies

Install the packages in the exact order shown to avoid version resolution issues:

```bash
# Core numerical stack
conda install numpy=1.20.3 -c conda-forge

# PyTorch and TorchVision compiled for CUDA 11.1
conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge

# Computer vision utilities
pip install opencv-python==4.4.0.46 pillow==8.4.0

# CUDA bindings required by the TensorRT detector wrappers
pip install pycuda==2021.1

# TensorRT runtime matching the pre-built YOLO engines
pip install tensorrt==7.2.2.3
```

After the Python packages finish installing, copy the shared library `libmyplugins.so` from the `weights/` directory into the
same directory where you run the demos (or ensure it is discoverable via `LD_LIBRARY_PATH`).

## Verify the installation

Run the following checks inside the activated environment to confirm every dependency matches the expected version:

```bash
python - <<'PY'
import torch, torchvision, cv2, tensorrt, pycuda.autoinit
import numpy as np
print('PyTorch:', torch.__version__)
print('TorchVision:', torchvision.__version__)
print('NumPy:', np.__version__)
print('OpenCV:', cv2.__version__)
print('TensorRT:', tensorrt.__version__)
PY
```

If the versions reported by the script match the table above, the GLAD demos (`GLAD.py` and `GLAD_MC.py`) can be run without
any additional setup.

## More information

About TensorRT please see the readme in [the tensorrtx home page](https://github.com/wang-xinyu/tensorrtx).
The YOLOv5 version is v6.0 in this project.
