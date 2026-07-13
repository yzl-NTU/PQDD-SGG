# PQDD-SGG
PQDD-SGG: Prior Guided Relation Queries with Decoupled Decoding for One-Stage Scene Graph Generation
It is recommended to create an isolated Conda environment.

## 1. Create a Conda Environment

```bash
conda create -n pqdd python=3.8 -y
conda activate pqdd
```

## 2. Install PyTorch

### Option A: RTX 4090 / CUDA 11.8

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Option B: CUDA 11.3 / PyTorch 1.10.1

```bash
conda install pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
```

Choose only one of the two installation options above. Do not install both.

## 3. Install Additional Dependencies

```bash
conda install scipy matplotlib -y
pip install cython
pip install -U "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
```
