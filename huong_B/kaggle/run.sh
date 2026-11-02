set -x
cd /kaggle/working
nvidia-smi --query-gpu=name,memory.total --format=csv
nvcc --version | tail -2
python --version; python -c "import torch;print(torch.__version__, torch.version.cuda)"
git clone -q https://github.com/Tencent-Hunyuan/Hunyuan3D-2 H3D
pip -q install diffusers transformers accelerate omegaconf einops trimesh rembg onnxruntime pyyaml opencv-python-headless safetensors xatlas pybind11 fast_simplification pytorch-lightning realesrgan basicsr 2>&1 | tail -3
cd H3D/hy3dgen/texgen/custom_rasterizer && python setup.py install 2>&1 | tail -15
cd ../differentiable_renderer && python setup.py install 2>&1 | tail -15
cd /kaggle/working
python run_tex.py
