"""Kaggle kernel vngipl/pixar-seg3 (GPU): cut the Pixar portraits (output of vngipl/real2pixar-qwen) into face parts and re-assemble new faces.
v11 (SegFormer + MediaPipe) -> SAM 2 + ViTMatte masks (pixar_seg2_masks.py) -> face base by skin diffusion + compose + mix (pixar_face_compose.py).
Deliverables in /kaggle/working/seg3: compose/, mix/, sheets/, compose.csv, stage1.csv, parts.zip (all part PNGs, one file to keep the download small).
Usage (from scripts/): python build_notebook_pixar_seg3.py"""
import json

from common import ROOT

USER = "vngipl"  # third Kaggle account (KAGGLE_API_TOKEN_3)
S = ROOT / "scripts"
k = ROOT / "kaggle_upload" / "pixar_seg3_kernel"
V11 = (ROOT / "results" / "pixar_parts_v11" / "seg.py").read_text(encoding="utf-8")
# v11 globbed every png under /kaggle/input and wrote to /kaggle/working: point it at the portrait copy and a scratch dir
V11 = V11.replace('"/kaggle/input/**/*.png"', '"/kaggle/temp/portraits/*.png"').replace('W = "/kaggle/working"', 'W = "/kaggle/temp/v11"')
assert "/kaggle/temp/portraits" in V11 and "/kaggle/temp/v11" in V11
# v11 cut the nose above 3/4 of the eye height, which dropped the bridge: start at the top of the eyes instead (the SegFormer nose class decides where it really begins)
NOSE_CUT = 'm["nose"][: int(ey.max() - 0.25 * (ey.max() - ey.min())), :] = False'
assert NOSE_CUT in V11
V11 = V11.replace(NOSE_CUT, 'm["nose"][: int(ey.min()), :] = False')

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n").splitlines(True)})
put = lambda name, text: f"open('/kaggle/temp/code/{name}', 'w', encoding='utf-8').write({text!r})"

md("# pixar-seg3\nPixar portraits -> v11 masks -> SAM 2 + ViTMatte -> face base (skin diffusion) -> compose + mix. Output: /kaggle/working/seg3.")
code("""
import os, glob, shutil
BASE = '/kaggle/temp'
os.makedirs(f'{BASE}/code', exist_ok=True)
os.environ['HF_HOME'] = f'{BASE}/hf_cache'
import torch
print(torch.cuda.is_available() and torch.cuda.get_device_name(0))
os.makedirs(f'{BASE}/portraits', exist_ok=True)
src = [f for f in glob.glob('/kaggle/input/**/pixar/[0-9][0-9][0-9][0-9].png', recursive=True)]
for f in src:
    shutil.copy(f, f'{BASE}/portraits/')
print(len(src), 'Pixar portraits from', os.path.dirname(src[0]) if src else None)
assert len(src) == 50, 'expected the 50 Pixar portraits of vngipl/real2pixar-qwen'
""")
code("!pip install -q transformers==4.57.1 mediapipe\n!pip uninstall -y -q torchao\n!python -c \"from transformers import Sam2Model; print('ok')\"")
code(put("v11_seg.py", V11) + "\n" + put("pixar_seg2_masks.py", (S / "pixar_seg2_masks.py").read_text(encoding="utf-8")) + "\n"
     + put("pixar_face_compose.py", (S / "pixar_face_compose.py").read_text(encoding="utf-8")) + "\n"
     + put("pixar_seg_lab.py", (S / "pixar_seg_lab.py").read_text(encoding="utf-8")))
code("""
import subprocess, sys
def run(*a):
    r = subprocess.run([sys.executable, '-u', *a]); assert r.returncode == 0, f'{a[0]} failed with code {r.returncode}'
P, V, O, OUT = '/kaggle/temp/portraits', '/kaggle/temp/v11', '/kaggle/temp/seg3', '/kaggle/working/seg3'
if len(glob.glob(f'{V}/parts/*')) < 50:
    run('/kaggle/temp/code/v11_seg.py')                                                  # stage 0: v11 SegFormer + MediaPipe masks
assert len(glob.glob(f'{V}/parts/*')) == 50
run('/kaggle/temp/code/pixar_seg2_masks.py', '--img', P, '--v11', V, '--out', O)        # stage 1: SAM 2 + ViTMatte masks
run('/kaggle/temp/code/pixar_seg_lab.py', '--img', P, '--out', O)                       # class maps: ears / neck / glasses / cloth / hat for the face base
run('/kaggle/temp/code/pixar_face_compose.py', '--img', P, '--seg', O, '--out', OUT, '--mix', '12')  # stage 2+3: face base, compose, mix
""")
code("""
import numpy as np, zipfile
from PIL import Image, ImageDraw
shutil.copy(f'{O}/stage1.csv', OUT)
with zipfile.ZipFile(f'{OUT}/parts.zip', 'w', zipfile.ZIP_STORED) as z:  # every part layer, full canvas RGBA
    for f in sorted(glob.glob(f'{O}/parts/*/*.png') + glob.glob(f'{O}/parts_soft_crop/*/*') + glob.glob(f'{O}/lab/*.png')):  # base, saved parts (+ layout.json), class maps
        z.write(f, os.path.relpath(f, O))
os.makedirs(f'{OUT}/sheets', exist_ok=True)
def onwhite(p):
    a = np.array(Image.open(p).convert('RGBA')).astype(float); al = a[..., 3:4] / 255
    return Image.fromarray((a[..., :3] * al + 255 * (1 - al)).astype(np.uint8))
ids = sorted(os.path.basename(d) for d in glob.glob(f'{O}/parts/*'))
for s in range(0, len(ids), 10):   # columns: Pixar | face base | re-assembled face
    chunk = ids[s:s + 10]; sh = Image.new('RGB', (3 * 256, 256 * len(chunk)), 'white')
    for r, pid in enumerate(chunk):
        for c, im in enumerate((Image.open(f'{P}/{pid}.png').convert('RGB'), onwhite(f'{O}/parts/{pid}/face.png'), onwhite(f'{OUT}/compose/{pid}.png'))):
            sh.paste(im.resize((256, 256)), (256 * c, 256 * r))
        ImageDraw.Draw(sh).text((4, 256 * r + 4), pid, fill=(255, 255, 0))
    sh.save(f'{OUT}/sheets/sheet_{s // 10 + 1:02d}.jpg', quality=88)
assert len(glob.glob(f'{OUT}/compose/*.png')) == 50 and os.path.exists(f'{OUT}/parts.zip')
print(open(f'{OUT}/compose.csv').read()[:400])
""")

k.mkdir(parents=True, exist_ok=True)
(k / "pixar_seg3.ipynb").write_text(json.dumps({"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}))
(k / "kernel-metadata.json").write_text(json.dumps({"id": f"{USER}/pixar-seg3", "title": "pixar-seg3", "code_file": "pixar_seg3.ipynb", "language": "python",
    "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
    "dataset_sources": [], "competition_sources": [], "kernel_sources": [f"{USER}/real2pixar-qwen"], "model_sources": []}, indent=2))
print("wrote", k)
