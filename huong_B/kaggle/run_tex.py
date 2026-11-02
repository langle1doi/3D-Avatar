import sys, os, glob, time, types, gc, torch
try:
    import pymeshlab
except Exception:
    m = types.ModuleType("pymeshlab"); m.MeshSet = object; sys.modules["pymeshlab"] = m
sys.path.insert(0, "/kaggle/working/H3D")
from PIL import Image
from rembg import remove
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
# nhận mọi ảnh .jpg/.jpeg/.png trong dataset đã gắn vào notebook (tên file = tên kết quả)
files = sorted(f for f in glob.glob("/kaggle/input/**/*", recursive=True) if f.lower().endswith((".jpg", ".jpeg", ".png")))
assert files, "không thấy ảnh nào trong /kaggle/input: hãy Add Data -> dataset chứa ảnh Pixar"
names = {os.path.splitext(os.path.basename(f))[0]: f for f in files}; print("ảnh:", list(names), flush=True)
OUT = "/kaggle/working/h3d_tex_out"; os.makedirs(OUT, exist_ok=True)
pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini", variant="fp16")
shapes = {}
for n in names:
    im = remove(Image.open(names[n]).convert("RGB"))
    mesh = pipe(image=im, num_inference_steps=30, octree_resolution=256, num_chunks=20000, generator=torch.manual_seed(2025), output_type="trimesh")[0]
    mesh = mesh.simplify_quadric_decimation(face_count=40000)
    shapes[n] = (im, mesh); print(n, "shape", len(mesh.faces), flush=True)
del pipe; gc.collect(); torch.cuda.empty_cache()
from hy3dgen.texgen import Hunyuan3DPaintPipeline
paint = Hunyuan3DPaintPipeline.from_pretrained("tencent/Hunyuan3D-2", subfolder="hunyuan3d-paint-v2-0-turbo")
try: paint.enable_model_cpu_offload()
except Exception as e: print("no offload", e)
for n, (im, mesh) in shapes.items():
    t = time.time()
    try:
        out = paint(mesh, image=im)
        out.export(f"{OUT}/{n}.glb"); print(n, "textured", round(time.time() - t, 1), "s", flush=True)
    except Exception as e:
        import traceback; traceback.print_exc(); print(n, "FAIL", e, flush=True)
    torch.cuda.empty_cache()
