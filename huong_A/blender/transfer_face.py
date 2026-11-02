# Chạy: blender -b --python transfer_face.py -- vao.glb ra.glb [ti_le_dau=0.7] [so_mat=25000]
# Chuyển 52 biểu cảm ARKit của đầu MPFB sang mô hình TripoSR (đỉnh gần nhất, trọng số giảm dần theo khoảng cách).
import importlib, sys, os
import bpy
import numpy as np
from mathutils import kdtree

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, DST = argv[0], argv[1]
HEAD_FRAC = float(argv[2]) if len(argv) > 2 else 0.7  # phần chiều cao mô hình (từ đỉnh) là đầu, tới cằm
TARGET_FACES = int(argv[3]) if len(argv) > 3 else 25000


def dynamic_import(pkg, key):
    for amod in sys.modules:
        if amod.endswith(pkg):
            return getattr(importlib.import_module(amod), key)
    raise ValueError(pkg)


HumanService = dynamic_import("mpfb.services.humanservice", "HumanService")
FaceService = dynamic_import("mpfb.services.faceservice", "FaceService")
ExportService = dynamic_import("mpfb.services.exportservice", "ExportService")
ObjectService = dynamic_import("mpfb.services.objectservice", "ObjectService")

# --- 1. nhân vật MPFB (đã bake, bỏ helper) làm nguồn biểu cảm
human = HumanService.create_human()
FaceService.load_targets(human, load_microsoft_visemes=False, load_arkit_faceunits=True)
root = ExportService.create_character_copy(human, name_suffix="_src")
src = ObjectService.find_object_of_type_amongst_nearest_relatives(root, "Basemesh")
ExportService.bake_modifiers_remove_helpers(src, bake_masks=True, bake_subdiv=False, remove_helpers=True, also_proxy=True)
keys = src.data.shape_keys.key_blocks
arkit = sorted(f[:-7] for f in os.listdir(os.path.join(os.environ["APPDATA"], "Blender Foundation", "Blender", "4.5", "extensions", ".user",
                                                      "user_default", "mpfb", "data", "targets", "faceunits")) if f.endswith(".target"))
arkit = [k for k in arkit if k in keys]
n = len(src.data.vertices)
basis = np.empty(n * 3); keys["Basis"].data.foreach_get("co", basis); basis = basis.reshape(-1, 3)
mw = np.array(src.matrix_world)
P = basis @ mw[:3, :3].T + mw[:3, 3]  # toạ độ thế giới
zmax_m = P[:, 2].max()
head_idx = np.where(P[:, 2] > 0.865 * zmax_m)[0]  # từ cằm trở lên
hp = P[head_idx]

# --- 2. nạp mô hình TripoSR, giảm mặt
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=SRC)
tgt = [o for o in bpy.data.objects if o not in before and o.type == "MESH"][0]
bpy.context.view_layer.objects.active = tgt
bpy.ops.object.select_all(action="DESELECT"); tgt.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
# mesh TripoSR từ Kaggle có chiều mặt ngược (marching cubes của skimage thay torchmcubes) -> lật lại
import bmesh
bm = bmesh.new(); bm.from_mesh(tgt.data); bmesh.ops.reverse_faces(bm, faces=bm.faces); bm.to_mesh(tgt.data); bm.free()
# TripoSR: trục lên = Z, mặt hướng +X (trong file). Sau khi Blender nhập glTF (Y-up -> Z-up), toạ độ thành (x, y, z) = (xf, -zf, yf).
# Đưa về quy ước MPFB (Z lên, mặt hướng -Y) bằng phép quay thuần (x, y, z) -> (z, -x, -y).
_n = len(tgt.data.vertices)
_a = np.empty(_n * 3); tgt.data.vertices.foreach_get("co", _a); _a = _a.reshape(-1, 3)
tgt.data.vertices.foreach_set("co", np.stack([_a[:, 2], -_a[:, 0], -_a[:, 1]], axis=1).reshape(-1)); tgt.data.update()
nf = len(tgt.data.polygons)
if nf > TARGET_FACES:
    m = tgt.modifiers.new("dec", "DECIMATE"); m.ratio = TARGET_FACES / nf
    bpy.ops.object.modifier_apply(modifier="dec")
print("MESH faces", nf, "->", len(tgt.data.polygons), "color attrs", [a.name for a in tgt.data.color_attributes])

nt = len(tgt.data.vertices)
T = np.empty(nt * 3); tgt.data.vertices.foreach_get("co", T); T = T.reshape(-1, 3)
zt_max, zt_min = T[:, 2].max(), T[:, 2].min()
H = zt_max - zt_min
zchin = zt_max - HEAD_FRAC * H
th = T[T[:, 2] > zchin]

# --- 3. căn đầu MPFB vào đầu TripoSR: tỉ lệ theo chiều cao, mép trên, giữa theo x, mặt trước theo y
s = (zt_max - zchin) / (hp[:, 2].max() - hp[:, 2].min())
hp2 = (hp - np.array([hp[:, 0].mean(), hp[:, 1].min(), hp[:, 2].max()])) * s
hp2 += np.array([(th[:, 0].min() + th[:, 0].max()) / 2, th[:, 1].min(), zt_max])
print("ALIGN scale", round(s, 3), "H", round(H, 3))

kd = kdtree.KDTree(len(hp2))
for i, p in enumerate(hp2): kd.insert(p, i)
kd.balance()
R = 0.06 * H  # bán kính ảnh hưởng
K = 4
rows, wts, near = [], [], []
for vi in np.where(T[:, 2] > zchin - 0.05 * H)[0]:
    hits = kd.find_n(T[vi], K)
    d0 = hits[0][2]
    if d0 > R: continue
    w = np.array([max(0.0, 1 - h[2] / R) for h in hits]); w = w / w.sum() * (1 - d0 / R)
    rows.append(vi); near.append([h[1] for h in hits]); wts.append(w)
rows = np.array(rows); near = np.array(near); wts = np.array(wts)
print("AFFECTED verts", len(rows), "of", nt)

# --- 4. tạo shape key trên mô hình đích
tgt.shape_key_add(name="Basis", from_mix=False)
for name in arkit:
    kd_arr = np.empty(n * 3); keys[name].data.foreach_get("co", kd_arr)
    delta = ((kd_arr.reshape(-1, 3) - basis)[head_idx]) * s  # đã cùng hướng, chỉ co giãn
    out = T.copy()
    out[rows] += (delta[near] * wts[..., None]).sum(axis=1)
    sk = tgt.shape_key_add(name=name, from_mix=False)
    sk.data.foreach_set("co", out.reshape(-1))
print("SHAPEKEYS", len(tgt.data.shape_keys.key_blocks) - 1, "smile:", "mouthSmileLeft" in arkit)

# --- 5. xuất GLB (chỉ mô hình đích)
for o in list(bpy.data.objects):
    if o is not tgt: bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.object.select_all(action="DESELECT"); tgt.select_set(True)
kw = dict(filepath=DST, export_format="GLB", use_selection=True, export_morph=True, export_morph_normal=False)
try:
    bpy.ops.export_scene.gltf(export_vertex_color="ACTIVE", **kw)
except TypeError:
    bpy.ops.export_scene.gltf(**kw)
print("WROTE", DST, os.path.getsize(DST) // 1024, "KB")
