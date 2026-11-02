# Chạy: blender -b --python fit_clean_head.py -- mesh.glb input.png ra.glb [ti_le_dau=0.7] [texture=2048]
# Lưới sạch của MPFB (đầu + cổ, 52 biểu cảm ARKit) được chỉnh cho khớp hình khối của mesh TripoSR,
# rồi tô texture: mặt trước lấy từ chính ảnh Pixar (chiếu theo camera của TripoSR), phần còn lại lấy từ màu đỉnh TripoSR.
import importlib, sys, os, math
import bpy, bmesh
import numpy as np
from mathutils import kdtree, Vector, Matrix
from mathutils.bvhtree import BVHTree

argv = sys.argv[sys.argv.index("--") + 1:]
SRC, IMG, DST = argv[:3]
HEAD_FRAC = float(argv[3]) if len(argv) > 3 else 0.7
TEX = int(argv[4]) if len(argv) > 4 else 2048
HAIR_LOW = float(argv[5]) if len(argv) > 5 else 0.28  # phía sau: tóc bắt đầu từ cằm + HAIR_LOW*chiều cao đầu (âm = tóc dài xuống dưới cằm)
# vị trí mắt trong ảnh theo tỉ lệ cạnh ảnh: "uTrai,vTrai,uPhai,vPhai,banKinh"
EYES = [float(x) for x in argv[6].split(",")] if len(argv) > 6 else None


def dynamic_import(pkg, key):
    for amod in sys.modules:
        if amod.endswith(pkg):
            return getattr(importlib.import_module(amod), key)
    raise ValueError(pkg)


HumanService = dynamic_import("mpfb.services.humanservice", "HumanService")
FaceService = dynamic_import("mpfb.services.faceservice", "FaceService")
ExportService = dynamic_import("mpfb.services.exportservice", "ExportService")
ObjectService = dynamic_import("mpfb.services.objectservice", "ObjectService")
Mhclo = dynamic_import("mpfb.entities.clothes.mhclo", "Mhclo")
MPFB_DATA = os.path.join(os.environ["APPDATA"], "Blender Foundation", "Blender", "4.5", "extensions", ".user", "user_default", "mpfb", "data")
# Bộ phận rời của MakeHuman (gói CC0): nhãn cầu, răng, lưỡi, lông mi, tóc. Đặt PARTS="" để quay về cách cũ (vẽ mắt lên da, tóc vỏ).
PARTS = [p for p in os.environ.get("PARTS", "eyes/low-poly,teeth/teeth_base,tongue/tongue01,eyelashes/eyelashes01,hair/long01").split(",") if p]
EYE_MAT = os.environ.get("EYE_MAT", "blue")
HAS_EYES = any(p.startswith("eyes/") for p in PARTS)
HAS_HAIR = any(p.startswith("hair/") for p in PARTS)
HAIR_DARK = float(os.environ.get("HAIR_DARK", "0.85"))

# ---------- 1. mesh TripoSR -> khung MPFB (Z lên, mặt hướng -Y); giữ vị trí + màu đỉnh gốc ----------
before = set(bpy.data.objects)
bpy.ops.import_scene.gltf(filepath=SRC)
tri = [o for o in bpy.data.objects if o not in before and o.type == "MESH"][0]
bpy.context.view_layer.objects.active = tri
bpy.ops.object.select_all(action="DESELECT"); tri.select_set(True)
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
me = tri.data
nv = len(me.vertices)
V = np.empty(nv * 3); me.vertices.foreach_get("co", V); V = V.reshape(-1, 3)
Vf = np.stack([V[:, 2], -V[:, 0], -V[:, 1]], axis=1)  # (x,y,z)->(z,-x,-y), phép quay thuần


def vertex_colors(mesh):
    a = mesh.color_attributes[0]
    arr = np.empty(len(a.data) * 4); a.data.foreach_get("color", arr); arr = arr.reshape(-1, 4)[:, :3]
    if a.domain == "POINT":
        return arr
    vi = np.empty(len(mesh.loops), dtype=np.int32); mesh.loops.foreach_get("vertex_index", vi)
    out = np.zeros((len(mesh.vertices), 3)); cnt = np.zeros(len(mesh.vertices))
    np.add.at(out, vi, arr); np.add.at(cnt, vi, 1)
    return out / np.maximum(cnt, 1)[:, None]


Ct = vertex_colors(me)
me.calc_loop_triangles()
tt = np.empty(len(me.loop_triangles) * 3, dtype=np.int32); me.loop_triangles.foreach_get("vertices", tt); tt = tt.reshape(-1, 3)
bvh = BVHTree.FromPolygons([tuple(p) for p in Vf], [tuple(t) for t in tt], all_triangles=True)
kdt = kdtree.KDTree(nv)
for i, p in enumerate(Vf): kdt.insert(tuple(p), i)
kdt.balance()
zt_max, zt_min = Vf[:, 2].max(), Vf[:, 2].min(); H = zt_max - zt_min
zchin = zt_max - HEAD_FRAC * H
th = Vf[Vf[:, 2] > zchin]

# ---------- 2. ảnh Pixar: hiệu chỉnh phép chiếu camera TripoSR (khoảng cách 1.9, fov 40 độ) ----------
im = bpy.data.images.load(IMG); im.colorspace_settings.name = "Non-Color"
iw, ih = im.size
img = np.array(im.pixels[:], dtype=np.float32).reshape(ih, iw, 4)[::-1, :, :3]  # hàng 0 = trên cùng
FOC = (iw / 2) / math.tan(math.radians(20))


def bilinear(u, v):
    x = np.clip(u - 0.5, 0, iw - 1.001); y = np.clip(v - 0.5, 0, ih - 1.001)
    x0 = x.astype(int); y0 = y.astype(int); fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy) + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def project(P, fu, fv):
    xf, yf, zf = -P[:, 1], P[:, 0], P[:, 2]  # về khung gốc của TripoSR
    d = 1.9 - xf
    return iw / 2 + fu * FOC * yf / d, ih / 2 - fv * FOC * zf / d


front_v = np.where(Vf[:, 1] < Vf[:, 1].min() + 0.25 * (Vf[:, 1].max() - Vf[:, 1].min()))[0]
best = None
for fu in (1, -1):
    for fv in (1, -1):
        u, v = project(Vf[front_v], fu, fv)
        err = np.abs(bilinear(u, v) - Ct[front_v]).mean()
        print("CALIB flip_u", fu, "flip_v", fv, "err", round(float(err), 4))
        if best is None or err < best[0]: best = (err, fu, fv)
_, FU, FV = best
print("CALIB chosen", FU, FV, "err", round(float(best[0]), 4))

# ---------- 3. đầu + cổ MPFB sạch, có 52 biểu cảm ----------
human = HumanService.create_human()
FaceService.load_targets(human, load_microsoft_visemes=False, load_arkit_faceunits=True)
Mh = np.array(human.matrix_world)
_hk = human.data.shape_keys.key_blocks
def get_full(kb):  # lưới người đầy đủ (kể cả phần phụ trợ răng/lưỡi mà bộ phận rời bám vào), toạ độ thế giới
    a = np.empty(len(human.data.vertices) * 3); kb.data.foreach_get("co", a)
    return a.reshape(-1, 3) @ Mh[:3, :3].T + Mh[:3, 3]
root = ExportService.create_character_copy(human, name_suffix="_src")
src = ObjectService.find_object_of_type_amongst_nearest_relatives(root, "Basemesh")
ExportService.bake_modifiers_remove_helpers(src, bake_masks=True, bake_subdiv=False, remove_helpers=True, also_proxy=True)
M = np.array(src.matrix_world)
me2 = src.data.copy()  # copy để mang theo shape key (mesh mới tạo sẽ mất chúng)
bm = bmesh.new(); bm.from_mesh(me2)
zs = np.array([(M @ np.append(np.array(v.co), 1.0))[2] for v in bm.verts])
zmax_m = zs.max()
bmesh.ops.bisect_plane(bm, geom=list(bm.verts) + list(bm.edges) + list(bm.faces), plane_co=(0, 0, 0.80 * zmax_m),
                       plane_no=(0, 0, 1), clear_inner=True)  # cắt bằng mặt phẳng -> mép đáy thẳng, không răng cưa
bm.to_mesh(me2); bm.free()
n = len(me2.vertices)
keys = me2.shape_keys.key_blocks
fu_dir = os.path.join(os.environ["APPDATA"], "Blender Foundation", "Blender", "4.5", "extensions", ".user", "user_default", "mpfb", "data", "targets", "faceunits")
arkit = sorted(f[:-7] for f in os.listdir(fu_dir) if f.endswith(".target"))  # đúng 52 biểu cảm ARKit
arkit = [k for k in arkit if k in keys]
obj = bpy.data.objects.new("Head", me2); bpy.context.collection.objects.link(obj)
for kb in list(keys):  # bỏ các shape key khác (macro giới tính/chủng tộc...) để không bị áp sẵn vào mô hình
    if kb.name != "Basis" and kb.name not in arkit: obj.shape_key_remove(kb)
keys = me2.shape_keys.key_blocks
SUBDIV = int(os.environ.get("SUBDIV", "1"))  # chia nhỏ lưới đầu: mỗi lần x4 số đỉnh (shape key và UV được nội suy theo)
if SUBDIV:
    bm = bmesh.new(); bm.from_mesh(me2)
    bmesh.ops.triangulate(bm, faces=[f for f in bm.faces if len(f.verts) > 4])  # mặt nhiều cạnh -> tam giác, tránh bị chia sai khi lưới bị kéo
    bmesh.ops.subdivide_edges(bm, edges=list(bm.edges), cuts=SUBDIV, use_grid_fill=True)
    bm.to_mesh(me2); bm.free(); me2.update()
    n = len(me2.vertices); keys = me2.shape_keys.key_blocks
    print("SUBDIV verts", n, "keys", len(keys))
print("HEAD verts", n, "keys", len(arkit))


def get(kb):
    a = np.empty(n * 3); kb.data.foreach_get("co", a); a = a.reshape(-1, 3)
    return a @ M[:3, :3].T + M[:3, 3]  # toạ độ thế giới


base = get(keys["Basis"])
shapes = {k: get(keys[k]) for k in arkit}
# Gộp sẵn một phần biểu cảm vào tư thế nghỉ (mặc định: mắt mở to kiểu Pixar). Cộng cùng độ dịch vào mọi biểu cảm để chúng vẫn đúng.
BAKE = {kv.split(":")[0]: float(kv.split(":")[1]) for kv in os.environ.get("BAKE", "eyeWideLeft:0.6,eyeWideRight:0.6,mouthPressLeft:0.5,mouthPressRight:0.5").split(",") if kv}
base0 = base  # bản chưa gộp: để dò đỉnh tương ứng với lưới người đầy đủ
if BAKE:
    _bd = sum(v * (shapes[k] - base) for k, v in BAKE.items() if k in shapes)
    # chớp mắt giữ nguyên dáng nhắm gốc (không cộng phần mở to) -> mi đi từ mắt mở to tới nhắm kín
    nobake = lambda k: k.startswith("eyeBlink")
    base = base + _bd; shapes = {k: a + (0 if nobake(k) else _bd) for k, a in shapes.items()}
hp = base[base[:, 2] > 0.865 * zmax_m]
# Tự tìm tỉ lệ đầu: thử nhiều giá trị, chọn giá trị làm mặt MPFB (phần trước) sát bề mặt TripoSR nhất
hf = hp[hp[:, 1] < hp[:, 1].min() + 0.35 * (hp[:, 1].max() - hp[:, 1].min())]
hf = hf[:: max(1, len(hf) // 400)]
best_fr = None
for fr in np.arange(0.45, 0.90, 0.02):
    zc = zt_max - fr * H; thc = Vf[Vf[:, 2] > zc]
    sc = (zt_max - zc) / (hp[:, 2].max() - hp[:, 2].min())
    cc1 = np.array([hp[:, 0].mean(), hp[:, 1].min(), hp[:, 2].max()])
    cc2 = np.array([(thc[:, 0].min() + thc[:, 0].max()) / 2, thc[:, 1].min(), zt_max])
    pts = (hf - cc1) * sc + cc2
    dist = np.mean([bvh.find_nearest(Vector(p))[3] or 9 for p in pts])
    if best_fr is None or dist < best_fr[0]: best_fr = (dist, fr)
print("AUTO head_frac", round(float(best_fr[1]), 2), "(tay:", HEAD_FRAC, ") dist", round(float(best_fr[0]), 4))
HEAD_FRAC = float(best_fr[1]); zchin = zt_max - HEAD_FRAC * H; th = Vf[Vf[:, 2] > zchin]
s = (zt_max - zchin) / (hp[:, 2].max() - hp[:, 2].min())
c1 = np.array([hp[:, 0].mean(), hp[:, 1].min(), hp[:, 2].max()])
c2 = np.array([(th[:, 0].min() + th[:, 0].max()) / 2, th[:, 1].min(), zt_max])
al = lambda a: (a - c1) * s + c2
base_a = al(base)


def measure_width(P):
    """Bề rộng mặt: đo da ở ngang má trong ảnh (quét từ mép ảnh vào) so với mặt P chiếu lên ảnh ở cùng hàng -> hệ số giãn ngang."""
    sx = 1.0
    uL, vL, uR, vR = EYES[:4]
    row = int(((vL + vR) / 2 + 0.75 * (uR - uL)) * ih)  # hàng giữa mũi và miệng: dưới gọng kính
    ref = np.median(np.concatenate([img[row - 3:row + 3, int(u * iw) - 3:int(u * iw) + 3].reshape(-1, 3) for u in (uL, uR)]), axis=0)
    def edge(x, step):  # đi từ mép ảnh vào trong tới pixel da đầu tiên (đi từ trong ra thì má hồng làm dừng sớm)
        # da = gần màu má (mép mặt nhạt hơn má hồng nên ngưỡng rộng) và có sắc màu (nền xám thì không)
        while 0 < x < iw - 1 and not (np.abs(img[row, x] - ref).sum() < 0.7 and np.ptp(img[row, x]) > 0.12): x += step
        return x
    w_img_px = edge(iw - 2, -1) - edge(1, 1)
    pu, pv = project(P, FU, FV)
    sel = (np.abs(pv - row) < 0.01 * ih) & (P[:, 1] < np.median(P[:, 1]))
    if sel.sum() > 4:
        sx = float(np.clip(w_img_px / max(pu[sel].max() - pu[sel].min(), 1), 0.9, 1.4))
    print("FACE_WIDTH image px", w_img_px, "model px", round(float(pu[sel].max() - pu[sel].min()), 1) if sel.sum() > 4 else None, "sx", round(sx, 3))
    return sx
print("ALIGN scale", round(float(s), 3), "H", round(float(H), 3))
print("DEBUG M", np.round(M, 3).tolist())
print("DEBUG base_a min", np.round(base_a.min(0), 3), "max", np.round(base_a.max(0), 3))
print("DEBUG tri  min", np.round(Vf.min(0), 3), "max", np.round(Vf.max(0), 3))

# ---------- 4. kéo lưới MPFB về bề mặt TripoSR (mượt hoá để không nhận độ gồ ghề) ----------
d = np.zeros_like(base_a)
for i, p in enumerate(base_a):
    loc, _, _, dist = bvh.find_nearest(Vector(p))
    if loc is not None and dist < 0.12 * H: d[i] = np.array(loc) - p
w = np.clip((base_a[:, 2] - (zchin - 0.10 * H)) / (0.10 * H), 0, 1)
ed = np.empty(len(me2.edges) * 2, dtype=np.int32); me2.edges.foreach_get("vertices", ed); ed = ed.reshape(-1, 2)
deg = np.zeros(n); np.add.at(deg, ed[:, 0], 1); np.add.at(deg, ed[:, 1], 1)
for _ in range(25 * 4 ** SUBDIV):  # lưới dày gấp đôi mỗi chiều -> cần gấp 4 lần số lần để bán kính làm mượt không đổi
    acc = d.copy(); np.add.at(acc, ed[:, 0], d[ed[:, 1]]); np.add.at(acc, ed[:, 1], d[ed[:, 0]])
    d = acc / (deg + 1)[:, None]
shift = 0.9 * w[:, None] * d
# Vùng mặt trước: TripoSR dựng cả gọng kính/bóng đổ thành gờ nổi -> da bị nếp. Làm mượt thêm chỉ ở mặt trước (mép giữ nguyên).
_hd = base_a[base_a[:, 2] > zchin, 1]  # chiều sâu tính riêng phần đầu (không tính vai) để vùng mặt trước phủ tới má
_front = (base_a[:, 1] < _hd.min() + 0.5 * (_hd.max() - _hd.min())) & (base_a[:, 2] > zchin)
for _ in range(int(os.environ.get("FACE_SMOOTH", "60")) * 4 ** SUBDIV):
    acc = shift.copy(); np.add.at(acc, ed[:, 0], shift[ed[:, 1]]); np.add.at(acc, ed[:, 1], shift[ed[:, 0]])
    shift[_front] = (acc / (deg + 1)[:, None])[_front]
base_f = base_a + shift
print("FIT mean shift", round(float(np.abs(shift).mean()), 4))

# ---------- 4b. mắt lồi thành vòm cầu: bắn tia từ camera TripoSR qua vị trí mắt trong ảnh để tìm điểm 3D ----------
eye_c = []
if EYES and not HAS_EYES:  # có nhãn cầu thật thì không cần vòm mắt giả
    polys = [tuple(p.vertices) for p in me2.polygons]
    hbvh = BVHTree.FromPolygons([tuple(p) for p in base_f], polys)
    for uu, vv in ((EYES[0], EYES[1]), (EYES[2], EYES[3])):
        d_ = Vector(((uu * iw - iw / 2) / (FU * FOC), 1.0, (ih / 2 - vv * ih) / (FV * FOC)))
        loc, _, _, _ = hbvh.ray_cast(Vector((0, -1.9, 0)), d_)
        if loc is None: print("EYE ray miss"); continue
        eye_c.append((np.array(loc), EYES[4] * iw * (1.9 + loc.y) / FOC))
    bump = np.zeros_like(base_f)
    for c, r3 in eye_c:
        dd2 = (base_f[:, 0] - c[0]) ** 2 + (base_f[:, 2] - c[2]) ** 2
        bump[:, 1] -= 0.55 * r3 * np.exp(-dd2 / (0.8 * r3) ** 2) * (base_f[:, 1] < c[1] + 0.35 * r3)  # chỉ đẩy lớp da ngoài cùng ra phía trước (-Y), không đẩy mặt trong hốc mắt
    shift = shift + bump; base_f = base_a + shift
    print("EYES", [(np.round(c, 3).tolist(), round(float(r), 3)) for c, r in eye_c])


def put(kb, arr):
    kb.data.foreach_set("co", arr.reshape(-1))


# Mặt Pixar tròn hơn: giãn ngang vùng đầu (đo sau khi đã khớp TripoSR), giảm dần về cổ để vai không rộng ra
SX = measure_width(base_f) if EYES and os.environ.get("FACE_WIDTH", "1") == "1" else 1.0
# Miệng: nếu miệng model lệch dọc so với miệng trong ảnh thì ảnh môi bị chiếu lên cằm (thành "môi thứ hai").
# Dò hàng miệng trong ảnh (hàng giữa mặt có màu môi đỏ nhất) và miệng model (đỉnh di chuyển nhiều nhất khi cười),
# rồi kéo dọc vùng mắt->cằm: 0 ở ngang mắt, đủ độ dịch từ miệng trở xuống, giảm về 0 dưới cằm (cằm dài/ngắn theo).
DZ, Z_EYE, Z_MOUTH = 0.0, 0.0, 0.0
if EYES and os.environ.get("MOUTH_ALIGN", "1") == "1":
    uL, vL, uR, vR = EYES[:4]; uc = (uL + uR) / 2; du = uR - uL
    band = img[:, int((uc - 0.25 * du) * iw):int((uc + 0.25 * du) * iw)]
    red = (band[..., 0] - band[..., 1]).mean(1)
    r0, r1 = int((vL + 0.35 * du) * ih), int((vL + 1.6 * du) * ih)
    v_img = r0 + int(np.argmax(red[r0:r1]))
    top = lambda names, m: np.argsort(sum(np.linalg.norm(shapes[k] - base, axis=1) for k in names))[-m:]
    pm = base_f[top(("mouthSmileLeft", "mouthSmileRight"), 30)].mean(0, keepdims=True)
    Z_EYE = float(base_f[top(("eyeBlinkLeft", "eyeBlinkRight"), 30), 2].mean()); Z_MOUTH = float(pm[0, 2])
    v_mod = float(project(pm, FU, FV)[1][0])
    DZ = float(np.clip(-(v_img - v_mod) * (1.9 + pm[0, 1]) / (FV * FOC), -0.05 * H, 0.05 * H))
    print("MOUTH image row", v_img, "model row", round(v_mod, 1), "dz", round(DZ, 4))
def widen(P):
    t = np.clip((P[:, 2] - (zchin - 0.10 * H)) / (0.15 * H), 0, 1)
    out = P.copy(); out[:, 0] = c2[0] + (P[:, 0] - c2[0]) * (1 + (SX - 1) * t)
    if DZ:
        tv = np.clip((Z_EYE - P[:, 2]) / max(Z_EYE - Z_MOUTH, 1e-6), 0, 1) * np.clip((P[:, 2] - (zchin - 0.10 * H)) / (0.10 * H), 0, 1)
        out[:, 2] += DZ * tv
    return out
base_f = widen(base_f)
put(keys["Basis"], base_f)
for k in arkit: put(keys[k], widen(al(shapes[k]) + shift))
me2.vertices.foreach_set("co", base_f.reshape(-1)); me2.update()

# ---------- 4c. bộ phận rời: đặt theo neo 3 đỉnh của MHCLO trên lưới người đã khớp, mỗi biểu cảm một shape key ----------
parts = []
if PARTS:
    hkd = kdtree.KDTree(n)
    for i, p in enumerate(base0): hkd.insert(tuple(p), i)
    hkd.balance()
    Hb = get_full(_hk["Basis"])
    near = np.array([hkd.find(tuple(p))[1] for p in Hb])  # đỉnh đầu gần nhất -> mượn độ dịch khớp mặt
    _bdf = sum(v * (get_full(_hk[k]) - Hb) for k, v in BAKE.items() if k in _hk) if BAKE else 0  # cùng phần gộp như lưới đầu
    Wb = widen(al(Hb + _bdf) + shift[near])
    Wk = {k: widen(al(get_full(_hk[k]) + (0 if BAKE and nobake(k) else _bdf)) + shift[near]) for k in arkit if k in _hk}
    _hh = zt_max - zchin  # màu tóc: chóp đầu phía trước trong ảnh Pixar (giống cách tính hair_c ở bước 6)
    _tvh = (Vf[:, 2] > zt_max - 0.14 * _hh) & (Vf[:, 2] < zt_max - 0.04 * _hh) & (Vf[:, 1] < np.median(Vf[:, 1]))
    hair_tint = np.median(bilinear(*project(Vf[_tvh], FU, FV)), axis=0) if _tvh.sum() > 20 else np.array((0.3, 0.2, 0.1))
    def binding(part):  # công thức neo của MHCLO (giống ClothesService.fit_clothes_to_human), chạy được cho mọi tư thế/biểu cảm
        f = os.path.join(MPFB_DATA, part, os.path.basename(part) + ".mhclo")
        mc = Mhclo(); mc.load(f)
        nvp = len(mc.verts)
        idx = np.array([mc.verts[i]["verts"] for i in range(nvp)]); wt = np.array([mc.verts[i]["weights"] for i in range(nvp)])
        of = np.array([mc.verts[i]["offsets"] for i in range(nvp)])
        def place(W):
            xs = abs(W[mc.x_scale[0], 0] - W[mc.x_scale[1], 0]) / mc.x_scale[2]
            ys = abs(W[mc.y_scale[0], 2] - W[mc.y_scale[1], 2]) / mc.y_scale[2]
            zs = abs(W[mc.z_scale[0], 1] - W[mc.z_scale[1], 1]) / mc.z_scale[2]
            return (wt[..., None] * W[idx]).sum(1) + of * np.array([xs, zs, ys])
        return f, nvp, place

    # Mắt to kiểu Pixar: phóng to mọi thứ quanh tâm nhãn cầu (da mi, nhãn cầu, lông mi), giảm dần ra ngoài.
    # Áp cho cả lưới đầu lẫn lưới người đầy đủ (bộ phận neo vào đó) và mọi biểu cảm -> chớp mắt vẫn khép kín.
    EYE_GROW = float(os.environ.get("EYE_GROW", "1.6"))
    if HAS_EYES and EYE_GROW != 1:
        E = binding(next(p for p in PARTS if p.startswith("eyes/")))[2](Wb)
        sides_ = [E[E[:, 0] < c2[0]], E[E[:, 0] >= c2[0]]]
        cs = [q.mean(0) for q in sides_]; re = float(np.mean([np.ptp(q[:, 0]) / 2 for q in sides_]))
        g = np.array([max(1.0, EYE_GROW / SX), 1 + 0.5 * (EYE_GROW - 1), EYE_GROW])  # chiều sâu (Y) chỉ phóng nửa: phóng đủ thì mi dưới phồng thành túi tối  # chiều ngang đã được giãn mặt lo -> chủ yếu phóng to dọc (mắt tròn)
        def grow(P):
            out = P.copy()
            for c in cs:
                d = np.linalg.norm(P - c, axis=1)
                # phóng đều trong 1.2 bán kính nhãn cầu (nhãn cầu không méo), ngoài đó giảm êm theo hình chuông -> không có mép gấp trên má
                r0_ = re / SX
                t = np.where(d < 1.2 * r0_, 1.0, np.exp(-((d - 1.2 * r0_) / (1.5 * r0_)) ** 2))
                out += (P - c) * (g - 1) * t[:, None]
            return out
        base_f = grow(base_f)
        put(keys["Basis"], base_f)
        for k in arkit: put(keys[k], grow(widen(al(shapes[k]) + shift)))
        me2.vertices.foreach_set("co", base_f.reshape(-1)); me2.update()
        Wb = grow(Wb); Wk = {k: grow(W) for k, W in Wk.items()}
        print("EYE_GROW", EYE_GROW, "eye radius", round(re, 4), "centers", [np.round(c, 3).tolist() for c in cs])

    for part in PARTS:
        f, nvp, place = binding(part)
        po = HumanService.add_mhclo_asset(f, human, asset_type=part.split("/")[0].capitalize(), subdiv_levels=0,
                                          set_up_rigging=False, interpolate_weights=False, import_subrig=False, import_weights=False)
        po.parent = None; po.matrix_world = Matrix.Identity(4)
        for md in list(po.modifiers): po.modifiers.remove(md)
        if po.data.shape_keys: po.shape_key_clear()
        assert len(po.data.vertices) == nvp, (part, len(po.data.vertices), nvp)
        # răng/lưỡi mượn độ dịch của môi nên bị kéo ra trước -> lùi vào trong (+Y) để không lộ qua khe môi khi ngậm
        # và thu nhỏ quanh tâm (của tư thế nghỉ) vì cằm đã khớp ngắn hơn cằm MPFB -> lợi/lưỡi không đâm xuyên da khi há miệng
        if part.startswith(("teeth/", "tongue/")):
            _place = place; _c = _place(Wb).mean(0)
            _back = np.array([0, float(os.environ.get("TEETH_BACK", "0.06")), 0]); _sc = float(os.environ.get("TEETH_SCALE", "0.85"))
            place = lambda W: _c + (_place(W) - _c) * _sc + _back
        if part.startswith("eyes/") and "cs" in dir():  # thu nhãn cầu một chút quanh tâm mỗi mắt -> không đâm xuyên mi dưới
            _ep = place; _es = float(os.environ.get("EYEBALL_SCALE", "0.96"))
            def place(W, _ep=_ep):
                Pp = _ep(W); out = Pp.copy()
                for c in cs:
                    m = np.linalg.norm(Pp - c, axis=1) < 3 * re
                    out[m] = c + (Pp[m] - c) * _es
                return out
        if part.startswith("hair/"):  # tóc Pixar dày, phồng: đẩy tóc ra xa tâm đầu
            _hp = place; _hc = np.array([c2[0], base_f[:, 1].mean(), zt_max - 0.45 * (zt_max - zchin)])
            _vol = float(os.environ.get("HAIR_VOLUME", "1.08"))
            place = lambda W: _hc + (_hp(W) - _hc) * _vol
        P0 = place(Wb); po.data.vertices.foreach_set("co", P0.reshape(-1)); po.data.update()
        po.shape_key_add(name="Basis")
        nk = 0
        for k, W in Wk.items():
            Pk = place(W)
            if np.abs(Pk - P0).max() > 1e-4 * H:  # chỉ giữ biểu cảm làm bộ phận này di chuyển (răng: hàm; mi: chớp)
                po.shape_key_add(name=k).data.foreach_set("co", Pk.reshape(-1)); nk += 1
        if part.startswith("eyes/"):  # màu mống mắt
            im_e = bpy.data.images.load(os.path.join(MPFB_DATA, "eyes", "materials", EYE_MAT + "_eye.png"))
            m_e = bpy.data.materials.new("eye"); m_e.use_nodes = True
            t_e = m_e.node_tree.nodes.new("ShaderNodeTexImage"); t_e.image = im_e
            b_e = m_e.node_tree.nodes["Principled BSDF"]; b_e.inputs["Roughness"].default_value = 0.1
            m_e.node_tree.links.new(t_e.outputs["Color"], b_e.inputs["Base Color"])
            po.data.materials.clear(); po.data.materials.append(m_e)
        if part.startswith("hair/"):
            # tượng bán thân bị cắt ở vai -> bỏ phần tóc rủ thấp hơn mép đáy
            zcut = base_f[:, 2].min() + 0.02 * H
            bm = bmesh.new(); bm.from_mesh(po.data)
            bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < zcut], context="VERTS")
            bm.to_mesh(po.data); bm.free()
            # nhuộm: giữ sáng/tối + alpha của texture gốc, thay màu bằng màu tóc lấy từ ảnh Pixar
            for nd in po.data.materials[0].node_tree.nodes:
                if nd.type == "TEX_IMAGE" and nd.image and "normal" not in nd.image.name.lower():
                    im_h = nd.image; w_, h_ = im_h.size
                    px = np.array(im_h.pixels[:], np.float32).reshape(-1, 4)
                    lum = px[:, :3] @ np.array([0.3, 0.59, 0.11], np.float32)
                    lum = lum / max(float(np.median(lum[px[:, 3] > 0.5])) if (px[:, 3] > 0.5).any() else 1.0, 1e-3)
                    lum = np.minimum(lum, 1.3)  # đường ngôi trong texture gốc rất sáng -> thành vệt trắng; chặn lại
                    px[:, :3] = np.clip(HAIR_DARK * hair_tint[None] * lum[:, None], 0, 1)  # tối hơn chút: đèn trong cảnh làm tóc sáng hơn ảnh
                    new = bpy.data.images.new("hair_tinted", w_, h_, alpha=True); new.pixels.foreach_set(px.reshape(-1))
                    new.filepath_raw = os.path.join(os.path.dirname(os.path.abspath(DST)), "hair_tinted.png"); new.file_format = "PNG"; new.save()
                    nd.image = bpy.data.images.load(new.filepath_raw)
                    break
        po.name = os.path.basename(part)
        parts.append(po)
        print("PART", part, "verts", nvp, "keys", nk, "mats", [m.name for m in po.data.materials])

# ---------- 5. vá lỗ nhỏ (mắt) và bật smooth ----------
bm = bmesh.new(); bm.from_mesh(me2)
uvl = bm.loops.layers.uv.active
vuv = {}
for f in bm.faces:
    for l in f.loops: vuv.setdefault(l.vert.index, tuple(l[uvl].uv))
res = bmesh.ops.holes_fill(bm, edges=[e for e in bm.edges if e.is_boundary], sides=3 if HAS_EYES else 24 * 2 ** SUBDIV)  # có nhãn cầu: hốc mắt phải để hở
for f in res["faces"]:
    for l in f.loops: l[uvl].uv = vuv.get(l.vert.index, (0, 0))
for f in bm.faces: f.smooth = True
print("HOLES filled", len(res["faces"]), "boundary edges left", sum(e.is_boundary for e in bm.edges), "faces", len(bm.faces),
      "tris/ngons", sum(len(f.verts) == 3 for f in bm.faces), sum(len(f.verts) > 4 for f in bm.faces))
bm.to_mesh(me2); bm.free()
me2.calc_loop_triangles()

# ---------- 6. nướng texture trong không gian UV ----------
nl = len(me2.loops)
uv = np.empty(nl * 2); me2.uv_layers.active.data.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
nrm = np.empty(n * 3); me2.vertex_normals.foreach_get("vector", nrm); nrm = nrm.reshape(-1, 3)
mt = len(me2.loop_triangles)
tl = np.empty(mt * 3, dtype=np.int32); me2.loop_triangles.foreach_get("loops", tl); tl = tl.reshape(-1, 3)
tv = np.empty(mt * 3, dtype=np.int32); me2.loop_triangles.foreach_get("vertices", tv); tv = tv.reshape(-1, 3)
pos_t = np.zeros((TEX, TEX, 3), np.float32); nrm_t = np.zeros((TEX, TEX, 3), np.float32); occ = np.zeros((TEX, TEX), bool)
for t in range(mt):
    q = uv[tl[t]] * TEX
    if np.abs(q).max() < 1e-6: continue
    x0, y0 = np.floor(q.min(0)).astype(int); x1, y1 = np.ceil(q.max(0)).astype(int)
    x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1 + 1, TEX), min(y1 + 1, TEX)
    if x1 <= x0 or y1 <= y0: continue
    gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    a, b, c = q
    den = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
    if abs(den) < 1e-9: continue
    w0 = ((b[1] - c[1]) * (gx - c[0]) + (c[0] - b[0]) * (gy - c[1])) / den
    w1 = ((c[1] - a[1]) * (gx - c[0]) + (a[0] - c[0]) * (gy - c[1])) / den
    w2 = 1 - w0 - w1
    m = (w0 >= -0.02) & (w1 >= -0.02) & (w2 >= -0.02)
    if not m.any(): continue
    ys, xs = np.nonzero(m); ys0, xs0 = ys + y0, xs + x0
    ww = np.stack([w0[m], w1[m], w2[m]], 1)
    pos_t[ys0, xs0] = ww @ base_f[tv[t]]; nrm_t[ys0, xs0] = ww @ nrm[tv[t]]; occ[ys0, xs0] = True
ys, xs = np.nonzero(occ)
P = pos_t[ys, xs].astype(np.float64); N = nrm_t[ys, xs].astype(np.float64)
N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-9)
print("TEXELS", len(ys))

u, v = project(P, FU, FV)
col_img = bilinear(u, v)
w_img = np.clip((-N[:, 1] - 0.25) / 0.45, 0, 1)  # mặt hướng về phía trước thì tin ảnh Pixar
# Phía sau (không có trong ảnh): TripoSR chỉ đoán nên lốm đốm -> tô bằng màu tóc/da/áo ước lượng từ phần nhìn thấy ở ảnh.
headH = zt_max - zchin; zr = P[:, 2]
def med(mask, fallback):
    return np.median(col_img[mask], axis=0) if mask.sum() > 50 else np.array(fallback)
_tv = (Vf[:, 2] > zt_max - 0.14 * headH) & (Vf[:, 2] < zt_max - 0.04 * headH) & (Vf[:, 1] < np.median(Vf[:, 1]))  # chóp đầu phía trước: chắc chắn là tóc
if _tv.sum() > 20:  # lấy màu từ chính ảnh Pixar (màu đỉnh TripoSR ở đây có thể bị xám)
    _hu, _hv_ = project(Vf[_tv], FU, FV); hair_c = np.median(bilinear(_hu, _hv_), axis=0)
else:
    hair_c = np.array((0.3, 0.2, 0.1))
skin_c = med((w_img > 0.8) & (zr > zchin + 0.25 * headH) & (zr < zchin + 0.45 * headH), (0.8, 0.6, 0.5))
coat_c = med((w_img > 0.8) & (zr < zchin - 0.03 * H), (0.1, 0.1, 0.12))
print("COLORS hair", np.round(hair_c, 2), "skin", np.round(skin_c, 2), "coat", np.round(coat_c, 2))
w_under = np.clip((-N[:, 2] - 0.3) / 0.4, 0, 1)  # mặt dưới cằm/cổ: màu da, không phải màu tóc
w_back = np.clip((N[:, 1] - 0.05) / 0.45, 0, 1) * (1 - w_under)
w_side = np.clip(1 - w_img - w_back - w_under, 0, 1)
# Hốc mắt: mặt trong nhãn cầu/hốc quay ra sau nhưng nằm ngay sau bề mặt mắt -> vẫn lấy màu từ ảnh (tròng trắng, mống mắt) thay vì màu tóc
ez = np.zeros(len(P), bool)
for c, r3 in eye_c:
    ez |= (np.sqrt((P[:, 0] - c[0]) ** 2 + (P[:, 2] - c[2]) ** 2) < 1.4 * r3) & (P[:, 1] < c[1] + 2.0 * r3)
_hz = base_f[base_f[:, 2] > zchin]; _yf = _hz[:, 1].min(); _yd = _hz[:, 1].max() - _yf
_xc = _hz[:, 0].mean(); _hw = (_hz[:, 0].max() - _hz[:, 0].min()) / 2
# Cả vùng mặt phía trước (mắt, mũi, má, miệng, cằm) lấy màu từ ảnh Pixar, không pha màu TripoSR -> hết vệt chuyển màu hai bên sống mũi
face_zone = (P[:, 1] < _yf + float(os.environ.get("FACE_DEPTH", "0.6")) * _yd)  # 0.6: mặt đã giãn ngang nên má hai bên nằm sâu hơn & (zr > zchin - 0.05 * H) & (np.abs(P[:, 0] - _xc) < 0.8 * _hw) & (N[:, 1] < 0.6) & (N[:, 2] < 0.5)  # không gồm da đầu ngửa lên (chiếu vào ảnh sẽ ra màu trán sáng)  # mặt dưới mũi/cằm có pháp tuyến hơi quay ra sau nên cần nới, nếu không sẽ lấy nhầm màu áo
ez = ez | face_zone
w_img = np.where(ez, 1.0, w_img); w_back = np.where(ez, 0.0, w_back); w_under = np.where(ez, 0.0, w_under); w_side = np.where(ez, 0.0, w_side)
tot = np.maximum(w_img + w_back + w_side + w_under, 1e-6)
z_hair = zchin + HAIR_LOW * headH
shade = 0.8 + 0.2 * np.clip((zr - z_hair) / max(zt_max - z_hair, 1e-6), 0, 1)
back_c = np.where((zr > z_hair)[:, None], hair_c[None] * shade[:, None], np.where((zr < zchin - 0.02 * H)[:, None], coat_c[None], skin_c[None]))
fall = np.zeros_like(P)
for i in np.where(w_side > 0.001)[0]:
    fall[i] = Ct[kdt.find(tuple(P[i]))[1]]
col = (col_img * w_img[:, None] + back_c * w_back[:, None] + skin_c[None] * w_under[:, None] + fall * w_side[:, None]) / tot[:, None]
# Vùng miệng/cằm phía trước nhưng quay ra sau (mặt trong môi, khoang miệng) -> màu da, hết vệt tối dưới môi
_hz = base_f[base_f[:, 2] > zchin]; _yf = _hz[:, 1].min(); _yd = _hz[:, 1].max() - _yf
_xc = _hz[:, 0].mean(); _hw = (_hz[:, 0].max() - _hz[:, 0].min()) / 2
mouth_zone = (P[:, 1] < _yf + 0.40 * _yd) & (zr > zchin - 0.05 * H) & (zr < zchin + 0.42 * headH) & (np.abs(P[:, 0] - _xc) < 0.6 * _hw) & (N[:, 1] >= 0.0) & ~ez
# Mặt trong môi/khoang miệng nằm sau mặt trước: dùng màu da. Nhưng nếp dưới môi sau khi khớp có thể lộ ra trước ->
# bắn tia từ camera: điểm nào camera nhìn thấy thì vẫn lấy màu ảnh (tránh mảng tối viền sắc dưới môi).
_mb = BVHTree.FromPolygons([tuple(p) for p in base_f], [tuple(p.vertices) for p in me2.polygons])
_cam = Vector((0, -1.9, 0))
for i in np.where(mouth_zone)[0]:
    p = Vector(P[i]); dv = p - _cam
    hit = _mb.ray_cast(_cam, dv.normalized(), dv.length + 1e-3)[0]
    col[i] = col_img[i] if hit is not None and (hit - p).length < 2e-3 * H else skin_c * 0.92
# Có nhãn cầu thật: mống/tròng mắt vẽ sẵn trong ảnh không được in lên da mi quanh nhãn cầu -> tô màu da (elip theo cỡ mắt đã phóng)
if HAS_EYES and "cs" in globals():
    r0 = re / SX  # bán kính nhãn cầu thật (re đo sau khi giãn ngang)
    for c in cs:
        q = P - c
        e = np.sqrt((q[:, 0] / (r0 * SX * g[0])) ** 2 + (q[:, 2] / (r0 * g[2])) ** 2)  # 1 = mép nhãn cầu
        wgt = np.clip((1.35 - e) / 0.25, 0, 1) * (q[:, 1] < 2 * r0)  # 2*r0: gồm cả mép trong mi dưới (sâu hơn bề mặt)  # mống mắt vẽ trong ảnh to hơn nhãn cầu -> phủ rộng hơn mép nhãn cầu  # mép mờ dần, không thành vòng da phẳng
        ring = (e > 1.45) & (e < 1.8) & (q[:, 1] < r0)  # màu da tại chỗ (trong tròng kính da hồng sáng hơn màu da trung bình)
        lid_c = np.median(col[ring], axis=0) if ring.sum() > 50 else skin_c
        col = col * (1 - wgt[:, None]) + lid_c[None] * wgt[:, None]
tex = np.zeros((TEX, TEX, 4), np.float32); tex[..., 3] = 1
tex[ys, xs, :3] = col
filled = occ.copy()
for _ in range(10):  # lan màu ra ngoài mép đảo UV để không lộ viền khi lọc texture
    acc = np.zeros((TEX, TEX, 3), np.float32); cnt = np.zeros((TEX, TEX), np.float32)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        sh = np.roll(tex[..., :3] * filled[..., None], (dy, dx), (0, 1)); sf = np.roll(filled, (dy, dx), (0, 1))
        acc += sh; cnt += sf
    grow = (~filled) & (cnt > 0)
    tex[..., :3][grow] = acc[grow] / cnt[grow][:, None]; filled |= grow
# bản đồ độ nhám: vùng mắt bóng (0.08) để có điểm sáng phản chiếu, da mờ hơn (0.55)
rough = np.full(len(P), 0.55)
for c, r3 in eye_c:
    dd = np.sqrt((P[:, 0] - c[0]) ** 2 + (P[:, 2] - c[2]) ** 2)
    mk = np.clip((1.25 * r3 - dd) / (0.25 * r3), 0, 1) * (P[:, 1] < c[1] + 1.5 * r3)
    rough = rough * (1 - mk) + 0.2 * mk
rt = np.zeros((TEX, TEX, 4), np.float32); rt[..., 0] = 1; rt[..., 1] = 0.55; rt[..., 3] = 1; rt[ys, xs, 1] = rough
rpath = os.path.join(os.path.dirname(os.path.abspath(DST)), "head_rough.jpg")
rimg = bpy.data.images.new("head_rough", TEX, TEX, alpha=False); rimg.colorspace_settings.name = "Non-Color"
rimg.pixels.foreach_set(np.ascontiguousarray(rt).reshape(-1)); rimg.filepath_raw = rpath; rimg.file_format = "JPEG"; rimg.save()
tex_path = os.path.join(os.path.dirname(os.path.abspath(DST)), "head_tex.jpg")
timg = bpy.data.images.new("head_tex", TEX, TEX, alpha=False); timg.colorspace_settings.name = "Non-Color"
timg.pixels.foreach_set(np.ascontiguousarray(tex).reshape(-1)); timg.filepath_raw = tex_path; timg.file_format = "JPEG"; timg.save()
timg2 = bpy.data.images.load(tex_path)

# ---------- 6b. tóc: vỏ tóc dựng từ chính lưới đầu (đẩy ra theo pháp tuyến), có sợi dọc ----------
_hv = base_f[base_f[:, 2] > zchin]  # độ rộng tính riêng cho phần đầu (không tính vai) để "hai bên thái dương" đúng
xc, yc = _hv[:, 0].mean(), base_f[:, 1].mean()
halfW = (_hv[:, 0].max() - _hv[:, 0].min()) / 2
zb = base_f[:, 2]; side = np.abs(base_f[:, 0] - xc) / halfW
front = nrm[:, 1] < -0.5; back = nrm[:, 1] > 0.2
# Mặt nạ mềm (0..1): mép tóc mỏng dần vào da đầu thay vì cắt răng cưa; chân tóc phía trước thấp dần ở hai bên thái dương
cl = lambda x: np.clip(x, 0, 1)
fw = cl((-nrm[:, 1] - 0.3) / 0.4)  # 1 = hướng về phía trước
z_line = zt_max - (0.17 + 0.30 * side ** 2) * headH  # thấp dần ở thái dương (chỗ rẽ ngôi bỏ: lưới thưa nên ra khối vuông)
m_front = cl((zb - z_line) / (0.06 * headH))
m_side = cl((zb - (zchin + 0.62 * headH)) / (0.08 * headH))
m_back = cl((zb - z_hair) / (0.10 * headH)) * cl((0.55 - side) / 0.15) * cl((nrm[:, 1] - 0.1) / 0.3) * cl((nrm[:, 2] + 0.5) / 0.3)  # chỉ mặt hướng ra sau, không bám mặt dưới cằm/cổ
behind = cl((base_f[:, 1] - (_yf + 0.28 * _yd)) / (0.08 * _yd))  # tóc không được mọc ở phần 28% độ sâu phía trước (mi mắt, môi, hốc mắt)
mm = fw * m_front + (1 - fw) * np.maximum(m_side, m_back) * behind
for _ in range(4 * 4 ** SUBDIV):  # làm mượt mặt nạ theo kề cạnh để mép tóc hết bậc thang
    acc = mm.copy(); np.add.at(acc, ed[:, 0], mm[ed[:, 1]]); np.add.at(acc, ed[:, 1], mm[ed[:, 0]])
    mm = acc / (deg + 1)
t_up = np.clip((zb - z_hair) / max(zt_max - z_hair, 1e-6), 0, 1)
off = 0.004 * H + (0.03 * H + 0.02 * H * (1 - t_up)) * mm * mm * (3 - 2 * mm)  # tóc dày hơn ở phía dưới, mỏng dần về mép
hb = bmesh.new(); vmap = {}
for p in me2.polygons:
    vs = list(p.vertices)
    if all(mm[i] > 0.02 for i in vs):
        for i in vs:
            if i not in vmap: vmap[i] = hb.verts.new(tuple(base_f[i] + nrm[i] * off[i]))
        try: hb.faces.new([vmap[i] for i in vs])
        except ValueError: pass
hobj = None
if len(hb.faces) > 0 and not HAS_HAIR:
    rev = {v: i for i, v in vmap.items()}
    uvh = hb.loops.layers.uv.new("UV")
    zlo, zhi = min(zb[list(vmap)]), max(zb[list(vmap)])
    for f in hb.faces:
        f.smooth = True
        for l in f.loops:
            i = rev[l.vert]
            l[uvh].uv = (math.atan2(base_f[i, 0] - xc, base_f[i, 1] - yc) / (2 * math.pi) + 0.5, (zb[i] - zlo) / max(zhi - zlo, 1e-6))
    hm = bpy.data.meshes.new("Hair"); hb.to_mesh(hm); hobj = bpy.data.objects.new("Hair", hm); bpy.context.collection.objects.link(hobj)
    rng = np.random.default_rng(1)
    nz = rng.random((256, 1024)).astype(np.float32)
    for _ in range(3):  # làm mượt dọc mạnh, ngang nhẹ -> sợi tóc dọc
        k = np.ones(41) / 41; nz = np.apply_along_axis(lambda c: np.convolve(np.pad(c, 20, mode="wrap"), k, "valid"), 0, nz)
    nz = np.apply_along_axis(lambda r: np.convolve(np.pad(r, 1, mode="wrap"), np.ones(3) / 3, "valid"), 1, nz)
    nz = (nz - nz.min()) / max(nz.max() - nz.min(), 1e-6)
    hc = np.clip(hair_c[None, None, :] * (0.65 + 0.7 * nz[..., None]), 0, 1).astype(np.float32)
    ht = np.concatenate([hc, np.ones((256, 1024, 1), np.float32)], axis=2)
    himg = bpy.data.images.new("hair_tex", 1024, 256, alpha=False); himg.colorspace_settings.name = "Non-Color"
    himg.pixels.foreach_set(np.ascontiguousarray(ht[::-1]).reshape(-1))
    hpath = os.path.join(os.path.dirname(os.path.abspath(DST)), "hair_tex.jpg")
    himg.filepath_raw = hpath; himg.file_format = "JPEG"; himg.save()
    hmat = bpy.data.materials.new("hair"); hmat.use_nodes = True
    hb2 = hmat.node_tree.nodes["Principled BSDF"]; htn = hmat.node_tree.nodes.new("ShaderNodeTexImage"); htn.image = bpy.data.images.load(hpath)
    hmat.node_tree.links.new(htn.outputs["Color"], hb2.inputs["Base Color"]); hb2.inputs["Roughness"].default_value = 0.4
    hmat.use_backface_culling = False
    hm.materials.append(hmat)
    print("HAIR faces", len(hb.faces))
hb.free()

# ---------- 7. vật liệu + xuất ----------
mat = bpy.data.materials.new("skin"); mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
tn = mat.node_tree.nodes.new("ShaderNodeTexImage"); tn.image = timg2
mat.node_tree.links.new(tn.outputs["Color"], bsdf.inputs["Base Color"])
rn = mat.node_tree.nodes.new("ShaderNodeTexImage"); rn.image = bpy.data.images.load(rpath); rn.image.colorspace_settings.name = "Non-Color"
sp = mat.node_tree.nodes.new("ShaderNodeSeparateColor")
mat.node_tree.links.new(rn.outputs["Color"], sp.inputs["Color"]); mat.node_tree.links.new(sp.outputs["Green"], bsdf.inputs["Roughness"])
me2.materials.append(mat)
for o in list(bpy.data.objects):
    if o is not obj and o is not hobj and o not in parts: bpy.data.objects.remove(o, do_unlink=True)
bpy.context.view_layer.objects.active = obj
bpy.ops.object.select_all(action="DESELECT"); obj.select_set(True)
if hobj: hobj.select_set(True)
for po in parts: po.select_set(True)
bpy.ops.export_scene.gltf(filepath=DST, export_format="GLB", use_selection=True, export_morph=True, export_morph_normal=False, export_image_format="JPEG")
print("WROTE", DST, os.path.getsize(DST) // 1024, "KB")
