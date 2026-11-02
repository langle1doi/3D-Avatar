# Chạy: blender -b "rain_v3.2.blend" --python rain_bake.py -- ra.glb [rain]
#       blender -b "snow_v4.2.blend" --python rain_bake.py -- ra.glb snow
# Nướng nhân vật Rain/Snow (Blender Studio, CC-BY: "Rain Rig / Snow Rig (CC) Blender Foundation | studio.blender.org") thành GLB nhẹ cho web:
# xương gọn (thân/tay/chân) + biểu cảm dưới dạng shape key tên ARKit, chụp từ rig gốc (rig gốc có 2166 xương, web không dùng nổi).
import math, os, re, sys
import bpy
import numpy as np
from mathutils import kdtree

_argv = sys.argv[sys.argv.index("--") + 1:]
DST = _argv[0]; CH = _argv[1] if len(_argv) > 1 else "rain"; SNOW = CH == "snow"
rig = bpy.data.objects["RIG-Snow" if SNOW else "RIG-rain"]
B = rig.pose.bones

# ---------- 1. xương giữ lại: tên -> cha (hierarchy tự đặt vì xương DEF của CloudRig gần như phẳng) ----------
KEEP = {"DEF-Pelvis": None, "DEF-Spine1": "DEF-Pelvis", "DEF-Spine2": "DEF-Spine1", "DEF-Spine3": "DEF-Spine2",
        "DEF-Neck": "DEF-Spine3", "DEF-Head": "DEF-Neck"}
for s in "LR":
    def add(n, p): KEEP[f"DEF-{n}.{s}"] = p if p in KEEP else f"DEF-{p}.{s}"
    add("Clavicle", "DEF-Spine3"); add("Upperarm1", "Clavicle"); add("Upperarm2", "Upperarm1")
    add("Forearm1", "Upperarm2"); add("Forearm2", "Forearm1"); add("Hand", "Forearm2")
    for f, n in (("Index", 4), ("Middle", 4), ("Ring", 4), ("Pinky", 4), ("Thumb", 3)):
        for i in range(1, n + 1): add(f"{f}{i}", "Hand" if i == 1 else f"{f}{i - 1}")
    add("Thigh1", "DEF-Pelvis"); add("Thigh2", "Thigh1"); add("Shin1", "Thigh2"); add("Shin2", "Shin1")
    add("Foot", "Shin2"); add("Toe", "Foot")
# tên xương chuẩn theo Rain; Snow ánh xạ về cùng tên để hai nhân vật dùng chung animation
BONE = {k: k for k in KEEP}
if SNOW:
    BONE.update({"DEF-Pelvis": "DEF-Hips", "DEF-Spine1": "DEF-Spine", "DEF-Spine2": "DEF-RibCage", "DEF-Spine3": "DEF-Chest"})
    for s_ in "LR":
        for a_, b_ in (("Clavicle", "Shoulder"), ("Upperarm1", "UpperArm_1"), ("Upperarm2", "UpperArm_2"), ("Forearm1", "Forearm_1"), ("Forearm2", "Forearm_2"), ("Hand", "Wrist"),
                       ("Thigh1", "Thigh_1"), ("Thigh2", "Thigh_2"), ("Shin1", "Knee_1"), ("Shin2", "Knee_2"), ("Foot", "Foot"), ("Toe", "Toes")):
            BONE[f"DEF-{a_}.{s_}"] = f"DEF-{b_}.{s_}"
        for f in ("Index", "Middle", "Ring", "Pinky"):
            BONE[f"DEF-{f}1.{s_}"] = f"DEF-Finger_{f}_Carpal.{s_}"
            for i in (2, 3, 4): BONE[f"DEF-{f}{i}.{s_}"] = f"DEF-Finger_{f}{i - 1}.{s_}"
        for i in (1, 2, 3): BONE[f"DEF-Thumb{i}.{s_}"] = f"DEF-Finger_Thumb{i}.{s_}"
CANON = {v: k for k, v in BONE.items()}
missing = [k for k in BONE.values() if k not in rig.data.bones]
assert not missing, missing

# mesh gắn vào đầu: mọi xương mặt/tóc của chúng gộp vào DEF-Head
HEADISH = re.compile(r"head|eye|gums|teeth|tongue|hair")
SKIP = {"GEO-rain-body_nomask", "GEO-rain-eyes_viewport", "GEO-snow_head_helper", "GEO-snow_shoes_base_deformer"}
SRC = [o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("GEO-" + CH) and not o.hide_render and o.name not in SKIP]
short = lambda o: re.sub(r"GEO-(rain|snow)[-_]", "", o.name)
HEAD_NAME, EYES_NAME = f"GEO-{CH}-head", f"GEO-{CH}-eyes"
for o in SRC:
    for m in o.modifiers:
        if m.type == "SUBSURF": m.show_viewport = False  # giữ nguyên số đỉnh giữa các tư thế, lưới nhẹ cho web
    if "UVMap" in o.data.uv_layers: o.data.uv_layers.active = o.data.uv_layers["UVMap"]  # Snow để tầng UV khác làm active


def evaluated():
    dg = bpy.context.evaluated_depsgraph_get(); dg.update()
    return dg


def coords(o, dg):
    e = o.evaluated_get(dg); me = e.to_mesh()
    a = np.empty(len(me.vertices) * 3); me.vertices.foreach_get("co", a); a = a.reshape(-1, 3)
    M = np.array(e.matrix_world); a = a @ M[:3, :3].T + M[:3, 3]
    e.to_mesh_clear()
    return a


def apply_action(name, frame=None):  # áp một tư thế trong thư viện của rig (Snow): đặt thẳng giá trị kênh lên xương
    a = bpy.data.actions[name]; fr = frame if frame is not None else a.frame_range[0]
    for f in a.fcurves:
        m = re.match(r'pose\.bones\["(.+?)"\]\.(location|rotation_quaternion|rotation_euler|scale)$', f.data_path)
        if m and m.group(1) in B: getattr(B[m.group(1)], m.group(2))[f.array_index] = f.evaluate(fr)


if SNOW:  # tư thế nghỉ của mặt: mặc định của thư viện
    apply_action("Eyemask Default"); apply_action("Mouth Default")
SNAP = {b.name: (b.location.copy(), b.rotation_quaternion.copy(), b.rotation_euler.copy(), b.scale.copy()) for b in B}


def snap_reset():
    for n, (l_, q_, e_, s_) in SNAP.items():
        b = B[n]; b.location, b.rotation_quaternion, b.rotation_euler, b.scale = l_, q_, e_, s_


# ---------- 2. tư thế nghỉ -> mesh mới có sẵn UV/vật liệu ----------
dg = evaluated()
BASE = {}; NEW = {}
for o in SRC:
    me = bpy.data.meshes.new_from_object(o.evaluated_get(dg), depsgraph=dg)
    p = coords(o, dg); me.vertices.foreach_set("co", p.reshape(-1)); me.update()
    if SNOW:  # vật liệu Snow gắn theo object: giữ tên qua vật liệu giữ chỗ để chỉ số mặt không đổi
        names = [s_.material.name if s_.material else "snow.skin" for s_ in o.material_slots]
        me.materials.clear()
        for nm_ in names: me.materials.append(bpy.data.materials.get(nm_ + ".ph") or bpy.data.materials.new(nm_ + ".ph"))
    n = bpy.data.objects.new(short(o), me); bpy.context.collection.objects.link(n)
    BASE[o.name] = p; NEW[o.name] = n
    print("MESH", n.name, "verts", len(me.vertices))

# ---------- 3. bảng biểu cảm: tên ARKit -> [(xương, kiểu, trục, giá trị)]; trái/phải theo chính nhân vật như ARKit ----------
JAW_DEG = 22
EX = {}
for s, A in (("L", "Left"), ("R", "Right")):
    sign = 1 if s == "L" else -1
    EX[f"eyeBlink{A}"] = [(f"ACT-Eyelid_Upper.{s}", "loc", 1, -0.04), (f"ACT-Eyelid_Lower.{s}", "loc", 1, 0.006)]
    EX[f"eyeWide{A}"] = [(f"ACT-Eyelid_Upper.{s}", "loc", 1, 0.016)]
    EX[f"eyeSquint{A}"] = [(f"ACT-Eyelid_Lower.{s}", "loc", 1, 0.02)]
    EX[f"mouthSmile{A}"] = [(f"ACT-Lips_Corner.{s}", "loc", 1, 0.03), (f"ACT-Lips_Corner.{s}", "loc", 2, 0.02), (f"ACT-Cheek_Outer.{s}", "loc", 1, 0.015)]
    EX[f"mouthFrown{A}"] = [(f"ACT-Lips_Corner.{s}", "loc", 1, -0.03)]
    EX[f"mouthStretch{A}"] = [(f"ACT-Lips_Corner.{s}", "loc", 2, 0.04)]
    EX[f"cheekSquint{A}"] = [(f"ACT-Cheek_Outer.{s}", "loc", 1, 0.03)]
    EX[f"noseSneer{A}"] = [(f"ACT-Sneer.{s}", "loc", 1, 0.02)]
    EX[f"browDown{A}"] = [(f"MSTR-Eyebrow.{s}", "loc", 1, -0.03)]
    EX[f"browOuterUp{A}"] = [(f"MSTR-Eyebrow.{s}", "loc", 1, 0.03)]
EX["mouthPucker"] = [("ACT-Lips_Corner.L", "loc", 0, 0.03), ("ACT-Lips_Corner.R", "loc", 0, -0.03)]
EX["cheekPuff"] = [("SK-Cheek_Puff.L", "loc", 1, 0.02), ("SK-Cheek_Puff.R", "loc", 1, 0.02)]
EX["mouthRollUpper"] = [("ACT-Lip_Roll_Upper", "rot", 0, -40)]
EX["mouthRollLower"] = [("ACT-Lip_Roll_Lower", "rot", 0, -40)]


def pose(items):
    for b, kind, ax, v in items:
        if b == "act":
            apply_action(kind); continue
        pb = B[b]
        if kind == "loc":
            pb.location[ax] = v
        else:
            pb.rotation_mode = "XYZ"; pb.rotation_euler[ax] = math.radians(v)


def reset(items):
    snap_reset()


SIDE = {}  # biểu cảm tách trái/phải bằng mặt nạ theo trục x (tư thế thư viện của Snow không tách sẵn)
if SNOW:
    EX = {}
    for A, sg in (("Left", 1), ("Right", -1)):
        for nm_, act in ((f"eyeBlink{A}", "Eyemask Closed"), (f"eyeWide{A}", "Eyemask Scared"), (f"eyeSquint{A}", "Eyemask Relaxed"), (f"mouthSmile{A}", "Mouth Teethsmile"), (f"mouthFrown{A}", "Mouth Teethfrown")):
            EX[nm_] = [("act", act, 0, 0)]; SIDE[nm_] = sg
        s_ = "L" if sg > 0 else "R"
        EX[f"browDown{A}"] = [(f"MSTR-Eyebrow.{s_}", "loc", 1, -0.03)]
        EX[f"browOuterUp{A}"] = [(f"MSTR-Eyebrow.{s_}", "loc", 1, 0.03)]
    EX["mouthPucker"] = [("act", "Mouth Oo", 0, 0)]; EX["mouthFunnel"] = [("act", "Mouth Uu", 0, 0)]
    EX["mouthPressLeft"] = [("act", "Mouth Compressed", 0, 0)]; SIDE["mouthPressLeft"] = 1
    EX["mouthPressRight"] = EX["mouthPressLeft"]; SIDE["mouthPressRight"] = -1
    EX["cheekPuff"] = [("ACT-CheekPuff.L", "loc", 1, 0.02), ("ACT-CheekPuff.R", "loc", 1, 0.02)]

# hàm: thử các trục và hai chiều, chọn cách làm phần di chuyển đi xuống nhiều nhất
JAW = "Jaw" if SNOW else "MSTR-Jaw"
head = next(o for o in SRC if o.name == HEAD_NAME)
best = None
for ax in (0, 2):
    for sg in (1, -1):
        it = [(JAW, "rot", ax, sg * JAW_DEG)]
        pose(it); dz = coords(head, evaluated()) - BASE[head.name]; reset(it)
        mv = np.abs(dz).sum(1) > 1e-3
        score = -dz[mv, 2].mean() if mv.any() else -9
        print("JAW try ax", ax, "sign", sg, "moved", int(mv.sum()), "score", round(float(score), 5))
        if best is None or score > best[0]: best = (score, it)
EX["jawOpen"] = best[1]
print("JAW chosen", best[1])

# ---------- 4. chụp từng biểu cảm ----------
KEYS = {o.name: {} for o in SRC}
for name, items in EX.items():
    pose(items); d = evaluated()
    moved = []
    for o in SRC:
        p = coords(o, d)
        if name in SIDE and p.shape == BASE[o.name].shape:  # mờ dần qua giữa mặt
            w_ = np.clip(SIDE[name] * BASE[o.name][:, 0] / 0.012 * 0.5 + 0.5, 0, 1)[:, None]; p = BASE[o.name] + (p - BASE[o.name]) * w_
        if p.shape == BASE[o.name].shape and np.abs(p - BASE[o.name]).max() > 2e-4:
            KEYS[o.name][name] = p; moved.append(NEW[o.name].name)
    reset(items)
    print("EXPR", name, moved)
evaluated()

# ---------- 4b. morph khuôn mặt (kiểu RPM): biến dạng mượt quanh các mốc, áp cho MỌI mesh ở đầu để mắt/mi/răng/tóc khớp nhau ----------
H0 = BASE[head.name]; E0 = BASE[EYES_NAME]
eyes_ = [E0[E0[:, 0] > 0], E0[E0[:, 0] < 0]]
cs = [q.mean(0) for q in eyes_]; r0 = float(np.mean([np.ptp(q[:, 0]) / 2 for q in eyes_]))
_sm = sum(np.linalg.norm(KEYS[head.name][k] - H0, axis=1) for k in ("mouthSmileLeft", "mouthSmileRight"))
cM = H0[np.argsort(_sm)[-40:]].mean(0); cM[0] = 0  # miệng: quanh các đỉnh chuyển động nhiều nhất khi cười
zE = float(np.mean([c[2] for c in cs])); zM = float(cM[2])
_m = (np.abs(H0[:, 0]) < 0.03) & (H0[:, 2] < zE) & (H0[:, 2] > zM)
nose = H0[_m][np.argmin(H0[_m, 1])]  # đỉnh mũi: điểm nhô ra trước nhất giữa mắt và miệng (mặt nhìn về -Y)
_c = (np.abs(H0[:, 0]) < 0.03) & (H0[:, 1] < H0[:, 1].mean()) & (H0[:, 2] < zM)
_front = H0[_c]; zC = float(_front[_front[:, 1] < np.percentile(_front[:, 1], 40), 2].min())  # cằm: thấp nhất trong phần nhô ra trước
F = zE - zC  # đơn vị đo mặt
print("LANDMARK eyes", np.round(cs, 3).tolist(), "r0", round(r0, 4), "nose", np.round(nose, 3).tolist(), "mouth", np.round(cM, 3).tolist(), "chin z", round(zC, 3), "F", round(F, 3))
cl = lambda x: np.clip(x, 0, 1)
smooth = lambda t: t * t * (3 - 2 * t)
neck = lambda P: smooth(cl((P[:, 2] - (zC - 0.45 * F)) / (0.35 * F)))  # 1 từ gần cằm trở lên, về 0 ở chỗ cổ nối với thân


def bell(P, c, r):
    return np.exp(-(((P - c) / np.array(r)) ** 2).sum(1))


def eyebell(P, c):  # đều trong phạm vi nhãn cầu (không méo), ngoài đó giảm êm -> không có nếp gấp
    d = np.linalg.norm(P - c, axis=1)
    return np.where(d < 1.2 * r0, 1.0, np.exp(-((d - 1.2 * r0) / (1.5 * r0)) ** 2))


def m_faceWidth(P):
    Q = P.copy(); Q[:, 0] += 0.18 * P[:, 0] * neck(P); return Q


def m_faceLong(P):
    Q = P.copy(); Q[:, 2] -= 0.13 * F * smooth(cl((nose[2] - P[:, 2]) / (nose[2] - zC))) * neck(P); return Q


def m_jawNarrow(P):
    Q = P.copy(); Q[:, 0] -= 0.28 * P[:, 0] * smooth(cl((zM + 0.1 * F - P[:, 2]) / (zM + 0.1 * F - zC))) * neck(P); return Q


def m_eyeSize(P):
    Q = P.copy()
    for c in cs: Q += (P - c) * np.array([0.3, 0.15, 0.3]) * eyebell(P, c)[:, None]
    return Q


def m_eyeSpacing(P):
    Q = P.copy()
    for c in cs: Q[:, 0] += np.sign(c[0]) * 0.5 * r0 * eyebell(P, c)
    return Q


def m_eyeHeight(P):
    Q = P.copy()
    for c in cs: Q[:, 2] += 0.5 * r0 * eyebell(P, c)
    return Q


def m_noseSize(P):
    R = 0.3 * (zE - zM); c = nose + np.array([0, 0.7 * R, 0])  # co về điểm sâu trong mặt: co về đầu mũi thì mũi bẹt, lộ lỗ mũi
    return P + (P - c) * 0.35 * bell(P, nose, (R, R, R))[:, None]


def m_noseWidth(P):
    R = 0.3 * (zE - zM); Q = P.copy(); Q[:, 0] += 0.45 * P[:, 0] * bell(P, nose, (R, R * 1.2, R))
    return Q


def m_mouthWidth(P):
    Q = P.copy(); Q[:, 0] += 0.35 * P[:, 0] * bell(P, cM, (0.55 * F, 0.5 * F, 0.22 * F)); return Q


def m_lipFull(P):
    Q = P.copy(); b = bell(P, cM, (0.42 * F, 0.3 * F, 0.16 * F)); Q[:, 2] += 0.55 * (P[:, 2] - cM[2]) * b; Q[:, 1] -= 0.02 * F * b; return Q


def m_cheek(P):
    Q = P.copy(); z = nose[2] - 0.1 * F; xw = 0.42 * F
    for sg in (1, -1):
        b = bell(P, np.array([sg * xw, H0[:, 1].min() + 0.35 * F, z]), (0.42 * F, 0.42 * F, 0.36 * F))
        Q[:, 0] += sg * 0.05 * F * b; Q[:, 1] -= 0.05 * F * b
    return Q


MORPHS = {k[2:]: v for k, v in list(globals().items()) if k.startswith("m_")}
for name, f in MORPHS.items():
    moved = 0
    for o in SRC:
        if not HEADISH.search(NEW[o.name].name): continue
        p = f(BASE[o.name])
        if np.abs(p - BASE[o.name]).max() > 2e-4:
            KEYS[o.name]["shape_" + name] = p; moved += 1
    print("MORPH", name, "meshes", moved)
# Lông mày mỏng: ép dày theo chiều dọc về đường giữa của mỗi bên
_br = next(o for o in SRC if short(o) == "eyebrows"); _bp = BASE[_br.name].copy()
for sg_ in (1, -1):
    m_ = sg_ * _bp[:, 0] > 0; fit_ = np.polyval(np.polyfit(_bp[m_, 0], _bp[m_, 2], 3), _bp[m_, 0]); _bp[m_, 2] = fit_ + (_bp[m_, 2] - fit_) * 0.5
KEYS[_br.name]["shape_browThin"] = _bp
# Morph bù: chớp mắt lưu độ dịch của mắt cỡ gốc nên mắt to khép không kín. Bù = biến dạng mắt to áp lên dáng mắt nhắm, trừ phần đã có sẵn.
# Web đặt độ mạnh = chớp x cỡ mắt (xem rpm.html).
for A in ("Left", "Right"):
    k = f"eyeBlink{A}"
    for o in SRC:
        if k in KEYS[o.name]:
            b0, pk = BASE[o.name], KEYS[o.name][k]
            KEYS[o.name][f"fix_{k}_eyeSize"] = b0 + (m_eyeSize(pk) - pk) - (m_eyeSize(b0) - b0)

for o in SRC:
    n = NEW[o.name]
    if not KEYS[o.name]: continue
    n.shape_key_add(name="Basis")
    for k, p in KEYS[o.name].items():
        n.shape_key_add(name=k).data.foreach_set("co", p.reshape(-1))

# ---------- 5. xương gọn + trọng số (lấy từ đỉnh gốc gần nhất; xương không giữ thì gộp vào xương giữ gần nhất) ----------
arm_d = bpy.data.armatures.new("Rain"); arm = bpy.data.objects.new("Rain", arm_d); bpy.context.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm; arm.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
MW = rig.matrix_world
for k in KEEP:
    b = rig.data.bones[BONE[k]]; eb = arm_d.edit_bones.new(k.replace("DEF-", ""))
    eb.head = MW @ b.head_local; eb.tail = MW @ b.tail_local; eb.roll = 0
for k, p in KEEP.items():
    if p: arm_d.edit_bones[k.replace("DEF-", "")].parent = arm_d.edit_bones[p.replace("DEF-", "")]
bpy.ops.object.mode_set(mode="OBJECT")
seg = [(k, np.array(MW @ rig.data.bones[BONE[k]].head_local), np.array(MW @ rig.data.bones[BONE[k]].tail_local)) for k in KEEP]


def nearest_keep(name):
    p = np.array(MW @ rig.data.bones[name].head_local); best_, bd = None, 1e9
    for k, a, c in seg:
        t = np.clip(np.dot(p - a, c - a) / max(np.dot(c - a, c - a), 1e-9), 0, 1); dd = np.linalg.norm(p - (a + t * (c - a)))
        if dd < bd: best_, bd = k, dd
    return best_


for o in SRC:
    n = NEW[o.name]; me0 = o.data
    headish = bool(HEADISH.search(n.name))
    gmap = {}
    for g in o.vertex_groups:
        if g.name in CANON: gmap[g.index] = CANON[g.name]
        elif g.name in rig.data.bones and rig.data.bones[g.name].use_deform: gmap[g.index] = "DEF-Head" if headish else nearest_keep(g.name)
    vg = {}
    M = np.array(o.matrix_world); P0 = np.array([v.co for v in me0.vertices]) @ M[:3, :3].T + M[:3, 3]
    kd = kdtree.KDTree(len(P0))
    for i, p in enumerate(P0): kd.insert(tuple(p), i)
    kd.balance()
    w0 = []
    for v in me0.vertices:
        acc = {}
        for g in v.groups:
            k = gmap.get(g.group)
            if k and g.weight > 0: acc[k] = acc.get(k, 0) + g.weight
        w0.append(acc)
    for i, p in enumerate(BASE[o.name]):
        acc = w0[kd.find(tuple(p))[1]] or {("DEF-Head" if headish else "DEF-Spine3"): 1.0}
        for k, w in acc.items():
            name = k.replace("DEF-", "")
            if name not in vg: vg[name] = n.vertex_groups.new(name=name)
            vg[name].add([i], w, "ADD")
    n.parent = arm; md = n.modifiers.new("Armature", "ARMATURE"); md.object = arm
    print("WEIGHTS", n.name, "groups", len(vg))

# ---------- 6. vật liệu: Rain dùng UDIM (đầu ở ô 1003, thân ở 1001+1002) mà glTF không hỗ trợ -> đầu dùng thẳng ô 1003, thân ghép 2 ô ----------
TEXD = os.path.join(os.path.dirname(bpy.data.filepath), "textures")
OUTT = os.path.join(os.path.dirname(os.path.abspath(DST)), CH + "_tex"); os.makedirs(OUTT, exist_ok=True)


def shrink(src, size):
    im = bpy.data.images.load(os.path.join(TEXD, src))
    if im.size[0] > size: im.scale(size, size)
    p = os.path.join(OUTT, os.path.splitext(src)[0] + ".png"); im.filepath_raw = p; im.file_format = "PNG"; im.save()
    return p


def pixels(src, size):
    im = bpy.data.images.load(os.path.join(TEXD, src)); im.scale(size, size)
    return np.array(im.pixels[:], np.float32).reshape(size, size, 4)


# thân: ghép các ô UDIM thành một hàng (Rain 2 ô, Snow 3 ô)
SKIN_TILE = "skin_diffuse.%d.png" if SNOW else "TEX-rain_body_diffuse.%d.png"
ATLAS_N = 3 if SNOW else 2
atlas = np.concatenate([pixels(SKIN_TILE % (1001 + i), 1024) for i in range(ATLAS_N)], axis=1)
# Texture trung tính cho đổi màu trên web: chia cho màu trung vị (trong không gian tuyến tính) -> chỉ còn sáng/tối và sắc lệch (môi, tàn nhang).
# Web nhân lại với màu chọn: material.color = màu / NEUTRAL.
NEUTRAL = 0.8
lin = lambda s: np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)
srgb = lambda x: np.where(x <= 0.0031308, x * 12.92, 1.055 * np.maximum(x, 1e-9) ** (1 / 2.4) - 0.055)


def median_color(px):
    m = px[..., :3].mean(-1) > 0.08
    return np.median(lin(px[..., :3][m]), axis=0)


def neutral(px, med, name):
    w, h = px.shape[1], px.shape[0]; out = px.copy()
    out[..., :3] = srgb(np.clip(lin(px[..., :3]) / med * NEUTRAL, 0, 1))
    im = bpy.data.images.new(name, w, h, alpha=False); im.pixels.foreach_set(out.astype(np.float32).reshape(-1))
    p = os.path.join(OUTT, name + ".png"); im.filepath_raw = p; im.file_format = "PNG"; im.save()
    return p


head_px = pixels(SKIN_TILE % (1001 if SNOW else 1003), 2048); hair_px = pixels("hair_diffuse.png" if SNOW else "TEX-rain_hair_diffuse.png", 1024)
_fh, _fw = head_px.shape[:2]
# trung vị chỉ trong vùng mặt: ảnh còn có khoang miệng hồng và da đầu xám, tính cả ảnh thì da mặt bị đẩy gần trắng, mất chi tiết
SKIN_MED = median_color(head_px[int(0.25 * _fh):int(0.6 * _fh), int(0.3 * _fw):int(0.7 * _fw)]); HAIR_MED = median_color(hair_px)
TEXS = {"head": neutral(head_px, SKIN_MED, "head_neutral"), "body": neutral(atlas, SKIN_MED, "body_neutral"), "eyes": shrink("eyes_diffuse.png" if SNOW else "TEX-rain_eyes.png", 1024),
        "hair": neutral(hair_px, HAIR_MED, "hair_neutral"), "jeans": shrink("pants_diffuse.png" if SNOW else "TEX-rain_jeans_diffuse.png", 1024)}
if SNOW:
    TEXS.update({"lip": TEXS["head"], "top": shrink("shirt_diffuse.png", 1024), "shoes": shrink("shoes_diffuse.png", 1024)})
_hex = lambda c: "#%02x%02x%02x" % tuple(int(round(float(v) * 255)) for v in np.clip(srgb(c), 0, 1))
import json
with open(os.path.splitext(os.path.abspath(DST))[0] + ".json", "w", encoding="utf-8") as f_:
    json.dump({"char": CH, "skin": _hex(SKIN_MED), "hair": _hex(HAIR_MED), "eyes": "#2a7fa8", "neutral": NEUTRAL, "head_z": round(float(cs[0][2]), 3),
               "lip_mask": None if SNOW else f"../blender/out/{CH}_tex/lip_mask.png", "iris": [0.459, 0.539, 0.122] if SNOW else None, "iris_disk": [0.459, 0.539, 0.122] if SNOW else [0.4548, 0.5431, 0.1097], "credit": f"{CH.capitalize()} Rig (CC) Blender Foundation | studio.blender.org"}, f_)
print("MEDIAN skin", _hex(SKIN_MED), "hair", _hex(HAIR_MED))
COL = {"hairband": (0.2, 0.5, 0.6), "eyebrows": (0.05, 0.03, 0.02), "eyelashes": (0.01, 0.01, 0.01), "eyedot": (1, 1, 1), "gums": (0.75, 0.3, 0.3), "teeth": (0.95, 0.95, 0.9),
       "tongue": (0.8, 0.35, 0.35), "top": (0.85, 0.85, 0.85), "scarf": (0.1, 0.4, 0.55), "shoes": (0.8, 0.8, 0.8), "metal": (0.4, 0.4, 0.4), "socks": (0.8, 0.8, 0.8), "laces": (0.41, 0.41, 0.41)}
ROUGH = {"eyes": 0.15, "hair": 0.6, "teeth": 0.3, "tongue": 0.4, "gums": 0.4}
made = {}
for o in SRC:
    n = NEW[o.name]; me = n.data
    for l_ in [l_ for l_ in me.uv_layers if l_.name != me.uv_layers.active.name]: me.uv_layers.remove(l_)  # bỏ UV phụ cho nhẹ
    ul = me.uv_layers.active.data; uv = np.empty(len(ul) * 2); ul.foreach_get("uv", uv); uv = uv.reshape(-1, 2)
    if n.name == "head" and not SNOW: uv[:, 0] -= 2  # đầu Rain ở ô 1003; đầu Snow ở ô đầu tiên
    if n.name == "body": uv[:, 0] /= ATLAS_N
    ul.foreach_set("uv", uv.reshape(-1))
    for i, m in enumerate(me.materials):
        key = (m.name.replace("MAT-rain.", "").replace("snow.", "").replace(".ph", "") if m else "body")
        key = {"skin": "body", "skin_darker": "body", "skin_upper_lip": "lip", "skin_lower_lip": "lip", "eyes_cornea": "cornea", "eye_dot": "eyedot", "pants": "jeans", "shirt": "top", "shoe": "shoes"}.get(key, key)
        if key == "body" and n.name == "head": key = "head"
        if n.name == "eyebrows": key = "eyebrows"
        if key not in made:
            nm = bpy.data.materials.new(key); nm.use_nodes = True; nt = nm.node_tree
            b = nt.nodes["Principled BSDF"]
            b.inputs["Roughness"].default_value = ROUGH.get(key, 0.55)
            if key in TEXS:
                tn = nt.nodes.new("ShaderNodeTexImage"); tn.image = bpy.data.images.load(TEXS[key])
                nt.links.new(tn.outputs["Color"], b.inputs["Base Color"])
            else:
                b.inputs["Base Color"].default_value = (*COL.get(key, (0.7, 0.7, 0.7)), 1)
            if key == "cornea":
                b.inputs["Alpha"].default_value = 0.08; b.inputs["Roughness"].default_value = 0.05; nm.surface_render_method = "BLENDED"
            made[key] = nm
        me.materials[i] = made[key]

# ---------- 6b. mặt nạ môi: web nhuộm riêng vùng này (xem rpm.html). Vùng UV quanh miệng, trong đó lấy phần tối hơn da ----------
hn = NEW[head.name]
Hh, Ww = head_px.shape[:2]; _l = lin(head_px[..., :3]); u0, v0, u1, v1 = 0.43, 0.335, 0.57, 0.405  # môi nằm giữa ô mặt (hàng 0 của pixels Blender ở dưới)
# môi = trong ô quanh miệng, chỗ tỉ lệ G/R thấp hơn da (đỏ hơn da)
mask = cl(((SKIN_MED[1] / SKIN_MED[0]) - _l[..., 1] / np.maximum(_l[..., 0], 1e-4) - 0.05) / 0.08).astype(np.float32); box = np.zeros((Hh, Ww), np.float32)
box[int(v0 * Hh):int(v1 * Hh), int(u0 * Ww):int(u1 * Ww)] = 1
mask *= box
for _ in range(3):  # làm mờ mép
    mask = (mask + np.roll(mask, 1, 0) + np.roll(mask, -1, 0) + np.roll(mask, 1, 1) + np.roll(mask, -1, 1)) / 5
mp = np.ones((Hh, Ww, 4), np.float32); mp[..., :3] = mask[..., None]
im = bpy.data.images.new("lip_mask", Ww, Hh, alpha=False); im.colorspace_settings.name = "Non-Color"; im.pixels.foreach_set(mp.reshape(-1))
im.filepath_raw = os.path.join(OUTT, "lip_mask.png"); im.file_format = "PNG"; im.save()
print("LIP MASK uv", round(float(u0), 3), round(float(v0), 3), round(float(u1), 3), round(float(v1), 3), "coverage", round(float((mask > 0.5).mean()), 5))

# ---------- 6c. thư viện đồ: tóc khối hoạt hình dựng từ chính bề mặt đầu + kính. Mọi món đi theo morph khuôn mặt, gắn cứng vào xương Head ----------
from mathutils.bvhtree import BVHTree
from mathutils import Vector
EXTRA = []


def extra_mesh(ob, name, P, mat, keep_morph=True):
    ob.name = name; ob.data.vertices.foreach_set("co", P.reshape(-1)); ob.data.update()
    ob.data.materials.clear(); ob.data.materials.append(mat)
    ob.shape_key_add(name="Basis")
    for nm_, f in (MORPHS.items() if keep_morph else ()):
        q = f(P)
        if np.abs(q - P).max() > 2e-4: ob.shape_key_add(name="shape_" + nm_).data.foreach_set("co", q.reshape(-1))
    g = ob.vertex_groups.new(name="Head"); g.add(list(range(len(P))), 1.0, "REPLACE")
    ob.parent = arm; ob.modifiers.new("Armature", "ARMATURE").object = arm
    EXTRA.append(ob); print("EXTRA", name, "verts", len(P), "keys", len(ob.data.shape_keys.key_blocks) - 1)


def bvh_of(o):
    me = NEW[o.name].data; me.calc_loop_triangles()
    t = np.empty(len(me.loop_triangles) * 3, dtype=np.int32); me.loop_triangles.foreach_get("vertices", t)
    return BVHTree.FromPolygons([tuple(p) for p in BASE[o.name]], [tuple(x) for x in t.reshape(-1, 3)], all_triangles=True)


bvh = bvh_of(head)
rh = H0[H0[:, 2] > zC]
hc = np.array([0.0, float((rh[:, 1].min() + rh[:, 1].max()) / 2), float(rh[:, 2].max() - 0.42 * np.ptp(rh[:, 2]))])  # tâm đầu


# Đầu Snow hở một lỗ lớn ở sọ (tóc gốc che chỗ đó) nên không bám theo lưới được: dùng khối elip bao trọn sọ
EY_B = float(rh[:, 1].max() - hc[1]); EY_F = float(hc[1] - (cs[0][1] - 0.9 * r0)); EX_ = float(np.percentile(np.abs(rh[rh[:, 2] > cs[0][2], 0]), 99)); EZ_ = float(rh[:, 2].max() - hc[2])


def scalp(d):  # điểm trên da đầu theo hướng d tính từ tâm đầu (bắn tia từ ngoài vào)
    if SNOW:
        d_ = np.array(d, float); d_ /= np.linalg.norm(d_); ey = EY_F if d_[1] < 0 else EY_B
        e_ = hc + d_ / math.sqrt((d_[0] / EX_) ** 2 + (d_[1] / ey) ** 2 + (d_[2] / EZ_) ** 2)
        loc_, _, _, dist_ = bvh.find_nearest(Vector(e_))
        return np.array(loc_) if loc_ is not None and dist_ < 0.02 else e_  # sát da nơi có da, nơi sọ hở thì theo khối elip -> tóc ôm đầu, che kín lỗ
    d = Vector(d).normalized(); o_ = Vector(hc) + d * 1.0
    hit = bvh.ray_cast(o_, -d)[0]
    return np.array(hit) if hit is not None else np.array(bvh.find_nearest(Vector(hc) + d * 0.3)[0])  # đầu Snow hở da đầu dưới tóc: tia có thể hụt


# vân sợi tóc: nhiễu kéo dài theo chiều dọc
rng = np.random.default_rng(3)
nz = rng.random((256, 1024)).astype(np.float32)
for _ in range(3):
    nz = np.apply_along_axis(lambda c: np.convolve(np.pad(c, 20, mode="wrap"), np.ones(41) / 41, "valid"), 0, nz)
nz = (nz - nz.min()) / max(float(np.ptp(nz)), 1e-6)
sp = np.ones((256, 1024, 4), np.float32); sp[..., :3] = srgb(np.clip(NEUTRAL * (0.72 + 0.56 * nz), 0, 1))[..., None]
im = bpy.data.images.new("hair_streak", 1024, 256, alpha=False); im.pixels.foreach_set(sp.reshape(-1))
STREAK = os.path.join(OUTT, "hair_streak.png"); im.filepath_raw = STREAK; im.file_format = "PNG"; im.save()


def hair_mat(name):
    nm = bpy.data.materials.new(name); nm.use_nodes = True; nt = nm.node_tree; b = nt.nodes["Principled BSDF"]
    tn = nt.nodes.new("ShaderNodeTexImage"); tn.image = bpy.data.images.load(STREAK)
    nt.links.new(tn.outputs["Color"], b.inputs["Base Color"]); b.inputs["Roughness"].default_value = 0.6
    return nm


def build_hair(name, z_end, flare, wave, amax_deg=102, ph_low=110, off=0.012, hl=36, part=0.0, NP=72, NCAP=26, NDROP=30):
    """Lưới tóc: hàng theo chiều dọc sợi, cột quanh đầu. phi = 0 ở sau gáy, +-180 độ ở giữa trán."""
    rows = []
    amax = math.radians(amax_deg)
    for j in range(NCAP):  # phần ôm đầu: polar a từ đỉnh xuống ngang tai
        a = amax * j / (NCAP - 1)
        # đường chân tóc: gần đỉnh phủ gần trọn vòng, xuống dưới lùi dần ra hai bên để chừa mặt
        ph_max = math.radians(180 - (180 - ph_low) * math.sqrt(cl((math.degrees(a) - hl) / (92 - hl))))  # căn bậc hai: chân tóc tròn, không nhọn ở giữa trán
        row = []
        for i in range(NP):
            ph = ph_max * (2 * i / (NP - 1) - 1)
            ph_ = ph + math.radians(part) * (1 - a / amax)  # rẽ ngôi lệch: xoay đường ngôi, giảm dần xuống dưới
            d = (math.sin(a) * math.sin(ph_), math.sin(a) * math.cos(ph_), math.cos(a))
            # tóc ngắn: mép dưới thu sát da, không thành vành
            row.append(scalp(d) + np.array(d) * ((off + 0.010 * math.sin(a)) * (1 if NDROP else 1 - float(smooth(cl((a / amax - 0.55) / 0.45)))) + 0.001))
        rows.append(np.array(row))
    rows[0][:] = rows[1].mean(0)  # đỉnh đầu: gom về một điểm, tránh gai
    low = rows[-1]
    phis = [math.radians(ph_low) * (2 * i / (NP - 1) - 1) for i in range(NP)]
    for j in range(1, NDROP + 1):  # phần rủ xuống
        s = j / NDROP; t = float(smooth(np.array(s)))
        row = low.copy()
        row[:, 2] = low[:, 2] + (z_end - low[:, 2]) * s
        for i, ph in enumerate(phis):
            lock = 0.5 * math.sin(7 * ph + 1.0) + 0.3 * math.sin(13 * ph + 2.1)  # lọn tóc
            r = 1 + flare * t + wave * s * (lock + 0.6 * math.sin(5 * s + 3 * ph))
            # phần dưới hẹp ngang, dày theo trước-sau: tóc chia ra trước và sau vai thay vì đâm vào vai
            row[i, 0] = hc[0] + (low[i, 0] - hc[0]) * r * (1 - 0.1 * t); row[i, 1] = hc[1] + (low[i, 1] - hc[1]) * r * (1 + 0.32 * t)
        rows.append(row)
    G = np.array(rows); nr, nc = G.shape[:2]
    for _ in range(10):  # làm mượt: tia dò da đầu trúng tai tạo bậc ở ngang tai
        G[1:-1, 1:-1] = (G[1:-1, 1:-1] * 2 + G[:-2, 1:-1] + G[2:, 1:-1] + G[1:-1, :-2] + G[1:-1, 2:]) / 6
    for k in range(nr):  # hai mép cột trùng nhau ở trán: khép lại để không hở khe đen ở đường ngôi
        if np.linalg.norm(G[k, 0] - G[k, -1]) < 0.03: G[k, 1] = G[k, -2] = (G[k, 1] + G[k, -2]) / 2; G[k, 0] = G[k, -1] = (G[k, 0] + G[k, -1]) / 2
    # làm mượt co lớp tóc vào trong: đỉnh nào lọt dưới da thì đẩy ra lại sát da (phần ôm đầu)
    for k in range(0 if SNOW else min(NCAP, nr)):
        for i in range(nc):
            p = Vector(G[k, i]); loc, nrm_, _, _ = bvh.find_nearest(p)
            if loc is not None and (p - loc).dot(nrm_) < 0.003: G[k, i] = np.array(loc + nrm_ * 0.003)
    V = G.reshape(-1, 3)
    if name is None: return V  # chỉ lấy hình, dùng làm morph
    F_ = [(k * nc + i, k * nc + i + 1, (k + 1) * nc + i + 1, (k + 1) * nc + i) for k in range(nr - 1) for i in range(nc - 1)]
    me = bpy.data.meshes.new(name); me.from_pydata(V.tolist(), [], F_); me.update()
    for pl in me.polygons: pl.use_smooth = True
    uvl = me.uv_layers.new(name="UVMap").data
    for pl in me.polygons:
        for li in pl.loop_indices:
            vi = me.loops[li].vertex_index
            uvl[li].uv = ((vi % nc) / (nc - 1) * 3, (vi // nc) / (nr - 1))
    ob = bpy.data.objects.new(name, me); bpy.context.collection.objects.link(ob)
    extra_mesh(ob, name, V, hair_mat(name))
    return ob, V


# một lưới tóc + morph: cùng số đỉnh ở mọi biến thể, web chỉ kéo thanh trượt (chiều âm = ngược lại)
if not SNOW:  # Snow giữ tóc gốc: đầu hở lỗ ở sọ, đã thử tóc khối nhưng bị thủng và chìm dưới da
    G0 = dict(z_end=zC - 0.17, flare=0.05, wave=0.05, off=0.012, hl=62, ph_low=112)  # hl gốc gần hl morph: khoảng cách lớn thì nội suy hở lỗ trên đỉnh
    MV = (("hairLen", dict(z_end=zC - 0.34)), ("hairVol", dict(flare=0.28, off=0.026)), ("hairWave", dict(wave=0.3)), ("hairline", dict(hl=74)), ("hairFront", dict(ph_low=132)), ("hairPart", dict(part=-25)))
    ob, V0 = build_hair("hair_gen", **G0)
    for nm_, kw in MV:
        ob.shape_key_add(name="shape_" + nm_).data.foreach_set("co", build_hair(None, **{**G0, **kw}).reshape(-1))

# râu: trọng số mượt trên từng đỉnh của đầu, lưu vào màu đỉnh; web pha màu râu vào da (xem rpm.html). Tách lưới riêng thì mép răng cưa.
if SNOW:
    _x, _y, _z = H0[:, 0], H0[:, 1], H0[:, 2]
    ell = np.sqrt((_x / (0.42 * F)) ** 2 + ((_z - zM) / (0.13 * F)) ** 2)
    bw = smooth(cl((nose[2] - 0.1 * F - _z) / (0.07 * F))) * smooth(cl((_z - (zC - 0.28 * F)) / (0.08 * F))) * smooth(cl((H0[:, 1].mean() + 0.1 * F - _y) / (0.1 * F))) * smooth(cl((ell - 1) / 0.2))
    for c_ in list(NEW[head.name].data.color_attributes): NEW[head.name].data.color_attributes.remove(c_)  # Snow có sẵn lớp màu đỉnh riêng, phải bỏ để râu là COLOR_0
    ca = NEW[head.name].data.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
    ca.data.foreach_set("color", np.stack([bw, bw, bw, np.ones_like(bw)], axis=1).astype(np.float32).reshape(-1))
    NEW[head.name].data.color_attributes.active_color = ca

# kính: hai gọng siêu elip + cầu mũi + càng kính, ống tiết diện tròn
import bmesh


def tube(bm, pts, r, closed=False, sides=8):
    pts = [np.array(p, float) for p in pts]; rings = []; N = len(pts)
    for i, p in enumerate(pts):
        a = pts[(i + 1) % N] if closed or i < N - 1 else p
        b_ = pts[i - 1] if closed or i > 0 else p
        t = a - b_; t /= max(np.linalg.norm(t), 1e-9)
        u = np.cross(t, (0, 1, 0))
        if np.linalg.norm(u) < 1e-3: u = np.cross(t, (1, 0, 0))
        u /= np.linalg.norm(u); v_ = np.cross(t, u)
        rings.append([bm.verts.new(tuple(p + r * (math.cos(2 * math.pi * k / sides) * u + math.sin(2 * math.pi * k / sides) * v_))) for k in range(sides)])
    for i in range(N if closed else N - 1):
        r1, r2 = rings[i], rings[(i + 1) % N]
        for k in range(sides):
            bm.faces.new((r1[k], r1[(k + 1) % sides], r2[(k + 1) % sides], r2[k]))


GA, GB = 0.040, 0.031  # nửa rộng, nửa cao tròng kính
ex, ez = abs(cs[0][0]), cs[0][2] - 0.002
sy = lambda x, z: (lambda h: h[0].y if h[0] is not None else None)(bvh.ray_cast(Vector((x, -1, z)), Vector((0, 1, 0))))
ring = lambda sg: [(sg * ex + GA * np.sign(np.cos(t)) * abs(np.cos(t)) ** 0.35, ez + GB * np.sign(np.sin(t)) * abs(np.sin(t)) ** 0.35) for t in np.linspace(0, 2 * math.pi, 40, endpoint=False)]
ys = [y for sg in (1, -1) for x, z in ring(sg) for y in [sy(x, z)] if y is not None]
gy = min(ys) - 0.006  # mặt phẳng kính, sát da nhất
bm = bmesh.new()
for sg in (1, -1):
    tube(bm, [(x, gy, z) for x, z in ring(sg)], 0.0028, closed=True)
    tube(bm, [(sg * (ex + GA), gy, ez + 0.012), (sg * (ex + GA + 0.004), gy + 0.02, ez + 0.012), (sg * (ex + GA + 0.008), float(H0[:, 1].mean()) + 0.02, ez + 0.006)], 0.0022)
bridge = [(x, gy - 0.002 * (1 - (x / (ex - GA)) ** 2), ez + 0.012 + 0.004 * (1 - (x / (ex - GA)) ** 2)) for x in np.linspace(-(ex - GA), ex - GA, 9)]
tube(bm, bridge, 0.0026)
gm = bpy.data.meshes.new("glasses"); bm.to_mesh(gm); bm.free()
go = bpy.data.objects.new("glasses", gm); bpy.context.collection.objects.link(go)
for pl in gm.polygons: pl.use_smooth = True
gmat = bpy.data.materials.new("glasses"); gmat.use_nodes = True
gmat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.01, 0.01, 0.012, 1); gmat.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.25
_g = np.array([v.co for v in gm.vertices]); extra_mesh(go, "glasses", _g, gmat)

# ---------- 7. xuất ----------
for o in list(bpy.context.selected_objects): o.select_set(False)
arm.select_set(True)
for n in NEW.values(): n.select_set(True)
for n in EXTRA: n.select_set(True)
bpy.context.view_layer.objects.active = arm
os.makedirs(os.path.dirname(os.path.abspath(DST)), exist_ok=True)
bpy.ops.export_scene.gltf(filepath=DST, export_format="GLB", use_selection=True, export_morph=True, export_morph_normal=False, export_skins=True,
                          export_animations=False, export_def_bones=False, export_vertex_color="ACTIVE", export_image_format="WEBP")  # WEBP: nhẹ như JPEG mà giữ alpha cho tóc
print("WROTE", DST, os.path.getsize(DST) // 1024, "KB", "bones", len(arm_d.bones), "exprs", len(EX))
