# Chạy: blender -b --python make_test_avatar.py
# Tạo nhân vật MPFB + 52 shape key ARKit + xương Mixamo, xuất GLB để thử trên web.
import importlib, sys, os
import bpy


def dynamic_import(absolute_package_str, key):
    """Blender extension đặt MPFB ở đường dẫn module không cố định nên phải dò theo đuôi tên."""
    for amod in sys.modules:
        if amod.endswith(absolute_package_str):
            return getattr(importlib.import_module(amod), key)
    raise ValueError(f"Không thấy module {absolute_package_str}")


HumanService = dynamic_import("mpfb.services.humanservice", "HumanService")
FaceService = dynamic_import("mpfb.services.faceservice", "FaceService")

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
os.makedirs(OUT, exist_ok=True)

basemesh = HumanService.create_human()
FaceService.load_targets(basemesh, load_microsoft_visemes=False, load_arkit_faceunits=True)
rig = HumanService.add_builtin_rig(basemesh, "mixamo")

keys = [k.name for k in basemesh.data.shape_keys.key_blocks]
print("SHAPEKEYS", len(keys), [k for k in keys if "mouth" in k.lower() and "smile" in k.lower()])
print("BONES", len(rig.data.bones), [b.name for b in rig.data.bones][:6])
print("VERTS", len(basemesh.data.vertices))

# Nhân bản rồi bake modifier và bỏ helper geometry (mảnh phụ của MakeHuman) như script mẫu 10 của MPFB.
ExportService = dynamic_import("mpfb.services.exportservice", "ExportService")
ObjectService = dynamic_import("mpfb.services.objectservice", "ObjectService")
export_root = ExportService.create_character_copy(basemesh, name_suffix="_export")
export_mesh = ObjectService.find_object_of_type_amongst_nearest_relatives(export_root, "Basemesh")
ExportService.bake_modifiers_remove_helpers(export_mesh, bake_masks=True, bake_subdiv=False, remove_helpers=True, also_proxy=True)
print("EXPORT_SHAPEKEYS", len(export_mesh.data.shape_keys.key_blocks) if export_mesh.data.shape_keys else 0, "VERTS", len(export_mesh.data.vertices))

bpy.ops.object.select_all(action="DESELECT")
export_root.select_set(True)
for child in ObjectService.get_list_of_children(export_root):
    child.select_set(True)
bpy.context.view_layer.objects.active = export_root
path = os.path.join(OUT, "test_avatar.glb")
bpy.ops.export_scene.gltf(filepath=path, use_selection=True, export_format="GLB",
                          export_morph=True, export_skins=True, export_animations=False)
print("WROTE", path, os.path.getsize(path))
