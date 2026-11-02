set -x
cd /kaggle/working
pip -q install uv 2>&1 | tail -1
uv python install 3.11 2>&1 | tail -3
uv venv --python 3.11 --python-preference only-managed /kaggle/working/v311 2>&1 | tail -3
. /kaggle/working/v311/bin/activate
python --version
PY=/kaggle/working/v311/bin/python
git clone -q https://github.com/VAST-AI-Research/UniRig
cd UniRig
uv pip install --python $PY -q torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -2
uv pip install --python $PY -q torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.3.0+cu121.html 2>&1 | tail -2
uv pip install --python $PY -q spconv-cu120 "numpy==1.26.4" 2>&1 | tail -2
grep -viE "flash|^torch" requirements.txt > req2.txt
cat req2.txt
uv pip install --python $PY -q -r req2.txt 2>&1 | tail -5
grep -rl flash_attention_2 . --include=*.py --include=*.yaml | xargs -r sed -i "s/flash_attention_2/sdpa/g"
grep -rn "_attn_implementation" configs src | head -5
sed -i "s/enable_flash=True/enable_flash=False/g" src/model/pointcept/models/PTv3Object.py
grep -rn "flash_attn\." src | head
grep -n "enable_flash=" src/model/pointcept/models/PTv3Object.py | head
$PY -c "import torch,bpy,yaml;from flash_attn.modules.mha import MHA;print('torch',torch.__version__,torch.cuda.is_available(),'bpy ok')"
mkdir -p /kaggle/working/shim/flash_attn/modules
touch /kaggle/working/shim/flash_attn/__init__.py /kaggle/working/shim/flash_attn/modules/__init__.py
cat > /kaggle/working/shim/flash_attn/modules/mha.py <<'EOM'
import torch, torch.nn as nn, torch.nn.functional as F
class MHA(nn.Module):
    def __init__(self, embed_dim, num_heads, cross_attn=False, **kw):
        super().__init__()
        self.h = num_heads; self.d = embed_dim // num_heads
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wkv = nn.Linear(embed_dim, 2 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    def forward(self, x, x_kv=None, **kw):
        b, s, _ = x.shape; t = x_kv.shape[1]
        q = self.Wq(x).view(b, s, self.h, self.d).transpose(1, 2)
        kv = self.Wkv(x_kv).view(b, t, 2, self.h, self.d)
        k = kv[:, :, 0].transpose(1, 2); v = kv[:, :, 1].transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(b, s, self.h * self.d)
        return self.out_proj(o)
EOM
export PYTHONPATH=/kaggle/working/shim
mkdir -p /kaggle/working/rigged
uv pip install --python $PY -q psutil 2>&1 | tail -1
mkdir -p inputs results
for f in $(find /kaggle/input -name "tripo_full_*.glb"); do cp $f inputs/; done
ls inputs
for f in inputs/*.glb; do
  n=$(basename $f .glb)
  echo "=== $n"
  bash launch/inference/generate_skeleton.sh --input $f --output results/${n}_skel.fbx 2>&1 | tail -25
  ls results tmp 2>&1 | head -20
  bash launch/inference/generate_skin.sh --input results/${n}_skel.fbx --output results/${n}_rig.fbx 2>&1 | tail -25
  bash launch/inference/merge.sh --source results/${n}_rig.fbx --target $f --output results/${n}_rig.glb 2>&1 | tail -10
done
mkdir -p /kaggle/working/rigged
cp results/*.fbx results/*.glb /kaggle/working/rigged/ 2>/dev/null
ls -la /kaggle/working/rigged
