"""Frontalise portraits with Qwen-Image-Edit-2509 on a 16GB T4, to compare against FLUX Kontext (real2pixar_front.py, same instruction).

  python qwen_front.py --input <dir of <id>.png> --output <out dir> [--limit N] [--gguf Qwen-Image-Edit-2509-Q4_K_M.gguf] [--dtype fp16|bf16]
Memory recipe (same idea as the FLUX scripts): the VL text encoder (NF4) embeds every prompt+image first and is dropped, then the transformer is
loaded as a GGUF (QuantStack/Qwen-Image-Edit-2509-GGUF). Writes <out>/<id>.png (resumable) and <out>/params.json.
"""
import argparse, gc, json, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

MODEL = "Qwen/Qwen-Image-Edit-2509"
GGUF_REPO = "QuantStack/Qwen-Image-Edit-2509-GGUF"
FRONT = ("Rotate the head so the person faces the camera directly: a straight front-facing portrait, head upright, both eyes looking at the camera, "
         "symmetric pose. Keep the same person: identity, face shape, hair, skin tone, expression, clothing and background.")  # same text as the Kontext run
NEG = " "

ap = argparse.ArgumentParser()
ap.add_argument("--input"); ap.add_argument("--output"); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--gguf", default="Qwen-Image-Edit-2509-Q4_K_M.gguf"); ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16"])
ap.add_argument("--steps", type=int, default=20); ap.add_argument("--cfg", type=float, default=4.0)
ap.add_argument("--res", type=int, default=512); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--shard", default="0/1", help="k/n: process every n-th image starting at k (one process per GPU)")
args = ap.parse_args()
assert torch.cuda.is_available(), "CUDA GPU required"
dev = "cuda"; dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
R = args.res
out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
files = sorted(Path(args.input).glob("*.png"))
files = files[: args.limit] if args.limit else files
shard_k, shard_n = map(int, args.shard.split("/"))
files = files[shard_k::shard_n]
assert files, f"no png in {args.input}"
todo = [p for p in files if not (out / p.name).exists()]
if not todo:
    print("nothing to do"); raise SystemExit(0)


def free():
    gc.collect(); torch.cuda.empty_cache()


import diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus as m
from diffusers import GGUFQuantizationConfig, QwenImageEditPlusPipeline, QwenImageTransformer2DModel
from huggingface_hub import hf_hub_download
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

m.VAE_IMAGE_SIZE = R * R  # conditioning latents at the output size instead of 1024^2 (far fewer tokens on a T4)
import fcntl  # shards load one at a time: two 10GB GGUF + text-encoder loads at once would exhaust the 29GB of host RAM
load_lock = open("/tmp/qwen_load.lock", "w"); fcntl.flock(load_lock, fcntl.LOCK_EX)

# 1) encode prompt+image with the NF4 text encoder, keep the embeddings on the CPU, drop the encoder
te = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, subfolder="text_encoder", torch_dtype=dtype, device_map={"": 0},
                                                        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype))
pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, transformer=None, vae=None, text_encoder=te, torch_dtype=dtype)
emb = {}
for p in todo:
    img = Image.open(p).convert("RGB").resize((R, R), Image.LANCZOS)
    cw, ch = m.calculate_dimensions(m.CONDITION_IMAGE_SIZE, img.width / img.height)
    cond = [pipe.image_processor.resize(img, ch, cw)]  # same condition image the pipeline builds itself
    with torch.no_grad():
        pe, pm = pipe.encode_prompt(prompt=FRONT, image=cond, device=dev)
        ne, nm = pipe.encode_prompt(prompt=NEG, image=cond, device=dev)
    assert torch.isfinite(pe).all() and torch.isfinite(ne).all(), "non-finite text embeddings (try --dtype bf16)"
    emb[p.name] = tuple(None if x is None else x.cpu() for x in (pe, pm, ne, nm))  # the mask is None when it would be all ones
del pipe, te; free()

# 2) GGUF transformer + VAE only
path = hf_hub_download(GGUF_REPO, args.gguf)
tr = QwenImageTransformer2DModel.from_single_file(path, quantization_config=GGUFQuantizationConfig(compute_dtype=dtype), config=MODEL,
                                                  subfolder="transformer", torch_dtype=dtype).to(dev)
pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL, transformer=tr, text_encoder=None, tokenizer=None, processor=None, torch_dtype=dtype).to(dev)
pipe.set_progress_bar_config(disable=True)
fcntl.flock(load_lock, fcntl.LOCK_UN)
timings = []
for k, p in enumerate(todo, 1):
    img = Image.open(p).convert("RGB").resize((R, R), Image.LANCZOS)
    pe, pm, ne, nm = (None if x is None else x.to(dev) for x in emb[p.name])
    torch.cuda.synchronize(); t0 = time.time()
    with torch.inference_mode():
        res = pipe(image=[img], prompt_embeds=pe.to(dtype), prompt_embeds_mask=pm, negative_prompt_embeds=ne.to(dtype), negative_prompt_embeds_mask=nm,
                   true_cfg_scale=args.cfg, height=R, width=R, num_inference_steps=args.steps, generator=torch.Generator(dev).manual_seed(args.seed))
    im = res.images[0]
    torch.cuda.synchronize(); timings.append(time.time() - t0)
    assert np.asarray(im).max() > 0 and np.asarray(im).std() > 1, f"black / flat output for {p.name} (fp16 overflow? try --dtype bf16)"
    im.save(out / p.name)
    print(f"{p.name}: {timings[-1]:.0f}s ({k}/{len(todo)})", flush=True)
(out / f"params_shard{shard_k}.json").write_text(json.dumps({"model": MODEL, "gguf": f"{GGUF_REPO}/{args.gguf}", "prompt": FRONT, "dtype": args.dtype, "steps": args.steps,
                                             "true_cfg_scale": args.cfg, "res": R, "seed": args.seed, "mean_seconds": float(np.mean(timings))}, indent=2))
print("done")
