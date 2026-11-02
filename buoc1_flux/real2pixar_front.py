"""Stage A+B of real2pixar: real photo -> front-facing photo (FLUX Kontext zero-shot) -> Pixar (FLUX Kontext + LoRA-100, constant prompt 'zqx style').

  python real2pixar_front.py --input <dir of <id>.png> --output <out dir> --ckpt trig100.pt [--limit N]
Both passes share one NF4 transformer: pass 1 without LoRA, then the LoRA state is loaded in place for pass 2. Resumable (skips existing pngs).
Writes <out>/front/<id>.png, <out>/pixar/<id>.png, <out>/grid.jpg (original | front | pixar) and <out>/params.json.
"""
import argparse, copy, gc, json, time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
TARGETS = ["to_q", "to_k", "to_v", "to_out.0", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"]  # same as flux_kontext.py (rank 16 LoRA)
FRONT = ("Rotate the head so the person faces the camera directly: a straight front-facing portrait, head upright, both eyes looking at the camera, "
         "symmetric pose. Keep the same person: identity, face shape, hair, skin tone, expression, clothing and background.")
PIXAR = "zqx style"

ap = argparse.ArgumentParser()
ap.add_argument("--input"); ap.add_argument("--output"); ap.add_argument("--ckpt")
ap.add_argument("--front_in", help="dir of already front-facing <id>.png (e.g. the Qwen output): skips pass 1 and runs only the Pixar pass on them")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--res", type=int, default=512); ap.add_argument("--steps", type=int, default=28)
ap.add_argument("--guidance", type=float, default=2.5)
args = ap.parse_args()
assert torch.cuda.is_available(), "CUDA GPU required"
dev = "cuda"; dtype = torch.float16  # T4
R = args.res
out = Path(args.output)
for d in ("front", "pixar"):
    (out / d).mkdir(parents=True, exist_ok=True)
files = sorted(Path(args.input).glob("*.png"))
files = files[: args.limit] if args.limit else files
assert files, f"no png in {args.input}"
front_dir = Path(args.front_in) if args.front_in else out / "front"
if args.front_in:
    missing = [p.name for p in files if not (front_dir / p.name).exists()]
    assert not missing, f"no front-facing image in {front_dir} for {missing[:5]}"


def free():
    gc.collect(); torch.cuda.empty_cache()


from diffusers import BitsAndBytesConfig, FluxKontextPipeline, FluxTransformer2DModel
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict

pipe = FluxKontextPipeline.from_pretrained(MODEL, transformer=None, vae=None, torch_dtype=dtype).to(dev)
emb = {}
with torch.no_grad():
    for k, t in (("front", FRONT), ("pixar", PIXAR)):
        pe, pooled, _ = pipe.encode_prompt(t, prompt_2=None, device=dev)
        emb[k] = (pe.to(dtype), pooled.to(dtype))
del pipe; free()
tr = FluxTransformer2DModel.from_pretrained(MODEL, subfolder="transformer", torch_dtype=dtype, quantization_config=BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)).to(dev)
pipe = FluxKontextPipeline.from_pretrained(MODEL, transformer=tr, text_encoder=None, text_encoder_2=None, tokenizer=None,
                                           tokenizer_2=None, torch_dtype=dtype).to(dev)
pipe.set_progress_bar_config(disable=True)
vae32 = copy.deepcopy(pipe.vae).float()  # fp16 VAE encodes inside the pipeline, fp32 copy decodes


def edit(img, key):
    pe, pooled = emb[key]
    g = torch.Generator(dev).manual_seed(args.seed)
    with torch.autocast("cuda", dtype=dtype):
        lat = pipe(image=img, prompt_embeds=pe, pooled_prompt_embeds=pooled, height=R, width=R, guidance_scale=args.guidance,
                   num_inference_steps=args.steps, generator=g, _auto_resize=False, max_area=R * R, output_type="latent").images
    assert torch.isfinite(lat).all(), "non-finite latents (fp16 overflow)"
    z = pipe._unpack_latents(lat.float(), R, R, pipe.vae_scale_factor) / vae32.config.scaling_factor + vae32.config.shift_factor
    with torch.no_grad():
        px = vae32.decode(z, return_dict=False)[0]
    im = pipe.image_processor.postprocess(px, output_type="pil")[0]
    assert np.asarray(im).max() > 0, "all-black output"
    return im


timings = {"front": [], "pixar": []}
for i, p in enumerate(files, 1):  # pass 1: frontalise (base model, no LoRA)
    if args.front_in or (out / "front" / p.name).exists():
        continue
    t0 = time.time()
    edit(Image.open(p).convert("RGB").resize((R, R), Image.LANCZOS), "front").save(out / "front" / p.name)
    timings["front"].append(time.time() - t0); print(f"front {p.name}: {timings['front'][-1]:.0f}s ({i}/{len(files)})", flush=True)

tr.requires_grad_(False)
tr.add_adapter(LoraConfig(r=16, lora_alpha=16, init_lora_weights="gaussian", target_modules=TARGETS))
set_peft_model_state_dict(tr, torch.load(args.ckpt, map_location="cpu")["state"])
tr.eval()
for i, p in enumerate(files, 1):  # pass 2: Pixar with the zqx LoRA, on the frontalised image
    if (out / "pixar" / p.name).exists():
        continue
    t0 = time.time()
    edit(Image.open(front_dir / p.name).convert("RGB").resize((R, R), Image.LANCZOS), "pixar").save(out / "pixar" / p.name)
    timings["pixar"].append(time.time() - t0); print(f"pixar {p.name}: {timings['pixar'][-1]:.0f}s ({i}/{len(files)})", flush=True)

sheet = Image.new("RGB", (256 * 3, 256 * len(files)), "white")
for i, p in enumerate(files):
    for j, im in enumerate((Image.open(p).convert("RGB"), Image.open(front_dir / p.name), Image.open(out / "pixar" / p.name))):
        sheet.paste(im.resize((256, 256)), (256 * j, 256 * i))
    ImageDraw.Draw(sheet).text((4, 256 * i + 4), p.stem, fill=(255, 255, 0))
sheet.save(out / "grid.jpg", quality=90)  # columns: original | front-facing | Pixar zqx
(out / "params.json").write_text(json.dumps({"model": MODEL, "front_prompt": FRONT, "pixar_prompt": PIXAR, "ckpt": str(args.ckpt), "seed": args.seed,
                                             "steps": args.steps, "guidance": args.guidance, "res": R,
                                             "mean_seconds": {k: float(np.mean(v)) if v else 0.0 for k, v in timings.items()}}, indent=2))
print("done")
