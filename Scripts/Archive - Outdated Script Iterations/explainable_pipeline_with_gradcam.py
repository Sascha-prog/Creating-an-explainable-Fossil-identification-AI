# explainable_pipeline_with_gradcam.py
import math
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import open_clip
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    AutoTokenizer, AutoModelForCausalLM
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Model loading
# -----------------------------
print("Loading models...")
# 1) OpenCLIP ViT (LAION checkpoint). Change name if you prefer another checkpoint.
clip_model, _, preprocess_clip = open_clip.create_model_and_transforms(
    "ViT-L-14", pretrained="laion2b_s32b_b82k"
)
clip_model = clip_model.to(device).eval()

# tokenizer helper from open_clip
tokenize = open_clip.tokenize

# 2) BLIP-2 captioner
blip_model_name = "Salesforce/blip2-flan-t5-xl"
blip_processor = Blip2Processor.from_pretrained(blip_model_name)
blip_model = Blip2ForConditionalGeneration.from_pretrained(
    blip_model_name, torch_dtype=torch.float16
).to(device).eval()

# 3) Bilingual LLM (example). Swap for a smaller or different bilingual model if needed.
# NOTE: large LLMs may need HF auth tokens and ample GPU memory.
llm_name = "meta-llama/Llama-2-7b-chat-hf"
llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
llm = AutoModelForCausalLM.from_pretrained(llm_name, torch_dtype=torch.float16).to(device).eval()

print("Models loaded.")

# -----------------------------
# Utility: register hooks to capture activations + grads
# -----------------------------
class ViTGradCAM:
    def __init__(self, model):
        self.model = model
        self.activations = None
        self.gradients = None
        self.handle_fwd = None

        # Try to locate the last transformer block for ViT in open_clip models.
        # Common path: model.visual.transformer.resblocks[-1]
        # Fallbacks may be needed for other implementations.
        try:
            self.target_module = model.visual.transformer.resblocks[-1]
        except Exception:
            # Try another common path
            try:
                self.target_module = model.visual.transformer.blocks[-1]
            except Exception:
                raise RuntimeError("Unable to find ViT transformer blocks. Inspect the clip model structure.")

    def _forward_hook(self, module, inp, out):
        # out: tensor of shape (B, N+1, D) typically (cls token + patches)
        # Retain grad so we can access grad later.
        self.activations = out
        out.register_hook(self._backward_hook)

    def _backward_hook(self, grad):
        # grad will have same shape as activations: (B, N+1, D)
        self.gradients = grad

    def __enter__(self):
        # attach forward hook
        self.handle_fwd = self.target_module.register_forward_hook(self._forward_hook)
        return self

    def __exit__(self, exc_type, exc, tb):
        # remove hook
        if self.handle_fwd is not None:
            self.handle_fwd.remove()
            self.handle_fwd = None

    def compute_heatmap(self, upsample_to=None):
        """
        After a backward pass has computed gradients, create heatmap:
          - use activations & gradients from the target module
          - exclude CLS token (index 0), use patch tokens [1:]
          - combine activations and gradients: cam_token = sum_c (act * grad) per token
        Returns: heatmap (H_p x W_p) normalized (0..1). If upsample_to specified (W,H) it resizes to that.
        """
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Activations or gradients not found. Ensure forward+backward pass ran with hooks attached.")

        acts = self.activations.detach()      # (B, N+1, D)
        grads = self.gradients.detach()       # (B, N+1, D)

        # Use first image in batch
        acts = acts[0]   # (N+1, D)
        grads = grads[0] # (N+1, D)

        # Exclude CLS token at index 0
        patch_acts = acts[1:, :]   # (N, D)
        patch_grads = grads[1:, :] # (N, D)

        # Classic Grad-CAM style combination per token:
        # cam_token = sum_c (patch_acts * patch_grads) over channels
        cam_per_patch = (patch_acts * patch_grads).sum(dim=1)  # (N,)

        cam_np = cam_per_patch.cpu().numpy()

        # Normalize and reshape to square grid (assumes square patch grid)
        n_patches = cam_np.shape[0]
        side = int(math.sqrt(n_patches))
        if side * side != n_patches:
            # fallback: attempt nearest square reshape (will truncate / pad if necessary)
            side = int(math.sqrt(n_patches))
            cam_np = cam_np[: side * side]
        heatmap = cam_np.reshape(side, side)
        # Normalize 0..1
        heatmap = heatmap - heatmap.min()
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        heatmap = np.clip(heatmap, 0.0, 1.0)

        if upsample_to is not None:
            # Upsample using PIL for simplicity
            from PIL import Image
            im = Image.fromarray((heatmap * 255).astype(np.uint8))
            im = im.resize(upsample_to, resample=Image.BICUBIC)
            heatmap = np.array(im).astype(np.float32) / 255.0

        return heatmap

# -----------------------------
# Helper: zero-shot CLIP label scoring (get top-k labels)
# -----------------------------
def clip_zero_shot_topk(clip_model, image_tensor, candidate_texts, k=3):
    """
    image_tensor: (1,3,H,W) preprocessed by open_clip transforms (tensor)
    candidate_texts: list[str]
    returns top_k list of dicts [{"label":..., "score":...}, ...], and the text_features used
    """
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor.to(device))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        tokenized = tokenize(candidate_texts).to(device)
        text_features = clip_model.encode_text(tokenized)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # similarity (1,K)
        sim = (image_features @ text_features.T).squeeze(0).cpu().numpy()
        # convert to softmax scores for nicer display
        scores = (torch.tensor(sim)).softmax(dim=0).numpy()
        topk_idxs = np.argsort(-sim)[:k]
        topk = [{"label": candidate_texts[i], "score": float(scores[i])} for i in topk_idxs]
        return topk, tokenized, text_features

# -----------------------------
# Overlay helper
# -----------------------------
def overlay_heatmap_on_image(image_pil, heatmap, alpha=0.4, cmap="jet"):
    import matplotlib.cm as cm
    image_np = np.array(image_pil).astype(np.float32) / 255.0
    # heatmap assumed 0..1 same HxW as image
    # If heatmap is smaller, caller should upsample first
    if heatmap.shape[:2] != image_np.shape[:2]:
        from PIL import Image
        im = Image.fromarray((heatmap * 255).astype(np.uint8))
        im = im.resize((image_np.shape[1], image_np.shape[0]), resample=Image.BICUBIC)
        heatmap = np.array(im).astype(np.float32) / 255.0

    cmap_fn = cm.get_cmap(cmap)
    colored = cmap_fn(heatmap)[:, :, :3]  # drop alpha
    overlay = (1 - alpha) * image_np + alpha * colored
    overlay = np.clip(overlay, 0, 1)
    return (overlay * 255).astype(np.uint8)

# -----------------------------
# Main pipeline
# -----------------------------
def explain_image(image_path, candidate_texts=None, lang="en"):
    """
    End-to-end: CLIP zero-shot -> Grad-CAM on top text label -> BLIP-2 caption -> LLM explanation
    """
    if candidate_texts is None:
        # Example class labels for zero-shot probing; replace with your label set
        candidate_texts = ["a photo of a giraffe", "a photo of an antelope", "a photo of a horse", "a photo of a cow"]

    # load image
    img = Image.open(image_path).convert("RGB")
    # preprocess for clip
    image_tensor = preprocess_clip(img).unsqueeze(0).to(device)

    # 1) CLIP zero-shot top-k
    topk, tokenized_texts, text_features = clip_zero_shot_topk(clip_model, image_tensor, candidate_texts, k=3)
    top_label = topk[0]["label"]
    print("Top CLIP predictions:", topk)

    # 2) Run forward within the Grad-CAM hook context so we capture activations
    with ViTGradCAM(clip_model) as gcam:
        # Important: we need gradients, so do NOT use torch.no_grad here.
        # Compute CLIP image & text features (without no_grad so grad flows)
        image_features = clip_model.encode_image(image_tensor)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # encode the candidate texts again but keep tokenized_texts and both on device
        # find index of chosen label in candidate_texts (we used same list above)
        text_features_all = clip_model.encode_text(tokenized_texts.to(device))
        text_features_all = text_features_all / text_features_all.norm(dim=-1, keepdim=True)

        # compute similarity vector
        sims = (image_features @ text_features_all.T).squeeze(0)  # (K,)
        # choose index of the top label (we used topk earlier, but recompute index)
        top_idx = int(torch.argmax(sims).cpu().numpy())
        target_score = sims[top_idx]

        # backward on target_score to populate gradients captured by the hook
        clip_model.zero_grad()
        # If target_score is a softmaxed value may be small; directly backward on similarity scalar is fine.
        target_score.backward(retain_graph=True)

        # compute heatmap upsampled to image size
        heatmap = gcam.compute_heatmap(upsample_to=(img.width, img.height))

    # 3) BLIP-2 caption
    blip_inputs = blip_processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        caption_ids = blip_model.generate(**blip_inputs, max_new_tokens=40)
        caption = blip_processor.decode(caption_ids[0], skip_special_tokens=True)
    print("BLIP-2 caption:", caption)

    # 4) Build structured prompt and ask bilingual LLM for explanation
    system_prompt = f"You are an assistant that explains image model predictions. Reply in {lang}. Use only the evidence provided."
    user_prompt = (
        f"caption: {caption}\n"
        f"predictions: {topk}\n"
        f"saliency_grid_excerpt: top regions shown via heatmap\n"
        "request: Explain briefly why the model predicted the top label and mention visual evidence (<=3 sentences)."
    )
    full_prompt = f"[SYSTEM]\n{system_prompt}\n\n[USER]\n{user_prompt}"

    # generate explanation (note: LLMs may require auth / large GPU)
    llm_inputs = llm_tokenizer(full_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        llm_out = llm.generate(**llm_inputs, max_new_tokens=100)
    explanation = llm_tokenizer.decode(llm_out[0], skip_special_tokens=True)
    print("\nLLM explanation:\n", explanation)

    # 5) Overlay & show
    over = overlay_heatmap_on_image(img, heatmap, alpha=0.45)
    plt.figure(figsize=(10, 8))
    plt.imshow(over)
    plt.axis("off")
    plt.title(f"Top: {topk[0]['label']}  — Explanation: {topk[0]['score']:.2f}")
    plt.show()

    # Return structured result
    return {
        "topk": topk,
        "caption": caption,
        "heatmap": heatmap,  # np array HxW (upsampled)
        "explanation": explanation
    }

# -----------------------------
# Run quick demo
# -----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image file")
    parser.add_argument("--lang", type=str, default="en", help="Output language: en or nl")
    args = parser.parse_args()

    res = explain_image(args.image, lang=args.lang)
    # print summary
    print("\nResult keys:", list(res.keys()))
