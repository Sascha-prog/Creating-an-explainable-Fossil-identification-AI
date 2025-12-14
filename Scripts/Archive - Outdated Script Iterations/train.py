import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import open_clip
import os

# --------------------------
# Config
# --------------------------
json_path = "dataset/dataset.json"
image_folder = "dataset/images/"
model_name = "ViT-B-32"
pretrained = "laion2b_s34b_b79k"
batch_size = 8
epochs = 5
lr = 1e-5
device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------
# Dataset
# --------------------------
class FossilDataset(Dataset):
    def __init__(self, json_file, image_dir, preprocess):
        self.data = json.load(open(json_file))
        self.items = list(self.data.items())
        self.image_dir = image_dir
        self.preprocess = preprocess

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_name, text = self.items[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = self.preprocess(Image.open(img_path).convert("RGB"))
        return image, text

# --------------------------
# Load CLIP
# --------------------------
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name,
    pretrained=pretrained
)

tokenizer = open_clip.get_tokenizer(model_name)
model = model.to(device)

# --------------------------
# Freeze EVERYTHING
# --------------------------
for p in model.parameters():
    p.requires_grad = False
    
# Unfreeze ALL layer norms in visual and text tower
for name, param in model.named_parameters():
    if "ln" in name.lower() or "layernorm" in name.lower():
        param.requires_grad = True

# --------------------------
# Unfreeze last VISION block
# --------------------------
''''
vision_blocks = model.visual.transformer.resblocks
for p in vision_blocks[-1].parameters():
    p.requires_grad = True
'''
vision_blocks = model.visual.transformer.resblocks
for block in vision_blocks[-3:]:  # last 3 blocks
    for p in block.parameters():
        p.requires_grad = True
# --------------------------
# Unfreeze last TEXT block
# --------------------------
'''
text_blocks = model.transformer.resblocks
for p in text_blocks[-1].parameters():
    p.requires_grad = True
'''
text_blocks = model.transformer.resblocks
for block in text_blocks[-3:]:  # last 3 blocks
    for p in block.parameters():
        p.requires_grad = True


# Make logit scale trainable (important!)
model.logit_scale.requires_grad = True

print("Trainable parameters:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))

# --------------------------
# Dataloader
# --------------------------
def collate(batch):
    images = torch.stack([item[0] for item in batch])
    texts = [item[1] for item in batch]
    return images, texts

dataset = FossilDataset(json_path, image_folder, preprocess)
#loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate,
    drop_last=True
)
# --------------------------
# Loss & Optimizer
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr
)

print("Dataset length:", len(dataset))
# --------------------------
# Training
# --------------------------
for epoch in range(epochs):
    model.train()
    for imgs, texts in loader:
        imgs = imgs.to(device)
       # texts is a list of strings
        text_tokens = tokenizer(texts).to(device)
        
        optimizer.zero_grad()
        # Encode features
        image_features = model.encode_image(imgs)
        text_features = model.encode_text(text_tokens)
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features  = text_features  / text_features.norm(dim=-1, keepdim=True)
        # CLIP-style scaled similarity
        logits = model.logit_scale.exp() * image_features @ text_features.t()
        labels = torch.arange(len(imgs), device=device)
        loss = (criterion(logits, labels) + criterion(logits.t(), labels)) / 2
        #print("Batch loss:", loss.item())
        loss.backward()
        optimizer.step()
        #print(len(imgs), len(texts))
    #print("Batch shape:", imgs.shape, "Texts:", len(texts))
    #break
    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "finetuned_clip.pt")
print("Training complete!")
