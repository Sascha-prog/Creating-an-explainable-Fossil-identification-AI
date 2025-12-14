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
image_folder = "dataset/images/"      # folder containing images
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
        self.image_dir = image_dir
        self.preprocess = preprocess
        self.items = list(self.data.items())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_name, text = self.items[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = self.preprocess(Image.open(img_path).convert("RGB"))
        return image, text


# --------------------------
# Load CLIP model
# --------------------------
model, _, preprocess = open_clip.create_model_and_transforms(
    model_name,
    pretrained=pretrained
)

tokenizer = open_clip.get_tokenizer(model_name)

model = model.to(device)

# --------------------------
# Freeze EVERYTHING except last transformer block
# --------------------------
for param in model.parameters():
    param.requires_grad = False

# Vision transformer: unfreeze last block
for param in model.visual.transformer.resblocks[-1].parameters():
    param.requires_grad = True

# Text transformer: unfreeze last block
for param in model.transformer.resblocks[-1].parameters():
    param.requires_grad = True


trainable_params = list(model.visual.transformer.resblocks[-1].parameters()) + \
                   list(model.transformer.resblocks[-1].parameters())

# print("Trainable parameters:", sum(p.numel() for p in trainable_params))
print("Param iteration done!")
# --------------------------
# Dataloader
# --------------------------
dataset = FossilDataset(json_path, image_folder, preprocess)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --------------------------
# Loss + Optimizer
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr
)

# --------------------------
# Training Loop
# --------------------------
for epoch in range(epochs):
    model.train()
    for imgs, texts in loader:
        imgs = imgs.to(device)
        text_tokens = tokenizer(texts).to(device)

        optimizer.zero_grad()

        image_features = model.encode_image(imgs)
        text_features = model.encode_text(text_tokens)

        # Normalize as CLIP expects
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Similarity
        logits = image_features @ text_features.t()

        labels = torch.arange(len(imgs), device=device)

        loss = (criterion(logits, labels) + criterion(logits.t(), labels)) / 2

        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

print("Fine-tuning complete!")
torch.save(model.state_dict(), "clip_finetuned.pt")
