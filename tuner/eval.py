# import
from transformers import AutoProcessor, AutoModel
from torch.utils.data import Dataset, DataLoader
import torch
from PIL import Image
from datasets import load_dataset
import json
import base64
from io import BytesIO

device = "cuda"
processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)
model = AutoModel.from_pretrained(model_pretrained_name_or_path).eval().to(device)
model.load_state_dict(torch.load('model_state_dict.pth'))

def calc_probs(prompt, images):
    
    # preprocess
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)
    
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)


    with torch.no_grad():
        # embed
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(image_embs, dim=-1, keepdim=True)
    
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(text_embs, dim=-1, keepdim=True)
    
        # score
        scores = model.logit_scale.exp() * (text_embs @ image_embs.T)[0]
        
        # get probabilities if you have multiple images to choose from
        probs = torch.softmax(scores, dim=-1)
    
    return probs.cpu().tolist()

ds = load_dataset("yuvalkirstain/pickapic_v1", split="test", streaming = True)


count = 0
accuracy = 0
for i, sample in enumerate(ds.take(1000)):  # Only loads 10 examples at a time

    if sample["has_label"]:

        
        im0 = Image.open(BytesIO(sample["jpg_0"]))
        im1 = Image.open(BytesIO(sample["jpg_1"]))
        if (sample["best_image_uid"] == sample["image_0_uid"]):
            index = 0 
        elif sample["best_image_uid"] == sample["image_1_uid"]:
            index = 1
        else:
            raise ValueError("Image UID not found")
        images = [im0, im1]
        prompt = sample["caption"]
        probs = calc_probs(prompt, images)
        count += 1
        if probs[index] > 0.5:
            accuracy += 1
    else:
        print(i)
print(accuracy / count)
print(accuracy, count)

