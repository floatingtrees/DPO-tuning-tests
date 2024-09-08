import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModel
from pymongo import MongoClient
import gridfs
from PIL import Image
import base64
from io import BytesIO
import os




class DatabaseLoader(Dataset):
    def __init__(self, smoothing_constant):
        mongo_uri = "mongodb://localhost:27017/"  # Replace with your MongoDB URI
        database_name = "prod-database-scored"  # Replace with your database name
        collection_name = "metadata"  # Replace with your collection name
        self.client = MongoClient(mongo_uri)
        self.db = self.client[database_name]
        self.metadata = self.db[collection_name]    
        query = {"num_ratings": {"$gt": 0}}
        self.documents = list(self.metadata.find(query))
        self.fs = gridfs.GridFS(self.db)
        self.smoothing_constant = smoothing_constant
        processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        self.processor = AutoProcessor.from_pretrained(processor_name_or_path)
        total = 0
        for i in self.documents:
            total = total + i["num_ratings"]
        #print(total)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        item = self.documents[idx]
        rating_matrix = item["rating_matrix"]
        prompt = item["prompt"]
        probs = torch.zeros((2, ))
        images = []
        total_ratings = 0
        for i, im_id in enumerate(item["images"]):
            file_data = self.fs.get(im_id)
            file_contents = file_data.read()
            decoded_data = base64.b64decode(file_contents)
            image = Image.open(BytesIO(decoded_data))
            
            victories = 0
            if str(im_id) in rating_matrix:
                losers = rating_matrix[str(im_id)]
                for key in losers.keys():
                    victories += int(losers[key])
            probs[i] = victories + self.smoothing_constant
            images.append(image)
        probs = probs / torch.sum(probs)
        if (torch.any(torch.isnan(probs))):
            print(rating_matrix)
            exit()
        image_inputs = self.processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
        )
        prompt_inputs = self.processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    )

        inputs = (images, prompt)
        
        return(inputs, probs)


if __name__ == "__main__":
    dataset = DatabaseLoader(0)
    for i, element in enumerate(dataset):
        path = f"img_folder/folder{i}"
        if not os.path.exists(path):
            os.makedirs(path)
        x, y = element
        images, prompt = x
        images[0].save(f"{path}/im0.jpg")
        images[1].save(f"{path}/im1.jpg")
        with open(f"{path}/prompt.txt", "w") as f:
            f.write(prompt)
        with open(f"{path}/results.txt", "w") as f:
            f.write(repr(y))
        if i == 100:
            exit()
