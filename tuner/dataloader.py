import torch
from torch.utils.data import Dataset, DataLoader
from pymongo import MongoClient
import gridfs
from PIL import Image
import base64
from io import BytesIO




class DatabaseLoader(Dataset):
    def __init__(self):
        mongo_uri = "mongodb://localhost:27017/"  # Replace with your MongoDB URI
        database_name = "prod-database-scored"  # Replace with your database name
        collection_name = "metadata"  # Replace with your collection name
        self.client = MongoClient(mongo_uri)
        self.db = self.client[database_name]
        self.metadata = self.db[collection_name]    
        query = {"num_ratings": {"$gt": 0}}
        self.documents = list(self.metadata.find(query))
        self.fs = gridfs.GridFS(self.db)

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, idx):
        item = self.documents[idx]
        for i, im_id in enumerate(item["images"]):
            file_data = self.fs.get(im_id)
            file_contents = file_data.read()
            decoded_data = base64.b64decode(file_contents)
            image = Image.open(BytesIO(decoded_data))
            image.save(f"output{i}.png")


# MongoDB connection details
dataset = DatabaseLoader()
print(len(dataset))
print(dataset[0])
# Connect to MongoDB

