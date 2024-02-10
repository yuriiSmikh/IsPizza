import os
from PIL import Image
import numpy as np

# import torch
# import torchvision
# import torchvision.transforms as transforms

PATH_DATASET = "../dataset/pizzaGANdata/pizzaGANdata/images"


class Classifier:
    
    def __init__(self, dataset_path: str) -> None:
        self.dataset_path = dataset_path
        
    

    def _read_dataset(self, path: str) -> list: 
        data = []
        labels = []

        if not os.path.isdir(path):
            raise NameError("The provided path doesn't exist.")

        for filename in os.listdir(path):
            print(filename)
            img = Image.open(path + '/' + filename)
            data.append(img)
            img.load()
            img.close()

        return data


model = Classifier(PATH_DATASET)
data = model._read_dataset(PATH_DATASET)
img = data[0]
img.show()


