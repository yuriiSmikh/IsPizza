import os
import sys
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader


PATH_DATASET = os.path.join(*['dataset', 'pizza_not_pizza'])
PATH_MODEL = os.path.join(*['models', 'pizza.model'])
NUM_EPOCHS = 10
BASE_DIR = os.path.dirname(os.path.abspath(sys.argv[0]))
# Model architecture = resnet50

class Classifier:
    
    def __init__(self, path: str) -> None:
        self.data_path = path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.num_classes = None
        self.data_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Resize images to 224x224
            transforms.ToTensor(),  # Convert images to PyTorch tensors || all the images have to be equal size
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize images
        ])

    @staticmethod
    def read_dataset_to_loader(path: str, transforms) -> tuple[DataLoader, DataLoader]: 

        if not os.path.isdir(path):
            raise NameError("The provided path doesn't exist.")

        dataset = datasets.ImageFolder(root=path, transform=transforms)

        train_size = int(0.8 * len(dataset))  # 80% of the data for training
        test_size = len(dataset) - train_size  # Remaining 20% for validation
        batch_size = 32

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader
    
    @staticmethod
    def inspect_picture_in_loader(loader: DataLoader):
        features, labels = next(iter(loader))  # just to see the first batch
        pil_img = transforms.ToPILImage()(next(iter(features))) # just to see the first in the batch
        label = "pizza" if labels[0] == 1 else "not pizza"
        print(f"Label is: {label}")
        pil_img.show()

    def train_model(self, output_path: str | None = None) -> resnet50:
        if output_path and not os.path.exists(os.path.dirname(output_path)):
            raise ValueError(f"The output path directory `{os.path.dirname(output_path)}` doesn't exist")

        train_loader, _ = Classifier.read_dataset_to_loader(self.data_path, self.data_transform)
        self.num_classes = len(train_loader.dataset.dataset.classes)
        
        # neural network
        model = resnet50(weights=ResNet50_Weights.DEFAULT)  # Example: ResNet-18 pretrained on ImageNet
        model.fc = nn.Linear(model.fc.in_features, self.num_classes)

        # loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        # actual training
        num_epochs = NUM_EPOCHS
        print(f"Device used: {self.device.type}")
        model.to(self.device)

        print("Training loop:")
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * images.size(0)
            
            epoch_loss = running_loss / len(train_loader.dataset)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")
        
        if output_path:
            torch.save(model.state_dict(), output_path)
            print(f"Saved the model to {output_path}")
            # print("Model's state_dict:")
            # for param_tensor in model.state_dict():
            #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
                
        self.model = model
        return model
    
    @staticmethod
    def return_model_from_disc(model_path: str, num_classes: int) -> resnet50:
        if not os.path.exists(model_path):
            raise NameError("The provided path doesn't exist.")
        
        model = resnet50()

        # num_features = model.fc.in_features
        # print(f"features{num_features}, classes: {num_classes}")

        new_fc_weight = torch.randn(2, 2048)
        new_fc_bias = torch.randn(2)

        model.fc.weight.data = new_fc_weight
        model.fc.bias.data = new_fc_bias

        model.load_state_dict(torch.load(model_path))
        model.eval()

        return model
    
    def read_model_form_disc(self, model_path: str) -> None:
        full_path = os.path.join(BASE_DIR, model_path)
        print(f"Reading in the model from {full_path}")
        self.model = Classifier.return_model_from_disc(full_path, 2)


    def validate_model(self):
        _, test_data = Classifier.read_dataset_to_loader(self.data_path, self.data_transform)

        self.model.eval()  # Set the model to evaluation mode
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in test_data:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        print(f"Validation Accuracy: {val_accuracy:.2f}%")
    
    def predict_picture(self, img_path: str) -> str:
        img = Image.open(img_path)
        img_tensor = self.data_transform(img)
        img_tensor = img_tensor.unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        self.model.eval()

        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)  # Apply softmax to get probabilities
            _, predicted = torch.max(probabilities, 1)  # Get the class index with highest probability
            predicted_class = str(predicted.item())

        print(img_path, " prediction: ", predicted_class)

        return "Pizza" if predicted_class == "1" else "Not pizza"

if __name__ == "__main__":
    classifier = Classifier(PATH_DATASET)
    model = classifier.train_model(os.path.join(BASE_DIR, PATH_MODEL))
    
    for file in os.listdir(os.path.join(BASE_DIR, "picture_to_predict")):
        full_path = os.path.join(BASE_DIR, "picture_to_predict", file)
        print(classifier.predict_picture(full_path))
