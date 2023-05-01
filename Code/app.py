import os

import albumentations as A
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from PIL import Image
from sklearn import metrics
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dataset(Dataset):
    def __init__(self, root_images, root_file, transform=None):
        self.root_images = root_images
        self.root_file = root_file
        self.transform = transform
        self.file = pd.read_csv(root_file)

    def __len__(self):
        return self.file.shape[0]

    def __getitem__(self, index):
        img_path = os.path.join(self.root_images, self.file["id"][index])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image


learning_rate = 0.0001
batch_size = 32
epochs = 10
height = 224
width = 224
IMG = "AI images or Not/test"
FILE = "Data/sample_submission.csv"


def get_loader(image, file, batch_size, test_transform):

    test_ds = Dataset(image, file, test_transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return test_loader


normalize = A.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.255], max_pixel_value=255.0
)


test_transform = A.Compose(
    [A.Resize(width=width, height=height), normalize, ToTensorV2()]
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = EfficientNet.from_pretrained("efficientnet-b4")
        self.fct = nn.Linear(1000, 1)

    def forward(self, img):
        x = self.model(img)
        # print(x.shape)
        x = self.fct(x)
        return x


def load_checkpoint(checkpoint, model, optimizer):
    print("====> Loading...")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


# test = pd.read_csv(FILE)
# test

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

checkpoint_file = "Checkpoint/baseline_V0.pth.tar"
test_loader = get_loader(IMG, FILE, batch_size, test_transform)
checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))
load_checkpoint(checkpoint, model, optimizer)

model.eval()


# define the predict function
def predict(image):
    # preprocess the image
    image = np.array(image)
    image = test_transform(image=image)["image"]
    image = image.unsqueeze(0).to(device)

    # get the model prediction
    with torch.no_grad():
        output = model(image)
        pred = torch.sigmoid(output).cpu().numpy().squeeze()

    # check if prediction is AI generated, not AI generated, or uncertain
    if pred >= 0.6:
        prediction = "AI generated"
        confidence = pred
    elif pred <= 0.4:
        prediction = "NOT AI generated"
        confidence = 1 - pred
    else:
        prediction = "uncertain"
        confidence = abs(0.5 - pred) * 2

    # return the prediction and confidence as a string
    return f"This image is {prediction} with {confidence:.2%} confidence."


# define the input interface with examples
inputs = gr.inputs.Image(shape=(224, 224))
outputs = gr.outputs.Textbox()
examples = [
    ["Data/train/3.jpg"],
    ["Data/train/10.jpg"],
    ["Data/train/14.jpg"],
    ["Data/train/4515.jpg"],
    ["Data/train/4518.jpg"],
    ["Data/train/6122.jpg"],
    ["Data/train/6123.jpg"],
    ["Data/train/6124.jpg"],
    ["Data/train/6125.jpg"],
    ["Data/train/7461.jpg"],
    ["Data/train/7462.jpg"],
    ["Data/train/7463.jpg"],
    ["Data/train/7464.jpg"],
    ["Data/train/7465.jpg"],
    ["Data/train/8546.jpg"],
    ["Data/train/8543.jpg"],
    ["Data/train/9120.jpg"],
    ["Data/train/10120.jpg"],
]
iface = gr.Interface(
    fn=predict,
    inputs=inputs,
    outputs=outputs,
    title="AI image detector 🔎",
    description="Check if an image is AI generated or real.",
    examples=examples,
)

# launch the gradio app
iface.launch()
