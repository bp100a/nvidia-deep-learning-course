"""5.0 Pre-Trained Models (doggy door)"""

import torch
from torch import Tensor
import torchvision.transforms.v2 as transforms
import torchvision.io as tv_io
import torchvision.transforms.functional as F

import json

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.is_available()

from torchvision.models import vgg16
from torchvision.models import VGG16_Weights

# load the VGG16 network *pre-trained* on the ImageNet dataset
weights = VGG16_Weights.DEFAULT
model = vgg16(weights=weights)

print(f"{model.to(device)=}")

pre_trans = weights.transforms()
print(f"{pre_trans=}")

# This is equivalent to the following:
# IMG_WIDTH, IMG_HEIGHT = (224, 224)
#
# pre_trans = transforms.Compose([
#     transforms.ToDtype(torch.float32, scale=True), # Converts [0, 255] to [0, 1]
#     transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
#     transforms.CenterCrop(224)
# ])

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def show_image(image_path):
    """show the image"""
    image = mpimg.imread(image_path)
    plt.imshow(image)
    plt.show()


show_image("data/doggy_door_images/happy_dog.jpg")


def load_and_process_image(file_path) -> Tensor:
    # Print image's original shape, for reference
    print('Original image shape: ', mpimg.imread(file_path).shape)

    image: Tensor = tv_io.read_image(file_path).to(device)
    image = pre_trans(image)  # weights.transforms()
    image = image.unsqueeze(0)  # Turn into a batch
    return image  # this is the output classification


processed_image: Tensor = load_and_process_image("data/doggy_door_images/happy_dog.jpg")
print(f"Processed image shape: {processed_image.shape=}")

plot_image = F.to_pil_image(torch.squeeze(processed_image))
plt.imshow(plot_image, cmap='gray')
plt.show()

# 5a.5 - make a prediction
vgg_classes: dict = json.load(open("data/imagenet_class_index.json"))
print(f"{vgg_classes=}")


def readable_prediction(image_path) -> torch.topk:
    # Show image
    show_image(image_path)
    # Load and pre-process image
    image = load_and_process_image(image_path)
    # Make predictions
    output = model(image)[0]  # unbatch
    predictions = torch.topk(output, 3)
    indices = predictions.indices.tolist()
    # Print predictions in readable form
    out_str = "Top results: "
    pred_classes = [vgg_classes[str(idx)][1] for idx in indices]
    out_str += ", ".join(pred_classes)
    print(out_str)

    return predictions


readable_prediction("data/doggy_door_images/happy_dog.jpg")
readable_prediction("data/doggy_door_images/brown_bear.jpg")
readable_prediction("data/doggy_door_images/sleepy_cat.jpg")


# 5a.6 Only Dogs
def doggy_door(image_path):
    show_image(image_path)
    image: Tensor = load_and_process_image(image_path)
    # from the classification, grab the most correct prediction
    idx: int = model(image).argmax(dim=1).item()
    print(f"Predicted index: {idx=}")
    if 151 <= idx <= 268:
        print("Doggy come on in!")
    elif 281 <= idx <= 285:
        print("Kitty stay inside!")
    else:
        print("You're not a dog! Stay outside!")


doggy_door("data/doggy_door_images/brown_bear.jpg")
doggy_door("data/doggy_door_images/happy_dog.jpg")
doggy_door("data/doggy_door_images/sleepy_cat.jpg")
