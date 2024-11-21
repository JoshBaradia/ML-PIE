import os
import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
# from enhancementnet_model_mse import *
from UFPN import *
import torch.optim as optim
from torchvision.transforms import ToTensor
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from torch.utils.data import Dataset
from skimage import metrics
# from UCIQE import *

# Define the transform
transform = ToTensor()  # Remove the Resize transform

# Load the model
model = EnhancementNet()
# model.load_state_dict(torch.load('model_5.pth'))  # Replace 'model.pth' with your actual model path
# model.load_state_dict(torch.load('model_30.pth', map_location=torch.device('cpu')))
checkpoint = torch.load('checkpoint_trial_37.pth', map_location=torch.device('mps'))
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# Directory containing input images
input_dir = 'testrun_enet'  # Replace with your actual input directory

# Directory for saving output images
output_dir = 'inputs'  # Replace with your actual output directory

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process all images in the input directory
for filename in os.listdir(input_dir):
    # Load the image
    input_image = Image.open(os.path.join(input_dir, filename))
    input_tensor = transform(input_image).unsqueeze(0)

    # Get the original dimensions of the image
    original_width, original_height = input_image.size

    # Enhance the image
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Resize the output tensor to the original dimensions
    output_tensor = torch.nn.functional.interpolate(output_tensor, size=(original_height, original_width))

    # Convert the output tensor to an image
    output_image = ToPILImage()(output_tensor.squeeze(0))

    # Save the enhanced image
    output_image.save(os.path.join(output_dir,(filename)))