import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import colorsys
# from transformers import AutoFeatureExtractor, CLIPSegModel # this currently does not work, will fix soon

# generates colormap
def generate_colormap(num_classes):
    hues = np.linspace(0, 1, num_classes)
    saturation = value = 0.9
    colors = [colorsys.hsv_to_rgb(hue, saturation, value) for hue in hues]

    # colors in the 0-255 range
    colormap = (np.array(colors) * 255).astype(np.uint8)

    return colormap

# load model from huggingface
# model_name = "CIDAS/clipseg-rd64-refined"
# model = CLIPSegModel.from_pretrained(model_name)
# tokenizer = AutoFeatureExtractor.from_pretrained(model_name)

model = torch.load("isnetis.ckpt") # the anime-segmentation model by skytnt
model.eval()

# loads image using cv2
input_image = cv2.imread('images/trimmed1_0.jpg')
input_tensor = torch.from_numpy(input_image)

# preprocess = transforms.Compose([
  #  transforms.ToTensor(),
  #  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)

masks = outputs.logits

# dummy num_classes as test
num_classes = 30

colormap = generate_colormap(num_classes)

# creates colored image
segmentation_map = np.zeros((masks.shape[2], masks.shape[3], 3), dtype = np.uint8)
for class_idx in range(masks.shape[1]):
    class_mask = masks[0, class_idx].cpu().numpy()
    color = colormap[class_idx]
    segmentation_map[class_mask > 0] = color

# saves file
cv2.imwrite('segmentation_map.jpg', segmentation_map)
