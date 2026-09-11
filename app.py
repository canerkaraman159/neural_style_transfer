import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cpu")


def load_image(image_path, max_size=300, shape=None):
    image = Image.open(image_path).convert("RGB")
    
    if shape is not None:
        size = shape
    else:
        size = max(image.size)
        if size > max_size:
            size = max_size

    in_transform = transforms.Compose([
        transforms.Resize((size, size) if isinstance(size, int) else size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image = in_transform(image)[:3, :, :].unsqueeze(0)
    return image.to(device)


def im_convert(tensor):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image = image + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image = image.clamp(0, 1)
    return image.permute(1, 2, 0).numpy()


def gram_matrix(tensor):
    _, d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    return torch.mm(tensor, tensor.t())


class VGGFeatures(nn.Module):
    def __init__(self):
        super(VGGFeatures, self).__init__()
        self.vgg = models.vgg19(pretrained=True).features[:29].to(device).eval()
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.layers = {
            "0": "conv1_1",
            "5": "conv2_1",
            "10": "conv3_1",
            "19": "conv4_1",
            "21": "conv4_2",  # Sadece icerik katmani
            "28": "conv5_1"
        }
        
    def forward(self, x):
        features = {}
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.layers:
                features[self.layers[name]] = x
        return features


def run_style_transfer(content_img, style_img, steps=300, style_weight=1e6, content_weight=1):
    target = content_img.clone().requires_grad_(True).to(device)
    optimizer = optim.Adam([target], lr=0.03)
    model = VGGFeatures()

    for step in tqdm(range(steps)):
        target_features = model(target)
        content_features = model(content_img)
        style_features = model(style_img)

        # Icerik kaybi yalnizca conv4_2
        content_loss = torch.mean((target_features["conv4_2"] - content_features["conv4_2"])**2)

        # Stil kaybi
        style_loss = 0
        for layer in ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]:
            layer_target_feat = target_features[layer]
            layer_style_feat = style_features[layer]
            target_gram = gram_matrix(layer_target_feat)
            style_gram = gram_matrix(layer_style_feat)
            style_loss += torch.mean((target_gram - style_gram)**2)

        total_loss = content_weight * content_loss + style_weight * style_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step: {step}, total loss: {total_loss.item():.2f}")
            
    return target


# Uygulama (dosya adlarini kendi gorsellerine gore ayarla)
content = load_image("content.jpg")
style = load_image("style.jpg", shape=tuple(content.shape[-2:]))

output = run_style_transfer(content, style, steps=300)

# Sonucu goster
plt.figure(figsize=(8, 8))
plt.imshow(im_convert(output))
plt.title("Stil Transferi Sonucu (conv4_2)")
plt.axis("off")
plt.show()