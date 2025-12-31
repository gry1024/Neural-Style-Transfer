import torch
import torchvision
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# =========================
# 基本配置
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

content_name = "WestGate.jpg"
style_name = "starry_night.jpg"

content_img = Image.open(f"images/contents/{content_name}").convert("RGB")
style_img = Image.open(f"images/styles/{style_name}").convert("RGB")

image_shape = (300, 450)

# ImageNet 归一化参数
rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
rgb_std = torch.tensor([0.229, 0.224, 0.225], device=device)

# =========================
# 预处理 / 后处理
# =========================
preprocess = transforms.Compose([
    transforms.Resize(image_shape),
    transforms.ToTensor(),
    transforms.Normalize(mean=rgb_mean.tolist(), std=rgb_std.tolist())
])

def load_image(img):
    return preprocess(img).unsqueeze(0).to(device)

def postprocess(img):
    img = img.squeeze(0).detach().cpu()
    img = img.permute(1, 2, 0)
    img = torch.clamp(img * rgb_std.cpu() + rgb_mean.cpu(), 0, 1)
    return transforms.ToPILImage()(img.permute(2, 0, 1))

# =========================
# VGG19
# =========================
weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
vgg = torchvision.models.vgg19(weights=weights).features.to(device).eval()

for param in vgg.parameters():
    param.requires_grad_(False)

style_layers = [0, 5, 10, 19, 28]
content_layers = [25]

def extract_features(x):
    contents, styles = [], []
    for i, layer in enumerate(vgg):
        x = layer(x)
        if i in style_layers:
            styles.append(x)
        if i in content_layers:
            contents.append(x)
    return contents, styles

# =========================
# 损失函数
# =========================
def content_loss(y_hat, y):
    return torch.mean((y_hat - y.detach()) ** 2)

def gram(x):
    b, c, h, w = x.shape
    x = x.view(c, h * w)
    return torch.matmul(x, x.t()) / (c * h * w)

def style_loss(y_hat, gram_y):
    return torch.mean((gram(y_hat) - gram_y.detach()) ** 2)

def tv_loss(x):
    return 0.5 * (
        torch.mean(torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :])) +
        torch.mean(torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]))
    )

# =========================
# 初始化（含噪声）
# =========================
content_X = load_image(content_img)
style_X = load_image(style_img)

contents_Y, _ = extract_features(content_X)
_, styles_Y = extract_features(style_X)
styles_Y_gram = [gram(y) for y in styles_Y]

noise_strength = 0.1
generated = nn.Parameter(
    content_X.clone() + noise_strength * torch.randn_like(content_X)
)

# =========================
# 优化器 + 学习率衰减（新增）
# =========================
lr = 0.2
optimizer = torch.optim.Adam([generated], lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=100,
    gamma=0.5
)

content_weight = 1
style_weight = 1e4
tv_weight = 10

# =========================
# 训练
# =========================
num_epochs = 500

for epoch in range(num_epochs):
    optimizer.zero_grad()

    contents_hat, styles_hat = extract_features(generated)

    c_loss = sum(content_loss(h, y) for h, y in zip(contents_hat, contents_Y))
    s_loss = sum(style_loss(h, y) for h, y in zip(styles_hat, styles_Y_gram))
    t_loss = tv_loss(generated)

    loss = content_weight * c_loss + style_weight * s_loss + tv_weight * t_loss
    loss.backward()
    optimizer.step()

    scheduler.step()   # ← 新增：学习率衰减

    if (epoch + 1) % 50 == 0:
        print(
            f"Epoch {epoch+1:3d} | "
            f"LR: {scheduler.get_last_lr()[0]:.5f} | "
            f"Content: {c_loss.item():.4f} | "
            f"Style: {s_loss.item():.6f} | "
            f"Total: {loss.item():.4f}"
        )
        plt.imshow(postprocess(generated))
        plt.axis("off")
        plt.show()

# =========================
# 保存结果
# =========================
result = postprocess(generated)
out_dir = "outputs"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(
    out_dir,
    f"{content_name.split('.')[0]}_{style_name.split('.')[0]}.jpg"
)
result.save(out_path)
print("图片已保存")
