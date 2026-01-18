import torch
import torchvision
from torch import nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F

# =========================
# 基本配置
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

content_name = "Teaching Building1.jpg"
style_name = "starry_night.jpg"
# content_name = "WestGate.jpg"
# style_name = "candy.jpg"

content_img = Image.open(f"images/contents/{content_name}").convert("RGB")
style_img = Image.open(f"images/styles/{style_name}").convert("RGB")

# 读取图片获取原始尺寸
original_w, original_h = content_img.size

# 设定你想要的最大边长 (例如 800 或 1024)
# 注意：VGG19在 800px 以上非常吃显存
max_size = 1024

scale = max_size / max(original_h, original_w)
h = int(original_h * scale)
w = int(original_w * scale)

image_shape = (h, w) # 自动计算出的保持比例的尺寸
print(f"当前训练分辨率: {image_shape}")

# 写死，不推荐
# image_shape = (600, 1000)

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

# 改进的层选择策略 - 参考neural-style
# relu1_1, relu2_1, relu3_1, relu4_1, relu5_1
style_layers = [1, 6, 11, 20, 29]
# relu4_2
content_layers = [21]

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
    features = x.view(b, c, h * w)
    gram_mat = torch.bmm(features, features.transpose(1, 2))
    return gram_mat / (c * h * w)

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

noise_strength = 0.02
generated = nn.Parameter(
    content_X.clone() + noise_strength * torch.randn_like(content_X)
)

# =========================
# 优化器 + 学习率衰减
# =========================
lr = 0.3
optimizer = torch.optim.Adam([generated], lr=lr)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=200,
    gamma=0.5
)

# 分层风格权重
style_layer_weights = [1.0, 0.8, 0.5, 0.6, 0.5]

content_weight = 10
style_weight = 3e8
tv_weight = 1


# =========================
# 训练
# =========================
num_epochs = 800

for epoch in range(num_epochs):
    optimizer.zero_grad()

    contents_hat, styles_hat = extract_features(generated)

    c_loss = sum(content_loss(h, y) for h, y in zip(contents_hat, contents_Y))
    s_loss = sum(
        w * style_loss(h, y)
        for h, y, w in zip(styles_hat, styles_Y_gram, style_layer_weights)
    )
    t_loss = tv_loss(generated)

    loss = content_weight * c_loss + style_weight * s_loss + tv_weight * t_loss
    loss.backward()
    optimizer.step()
    scheduler.step()

    # 防止像素值爆炸（非常重要）
    with torch.no_grad():
        generated.clamp_(
            (0 - rgb_mean.view(1,3,1,1)) / rgb_std.view(1,3,1,1),
            (1 - rgb_mean.view(1,3,1,1)) / rgb_std.view(1,3,1,1)
        )

    if (epoch + 1) % 100 == 0:
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
out_dir = "outputs/Gatys"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(
    out_dir,
    f"{content_name.split('.')[0]}_{style_name.split('.')[0]}.jpg"
)
result.save(out_path)
print("图片已保存")
