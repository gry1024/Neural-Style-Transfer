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

# 路径保持一致
content_path = f"images/contents/{content_name}"
style_path = f"images/styles/{style_name}"

content_img_pil = Image.open(content_path).convert("RGB")
style_img_pil = Image.open(style_path).convert("RGB")

# 读取图片获取原始尺寸
original_w, original_h = content_img_pil.size

# 设定最终分辨率的长边限制
max_target_size = 1024

scale = max_target_size / max(original_h, original_w)
target_h = int(original_h * scale)
target_w = int(original_w * scale)

target_shape = (target_h, target_w)
# 草稿尺寸直接除以 2
draft_shape = (target_h // 2, target_w // 2)

print(f"最终分辨率: {target_shape}, 草稿分辨率: {draft_shape}")

# 写死，不推荐
# # 最终目标尺寸
# target_shape = (300, 500)
# # 草稿阶段尺寸 (低分辨率，例如缩小一倍)
# draft_shape = (150, 250)

# ImageNet 归一化参数
rgb_mean = torch.tensor([0.485, 0.456, 0.406], device=device)
rgb_std = torch.tensor([0.229, 0.224, 0.225], device=device)


# =========================
# 工具函数
# =========================
def get_transform(shape):
    return transforms.Compose([
        transforms.Resize(shape),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean.tolist(), std=rgb_std.tolist())
    ])


def load_image(img, shape):
    tr = get_transform(shape)
    return tr(img).unsqueeze(0).to(device)


def postprocess(img):
    img = img.squeeze(0).detach().cpu()
    img = img.permute(1, 2, 0)
    img = torch.clamp(img * rgb_std.cpu() + rgb_mean.cpu(), 0, 1)
    return transforms.ToPILImage()(img.permute(2, 0, 1))


# =========================
# LapStyle 核心组件: 拉普拉斯算子
# =========================
def calc_laplacian(img):
    """
    计算图像的拉普拉斯边缘图。
    用于在 Revision 阶段强制保留内容图的结构细节。
    """
    # 定义拉普拉斯卷积核
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=device)
    # 扩展为 (Output_Channels, Input_Channels/Groups, H, W)
    # 我们对 RGB 每个通道独立计算，所以 groups=3
    kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

    # 使用 padding=1 保持尺寸不变
    return F.conv2d(img, kernel, padding=1, groups=3)


# =========================
# VGG19 模型
# =========================
weights = torchvision.models.VGG19_Weights.IMAGENET1K_V1
vgg = torchvision.models.vgg19(weights=weights).features.to(device).eval()

for param in vgg.parameters():
    param.requires_grad_(False)

style_layers = [0, 5, 10, 19, 28]
content_layers = [25]  # relu4_2


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


def laplacian_loss(y_hat, y_lap_target):
    """
    计算生成图的拉普拉斯边缘与目标（内容图）边缘的差异
    """
    lap_hat = calc_laplacian(y_hat)
    return torch.mean((lap_hat - y_lap_target.detach()) ** 2)


# =========================
# 训练通用函数
# =========================
def run_style_transfer(stage_name, image_shape, num_epochs, init_img=None, use_lap_loss=False):
    print(f"\n=== 开始阶段: {stage_name} (Size: {image_shape}) ===")

    # 1. 准备该阶段的数据
    content_X = load_image(content_img_pil, image_shape)
    style_X = load_image(style_img_pil, image_shape)

    contents_Y, _ = extract_features(content_X)
    _, styles_Y = extract_features(style_X)
    styles_Y_gram = [gram(y) for y in styles_Y]

    # 如果使用拉普拉斯损失，计算内容图的拉普拉斯特征
    target_lap = None
    if use_lap_loss:
        target_lap = calc_laplacian(content_X)

    # 2. 初始化生成图
    if init_img is None:
        # 纯噪声初始化 (Drafting 阶段)
        noise_strength = 0.02  # 略微降低噪声以便观察
        generated = nn.Parameter(
            content_X.clone() + noise_strength * torch.randn_like(content_X)
        )
    else:
        # 使用上一阶段的结果初始化 (Revision 阶段)
        # 需插值到当前尺寸
        init_img = F.interpolate(init_img, size=image_shape, mode='bilinear', align_corners=False)
        generated = nn.Parameter(init_img.detach())  # Detach create new leaf node

    # 3. 优化器
    lr = 0.1 if stage_name == "Drafting" else 0.05  # Revision 阶段学习率稍低
    optimizer = torch.optim.Adam([generated], lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    # 权重配置
    style_layer_weights = [1.0, 0.8, 0.5, 0.3, 0.1]

    c_weight = 10  # 内容权重
    s_weight = 5e6  # 风格权重 (根据LapStyle逻辑，风格在草稿阶段很重要)
    tv_weight = 10
    lap_weight = 1500  # 拉普拉斯权重 (Revision阶段核心)

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        contents_hat, styles_hat = extract_features(generated)

        # 基础损失
        c_loss = sum(content_loss(h, y) for h, y in zip(contents_hat, contents_Y))
        s_loss = sum(w * style_loss(h, y) for h, y, w in zip(styles_hat, styles_Y_gram, style_layer_weights))
        t_loss = tv_loss(generated)

        loss = c_weight * c_loss + s_weight * s_loss + tv_weight * t_loss

        # LapStyle 特有: 拉普拉斯损失
        l_loss = torch.tensor(0.0).to(device)
        if use_lap_loss and target_lap is not None:
            l_loss = laplacian_loss(generated, target_lap)
            loss += lap_weight * l_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        # Clamp
        with torch.no_grad():
            generated.clamp_(
                (0 - rgb_mean.view(1, 3, 1, 1)) / rgb_std.view(1, 3, 1, 1),
                (1 - rgb_mean.view(1, 3, 1, 1)) / rgb_std.view(1, 3, 1, 1)
            )

        if (epoch + 1) % 100 == 0:
            msg = f"Epoch {epoch + 1:3d} | Total: {loss.item():.2f} | Cont: {c_loss.item():.4f} | Sty: {s_loss.item():.4f}"
            if use_lap_loss:
                msg += f" | Lap: {l_loss.item():.4f}"
            print(msg)

    return generated


# =========================
# 执行流程 (Drafting -> Revision)
# =========================

# 阶段 1: Drafting (低分辨率草稿)
# 目的：快速建立全局风格结构，忽略细节
draft_result = run_style_transfer(
    stage_name="Drafting",
    image_shape=draft_shape,
    num_epochs=400,
    init_img=None,
    use_lap_loss=False  # 草稿阶段不需要强制边缘
)

# 展示草稿结果
print("Drafting Stage Completed.")
plt.figure(figsize=(5, 5))
plt.title("Drafting Result (Low Res)")
plt.imshow(postprocess(draft_result))
plt.axis("off")
plt.show()

# 阶段 2: Revision (高分辨率修订)
# 目的：细化纹理，利用 Laplacian Loss 找回丢失的物体轮廓
final_result_tensor = run_style_transfer(
    stage_name="Revision",
    image_shape=target_shape,
    num_epochs=600,
    init_img=draft_result,  # 使用草稿作为初始化
    use_lap_loss=True  # 开启拉普拉斯约束
)

# =========================
# 保存结果
# =========================
result = postprocess(final_result_tensor)
out_dir = "outputs/Lapstyle"
os.makedirs(out_dir, exist_ok=True)

out_path = os.path.join(
    out_dir,
    f"{content_name.split('.')[0]}_{style_name.split('.')[0]}.jpg"
)
result.save(out_path)
print(f"\n最终图片已保存至: {out_path}")
plt.figure(figsize=(8, 8))
plt.title("LapStyle Final Result")
plt.imshow(result)
plt.axis("off")
plt.show()