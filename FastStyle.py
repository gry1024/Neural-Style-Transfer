import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import time

# =========================
# 1. 配置参数
# =========================
# 优先检测 CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"当前运行设备: {device}")

# 路径设置
# 训练别的风格图时更改风格图和最终输出pth命名即可
DATASET_PATH = "data/coco2017"  # 训练集路径
STYLE_IMAGE_PATH = "images/styles/starry_night.jpg"  # 风格图
TEST_IMG_PATH = "images/contents/Teaching Building1.jpg"  # 测试图 (用于生成中间结果)

OUTPUT_DIR = "outputs/FastStyle"
CHECKPOINT_DIR = "checkpoints"
MODEL_FINAL_NAME = "final_starry_night.pth"
STYLE_NAME = MODEL_FINAL_NAME.split(".")[0]

# 如果合成图全黑、Style Loss = inf，尝试将AMP_设置为False
AMP_ = False

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 训练超参数
NUM_EPOCHS = 20
BATCH_SIZE = 4
LEARNING_RATE = 1e-3
LOG_INTERVAL = 200

# 损失权重
CONTENT_WEIGHT = 1e5
STYLE_WEIGHT = 1e10
TV_WEIGHT = 1e-7


# =========================
# 2. 图像处理工具
# =========================
def load_image(filename, size=None, scale=None):
    if not os.path.exists(filename):
        # 如果找不到文件，生成一个纯黑图像防止报错，方便调试
        print(f"警告: 文件未找到 {filename}, 使用纯黑图像替代")
        return Image.new('RGB', (256, 256))

    img = Image.open(filename).convert('RGB')
    if size is not None:
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.Resampling.LANCZOS)
    return img


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)
    print(f"图片已保存: {filename}")


# VGG 标准化参数
mean = torch.tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)


def normalize_batch(batch):
    # 输入 batch 范围 [0, 255] -> 输出 VGG Normalized
    return (batch / 255.0 - mean) / std


class ScaleTo255(object):
    def __call__(self, tensor):
        return tensor * 255.0


# 数据集加载器
class FlatFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        if os.path.exists(root):
            self.paths = [os.path.join(root, f) for f in os.listdir(root)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        else:
            self.paths = []
        self.transform = transform
        if len(self.paths) == 0:
            print(f"警告: 数据集路径 {root} 是空的或不存在！")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img
        except Exception:
            return torch.zeros(3, 256, 256)


# 训练用数据增强
transform = transforms.Compose([
    transforms.Resize(280),
    transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    ScaleTo255()
])


# =========================
# 3. 模型定义
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, stride=1, upsample=False, norm=True, relu=True):
        super().__init__()
        self.block = nn.Sequential()

        if upsample:
            self.block.add_module("conv",
                                  nn.ConvTranspose2d(in_c, out_c, kernel_size, stride, padding=1, output_padding=1))
        else:
            reflect_pad = kernel_size // 2
            self.block.add_module("pad", nn.ReflectionPad2d(reflect_pad))
            self.block.add_module("conv", nn.Conv2d(in_c, out_c, kernel_size, stride))

        if norm:
            self.block.add_module("norm", nn.InstanceNorm2d(out_c, affine=True))
        if relu:
            self.block.add_module("relu", nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, norm=True, relu=True)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3, norm=True, relu=False)

    def forward(self, x): return self.conv2(self.conv1(x)) + x


class TransformerNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ResidualBlock(128), ResidualBlock(128), ResidualBlock(128),
            ResidualBlock(128), ResidualBlock(128),
            ConvBlock(128, 64, kernel_size=3, stride=2, upsample=True),
            ConvBlock(64, 32, kernel_size=3, stride=2, upsample=True),
            ConvBlock(32, 3, kernel_size=9, stride=1, norm=False, relu=False)
        )

    def forward(self, x): return self.model(x)


class Vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            features = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        except Exception:
            # 如果下载失败，尝试不带权重加载（仅用于调试，实际训练必须要有权重）
            print("警告: 无法下载 VGG16 权重，尝试加载空模型...")
            features = models.vgg16(pretrained=False).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(4): self.slice1.add_module(str(x), features[x])
        for x in range(4, 9): self.slice2.add_module(str(x), features[x])
        for x in range(9, 16): self.slice3.add_module(str(x), features[x])
        for x in range(16, 23): self.slice4.add_module(str(x), features[x])
        for param in self.parameters(): param.requires_grad = False

    def forward(self, x):
        h = self.slice1(x)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        return h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


# =========================
# 4. 训练函数
# =========================
def train():
    print(f"=== 准备训练 ===")

    dataset = FlatFolderDataset(DATASET_PATH, transform=transform)
    if len(dataset) == 0:
        return None

    # Windows 下建议设为 0 以避免 PicklingError 和其他多进程问题
    kwargs = {'num_workers': 0, 'pin_memory': True} if device.type == 'cuda' else {}
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
    print(f"加载了 {len(dataset)} 张图片")

    transformer = TransformerNet().to(device)
    vgg = Vgg16().to(device)
    optimizer = optim.Adam(transformer.parameters(), lr=LEARNING_RATE)
    mse_loss = nn.MSELoss()

    # 使用新版 AMP API
    use_amp = (device.type == 'cuda')
    if AMP_ == False:
        use_amp = AMP_  # 关闭 amp，避免出现奇怪的问题
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    # 计算风格特征
    style_img = load_image(STYLE_IMAGE_PATH, size=256)
    style_tensor = transforms.ToTensor()(style_img).unsqueeze(0).to(device) * 255.0
    with torch.no_grad():
        style_vgg = normalize_batch(style_tensor).repeat(BATCH_SIZE, 1, 1, 1)
        features_style = vgg(style_vgg)
        gram_style = [gram_matrix(y) for y in features_style]

    # === 准备中间可视化用的测试图 ===
    sample_tensor = None
    if os.path.exists(TEST_IMG_PATH):
        sample_img = load_image(TEST_IMG_PATH)
        # 预处理测试图，不裁剪，保持原比例
        sample_transform = transforms.Compose([
            transforms.ToTensor(),
            ScaleTo255()
        ])
        sample_tensor = sample_transform(sample_img).unsqueeze(0).to(device)
        print(f"中间结果将基于: {TEST_IMG_PATH} 生成")

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        transformer.train()
        agg_content = 0.
        agg_style = 0.
        count = 0

        for batch_id, x in enumerate(loader):
            n_batch = len(x)
            count += n_batch
            optimizer.zero_grad()
            x = x.to(device)

            # 使用新版 autocast
            with torch.amp.autocast('cuda', enabled=use_amp):
                y = transformer(x)
                x_vgg = normalize_batch(x)
                y_vgg = normalize_batch(y)

                fy = vgg(y_vgg)
                fx = vgg(x_vgg)

                content_loss = CONTENT_WEIGHT * mse_loss(fy[1], fx[1])
                style_loss = 0.
                for ft_y, gm_s in zip(fy, gram_style):
                    style_loss += mse_loss(gram_matrix(ft_y), gm_s[:n_batch, :, :])
                style_loss *= STYLE_WEIGHT

                tv_loss = TV_WEIGHT * (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) +
                                       torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))

                loss = content_loss + style_loss + tv_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            agg_content += content_loss.item()
            agg_style += style_loss.item()

            if (batch_id + 1) % LOG_INTERVAL == 0:
                print(f"Epoch {epoch + 1} [{batch_id + 1}/{len(loader)}] "
                      f"Content: {agg_content / count:.1f} Style: {agg_style / count:.1f}")

        # === 每个 Epoch 结束时，生成并保存中间结果 ===
        print(f"Epoch {epoch + 1} 完成，正在生成中间预览图...")
        save_model_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch + 1}.pth")
        torch.save(transformer.state_dict(), save_model_path)

        if sample_tensor is not None:
            transformer.eval()
            with torch.no_grad():
                output = transformer(sample_tensor)
                save_image(os.path.join(OUTPUT_DIR, f"debug_epoch_{epoch + 1}.jpg"), output[0].cpu())
            transformer.train()

    print(f"训练完成，耗时 {(time.time() - start_time) / 60:.1f} 分钟")

    final_path = os.path.join(CHECKPOINT_DIR, MODEL_FINAL_NAME)
    torch.save(transformer.state_dict(), final_path)
    return final_path


# =========================
# 5. 推断函数
# =========================
def stylize(model_path, content_path):
    print(f"\n=== 开始最终风格推断 ===")
    if not os.path.exists(content_path):
        print(f"错误: 找不到测试图片 {content_path}")
        return

    content_net = TransformerNet().to(device)
    content_net.load_state_dict(torch.load(model_path, map_location=device))
    content_net.eval()

    content_image = load_image(content_path)
    content_transform = transforms.Compose([
        transforms.ToTensor(),
        ScaleTo255()
    ])
    content_tensor = content_transform(content_image).unsqueeze(0).to(device)

    with torch.no_grad():
        start = time.time()
        output = content_net(content_tensor)
        print(f"推断耗时: {time.time() - start:.4f}s")

    file_name = os.path.basename(content_path)
    save_path = os.path.join(OUTPUT_DIR, f"stylized_{file_name}")
    save_image(save_path, output[0].cpu())
    print(f"风格化图片已生成: {save_path}")


# =========================
# 6. 主程序入口
# =========================
if __name__ == "__main__":
    trained_model_path = final_path = os.path.join(CHECKPOINT_DIR, MODEL_FINAL_NAME)

    if not os.path.exists(trained_model_path):
        trained_model_path = train()

    if trained_model_path:
        stylize(model_path=trained_model_path, content_path=TEST_IMG_PATH)