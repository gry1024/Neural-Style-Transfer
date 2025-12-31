import torch
import torchvision
from torch import nn
from d2l import torch as d2l

# 1. 读取图片
# d2l.Image.open 只是对 PIL 的封装，实际使用 PIL.Image.open 也可以
d2l.set_figsize()
content_name = 'Teaching Building1.png'
style_name = 'candy.jpg'
content_img = d2l.Image.open(f'../images/contents/{content_name}') # 内容图
style_img = d2l.Image.open(f'../images/styles/{style_name}') # 风格图

# 2. 定义预处理和后处理函数
# ImageNet 的 RGB 均值和标准差，用于归一化
rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    """
    预处理：缩放 -> 转Tensor -> 归一化 -> 增加Batch维度
    """
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    """
    后处理：将 Tensor 还原为可视化的图片
    """
    # 将图像从显存移动到内存，并去除 Batch 维度
    img = img[0].to(rgb_std.device)
    # 反归一化：img * std + mean
    # permute(1, 2, 0) 是将 (C, H, W) 转换为 (H, W, C) 以便显示
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    # 转回 PIL 图片格式
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

# 1. 加载预训练的 VGG19 模型
pretrained_net = torchvision.models.vgg19(pretrained=True)

# 2. 定义用于提取特征的层
# 越靠近输出层（深层），越容易提取全局内容信息；越靠近输入层（浅层），越关注细节纹理
style_layers = [0, 5, 10, 19, 28]  # 风格层：选取每个卷积块的第一个卷积层
content_layers = [25]              # 内容层：选取第四个卷积块的最后一个卷积层

# 3. 构建新网络
# 我们只需要用到 VGG 中计算到最深层（max layer index）的部分
# 丢弃后面不需要的层以节省计算资源
net = nn.Sequential(*[pretrained_net.features[i] for i in
                      range(max(content_layers + style_layers) + 1)])

def extract_features(X, content_layers, style_layers):
    """
    输入图像 X，返回指定的内容层和风格层的输出
    """
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X) # 逐层前向传播
        if i in style_layers:
            styles.append(X) # 收集风格特征
        if i in content_layers:
            contents.append(X) # 收集内容特征
    return contents, styles


# 1. 内容损失
def content_loss(Y_hat, Y):
    """
    Y_hat: 合成图的内容特征
    Y: 内容图的内容特征（需要 detach，因为它是固定的目标值，不参与梯度计算）
    """
    return torch.square(Y_hat - Y.detach()).mean()


# 2. 风格损失相关：Gram 矩阵
def gram(X):
    """
    计算 Gram 矩阵：特征向量的点积，表示通道间的相关性
    X shape: (Batch, Channel, Height, Width)
    """
    num_channels, n = X.shape[1], X.numel() // X.shape[1]
    X = X.reshape((num_channels, n))
    # 矩阵乘法 X * X.T
    return torch.matmul(X, X.T) / (num_channels * n)


def style_loss(Y_hat, gram_Y):
    """
    Y_hat: 合成图的风格特征
    gram_Y: 风格图的 Gram 矩阵 (预先计算好的常量)
    """
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()


# 3. 全变分损失 (Total Variation Loss)
def tv_loss(Y_hat):
    """
    惩罚相邻像素的差值，减少噪点，使图像更平滑
    """
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


# 4. 总损失计算
# 权重超参数：风格权重通常设得很大，以强调风格迁移的效果
content_weight, style_weight, tv_weight = 1, 1e4, 10


def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    # 计算各项损失
    contents_l = [content_loss(Y_hat, Y) * content_weight for Y_hat, Y in zip(
        contents_Y_hat, contents_Y)]
    styles_l = [style_loss(Y_hat, Y) * style_weight for Y_hat, Y in zip(
        styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight

    # 求和
    l = sum(styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l


# 1. 定义合成图像模型
class SynthesizedImage(nn.Module):
    def __init__(self, img_shape, **kwargs):
        super(SynthesizedImage, self).__init__(**kwargs)
        # self.weight 就是我们要生成的图片，初始化为随机噪声或内容图
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight


# 2. 初始化函数
def get_inits(X, device, lr, styles_Y):
    gen_img = SynthesizedImage(X.shape).to(device)
    # 使用内容图 X 来初始化合成图，这样收敛更快
    gen_img.weight.data.copy_(X.data)
    # 优化器只优化 gen_img.parameters()，即图片的像素
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    # 预先计算风格图的 Gram 矩阵（因为风格图是不变的）
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    return gen_img(), styles_Y_gram, trainer


# 3. 提取目标特征的辅助函数
def get_contents(image_shape, device):
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y


def get_styles(image_shape, device):
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y


# 4. 训练主循环
def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8)

    # 绘图动画相关 (d2l 库功能)
    animator = d2l.Animator(xlabel='epoch', ylabel='loss',
                            xlim=[10, num_epochs],
                            legend=['content', 'style', 'TV'],
                            ncols=2, figsize=(7, 2.5))

    for epoch in range(num_epochs):
        trainer.zero_grad()
        # 前向传播：提取当前合成图的特征
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        # 计算损失
        contents_l, styles_l, tv_l, l = compute_loss(
            X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram)
        # 反向传播：更新图像像素
        l.backward()
        trainer.step()
        scheduler.step()

        # 可视化进度
        if (epoch + 1) % 10 == 0:
            animator.axes[1].imshow(postprocess(X))
            animator.add(epoch + 1, [float(sum(contents_l)),
                                     float(sum(styles_l)), float(tv_l)])
    return X


# 5. 启动训练
device, image_shape = d2l.try_gpu(), (300, 450)  # 设置图像尺寸
net = net.to(device)
# 预提取内容图和风格图的特征
content_X, contents_Y = get_contents(image_shape, device)
_, styles_Y = get_styles(image_shape, device)
# 开始训练
output = train(content_X, contents_Y, styles_Y, device, 0.3, 500, 50)

# 1. 将 Tensor 转换回 PIL 图片格式
final_img = postprocess(output)

# 2. 调用 PIL 的 save 方法保存到当前文件夹
final_img.save(f"outputs/{content_name.split('.')[0]}_{style_name.split('.')[0]}.jpg")

print("图片已保存")