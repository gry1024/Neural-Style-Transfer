# Neural Style Transfer

本项目集成了多种神经风格迁移（Neural Style Transfer）算法的实现，旨在探索计算机视觉在艺术创作与图像处理中的应用。本项目包含从经典的基础模型到高性能的实时风格迁移算法的完整工作流。


<p align="center">
  <img src="demo.gif" width="600px" alt="Project Demo">
  <br>
  <i>实时交互局部神经风格迁移效果演示</i>
</p>

---

## 🚀 已实现算法 (Implemented Algorithms)

本项目主要包含以下核心实现：

1. **Classic Neural Style Transfer** (Gatys et al.): 基于 VGG19 网络的原始实现，通过迭代优化像素来最小化内容损失和风格损失。
2. **LapStyle (Drafting/Optimizing)**: 基于拉普拉斯金字塔的高质量风格迁移，能够处理更高分辨率的细节并减少伪影。
3. **Fast Neural Style Transfer**: 使用残差网络（Residual Network）构建转换网络，实现图像的实时风格化。
4. **Interactive Sematic Style Transfer**: 通过用户交互，实现不同物体风格选择自定义调整。

## 📂 项目结构 (Project Structure)

```text
Neural-Style-Transfer/
├── checkpoints/         # 存放预训练模型权重 (.pth) 用于FastStyle
├── images/              # 输入图像
│   ├── contents/        # 待转换的内容图 (Content Images)
│   └── styles/          # 风格参考图 (Style Images)
├── outputs/             # 风格迁移生成结果
├── FastStyle.py         # 快速风格迁移实现脚本
├── Gatys.py             # 经典Gatys神经风格迁移实现脚本
├── Lapstyle.py          # 基于拉普拉斯金字塔的风格迁移实现脚本
├── segment_style_tranfer.py # 语义分割/实时局部风格迁移实现脚本
├── sam_vit_h_4b8939.pth # SAM (Segment Anything Model) 权重文件(需自行下载)
├── Report.pdf           # 项目技术报告
├── requirements.txt     # 环境依赖清单
└── README.md            # 项目说明文档
```

## 1. Installation

代码已在 Windows 环境下使用 NVIDIA GPU 进行测试。 实验基于 Python 3.11, PyTorch 2.4.0 和 CUDA 12.1 进行。

安装 Conda 并创建 Conda 环境。

```bash
conda create --name StyleTransfer python=3.11
conda activate StyleTransfer
```

安装依赖库。

```bash
pip install -r requirements.txt
```


## 2. Interactive Local Style Transfer
**注意：** SAM 权重文件 `sam_vit_h_4b8939.pth`(2.4G) 从 [官方仓库](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) 下载后放在根目录下
运行以下命令启动交互界面。移动标高亮物体，点击左键以切换风格。

```bash
python segment_style_tranfer.py
```

## 3. Fast Style Transfer

预训练权重位于 `checkpoints/` 中。若想训练新风格图需要下载COCO数据集。

**Inference:** 运行以下命令使用现有权重生成风格化图像。

```bash
python FastStyle.py
```

## 4. Gatys

**Run:** 运行以下命令执行基于优化的风格迁移。

```bash
python Gatys.py
```

## 5. LapStyle

**Run:** 运行以下命令执行使用拉普拉斯金字塔策略的风格迁移。

```bash
python Lapstyle.py
```


---

## 📝 技术报告 (Technical Report)

本项目相关的技术细节、实现思路以及对比分析已整理至根目录下 Report.pdf 中。

