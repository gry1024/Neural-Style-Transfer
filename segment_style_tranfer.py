import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import glob
from torchvision import transforms
from PIL import Image
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# ==========================================
# 0. 基础配置
# ==========================================
os.environ["no_proxy"] = "localhost,127.0.0.1"

# ⬇️⬇️⬇️ 路径配置 ⬇️⬇️⬇️
TEST_IMG_PATH = "images/contents/Teaching Building1.jpg"
CHECKPOINT_DIR = "checkpoints"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"

INTERNAL_PROCESS_HEIGHT = 1200
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. 模型架构 (保持不变)
# ==========================================
class InstanceNormalization(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(InstanceNormalization, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(dim))
        self.bias = nn.Parameter(torch.FloatTensor(dim))
        self.eps = eps
        self._reset_parameters()

    def _reset_parameters(self):
        self.weight.data.fill_(1.0)
        self.bias.data.zero_()

    def forward(self, x):
        n, c, h, w = x.size()
        t = x.view(n, c, h * w)
        mean = torch.mean(t, 2, keepdim=True).view(n, c, 1, 1)
        var = torch.var(t, 2, keepdim=True).view(n, c, 1, 1)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = out * self.weight.view(1, c, 1, 1) + self.bias.view(1, c, 1, 1)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 transpose=False, use_norm=True, use_relu=True):
        super(ConvBlock, self).__init__()
        self.block = nn.Module()
        self.use_norm = use_norm
        self.use_relu = use_relu
        if transpose:
            self.block.conv = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                padding=1, output_padding=1
            )
        else:
            padding = kernel_size // 2
            self.block.pad = nn.ReflectionPad2d(padding)
            self.block.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        if use_norm: self.block.norm = InstanceNormalization(out_channels)
        if use_relu: self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if hasattr(self.block, 'pad'): x = self.block.pad(x)
        x = self.block.conv(x)
        if self.use_norm and hasattr(self.block, 'norm'): x = self.block.norm(x)
        if self.use_relu and hasattr(self, 'relu'): x = self.relu(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvBlock(channels, channels, kernel_size=3, stride=1, use_relu=False)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(3, 32, kernel_size=9, stride=1),
            ConvBlock(32, 64, kernel_size=3, stride=2),
            ConvBlock(64, 128, kernel_size=3, stride=2),
            ResBlock(128), ResBlock(128), ResBlock(128), ResBlock(128), ResBlock(128),
            ConvBlock(128, 64, kernel_size=3, stride=2, transpose=True),
            ConvBlock(64, 32, kernel_size=3, stride=2, transpose=True),
            ConvBlock(32, 3, kernel_size=9, stride=1, use_norm=False, use_relu=False)
        )

    def forward(self, x):
        return self.model(x)


# ==========================================
# 2. 智能切分交互类
# ==========================================
class StyleTransferApp:
    def __init__(self):
        self.window_name = "Interactive Style Studio"

        self.masks = []
        self.style_models = {}
        self.style_names = []
        self.style_img_cache = {}
        self.soft_masks_cache = []

        self.current_style_indices = []
        self.current_hover_idx = -1
        self.cached_base_img = None
        self.clicked_lock_id = -1

        self.view_scale = 1.0

        self._load_resources()
        self._process_image()
        self._precompute()

    def _load_resources(self):
        print("⏳ Loading SAM...")
        if not os.path.exists(SAM_CHECKPOINT): raise FileNotFoundError(f"Missing {SAM_CHECKPOINT}")
        sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
        sam.to(device)
        self.mask_generator = SamAutomaticMaskGenerator(sam)

        if not os.path.exists(CHECKPOINT_DIR): os.makedirs(CHECKPOINT_DIR)
        pths = glob.glob(os.path.join(CHECKPOINT_DIR, "final_*.pth"))
        if not pths: pths = glob.glob(os.path.join(CHECKPOINT_DIR, "*.pth"))[:5]

        print(f"⏳ Loading {len(pths)} style models...")
        self.style_names = ["Original"]
        for pth in pths:
            raw_name = os.path.basename(pth).split('.')[0]
            name = raw_name.replace("final_", "")
            try:
                net = TransformerNet().to(device)
                state = torch.load(pth, map_location=device, weights_only=False)
                net.load_state_dict(state, strict=True)
                net.eval()
                self.style_models[name] = net
                self.style_names.append(name)
            except Exception as e:
                print(f"⚠️ Skip {name}: {e}")

    def _process_image(self):
        print("⏳ Processing image...")
        if not os.path.exists(TEST_IMG_PATH): raise FileNotFoundError(TEST_IMG_PATH)
        full_img = cv2.imread(TEST_IMG_PATH)

        h, w = full_img.shape[:2]
        if h > INTERNAL_PROCESS_HEIGHT:
            scale = INTERNAL_PROCESS_HEIGHT / h
            new_w = int(w * scale)
            full_img = cv2.resize(full_img, (new_w, INTERNAL_PROCESS_HEIGHT), interpolation=cv2.INTER_AREA)

        self.img_bgr = full_img
        self.img_rgb = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)
        self.H, self.W = full_img.shape[:2]
        self.aspect_ratio = self.W / self.H

    def _precompute(self):
        print("⏳ Segmenting objects...")
        raw_masks = self.mask_generator.generate(self.img_rgb)

        # 1. 先将 SAM 识别出的 masks 加入列表
        # 按面积倒序，确保大物体在底
        self.masks = sorted(raw_masks, key=(lambda x: x['area']), reverse=True)
        print(f"✅ SAM detected {len(self.masks)} objects.")

        # ======================================================
        # 🚀 智能补漏：将背景打散为独立连通域
        # ======================================================
        print("🔧 Analyzing background gaps...")
        combined_mask = np.zeros((self.H, self.W), dtype=np.uint8)
        for m in self.masks:
            # 用 OR 运算合并所有已知区域
            combined_mask = cv2.bitwise_or(combined_mask, m['segmentation'].astype(np.uint8))

        # 取反，得到背景图 (0是物体，1是背景)
        background_map = 1 - combined_mask

        # 连通域分析：把不相连的背景切分开 (例如天空和地面分开)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(background_map, connectivity=8)

        # 从 1 开始遍历 (0是背景的背景，即物体区域，忽略)
        added_bg_count = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            # 过滤掉太小的噪点 (例如 < 500 像素)
            if area < 500: continue

            # 生成该连通域的 Mask
            component_mask = (labels == i)

            # 获取包围盒 (x, y, w, h)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            bg_data = {
                'segmentation': component_mask,
                'area': area,
                'bbox': [x, y, w, h],  # 精确的 bbox，极大提升渲染速度
                'predicted_iou': 1.0,
                'point_coords': [],
                'stability_score': 1.0,
                'crop_box': [x, y, w, h]
            }
            self.masks.append(bg_data)
            added_bg_count += 1

        print(f"✅ Auto-filled {added_bg_count} separate background regions.")
        # ======================================================

        print("⏳ Pre-computing soft masks...")
        for m in self.masks:
            m_float = m['segmentation'].astype(np.float32)
            m_soft = cv2.GaussianBlur(m_float, (5, 5), 0)
            self.soft_masks_cache.append(m_soft[:, :, np.newaxis])

        print("⏳ Pre-computing styles...")
        self.style_img_cache["Original"] = self.img_bgr
        prep = transforms.Compose([transforms.ToTensor(), lambda x: x * 255])
        input_tensor = prep(Image.fromarray(self.img_rgb)).unsqueeze(0).to(device)

        with torch.no_grad():
            for name, net in self.style_models.items():
                out = net(input_tensor).squeeze(0).cpu().clamp(0, 255).numpy()
                out = out.transpose(1, 2, 0).astype(np.uint8)
                out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
                if out_bgr.shape[:2] != (self.H, self.W):
                    out_bgr = cv2.resize(out_bgr, (self.W, self.H))
                self.style_img_cache[name] = out_bgr

        import random
        self.current_style_indices = [random.randint(0, len(self.style_names) - 1) for _ in range(len(self.masks))]

        print("✨ Initial rendering...")
        self.cached_base_img = self._full_render()

    def _full_render(self):
        canvas = self.img_bgr.copy().astype(np.float32)
        for i in range(len(self.masks)):
            s_idx = self.current_style_indices[i]
            s_name = self.style_names[s_idx]
            if s_name == "Original": continue
            alpha = self.soft_masks_cache[i]
            style_layer = self.style_img_cache[s_name].astype(np.float32)
            canvas = style_layer * alpha + canvas * (1 - alpha)
        return canvas.clip(0, 255).astype(np.uint8)

    def _update_region(self, mask_idx):
        # 极速局部更新：现在背景也有了精确的 bbox，所以点击背景也会很快
        bbox = self.masks[mask_idx]['bbox']
        x, y, w, h = bbox
        pad = 10
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(self.W, x + w + pad), min(self.H, y + h + pad)

        roi_canvas = self.img_bgr[y1:y2, x1:x2].astype(np.float32)
        for i in range(len(self.masks)):
            # 优化：快速跳过不相关的 mask
            alpha_roi = self.soft_masks_cache[i][y1:y2, x1:x2]
            # np.any 比 sum 快得多
            if not np.any(alpha_roi): continue

            s_idx = self.current_style_indices[i]
            s_name = self.style_names[s_idx]
            if s_name == "Original": continue

            style_roi = self.style_img_cache[s_name][y1:y2, x1:x2].astype(np.float32)
            roi_canvas = style_roi * alpha_roi + roi_canvas * (1 - alpha_roi)

        self.cached_base_img[y1:y2, x1:x2] = roi_canvas.clip(0, 255).astype(np.uint8)

    def mouse_callback(self, event, x, y, flags, param):
        if self.view_scale == 0: return
        img_x = int(x / self.view_scale)
        img_y = int(y / self.view_scale)

        if img_x < 0 or img_x >= self.W or img_y < 0 or img_y >= self.H:
            self.current_hover_idx = -1
            return

        # 查找物体（因为背景被打散了，现在背景也是一个个小的 mask，能被正常查找到）
        found = -1
        for i in reversed(range(len(self.masks))):
            if self.masks[i]['segmentation'][img_y, img_x]:
                found = i
                break

        self.current_hover_idx = found

        # 逻辑：如果移到了新的物体（包括从背景A移到了背景B），解除锁定
        if found != self.clicked_lock_id:
            self.clicked_lock_id = -1

        if event == cv2.EVENT_LBUTTONDOWN and found != -1:
            old_idx = self.current_style_indices[found]
            new_idx = (old_idx + 1) % len(self.style_names)
            self.current_style_indices[found] = new_idx

            # 显示干净的名称
            print(f"👉 Object {found} switched to: {self.style_names[new_idx]}")
            self._update_region(found)
            self.clicked_lock_id = found

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        initial_h = 800
        initial_w = int(initial_h * self.aspect_ratio)
        cv2.resizeWindow(self.window_name, initial_w, initial_h)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 1)
        cv2.imshow(self.window_name, self.cached_base_img)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_TOPMOST, 0)

        print("\n🎉 Ready! Clean names & Perfect interactions.")

        while True:
            try:
                rect = cv2.getWindowImageRect(self.window_name)
                win_w, win_h = rect[2], rect[3]
                if win_w == 0 or win_h == 0: win_w, win_h = initial_w, initial_h
            except:
                win_w, win_h = initial_w, initial_h

            # 比例锁定
            target_h = int(win_w / self.aspect_ratio)
            if abs(win_h - target_h) > 2:
                cv2.resizeWindow(self.window_name, win_w, target_h)
                win_h = target_h

            self.view_scale = win_w / self.W
            render_base = self.cached_base_img.copy()

            # 高亮逻辑：只有当不是刚被点击的物体时，才显示红色
            if self.current_hover_idx != -1 and self.current_hover_idx != self.clicked_lock_id:
                mask_bool = self.masks[self.current_hover_idx]['segmentation']
                overlay = render_base.copy()
                overlay[mask_bool] = (0, 0, 255)
                cv2.addWeighted(overlay, 0.4, render_base, 0.6, 0, render_base)

            final_display = cv2.resize(render_base, (win_w, win_h), interpolation=cv2.INTER_LINEAR)
            cv2.imshow(self.window_name, final_display)

            key = cv2.waitKey(10)
            if key == 27 or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        print("👋 Exiting...")


if __name__ == "__main__":
    app = StyleTransferApp()
    app.run()