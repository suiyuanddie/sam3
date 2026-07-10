"""
SAM3 单图分割脚本
用法:
    python sam3_run.py
    → 自动读取 input/ 文件夹中的第一张图片
    → 终端输入提示词 (如 "cup, bowl")
    → 结果输出到 output/ 文件夹
"""
import os
import cv2
import numpy as np
import torch
from PIL import Image
from datetime import datetime
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from rich import print as rprint


def load_model(device=None):
    """加载 SAM3 模型和处理器"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rprint(f"[blue]加载 SAM3 模型到: {device}[/blue]")
    model = build_sam3_image_model().to(device)
    processor = Sam3Processor(model)
    rprint("[green]模型加载完成[/green]")
    return model, processor, device


def run_sam3(image_path: str, text_prompt: str, output_dir: str,
             model=None, processor=None, device=None, save: bool = True):
    """
    SAM3 分割推理

    Args:
        image_path:   输入图片路径
        text_prompt:  提示词，逗号分隔 (如 "cup, bowl")
        output_dir:   输出目录
        model:        已加载的模型 (为 None 则自动加载)
        processor:    已加载的处理器 (为 None 则自动加载)
        device:       设备 (为 None 则自动选择)
        save:         是否保存到硬盘

    Returns:
        dict: {
            'detected_objects': int,
            'results': [{'id', 'label', 'score', 'mask', 'segmentation_image'}, ...],
            'merged_mask': np.ndarray,
            'detection_image': np.ndarray,
        }
    """
    # -------------------------- 1. 加载模型 --------------------------
    if model is None or processor is None:
        model, processor, device = load_model(device)

    # -------------------------- 2. 读取图像 --------------------------
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    H, W, _ = rgb.shape
    pil_image = Image.fromarray(rgb)

    # -------------------------- 3. 解析提示词 --------------------------
    prompts = [p.strip() for p in text_prompt.split(',') if p.strip()]
    if not prompts:
        raise ValueError("text_prompt 不能为空")

    # -------------------------- 4. SAM3 推理 --------------------------
    inference_state = processor.set_image(pil_image)

    all_masks = []
    all_scores = []
    all_labels = []

    for prompt in prompts:
        rprint(f"[cyan]处理提示词: '{prompt}'[/cyan]")
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks, _, scores = output["masks"], output["boxes"], output["scores"]

        if masks is None or len(masks) == 0:
            rprint(f"[yellow]未检测到 '{prompt}'，跳过[/yellow]")
            continue

        # 取置信度最高的结果
        best_mask = masks[0].cpu().squeeze().numpy() > 0
        best_score = scores[0].item()

        all_masks.append(best_mask)
        all_scores.append(best_score)
        all_labels.append(prompt)

    if not all_masks:
        rprint("[red]未检测到任何目标[/red]")
        return {'detected_objects': 0, 'results': [], 'merged_mask': None, 'detection_image': None}

    # -------------------------- 5. 生成结果 --------------------------
    # 5.1 检测总览图（所有目标叠加）
    detection_img = rgb.copy()
    for mask, score, label in zip(all_masks, all_scores, all_labels):
        # 半透明绿色掩码
        colored = np.zeros_like(rgb)
        colored[mask] = [0, 255, 0]
        cv2.addWeighted(detection_img, 1.0, colored, 0.4, 0, detection_img)
        # 边界框 + 文字
        y, x = np.where(mask)
        if len(y) == 0 or len(x) == 0:
            continue
        x1, y1, x2, y2 = x.min(), y.min(), x.max(), y.max()
        cv2.rectangle(detection_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = f"{label} ({score:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(detection_img, (x1, y1 - th - 10), (x1 + tw, y1 - 5), (255, 255, 255), -1)
        cv2.putText(detection_img, text, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 5.2 合并掩码
    merged_mask = np.zeros((H, W), dtype=np.uint8)
    for mask in all_masks:
        merged_mask = np.logical_or(merged_mask, mask).astype(np.uint8) * 255

    # 5.3 每个目标的单独分割图 + 掩码图
    results = []
    for i, (mask, label, score) in enumerate(zip(all_masks, all_labels, all_scores)):
        # 分割叠加图：原图 + 单目标绿色掩码
        seg_img = rgb.copy()
        colored = np.zeros_like(rgb)
        colored[mask] = [0, 255, 0]
        cv2.addWeighted(seg_img, 1.0, colored, 0.4, 0, seg_img)
        # 边界框
        y, x = np.where(mask)
        if len(y) > 0 and len(x) > 0:
            x1, y1, x2, y2 = x.min(), y.min(), x.max(), y.max()
            cv2.rectangle(seg_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            text = f"{label} ({score:.2f})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(seg_img, (x1, y1 - th - 10), (x1 + tw, y1 - 5), (255, 255, 255), -1)
            cv2.putText(seg_img, text, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 掩码图：白色前景 + 黑色背景
        mask_uint8 = (mask * 255).astype(np.uint8)

        results.append({
            'id': i,
            'label': label,
            'score': score,
            'mask': mask_uint8,
            'segmentation_image': seg_img,
        })

    # -------------------------- 6. 保存到硬盘 --------------------------
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(output_dir, f"res_{timestamp}")
        mask_dir = os.path.join(save_dir, "masks")
        seg_dir = os.path.join(save_dir, "segmentations")
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)

        # 保存检测总览图
        det_bgr = cv2.cvtColor(detection_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(save_dir, "detection_result.jpg"), det_bgr)

        # 保存合并掩码
        cv2.imwrite(os.path.join(save_dir, "merged_mask.png"), merged_mask)

        # 保存每张分割图 + 掩码图
        for r in results:
            label_clean = r['label'].replace(' ', '_')
            filename = f"{r['id']}_{label_clean}_{r['score']:.2f}"
            seg_bgr = cv2.cvtColor(r['segmentation_image'], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(seg_dir, f"seg_{filename}.jpg"), seg_bgr)
            cv2.imwrite(os.path.join(mask_dir, f"mask_{filename}.png"), r['mask'])

        rprint(f"[green]结果已保存到: {save_dir}[/green]")
        rprint(f"  ├─ detection_result.jpg  (检测总览)")
        rprint(f"  ├─ merged_mask.png       (合并掩码)")
        rprint(f"  ├─ segmentations/        ({len(results)} 张分割图)")
        rprint(f"  └─ masks/                ({len(results)} 张掩码图)")

    return {
        'detected_objects': len(results),
        'results': results,
        'merged_mask': merged_mask,
        'detection_image': detection_img,
    }


if __name__ == "__main__":
    import sys

    # 1. 自动读取 input/ 文件夹中的第一张图片
    INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")
    OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    if not os.path.isdir(INPUT_DIR):
        rprint(f"[red]input 文件夹不存在: {INPUT_DIR}[/red]")
        exit(1)

    images = sorted([
        f for f in os.listdir(INPUT_DIR)
        if os.path.splitext(f)[1].lower() in IMG_EXTS
    ])
    if not images:
        rprint(f"[red]input 文件夹中没有图片[/red]")
        exit(1)

    image_path = os.path.join(INPUT_DIR, images[0])
    rprint(f"[blue]输入图片: {image_path}[/blue]")

    # 2. 提示词：命令行参数 > 终端输入
    if len(sys.argv) > 1:
        text_prompt = ' '.join(sys.argv[1:])
    else:
        text_prompt = input("请输入提示词 (逗号分隔，如 cup, bowl): ").strip()

    if not text_prompt:
        rprint("[red]提示词不能为空[/red]")
        exit(1)

    rprint(f"[blue]提示词: {text_prompt}[/blue]")

    # 3. 执行推理，结果输出到 output/
    result = run_sam3(image_path, text_prompt, OUTPUT_DIR)
    rprint(f"\n[bold green]完成！共检测到 {result['detected_objects']} 个目标[/bold green]")
