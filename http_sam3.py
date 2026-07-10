from flask import Flask, request, jsonify, send_file
import base64
import cv2
import gc
import io
import numpy as np
from PIL import Image
import torch
import uuid
from collections import OrderedDict
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from rich import print as rprint
import os
import time

# 初始化 Flask 应用
app = Flask(__name__)

# 全局变量：SAM3 模型和处理器（服务启动时加载一次）
MODEL = None
PROCESSOR = None
DEVICE = None

# 结果缓存：{request_id: {"detection": bytes, "merged": bytes, "masks": list, "meta": dict, "ts": float}}
# 使用 OrderedDict 实现简单的 LRU，最多缓存最近 20 次请求
_RESULT_CACHE = OrderedDict()
_CACHE_MAX = 20
_CACHE_TTL = 600  # 10 分钟过期


def _cache_put(request_id, data):
    """将结果放入缓存，超出上限时淘汰最旧的"""
    _RESULT_CACHE[request_id] = data
    _RESULT_CACHE.move_to_end(request_id)
    while len(_RESULT_CACHE) > _CACHE_MAX:
        _RESULT_CACHE.popitem(last=False)


def _cache_get(request_id):
    """从缓存取出结果，过期则删除"""
    if request_id not in _RESULT_CACHE:
        return None
    entry = _RESULT_CACHE[request_id]
    if time.time() - entry["ts"] > _CACHE_TTL:
        del _RESULT_CACHE[request_id]
        return None
    _RESULT_CACHE.move_to_end(request_id)
    return entry


def load_sam3_model():
    """加载 SAM3 模型到全局变量"""
    global MODEL, PROCESSOR, DEVICE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rprint(f"[blue]正在加载 SAM3 模型到设备: {DEVICE}[/blue]")
    MODEL = build_sam3_image_model().to(DEVICE)
    PROCESSOR = Sam3Processor(MODEL)
    rprint(f"[green]SAM3 模型加载成功！[/green]")


def draw_detection_result(rgb, masks, scores, labels):
    """绘制带掩码、置信度的检测结果图"""
    projected = rgb.copy()
    for i, (mask, score, label) in enumerate(zip(masks, scores, labels)):
        # 1. 绘制半透明绿色掩码
        colored_mask = np.zeros_like(rgb)
        colored_mask[mask] = [0, 255, 0]
        cv2.addWeighted(projected, 1.0, colored_mask, 0.4, 0, projected)

        # 2. 计算掩码边界框
        y, x = np.where(mask)
        if len(y) == 0 or len(x) == 0:
            continue
        x1, y1, x2, y2 = x.min(), y.min(), x.max(), y.max()

        # 3. 绘制边界框和文本（标签+置信度）
        cv2.rectangle(projected, (x1, y1), (x2, y2), (255, 0, 0), 2)
        text = f"{label} ({score:.2f})"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(projected, (x1, y1 - th - 10), (x1 + tw, y1 - 5), (255, 255, 255), -1)
        cv2.putText(projected, text, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return projected


def _cleanup_gpu():
    """推理完成后释放 GPU 缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()


@app.route('/process', methods=['POST'])
def process_request():
    """处理外部请求的核心接口"""
    try:
        # -------------------------- 1. 校验输入 --------------------------
        if 'image' not in request.files:
            return jsonify({'status': 'fail', 'message': '缺少 image 文件'}), 400
        if 'text_prompt' not in request.form:
            return jsonify({'status': 'fail', 'message': '缺少 text_prompt 参数'}), 400

        # compact=true 时不内联 base64 图片，通过独立端点下载
        compact = request.form.get('compact', 'false').lower() == 'true'

        # -------------------------- 2. 读取图像 --------------------------
        image_file = request.files['image']
        image_bytes = np.frombuffer(image_file.read(), np.uint8)
        bgr = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        if bgr is None:
            return jsonify({'status': 'fail', 'message': '无效的图像文件'}), 400
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        H, W, _ = rgb.shape

        # -------------------------- 3. 解析提示词（逗号分隔） --------------------------
        text_prompt = request.form['text_prompt'].strip()
        prompts = [p.strip() for p in text_prompt.split(',') if p.strip()]
        if not prompts:
            return jsonify({'status': 'fail', 'message': 'text_prompt 不能为空'}), 400

        # -------------------------- 4. SAM3 推理（no_grad 省显存） --------------------------
        global MODEL, PROCESSOR
        if MODEL is None or PROCESSOR is None:
            return jsonify({'status': 'fail', 'message': 'SAM3 模型未加载'}), 500

        pil_image = Image.fromarray(rgb)

        with torch.no_grad():
            inference_state = PROCESSOR.set_image(pil_image)

            all_masks = []
            all_scores = []
            all_labels = []

            for prompt in prompts:
                rprint(f"[cyan]处理提示词: '{prompt}'[/cyan]")
                output = PROCESSOR.set_text_prompt(state=inference_state, prompt=prompt)
                masks, _, scores = output["masks"], output["boxes"], output["scores"]

                if masks is None or len(masks) == 0:
                    rprint(f"[yellow]未检测到 '{prompt}'[/yellow]")
                    continue

                # 取置信度最高的结果
                best_idx = 0
                best_mask = masks[best_idx].cpu().squeeze().numpy() > 0
                best_score = scores[best_idx].item()

                all_masks.append(best_mask)
                all_scores.append(best_score)
                all_labels.append(prompt)

        # 推理结束，立即释放 GPU 缓存
        _cleanup_gpu()

        if not all_masks:
            return jsonify({'status': 'fail', 'message': '未检测到任何目标'}), 400

        # -------------------------- 5. 生成结果图像 --------------------------
        # 5.1 检测图（带掩码和置信度）
        detection_img = draw_detection_result(rgb, all_masks, all_scores, all_labels)
        detection_bgr = cv2.cvtColor(detection_img, cv2.COLOR_RGB2BGR)

        # 5.2 合并掩码（所有物体）
        merged_mask = np.zeros((H, W), dtype=np.uint8)
        for mask in all_masks:
            merged_mask = np.logical_or(merged_mask, mask).astype(np.uint8) * 255

        # 5.3 单独掩码（每个提示词对应一个）
        individual_masks = []
        for i, (mask, label, score) in enumerate(zip(all_masks, all_labels, all_scores)):
            mask_uint8 = (mask * 255).astype(np.uint8)
            individual_masks.append({
                'id': i, 'label': label, 'score': score, 'mask': mask_uint8
            })

        # -------------------------- 6. 编码图像 --------------------------
        def encode_bgr(img):
            _, buf = cv2.imencode('.jpg', img)
            return buf.tobytes()

        def encode_mask(img):
            _, buf = cv2.imencode('.png', img)
            return buf.tobytes()

        detection_bytes = encode_bgr(detection_bgr)
        merged_bytes = encode_mask(merged_mask)
        mask_bytes_list = [encode_mask(im['mask']) for im in individual_masks]

        # -------------------------- 7. 生成 request_id 并缓存结果 --------------------------
        request_id = str(uuid.uuid4())[:12]
        _cache_put(request_id, {
            "detection": detection_bytes,
            "merged": merged_bytes,
            "masks": mask_bytes_list,
            "labels": [im['label'] for im in individual_masks],
            "scores": [im['score'] for im in individual_masks],
            "ts": time.time(),
        })

        # -------------------------- 8. 返回结果 --------------------------
        if compact:
            # compact 模式：只返回元数据 + 下载 URL，不内联图片
            return jsonify({
                'status': 'success',
                'message': '处理完成',
                'request_id': request_id,
                'detected_objects': len(all_masks),
                'detection_url': f'/image/{request_id}/detection',
                'merged_mask_url': f'/image/{request_id}/merged',
                'individual_mask_urls': [
                    {'id': i, 'label': im['label'], 'score': im['score'],
                     'url': f'/image/{request_id}/mask/{i}'}
                    for i, im in enumerate(individual_masks)
                ]
            })
        else:
            # 默认模式：内联 base64（兼容旧客户端）
            return jsonify({
                'status': 'success',
                'message': '处理完成',
                'request_id': request_id,
                'detected_objects': len(all_masks),
                'detection_image': base64.b64encode(detection_bytes).decode('utf-8'),
                'merged_mask': base64.b64encode(merged_bytes).decode('utf-8'),
                'individual_masks': [
                    {
                        'id': im['id'],
                        'label': im['label'],
                        'score': im['score'],
                        'mask_b64': base64.b64encode(mask_bytes_list[i]).decode('utf-8')
                    } for i, im in enumerate(individual_masks)
                ]
            })

    except Exception as e:
        _cleanup_gpu()
        rprint(f"[red]处理请求出错: {str(e)}[/red]")
        return jsonify({'status': 'fail', 'message': str(e)}), 500


@app.route('/image/<request_id>/detection', methods=['GET'])
def get_detection_image(request_id):
    """下载检测结果图（JPEG）"""
    entry = _cache_get(request_id)
    if entry is None:
        return jsonify({'status': 'fail', 'message': '结果已过期或不存在'}), 404
    return send_file(io.BytesIO(entry["detection"]), mimetype='image/jpeg',
                     download_name='detection.jpg')


@app.route('/image/<request_id>/merged', methods=['GET'])
def get_merged_mask(request_id):
    """下载合并掩码（PNG）"""
    entry = _cache_get(request_id)
    if entry is None:
        return jsonify({'status': 'fail', 'message': '结果已过期或不存在'}), 404
    return send_file(io.BytesIO(entry["merged"]), mimetype='image/png',
                     download_name='merged_mask.png')


@app.route('/image/<request_id>/mask/<int:mask_id>', methods=['GET'])
def get_individual_mask(request_id, mask_id):
    """下载单个物体掩码（PNG）"""
    entry = _cache_get(request_id)
    if entry is None:
        return jsonify({'status': 'fail', 'message': '结果已过期或不存在'}), 404
    if mask_id < 0 or mask_id >= len(entry["masks"]):
        return jsonify({'status': 'fail', 'message': f'掩码索引 {mask_id} 超出范围'}), 400
    label = entry["labels"][mask_id].replace(' ', '_')
    score = entry["scores"][mask_id]
    return send_file(io.BytesIO(entry["masks"][mask_id]), mimetype='image/png',
                     download_name=f'mask_{mask_id}_{label}_{score:.2f}.png')


if __name__ == "__main__":
    # 启动服务前加载模型
    load_sam3_model()
    # 启动服务（仅本机访问）
    app.run(host='127.0.0.1', port=5001, debug=False)
