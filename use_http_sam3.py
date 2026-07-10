import requests
import base64
import cv2
import numpy as np
import os
from datetime import datetime


def call_sam3_service(
        service_url: str,
        image_input,  # 可以是路径 (str) 或 numpy 数组
        text_prompt: str,
        save_dir: str = "output/sam3_results",
        save_result: bool = True,  # 新增：是否保存到硬盘
        return_data: bool = True    # 新增：是否返回数据给调用者
):
    """
    调用 SAM3 服务
    Args:
        service_url: 服务地址
        image_input: 图像路径(str) 或 OpenCV numpy数组(BGR格式)
        text_prompt: 提示词
        save_dir: 保存根目录
        save_result: 是否保存图像文件到本地
        return_data: 是否返回解析后的 JSON 数据

    Returns:
        dict or None: 如果 return_data=True 且成功，返回服务端 JSON 数据；否则返回 None
    """
    result_json = None
    save_path = None
    mask_save_path = None

    # -------------------------- 1. 处理输入图像 --------------------------
    if isinstance(image_input, str):
        # 输入是文件路径
        image = cv2.imread(image_input)
        if image is None:
            print(f"无法读取图像: {image_input}")
            return None
    elif isinstance(image_input, np.ndarray):
        # 输入是 numpy 数组
        image = image_input
    else:
        print(f"image_input 类型不支持: {type(image_input)}")
        return None

    # -------------------------- 2. 创建保存目录 (如果需要保存) --------------------------
    if save_result:
        time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"res_{time_stamp}")
        mask_save_path = os.path.join(save_path, "masks")
        os.makedirs(mask_save_path, exist_ok=True)

    # -------------------------- 3. 构造请求 --------------------------
    _, img_buf = cv2.imencode('.jpg', image)
    files = {'image': ('input.jpg', img_buf.tobytes(), 'image/jpeg')}
    data = {'text_prompt': text_prompt}

    try:
        print(f"发送请求到: {service_url}")
        response = requests.post(service_url, files=files, data=data, timeout=60)

        if response.status_code == 200:
            result_json = response.json()
            if result_json['status'] == 'success':
                print(f"检测到 {result_json['detected_objects']} 个目标")

                if save_result:
                    # 1. 保存检测图
                    det_b64 = result_json['detection_image']
                    det_img = cv2.imdecode(np.frombuffer(base64.b64decode(det_b64), np.uint8), cv2.IMREAD_COLOR)
                    cv2.imwrite(os.path.join(save_path, "detection_result.jpg"), det_img)

                    # 2. 保存合并掩码
                    merged_b64 = result_json['merged_mask']
                    merged_img = cv2.imdecode(np.frombuffer(base64.b64decode(merged_b64), np.uint8), cv2.IMREAD_GRAYSCALE)
                    cv2.imwrite(os.path.join(save_path, "merged_mask.png"), merged_img)

                    # 3. 保存单独掩码
                    for im in result_json['individual_masks']:
                        mask_b64 = im['mask_b64']
                        mask_img = cv2.imdecode(np.frombuffer(base64.b64decode(mask_b64), np.uint8), cv2.IMREAD_GRAYSCALE)
                        label_clean = im['label'].replace(' ', '_')
                        cv2.imwrite(
                            os.path.join(mask_save_path, f"mask_{im['id']}_{label_clean}_{im['score']:.2f}.png"),
                            mask_img
                        )
                    print(f"结果已保存到: {save_path}")
            else:
                print(f"服务返回错误: {result_json['message']}")
                if return_data:
                    return None
        else:
            print(f"请求失败: HTTP {response.status_code}")
    except Exception as e:
        print(f"调用出错: {str(e)}")

    # -------------------------- 4. 返回数据 (如果需要) --------------------------
    if return_data:
        return result_json
    else:
        return None


if __name__ == "__main__":
    SERVICE_URL = "http://127.0.0.1:5001/process"
    INPUT_DIR = "input"
    OUTPUT_DIR = "output"

    # 读取 input 文件夹中的第一张图
    exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
    images = sorted([f for f in os.listdir(INPUT_DIR) if f.lower().endswith(exts)])
    if not images:
        print(f"input 文件夹中没有找到图片: {INPUT_DIR}")
        exit(1)
    image_path = os.path.join(INPUT_DIR, images[0])
    print(f"使用图片: {image_path}")

    # 循环输入提示词
    while True:
        text_prompt = input("\n请输入提示词（逗号分隔多个目标，输入 q 退出）: ").strip()
        if text_prompt.lower() == 'q':
            print("退出。")
            break
        if not text_prompt:
            print("提示词不能为空，请重新输入。")
            continue

        call_sam3_service(
            service_url=SERVICE_URL,
            image_input=image_path,
            text_prompt=text_prompt,
            save_dir=OUTPUT_DIR,
            save_result=True,
            return_data=False
        )
