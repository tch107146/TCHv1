"""
DEIMv2: Real-Time Object Detection Meets DINOv3
Copyright (c) 2025 The DEIMv2 Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from D-FINE (https://github.com/Peterande/D-FINE)
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import os
import random
import sys

import cv2  # Added for video processing
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from engine.core import YAMLConfig

label_map = {
    1: 'rov', 2: 'Bio', 3: 'trash', 4: 'Rov', 5: 'aeroplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'trafficlight',
    11: 'firehydrant', 12: 'streetsign', 13: 'stopsign', 14: 'parkingmeter',
    15: 'bench', 16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse',
    20: 'sheep', 21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra',
    25: 'giraffe', 26: 'hat', 27: 'backpack', 28: 'umbrella', 29: 'shoe',
    30: 'eyeglasses', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sportsball', 38: 'kite', 39: 'baseballbat',
    40: 'baseballglove', 41: 'skateboard', 42: 'surfboard', 43: 'tennisracket',
    44: 'bottle', 45: 'plate', 46: 'wineglass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hotdog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'sofa',
    64: 'pottedplant', 65: 'bed', 66: 'mirror', 67: 'diningtable', 68: 'window',
    69: 'desk', 70: 'toilet', 71: 'door', 72: 'tv', 73: 'laptop',
    74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cellphone', 78: 'microwave',
    79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator', 83: 'blender',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddybear',
    89: 'hairdrier', 90: 'toothbrush', 91: 'hairbrush'
}


COLORS = plt.cm.tab20.colors  
COLOR_MAP = {label: tuple([int(c * 255) for c in COLORS[i % len(COLORS)]]) for i, label in enumerate(label_map)}



def draw(image, labels, boxes, scores, thrh=0.25):
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default() 
    labels, boxes, scores = labels[scores > thrh], boxes[scores > thrh], scores[scores > thrh]

    for j, box in enumerate(boxes):
        category = labels[j].item()
        color = COLOR_MAP.get(category, (255, 255, 255))  
        box = list(map(int, box))

        
        draw.rectangle(box, outline=color, width=3)

        text = f"{label_map[category]} {scores[j].item():.2f}"
        text_bbox = draw.textbbox((0, 0), text, font=font)  
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        
        text_background = [box[0], box[1] - text_height - 2, box[0] + text_width + 4, box[1]]
        draw.rectangle(text_background, fill=color)
       
        draw.text((box[0] + 2, box[1] - text_height - 2), text, fill="black", font=font)

    return image


def process_dataset(model, dataset_path, output_path, thrh=0.25, size=(640, 640), vit_backbone=False):
    import time
    os.makedirs(output_path, exist_ok=True)
    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith(('.jpg', '.png'))]

    transforms = T.Compose([
        T.Resize(size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
                if vit_backbone else T.Lambda(lambda x: x)
    ])

    print(f"Found {len(image_paths)} images in validation set...")
    
    # Warm up model to ensure accurate time measurement
    if len(image_paths) > 0:
        print("Warming up model on GPU...")
        dummy_im = torch.zeros((1, 3, size[0], size[1])).cuda()
        dummy_size = torch.tensor([[size[0], size[1]]]).cuda()
        for _ in range(10):
            _ = model(dummy_im, dummy_size)
        torch.cuda.synchronize()

    total_inference_time = 0.0
    start_loop_time = time.time()

    for idx, file_path in enumerate(image_paths):
        im_pil = Image.open(file_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).cuda()

        # 图像预处理
        im_data = transforms(im_pil).unsqueeze(0).cuda()
        
        # Inference with timing
        torch.cuda.synchronize()
        t_start = time.time()
        output = model(im_data, orig_size)
        torch.cuda.synchronize()
        total_inference_time += (time.time() - t_start)
        
        labels, boxes, scores = output[0]['labels'], output[0]['boxes'], output[0]['scores']

        # 绘制结果
        vis_image = draw(im_pil.copy(), labels, boxes, scores, thrh)
        save_path = os.path.join(output_path, f"vis_{os.path.basename(file_path)}")
        vis_image.save(save_path)

        if idx % 100 == 0:
            print(f"Processed {idx}/{len(image_paths)} images...")

    total_loop_time = time.time() - start_loop_time
    num_images = len(image_paths)

    if num_images > 0:
        avg_inf_ms = (total_inference_time / num_images) * 1000.0
        inf_fps = num_images / total_inference_time if total_inference_time > 0 else 0.0
        overall_fps = num_images / total_loop_time if total_loop_time > 0 else 0.0
        
        print("\n" + "="*40)
        print(f"📊 DEIMv2 效能評估報告 (Performance Report)")
        print(f"  - 總處理圖片數: {num_images} 張")
        print(f"  - 純模型推論平均時間: {avg_inf_ms:.2f} ms / 張")
        print(f"  - 純模型推論速度 (Inference FPS): {inf_fps:.2f} 張/秒")
        print(f"  - 整體系統速度 (Overall FPS, 含I/O與存檔): {overall_fps:.2f} 張/秒")
        print("="*40 + "\n")
    else:
        print("未偵測到任何圖片。")

    print("Visualization complete. Results saved in:", output_path)


def main(args):
    """Main function"""
    cfg = YAMLConfig(args.config, resume=args.resume)

    if 'HGNetv2' in cfg.yaml_cfg:
        cfg.yaml_cfg['HGNetv2']['pretrained'] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']
    else:
        raise AttributeError('Only support resume to load model.state_dict by now.')

    # Load train mode state and convert to deploy mode
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.eval().cuda()
            self.postprocessor = cfg.postprocessor.eval().cuda()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()
    img_size = cfg.yaml_cfg["eval_spatial_size"]
    vit_backbone = cfg.yaml_cfg.get('DINOv3STAs', False)

    process_dataset(model, args.dataset, args.output, thrh=0.25, size=img_size, vit_backbone=vit_backbone)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-d', '--dataset', type=str, default='./data/fiftyone/validation/data')
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to save visualized results")
    args = parser.parse_args()
    main(args)
