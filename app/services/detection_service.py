# app/services/detection_service.py
import os
import yaml
import cv2
import torch
import torchvision.ops as ops
import numpy as np
import multiprocessing
from tqdm import tqdm
from typing import List, Tuple

# 서비스 및 헬퍼 함수들을 명시적으로 import
from .pdf_service import convert_pdf_to_images
from .ocr_service import perform_ocr_on_crops_and_export


# --- 헬퍼 함수들 ---

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model(model_path, device='cpu'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"오류: 모델 파일을 찾을 수 없습니다 -> {model_path}")
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval().to(device)
    return model


# ... (sliding_window_coords, detect_objects_in_tiles, annotate_image, crop_and_save_objects 함수는 여기에 그대로 복사) ...

def sliding_window_coords(h: int, w: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    coords = []
    step = tile_size - overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            coords.append((x, y, min(x + tile_size, w), min(y + tile_size, h)))
    return coords


def detect_objects_in_tiles(model, full_image, tiles_coords, device, threshold, batch_size=8):
    all_detections = []
    tiles_with_offsets = [(full_image[y1:y2, x1:x2], (x1, y1)) for x1, y1, x2, y2 in tiles_coords]
    for i in range(0, len(tiles_with_offsets), batch_size):
        batch = tiles_with_offsets[i:i + batch_size]
        imgs_tensor = []
        for img, _ in batch:
            if img.shape[0] == 0 or img.shape[1] == 0: continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_chw = np.transpose(img_rgb, (2, 0, 1))
            tensor = torch.from_numpy(np.ascontiguousarray(img_chw)).float().div(255).to(device)
            imgs_tensor.append(tensor)
        if not imgs_tensor: continue
        with torch.no_grad():
            preds = model(imgs_tensor)
        for (img_arr, (x_off, y_off)), pred in zip(batch, preds):
            for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
                if score.item() >= threshold:
                    x1, y1, x2, y2 = map(int, box.cpu().numpy())
                    all_detections.append((x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off, score.item(), label.item()))
    return all_detections


def annotate_image(image_path: str, detections: list, class_names: list, output_path: str):
    image = cv2.imread(image_path)
    for x1, y1, x2, y2, score, label_id in detections:
        label, text = class_names[int(label_id)], f"{class_names[int(label_id)]}: {score:.2f}"
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(image, (x1, y1 - h - 10), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def crop_and_save_objects(full_image, detections: list, class_names: list, output_dir: str, original_image_name: str):
    crop_output_dir = os.path.join(output_dir, "3_cropped_objects")
    os.makedirs(crop_output_dir, exist_ok=True)
    base_name = os.path.splitext(original_image_name)[0]
    for i, (x1, y1, x2, y2, score, label_id) in enumerate(detections):
        class_name = class_names[int(label_id)]
        cropped_img = full_image[y1:y2, x1:x2]
        crop_filename = f"{base_name}_{i:03d}_{class_name}_{score:.2f}.png"
        crop_path = os.path.join(crop_output_dir, crop_filename)
        if cropped_img.shape[0] > 0 and cropped_img.shape[1] > 0:
            cv2.imwrite(crop_path, cropped_img)


# --- 병렬 처리 함수들 ---

def init_worker(model_path, device_str):
    global worker_model, worker_device
    worker_device = torch.device(device_str)
    worker_model = load_model(model_path, worker_device)
    print(f"프로세스 {os.getpid()}: 모델 로드 완료.")


def process_single_image(args):
    image_path, config, output_dir, tile_size, overlap, iou_thresh, conf_thresh, target_label_ids = args
    global worker_model, worker_device
    annotated_images_dir = os.path.join(output_dir, "2_annotated_images")
    full_image = cv2.imread(image_path)
    if full_image is None: return f"이미지 로드 실패: {image_path}"
    h, w, _ = full_image.shape
    tiles_coords = sliding_window_coords(h, w, tile_size, overlap)
    raw_detections = detect_objects_in_tiles(worker_model, full_image, tiles_coords, worker_device, conf_thresh)
    if not raw_detections: return f"객체 미탐지: {os.path.basename(image_path)}"
    target_detections = [d for d in raw_detections if d[5] in target_label_ids]
    if not target_detections: return f"타겟 객체 미탐지: {os.path.basename(image_path)}"
    boxes_tensor = torch.tensor([d[:4] for d in target_detections], dtype=torch.float32)
    scores_tensor = torch.tensor([d[4] for d in target_detections], dtype=torch.float32)
    keep_indices = ops.nms(boxes_tensor, scores_tensor, iou_thresh).cpu().numpy()
    final_detections = [target_detections[i] for i in keep_indices]
    output_image_path = os.path.join(annotated_images_dir, os.path.basename(image_path))
    annotate_image(image_path, final_detections, config['class_names'], output_image_path)
    crop_and_save_objects(full_image, final_detections, config['class_names'], output_dir, os.path.basename(image_path))
    return f"처리 완료: {os.path.basename(image_path)} ({len(final_detections)}개 탐지)"


# --- 메인 파이프라인 함수 ---

def run_analysis_pipeline(job_id, pdf_path, settings, output_dir, update_job_status_func, config_path="config/config.yaml"):
    config = load_config(config_path)
    class_names = config.get('class_names', [])
    # targetLabels는 여전히 사용자 설정에 따라 받습니다.
    target_label_ids = {class_names.index(label) for label in settings['targetLabels'] if label in class_names}

    update_job_status_func(job_id, 'running', 10, 'PDF를 이미지로 변환 중...')
    source_images_dir = os.path.join(output_dir, "1_source_images")
    image_paths = convert_pdf_to_images(pdf_path, source_images_dir, config.get('pdf_render_dpi', 300))

    update_job_status_func(job_id, 'running', 30, f'{len(image_paths)}개 페이지에 대한 객체 탐지 시작...')
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # --- ✨ [수정] 파라미터를 고정된 값으로 하드코딩 ---
    TILE_SIZE = 1024
    OVERLAP = 200
    IOU_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 0.4

    task_args = [(
        image_path, config, output_dir,
        TILE_SIZE,
        OVERLAP,
        IOU_THRESHOLD,
        CONFIDENCE_THRESHOLD,
        target_label_ids
    ) for image_path in image_paths]
    # -----------------------------------------------

    num_processes = min(os.cpu_count(), 8)
    with multiprocessing.Pool(processes=num_processes, initializer=init_worker,
                              initargs=(config['model_path'], device_str)) as pool:
        for i, result in enumerate(pool.imap_unordered(process_single_image, task_args)):
            progress = 30 + int(60 * (i + 1) / len(image_paths))
            update_job_status_func(job_id, 'running', progress, f'페이지 처리 중... {result}')

    update_job_status_func(job_id, 'running', 90, '객체 탐지 완료, OCR 및 엑셀 추출 시작...')
    cropped_dir = os.path.join(output_dir, "3_cropped_objects")
    excel_path = os.path.join(output_dir, "ocr_and_detection_results.xlsx")

    # --- ✨ [수정] OCR 함수 호출 시 필요한 파라미터 추가 ---
    perform_ocr_on_crops_and_export(
        cropped_images_dir=cropped_dir,
        output_excel_path=excel_path,
        pdf_path=pdf_path,
        reference_text=settings['referenceText'],
        reference_page=settings['referencePage']
    )

    annotated_dir = os.path.join(output_dir, '2_annotated_images')

    # ✨ [개선] 최종 결과에 통계 정보 추가
    total_symbols = len(os.listdir(cropped_dir)) if os.path.exists(cropped_dir) else 0

    return {
        'success': True,
        'total_pages': len(image_paths),
        'total_symbols': total_symbols,  # 통계 정보 추가
        'excel_path': excel_path,
        'annotated_images_dir': annotated_dir
    }