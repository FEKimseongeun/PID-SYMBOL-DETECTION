# app/services/detection_service.py
import os
import yaml
import cv2
import torch
import torchvision.ops as ops
import numpy as np
from tqdm import tqdm
from typing import List, Tuple
import time
from collections import defaultdict

# 서비스 및 헬퍼 함수들을 명시적으로 import
from .pdf_service import convert_pdf_to_images
from .ocr_service import perform_ocr_on_detections_and_export


# --- 헬퍼 함수들 ---

def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def load_model(model_path, device='cpu'):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"오류: 모델 파일을 찾을 수 없습니다 -> {model_path}")
    # torchvision Faster R-CNN 직렬화된 전체 객체(.pth) 기준
    # 필요 시 사용자 환경에 맞게 load_state_dict 방식으로 교체
    model = torch.load(model_path, map_location=device, weights_only=False)
    model.eval().to(device)
    return model


def sliding_window_coords(h: int, w: int, tile_size: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    assert tile_size > overlap, f"tile_size({tile_size}) must be greater than overlap({overlap})"
    coords = []
    step = tile_size - overlap
    for y in range(0, h, step):
        for x in range(0, w, step):
            x2 = min(x + tile_size, w)
            y2 = min(y + tile_size, h)
            if x2 - x > 0 and y2 - y > 0:
                coords.append((x, y, x2, y2))
    return coords


def detect_objects_in_tiles(model, full_image, tiles_coords, device, threshold, batch_size=8):
    """
    full_image: BGR ndarray
    return: [(x1, y1, x2, y2, score, label_id), ...]
    """
    all_detections = []

    # (타일 이미지, (x_off, y_off)) 리스트
    tiles_with_offsets = [(full_image[y1:y2, x1:x2], (x1, y1)) for x1, y1, x2, y2 in tiles_coords]

    for i in range(0, len(tiles_with_offsets), batch_size):
        batch = tiles_with_offsets[i:i + batch_size]
        imgs_tensor = []
        valids = []  # 유효 타일 인덱스 (잘린 타일 방지)
        for bi, (img, _) in enumerate(batch):
            if img is None or img.size == 0:
                continue
            h, w = img.shape[:2]
            if h < 2 or w < 2:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_chw = np.transpose(img_rgb, (2, 0, 1)).copy()  # C,H,W
            tensor = torch.from_numpy(img_chw).float().div(255.0).to(device)
            imgs_tensor.append(tensor)
            valids.append(bi)

        if not imgs_tensor:
            continue

        with torch.no_grad():
            preds = model(imgs_tensor)  # torchvision detection model: list of dict

        # preds 는 유효 타일 수만큼 반환됨
        for local_idx, pred in zip(valids, preds):
            (x_off, y_off) = batch[local_idx][1]
            boxes = pred.get('boxes', [])
            labels = pred.get('labels', [])
            scores = pred.get('scores', [])
            if len(boxes) == 0:
                continue
            boxes = boxes.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            scores = scores.detach().cpu().numpy()

            for box, label, score in zip(boxes, labels, scores):
                if float(score) >= float(threshold):
                    x1, y1, x2, y2 = map(int, box.tolist())
                    if x2 <= x1 or y2 <= y1:
                        continue
                    all_detections.append((
                        x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off,
                        float(score), int(label)
                    ))
    return all_detections


def annotate_image(image_path: str, detections: list, class_names: list, output_path: str):
    """이미지에 탐지 결과를 시각화해서 저장"""
    image = cv2.imread(image_path)
    if image is None:
        return
    for x1, y1, x2, y2, score, label_id in detections:
        label_name = class_names[int(label_id)] if 0 <= int(label_id) < len(class_names) else str(label_id)
        text = f"{label_name}: {score:.2f}"

        # 박스
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 라벨 배경 위치 보정 (상단 바깥으로 나가지 않게)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(0, y1 - th - 8)
        cv2.rectangle(image, (x1, ty), (x1 + tw + 6, ty + th + 6), (0, 255, 0), -1)
        cv2.putText(image, text, (x1 + 3, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, image)


def _class_wise_nms(dets: list, iou_thresh: float) -> list:
    """
    dets: [(x1,y1,x2,y2,score,label_id), ...]
    클래스별로 NMS 수행 후 합치기
    """
    if not dets:
        return []
    by_label = defaultdict(list)
    for d in dets:
        by_label[int(d[5])].append(d)

    merged = []
    for lbl, group in by_label.items():
        boxes = torch.tensor([g[:4] for g in group], dtype=torch.float32)
        scores = torch.tensor([g[4] for g in group], dtype=torch.float32)
        keep = ops.nms(boxes, scores, iou_thresh).cpu().numpy().tolist()
        merged.extend([group[i] for i in keep])
    return merged


def process_single_image(args, model, device):
    """단일 이미지에서 객체 탐지 수행"""
    image_path, config, output_dir, tile_size, overlap, iou_thresh, conf_thresh, target_label_ids = args

    annotated_images_dir = os.path.join(output_dir, "2_annotated_images")

    full_image = cv2.imread(image_path)
    if full_image is None:
        return {"success": False, "message": f"이미지 로드 실패: {image_path}", "detections": []}

    h, w, _ = full_image.shape
    tiles_coords = sliding_window_coords(h, w, tile_size, overlap)

    # 객체 탐지 수행
    raw_detections = detect_objects_in_tiles(model, full_image, tiles_coords, device, conf_thresh)
    if not raw_detections:
        return {"success": False, "message": f"객체 미탐지: {os.path.basename(image_path)}", "detections": []}

    # 타겟 라벨만 필터링
    target_detections = [d for d in raw_detections if d[5] in target_label_ids]
    if not target_detections:
        return {"success": False, "message": f"타겟 객체 미탐지: {os.path.basename(image_path)}", "detections": []}

    # --- 클래스별 NMS 적용 (중요 수정) ---
    final_detections = _class_wise_nms(target_detections, iou_thresh)

    # 어노테이션된 이미지 저장 (시각화용)
    output_image_path = os.path.join(annotated_images_dir, os.path.basename(image_path))
    annotate_image(image_path, final_detections, config['class_names'], output_image_path)

    return {
        "success": True,
        "message": f"처리 완료: {os.path.basename(image_path)} ({len(final_detections)}개 탐지)",
        "detections": final_detections,
        "image_shape": (h, w, full_image.shape[2]),
        "image_path": image_path,
        # OCR에서 다시 디스크 I/O 하지 않도록 메모리 전달 (중요 수정)
        "image_ndarray": full_image
    }


# --- 메인 파이프라인 함수 ---

def run_analysis_pipeline(job_id, pdf_path, settings, output_dir, update_job_status_func,
                          config_path="config/config.yaml"):
    total_start_time = time.time()

    config = load_config(config_path)
    class_names = config.get('class_names', [])

    # targetLabels(str) -> 실제 존재하는 클래스명만 index 추출
    requested = settings.get('targetLabels', [])
    target_label_ids = set()
    missing_labels = []
    for label in requested:
        try:
            idx = class_names.index(label)
            target_label_ids.add(idx)
        except ValueError:
            missing_labels.append(label)
    if missing_labels:
        print(f"[경고] config.class_names에 없는 라벨 무시됨: {missing_labels}")

    # --- 단계 1: PDF 변환 ---
    step1_start_time = time.time()
    update_job_status_func(job_id, 'running', 10, 'PDF를 이미지로 변환 중...')
    source_images_dir = os.path.join(output_dir, "1_source_images")
    dpi = int(config.get('pdf_render_dpi', 300))
    image_paths = convert_pdf_to_images(pdf_path, source_images_dir, dpi)
    print(f"⏱️  PDF 변환 완료. 소요 시간: {time.time() - step1_start_time:.2f}초")

    # --- 단계 2: 객체 탐지 ---
    step2_start_time = time.time()
    update_job_status_func(job_id, 'running', 30, f'{len(image_paths)}개 페이지에 대한 객체 탐지 시작...')

    # 모델 로드
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    model = load_model(config['model_path'], device)
    print(f"🧠 모델 로드 완료. ({device_str} 사용)")

    # 탐지 파라미터 (기본값)
    TILE_SIZE = 1024
    OVERLAP = 200
    IOU_THRESHOLD = 0.5
    CONFIDENCE_THRESHOLD = 0.4

    task_args = [(path, config, output_dir, TILE_SIZE, OVERLAP, IOU_THRESHOLD, CONFIDENCE_THRESHOLD, target_label_ids)
                 for path in image_paths]

    # 모든 페이지의 탐지 결과를 저장
    all_detection_results = []

    print("➡️  순차적 이미지 처리 시작...")
    for i, args in enumerate(task_args):
        result = process_single_image(args, model, device)
        all_detection_results.append(result)
        progress = 30 + int(50 * (i + 1) / max(1, len(image_paths)))  # 30~80% 할당
        update_job_status_func(job_id, 'running', progress, f'페이지 처리 중 ({i + 1}/{len(image_paths)})...')
        print(f"    - {result['message']}")

    print(f"⏱️  모든 이미지 객체 탐지 완료. 소요 시간: {time.time() - step2_start_time:.2f}초")

    # --- 단계 3: OCR 및 엑셀 추출 (개선된 방식) ---
    step3_start_time = time.time()
    update_job_status_func(job_id, 'running', 85, '탐지된 객체에 대한 OCR 처리 시작...')

    excel_path = os.path.join(output_dir, "ocr_and_detection_results.xlsx")

    # 탐지 결과 직접 전달 (메모리 이미지 포함)
    perform_ocr_on_detections_and_export(
        detection_results=all_detection_results,
        output_excel_path=excel_path,
        pdf_path=pdf_path,
        reference_text=settings['referenceText'],
        reference_page=settings['referencePage'],
        class_names=class_names
    )

    print(f"⏱️  OCR 및 엑셀 저장 완료. 소요 시간: {time.time() - step3_start_time:.2f}초")

    annotated_dir = os.path.join(output_dir, '2_annotated_images')

    # 총 탐지된 심볼 수 계산
    total_symbols = sum(len(result.get('detections', [])) for result in all_detection_results if result['success'])

    print(f"\n✨ 총 소요 시간: {time.time() - total_start_time:.2f}초")
    return {
        'success': True,
        'total_pages': len(image_paths),
        'total_symbols': total_symbols,
        'excel_path': excel_path,
        'annotated_images_dir': annotated_dir
    }
