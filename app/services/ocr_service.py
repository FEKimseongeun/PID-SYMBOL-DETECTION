# app/services/ocr_service.py

import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from paddleocr import PaddleOCR
from .pdf_service import find_text_coordinates, extract_text_from_coords


def perform_ocr_on_detections_and_export(detection_results: list, output_excel_path: str,
                                         pdf_path: str, reference_text: str, reference_page: int,
                                         class_names: list):
    """탐지된 객체들에 대해 메모리에서 직접 OCR을 수행하고 결과를 엑셀 파일로 저장합니다."""

    # PID NO 추출 로직
    pid_no_map = {}
    print(f"기준 텍스트 '{reference_text}'의 좌표를 {reference_page} 페이지에서 찾는 중...")
    pid_no_coords = find_text_coordinates(pdf_path, reference_text, reference_page)

    if not pid_no_coords:
        print("⚠️ 기준 텍스트의 좌표를 찾지 못해 PID NO 추출을 건너뜁니다.")
    else:
        print(f"좌표 찾음: {pid_no_coords}. 모든 페이지에서 해당 좌표의 PID NO를 추출합니다.")
        import fitz
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            page_key = f"page_{i:03d}"
            pid_no = extract_text_from_coords(pdf_path, i, pid_no_coords, reference_text)
            pid_no_map[page_key] = pid_no
        doc.close()

    # PaddleOCR 초기화
    print("PaddleOCR 모델을 로드 중입니다...")
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)

    ocr_results = []
    total_detections = sum(len(result.get('detections', [])) for result in detection_results if result['success'])

    if total_detections == 0:
        print("⚠️ 탐지된 객체가 없어 OCR을 건너뜁니다.")
        # 빈 엑셀 파일 생성
        df = pd.DataFrame(
            columns=["PID NO", "Original Page", "Object Index", "Class Name", "Detection Score", "Recognized Text"])
        df.to_excel(output_excel_path, index=False, engine='openpyxl')
        return

    print(f"📝 총 {total_detections}개 탐지된 객체에 대해 OCR 수행 중...")

    processed_count = 0

    for result in tqdm(detection_results, desc="페이지별 OCR 처리"):
        if not result['success'] or not result.get('detections'):
            continue

        image_path = result['image_path']
        detections = result['detections']

        # 원본 이미지 로드
        full_image = cv2.imread(image_path)
        if full_image is None:
            print(f"⚠️ 이미지 로드 실패: {image_path}")
            continue

        # 이미지 파일명에서 페이지 정보 추출
        base_name = os.path.splitext(os.path.basename(image_path))[0]  # "page_000" 형태
        original_page_key = base_name

        # 해당 페이지의 PID NO 가져오기
        pid_no = pid_no_map.get(original_page_key, "N/A")

        # 각 탐지된 객체에 대해 OCR 수행
        for obj_idx, (x1, y1, x2, y2, score, label_id) in enumerate(detections):
            try:
                # 메모리에서 직접 crop
                cropped_img = full_image[y1:y2, x1:x2]

                if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                    print(f"⚠️ 잘못된 crop 크기: {image_path}, 객체 {obj_idx}")
                    continue

                # OCR 수행 (메모리의 numpy 배열에서 직접)
                # PaddleOCR은 numpy 배열을 직접 받을 수 있습니다
                ocr_result = ocr.ocr(cropped_img)

                recognized_text = ""
                if ocr_result and len(ocr_result) > 0:
                    # 디버깅 출력 제거하고 실제 데이터 파싱
                    result_data = ocr_result[0]

                    # 결과가 딕셔너리 형태인 경우 (실제 확인된 형태)
                    if isinstance(result_data, dict):
                        if 'rec_texts' in result_data:
                            # rec_texts 필드에서 직접 텍스트 추출
                            texts = result_data['rec_texts']
                            if isinstance(texts, list):
                                recognized_text = " ".join(str(text) for text in texts)
                            else:
                                recognized_text = str(texts)

                    # 결과가 리스트 형태인 경우 (일반적인 PaddleOCR 형태)
                    elif isinstance(result_data, list):
                        texts = []
                        for line in result_data:
                            if isinstance(line, list) and len(line) >= 2:
                                # [[[좌표]], (텍스트, 신뢰도)] 형태
                                if isinstance(line[1], tuple) and len(line[1]) >= 1:
                                    texts.append(str(line[1][0]))
                                elif isinstance(line[1], str):
                                    texts.append(line[1])
                        recognized_text = " ".join(texts)

                class_name = class_names[int(label_id)]

                ocr_results.append({
                    "PID NO": pid_no,
                    "Original Page": original_page_key,
                    "Object Index": obj_idx,
                    "Class Name": class_name,
                    "Detection Score": round(score, 3),
                    "Recognized Text": recognized_text.strip()
                })

                processed_count += 1

            except Exception as e:
                print(f"⚠️ OCR 처리 오류 ({image_path}, 객체 {obj_idx}): {e}")
                ocr_results.append({
                    "PID NO": pid_no,
                    "Original Page": original_page_key,
                    "Object Index": obj_idx,
                    "Class Name": class_names[int(label_id)] if int(label_id) < len(class_names) else "Unknown",
                    "Detection Score": round(score, 3),
                    "Recognized Text": f"OCR Error: {e}"
                })

    # 결과를 엑셀로 저장
    if ocr_results:
        df = pd.DataFrame(ocr_results)
        column_order = ["PID NO", "Original Page", "Object Index", "Class Name", "Detection Score", "Recognized Text"]
        df = df[column_order]
        df.to_excel(output_excel_path, index=False, engine='openpyxl')
        print(f"\n✅ OCR 결과가 '{output_excel_path}' 파일에 성공적으로 저장되었습니다.")
        print(f"📊 총 {len(ocr_results)}개 객체 처리 완료")
    else:
        print("⚠️ OCR을 수행할 객체가 없거나 결과를 처리하지 못했습니다.")


# ✨ 기존 함수는 하위 호환성을 위해 유지 (사용하지 않음)
def perform_ocr_on_crops_and_export(cropped_images_dir: str, output_excel_path: str,
                                    pdf_path: str, reference_text: str, reference_page: int):
    """기존 방식 - 하위 호환성을 위해 유지"""
    print("⚠️ 기존 방식의 OCR 함수가 호출되었습니다. 새로운 최적화된 방식을 사용해주세요.")
    pass