# app/services/ocr_service.py

import os
import re
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from paddleocr import PaddleOCR
from .pdf_service import find_text_coordinates, extract_text_from_coords


# --- 태그 정규식 (핵심) ---
# 예시 매칭: "AA-123B", "AA 123 B", "aa123", "P-1001", "T 25"
# - 영문 prefix 1~6자
# - 구분자(공백/하이픈/언더스코어/점) 허용
# - 숫자 1~6자리
# - 선택적 suffix(영문 1~3자)
TAG_PATTERN = re.compile(
    r'(?<![A-Za-z0-9])'           # 왼쪽 경계가 영숫자 아님
    r'([A-Za-z]{1,6})'            # prefix
    r'[\s\-_\.]*'                 # 구분자
    r'([0-9]{1,6})'               # number
    r'(?:[\s\-_\.]*([A-Za-z]{1,3}))?'  # optional suffix
    r'(?![A-Za-z0-9])',           # 오른쪽 경계가 영숫자 아님
    re.IGNORECASE
)

def _normalize_tag_string(raw_text: str):
    """
    OCR 결과 문자열에서 태그 패턴을 찾아 정규화된 태그와 컴포넌트를 반환.
    반환: (normalized, prefix, number, suffix) 또는 (None, None, None, None)
    normalized 예: 'AA-123B' (suffix 없으면 'AA-123')
    """
    if not raw_text:
        return (None, None, None, None)

    # 약한 노이즈 제거 (유사 구분자 통일)
    cleaned = raw_text.replace('—', '-').replace('–', '-').replace('·', '.')
    cleaned = re.sub(r'\s+', ' ', cleaned)  # 공백 압축

    m = TAG_PATTERN.search(cleaned)
    if not m:
        return (None, None, None, None)

    pref = (m.group(1) or '').upper()
    num  = (m.group(2) or '')
    suf  = (m.group(3) or '')
    sufU = suf.upper() if suf else ''

    normalized = f"{pref}-{num}{sufU}" if sufU else f"{pref}-{num}"
    return (normalized, pref, num, sufU)


# --- PaddleOCR 싱글턴 ---
_OCR_SINGLETON = None
def _get_ocr():
    global _OCR_SINGLETON
    if _OCR_SINGLETON is None:
        print("PaddleOCR 모델을 로드 중입니다...")
        _OCR_SINGLETON = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )
    return _OCR_SINGLETON


def _safe_crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img[y1:y2, x1:x2]
    if crop is None or crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return None
    return crop


def _parse_paddle_result(ocr_result) -> str:
    """
    PaddleOCR 결과를 버전별 포맷 차이에 견고하게 파싱하여 텍스트만 추출.
    """
    if not ocr_result:
        return ""

    texts = []

    first = ocr_result[0]
    # dict 포맷 (predict() 계열)
    if isinstance(first, dict):
        for key in ('rec_texts', 'texts'):
            v = first.get(key)
            if isinstance(v, list):
                return " ".join(map(str, v)).strip()

        arr = first.get('res') or []
        if isinstance(arr, list):
            for line in arr:
                try:
                    if isinstance(line, list) and len(line) >= 2:
                        val = line[1]
                        if isinstance(val, (tuple, list)) and len(val) >= 1:
                            texts.append(str(val[0]))
                        elif isinstance(val, str):
                            texts.append(val)
                except Exception:
                    continue
        return " ".join(texts).strip()

    # list 포맷 ([ [[pts]], (text, score) ] ...)
    for line in ocr_result:
        try:
            if isinstance(line, list) and len(line) >= 2:
                val = line[1]
                if isinstance(val, (tuple, list)) and len(val) >= 1:
                    texts.append(str(val[0]))
                elif isinstance(val, str):
                    texts.append(val)
        except Exception:
            continue
    return " ".join(texts).strip()


def perform_ocr_on_detections_and_export(detection_results: list, output_excel_path: str,
                                         pdf_path: str, reference_text: str, reference_page: int,
                                         class_names: list):
    """
    탐지된 객체들에 대해 메모리에서 직접 OCR을 수행하고 결과를 엑셀 파일로 저장.
    + 태그 정규식 기반 정제 컬럼 추가 (핵심 반영)
    출력 컬럼:
      ["PID NO", "Original Page", "Object Index", "Class Name", "Detection Score",
       "Recognized Text", "Tag Matched", "Tag (Normalized)", "Tag_Prefix", "Tag_Number", "Tag_Suffix"]
    """

    # --- PID NO 추출 ---
    pid_no_map = {}
    print(f"기준 텍스트 '{reference_text}'의 좌표를 {reference_page} 페이지에서 찾는 중...")
    pid_no_coords = find_text_coordinates(pdf_path, reference_text, reference_page)

    if not pid_no_coords:
        print("⚠️ 기준 텍스트의 좌표를 찾지 못해 PID NO 추출을 건너뜁니다.")
    else:
        print(f"좌표 찾음: {pid_no_coords}. 모든 페이지에서 해당 좌표의 PID NO를 추출합니다.")
        import fitz
        doc = fitz.open(pdf_path)
        for i, _ in enumerate(doc):
            page_key = f"page_{i:03d}"
            pid_no = extract_text_from_coords(pdf_path, i, pid_no_coords, reference_text)
            pid_no_map[page_key] = pid_no
        doc.close()

    # --- OCR 인스턴스 ---
    ocr = _get_ocr()

    # --- 총 객체 수 확인 ---
    total_detections = sum(len(res.get('detections', [])) for res in detection_results if res.get('success'))
    if total_detections == 0:
        print("⚠️ 탐지된 객체가 없어 OCR을 건너뜁니다.")
        cols = ["PID NO", "Original Page", "Object Index", "Class Name", "Detection Score",
                "Recognized Text", "Tag Matched", "Tag (Normalized)", "Tag_Prefix", "Tag_Number", "Tag_Suffix"]
        df = pd.DataFrame(columns=cols)
        os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
        df.to_excel(output_excel_path, index=False, engine='openpyxl')
        return

    print(f"📝 총 {total_detections}개 탐지된 객체에 대해 OCR 수행 중...")

    rows = []

    for result in tqdm(detection_results, desc="페이지별 OCR 처리"):
        if not result.get('success') or not result.get('detections'):
            continue

        # 메모리 이미지 우선 사용
        full_image = result.get('image_ndarray')
        image_path = result.get('image_path')
        if full_image is None:
            full_image = cv2.imread(image_path)
        if full_image is None:
            print(f"⚠️ 이미지 로드 실패: {image_path}")
            continue

        base_name = os.path.splitext(os.path.basename(image_path))[0]  # "page_000"
        original_page_key = base_name
        pid_no = pid_no_map.get(original_page_key, "N/A")

        for obj_idx, (x1, y1, x2, y2, score, label_id) in enumerate(result['detections']):
            try:
                crop = _safe_crop(full_image, x1, y1, x2, y2)
                if crop is None:
                    print(f"⚠️ 잘못된 crop 크기: {image_path}, 객체 {obj_idx}")
                    continue

                # OCR 수행
                ocr_result = ocr.ocr(crop)
                recognized_text = _parse_paddle_result(ocr_result).strip()

                # --- 태그 정규식 정제 (핵심) ---
                normalized, pref, num, suf = _normalize_tag_string(recognized_text)
                tag_matched = bool(normalized)

                # 클래스명 안전 조회
                if 0 <= int(label_id) < len(class_names):
                    cls_name = class_names[int(label_id)]
                else:
                    cls_name = str(label_id)

                rows.append({
                    "PID NO": pid_no,
                    "Original Page": original_page_key,
                    "Object Index": obj_idx,
                    "Class Name": cls_name,
                    "Detection Score": round(float(score), 3),
                    "Recognized Text": recognized_text,
                    "Tag Matched": tag_matched,
                    "Tag (Normalized)": normalized or "",
                    "Tag_Prefix": pref or "",
                    "Tag_Number": num or "",
                    "Tag_Suffix": suf or ""
                })

            except Exception as e:
                print(f"⚠️ OCR 처리 오류 ({image_path}, 객체 {obj_idx}): {e}")
                if 0 <= int(label_id) < len(class_names):
                    cls_name = class_names[int(label_id)]
                else:
                    cls_name = "Unknown"
                rows.append({
                    "PID NO": pid_no,
                    "Original Page": original_page_key,
                    "Object Index": obj_idx,
                    "Class Name": cls_name,
                    "Detection Score": round(float(score), 3),
                    "Recognized Text": f"OCR Error: {e}",
                    "Tag Matched": False,
                    "Tag (Normalized)": "",
                    "Tag_Prefix": "",
                    "Tag_Number": "",
                    "Tag_Suffix": ""
                })

    # --- 결과 저장 (정제 컬럼 포함) ---
    os.makedirs(os.path.dirname(output_excel_path), exist_ok=True)
    cols = ["PID NO", "Original Page", "Object Index", "Class Name", "Detection Score",
            "Recognized Text", "Tag Matched", "Tag (Normalized)", "Tag_Prefix", "Tag_Number", "Tag_Suffix"]

    if rows:
        df = pd.DataFrame(rows, columns=cols)
        df.to_excel(output_excel_path, index=False, engine='openpyxl')
        print(f"\n✅ OCR + 태그 정제 결과가 '{output_excel_path}' 파일에 저장되었습니다.")
        print(f"📊 총 {len(rows)}개 객체 처리 완료 (태그 매칭: {sum(1 for r in rows if r['Tag Matched'])}건)")
    else:
        df = pd.DataFrame(columns=cols)
        df.to_excel(output_excel_path, index=False, engine='openpyxl')
        print("⚠️ OCR 결과가 비어 있거나 처리하지 못했습니다.")
