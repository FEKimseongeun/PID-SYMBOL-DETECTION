# app/services/ocr_service.py

import os
import pandas as pd
from tqdm import tqdm
from paddleocr import PaddleOCR
# ✨ pdf_service에서 텍스트 추출 함수를 import
from .pdf_service import find_text_coordinates, extract_text_from_coords

# ✨ 함수가 PDF 관련 인자를 받도록 수정
def perform_ocr_on_crops_and_export(cropped_images_dir: str, output_excel_path: str, pdf_path: str, reference_text: str, reference_page: int):
    """크롭된 이미지들에 대해 OCR을 수행하고 PID NO를 추가하여 결과를 엑셀 파일로 저장합니다."""
    if not os.path.exists(cropped_images_dir):
        print("⚠️ 크롭된 이미지가 없어 OCR을 건너뜁니다.")
        return



    # ✨ [핵심] PID NO 추출 로직 추가
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
    print("PaddleOCR 모델을 로드 중입니다...")
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False)
    ocr_results = []
    image_files = [f for f in os.listdir(cropped_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for filename in tqdm(image_files, desc="크롭 이미지 OCR 처리"):
        try:
            parts = os.path.splitext(filename)[0].split('_')
            original_page_key = f"{parts[0]}_{parts[1]}"
            img_path = os.path.join(cropped_images_dir, filename)
            result = ocr.ocr(img_path)
            recognized_text = ""
            if result and result[0]:
                lines = result[0]['rec_texts']
                recognized_text = " ".join(lines)
            # ✨ PID NO를 맵에서 찾아서 추가
            pid_no = pid_no_map.get(original_page_key, "N/A")
            ocr_results.append({
                "PID NO": pid_no,
                "Original Page": original_page_key,
                "Cropped Filename": filename,
                "Class Name": parts[3],
                "Detection Score": float(parts[4]),
                "Recognized Text": recognized_text
            })
        except Exception as e:
            pid_no = pid_no_map.get(original_page_key, "N/A")
            ocr_results.append({
                "PID NO": pid_no, "Original Page": "Error", "Cropped Filename": filename,
                "Class Name": "Error", "Detection Score": 0.0, "Recognized Text": f"Processing Error: {e}"
            })
    if ocr_results:
        df = pd.DataFrame(ocr_results)
        # ✨ 엑셀 열 순서를 명확하게 지정
        column_order = ["PID NO", "Original Page", "Cropped Filename", "Class Name", "Detection Score", "Recognized Text"]
        df = df[column_order]
        df.to_excel(output_excel_path, index=False, engine='openpyxl')
        print(f"\n✅ OCR 결과가 '{output_excel_path}' 파일에 성공적으로 저장되었습니다.")
    else:
        print("⚠️ OCR을 수행할 이미지가 없거나 결과를 처리하지 못했습니다.")