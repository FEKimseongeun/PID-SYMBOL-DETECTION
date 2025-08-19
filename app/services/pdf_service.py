# app/services/pdf_service.py
import os
import fitz  # PyMuPDF
from tqdm import tqdm
from typing import List, Tuple, Optional

def convert_pdf_to_images(pdf_path: str, output_folder: str, dpi: int = 300) -> List[str]:
    """PDF 파일을 이미지로 변환합니다."""
    os.makedirs(output_folder, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []
    print(f"PDF를 이미지로 변환 중... (총 {len(doc)} 페이지)")
    for i, page in enumerate(tqdm(doc, desc="PDF 변환")):
        pix = page.get_pixmap(dpi=dpi)
        image_path = os.path.join(output_folder, f"page_{i:03d}.png")
        pix.save(image_path)
        image_paths.append(image_path)
    doc.close()
    return image_paths

# --- ✨ [추가] PID NO 추출을 위한 함수들 ---
def find_text_coordinates(pdf_path: str, text: str, page_num: int) -> Optional[Tuple[float, float, float, float]]:
    """지정된 페이지에서 특정 텍스트를 찾아 그 좌표를 반환합니다."""
    doc = fitz.open(pdf_path)
    try:
        if page_num >= len(doc): return None
        page = doc[page_num]
        text_instances = page.search_for(text)
        return tuple(text_instances[0]) if text_instances else None
    finally:
        doc.close()

def extract_text_from_coords(pdf_path: str, page_num: int, coords: Tuple[float, float, float, float], ref_text: str) -> str:
    """지정된 페이지의 특정 좌표 영역 내의 텍스트를 추출하고 기준 텍스트를 제거합니다."""
    doc = fitz.open(pdf_path)
    try:
        if page_num >= len(doc) or not coords: return ""
        page = doc[page_num]
        rect = fitz.Rect(coords)
        text = page.get_text("text", clip=rect).strip()
        return text.replace(ref_text, "").strip()
    finally:
        doc.close()