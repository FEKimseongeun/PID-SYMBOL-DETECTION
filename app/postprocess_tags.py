#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
엑셀 결과(원시 OCR 텍스트 포함)를 받아 태그 정규식 기반으로 정제/중복제거하는 모듈.

사용 예)
    python app/postprocess_tags.py -i ./out/result.xlsx
    # => ./out/result_cleaned.xlsx, ./out/result_cleaned_unique.xlsx 생성

파이프라인 내에서:
    from app.postprocess_tags import postprocess_excel
    postprocess_excel("./out/result.xlsx")  # DataFrame 반환 + 파일 저장
"""

import os
import re
import argparse
import unicodedata
import pandas as pd
from typing import Iterable, Tuple

# 기본 태그 정규식 (기존 코드와 동일한 의도 유지)
# 예: "AB-123A", "AB 123 A", "AB123A" 모두 매칭되게
TAG_PATTERN = re.compile(
    r'([A-Za-z]+)[\s\-]*([0-9]+)(?:[\s\-]*([A-Za-z]))?',
    re.IGNORECASE
)

# 하이픈류/공백류 통일, 유사문자 보정에 쓰는 테이블
_HYPHENS = ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212", "–", "—", "‑"]
_SPACES  = ["\u00A0", "\u2002", "\u2003", "\u2009"]  # NBSP 등
_TRANSLATE_TABLE = {
    ord(h): "-" for h in _HYPHENS
}
_TRANSLATE_TABLE.update({ord(s): " " for s in _SPACES})

# 흔한 OCR 혼동 문자 보정 (필요하면 확장)
# 좌변 패턴 → 우변으로 치환
_CONFUSION_MAP: Iterable[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bO(?=\d)\b"), "0"),      # 독립된 'O'가 숫자와 붙을 때 0
    (re.compile(r"(?<=\D)0(?=[A-Za-z])"), "O"),  # 문자+0+문자 패턴 중 0→O (상황에 따라)
    (re.compile(r"\bI(?=\d)\b"), "1"),
    (re.compile(r"(?<=\D)1(?=[A-Za-z])"), "I"),
]

def _normalize_text(s: str) -> str:
    """유니코드 정규화 + 하이픈/공백 통일 + 대문자화 + OCR 유사문자 보정(보수적으로)."""
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_TRANSLATE_TABLE)
    s = s.replace("_", "-")
    # 다중 구분자 분해를 위해 ,;/ 로 쪼갤 때 도움이 되도록 스페이스 정리
    s = re.sub(r"\s+", " ", s).strip()
    s = s.upper()

    # 유사문자 보정은 과도치 않게 보수적으로 적용
    for pat, repl in _CONFUSION_MAP:
        s = pat.sub(repl, s)
    return s

def _split_candidates(s: str) -> Iterable[str]:
    """
    한 OCR 문자열 안에서 ',', ';', '/' 등으로 복수 태그가 섞인 케이스를 분해.
    """
    if not s:
        return []
    parts = re.split(r"[,\;/]+", s)
    return [p.strip() for p in parts if p.strip()]

def _build_final_tag(m: Tuple[str, str, str]) -> str:
    """
    TAG_PATTERN 매치 그룹을 표준 형태 'LETTERS-NUMBERS(SUFFIX)'로 조립.
    예) ('AB', '123', 'A') -> 'AB-123A'
        ('PI', '100', '')  -> 'PI-100'
    """
    prefix = m[0] or ""
    number = m[1] or ""
    suffix = m[2] or ""
    return f"{prefix}-{number}{suffix}".upper()

def extract_tags_from_text(raw_text: str) -> Iterable[str]:
    """
    한 줄의 OCR raw_text에서 정규화/분해 후 TAG_PATTERN으로 모든 후보를 뽑아 표준화하여 반환.
    """
    norm = _normalize_text(raw_text)
    candidates = _split_candidates(norm) or [norm]
    out = []
    for c in candidates:
        for m in TAG_PATTERN.findall(c):
            tag = _build_final_tag(m)
            if tag and len(tag) >= 3:
                out.append(tag)
    return out

def postprocess_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    원본 추론 결과 DF(image_number,image_path,bbox,label,text,score) → 정제된 태그 DF 반환.
    열 이름이 없거나 다르면 KeyError가 나므로, 저장 단계와 열을 반드시 맞춰둘 것.
    """
    required_cols = {"image_number", "image_path", "bbox", "label", "text", "score"}
    missing = required_cols - set(df_raw.columns)
    if missing:
        raise KeyError(f"필수 컬럼 누락: {missing}. 현재 컬럼: {list(df_raw.columns)}")

    cleaned_rows = []
    for _, row in df_raw.iterrows():
        raw_text = row["text"]
        tags = extract_tags_from_text(raw_text)
        if not tags:
            continue
        for tag in tags:
            cleaned_rows.append({
                "image_number": row["image_number"],
                "image_path":   row["image_path"],
                "bbox":         row["bbox"],
                "label":        row["label"],
                "score":        row["score"],
                "raw_text":     raw_text,
                "final_tag":    tag,
            })

    if not cleaned_rows:
        return pd.DataFrame(columns=[
            "image_number","image_path","bbox","label","score","raw_text","final_tag"
        ])

    df = pd.DataFrame(cleaned_rows)

    # 전수 결과(중복 포함)
    df_all = df.sort_values(["image_number", "image_path", "final_tag", "score"], ascending=[True, True, True, False])

    # 태그 단위 유니크: 같은 도면(image_path)에서 동일 final_tag 여러 번 나오면 score 최대 하나만
    idx = df_all.groupby(["image_path", "final_tag"])["score"].idxmax()
    df_unique = df_all.loc[idx].sort_values(["image_number", "image_path", "final_tag"]).reset_index(drop=True)

    return df_all, df_unique

def postprocess_excel(input_xlsx: str, out_all: str = None, out_unique: str = None):
    """
    엑셀 파일을 받아 정제/중복제거 결과를 새 엑셀로 저장하고 DataFrame 반환.
    """
    if not os.path.exists(input_xlsx):
        raise FileNotFoundError(f"입력 파일이 존재하지 않음: {input_xlsx}")

    df_raw = pd.read_excel(input_xlsx, dtype={"image_number": int, "image_path": str, "bbox": str, "label": int, "text": str, "score": float})
    df_all, df_unique = postprocess_df(df_raw)

    base, ext = os.path.splitext(input_xlsx)
    out_all = out_all or f"{base}_cleaned.xlsx"
    out_unique = out_unique or f"{base}_cleaned_unique.xlsx"

    # 저장
    os.makedirs(os.path.dirname(out_all) or ".", exist_ok=True)
    df_all.to_excel(out_all, index=False)
    df_unique.to_excel(out_unique, index=False)

    print(f"[DONE] 정제 결과(전체): {out_all}")
    print(f"[DONE] 정제 결과(유니크): {out_unique}")

    return df_all, df_unique

def _cli():
    ap = argparse.ArgumentParser(description="P&ID 태그 후처리 (엑셀 → 정제/중복제거)")
    ap.add_argument("-i", "--input", required=True, help="원본 결과 엑셀 경로 (*.xlsx)")
    ap.add_argument("--out-all", help="정제 전수 결과 파일 경로 (기본: <input>_cleaned.xlsx)")
    ap.add_argument("--out-unique", help="유니크 결과 파일 경로 (기본: <input>_cleaned_unique.xlsx)")
    args = ap.parse_args()
    postprocess_excel(args.input, args.out_all, args.out_unique)

if __name__ == "__main__":
    _cli()
