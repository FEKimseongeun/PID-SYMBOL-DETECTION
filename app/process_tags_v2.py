#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Recognized Text → 태그 정제 후 'Tag (Normalized)' 컬럼만 추가
※ 셀 내부/행 수준 어떤 중복제거도 수행하지 않음. 첫 매칭 1개만 기록.

사용:
    python app/postprocess_recognized_singlecol_nodedup.py -i ./out/result.xlsx
    # => ./out/result_tagclean.xlsx
옵션:
    --col-name  (기본: 'Recognized Text')
    -o / --output
"""

import os
import re
import argparse
import unicodedata
import pandas as pd
from typing import Iterable, Tuple, List

# --- 패턴 ---
TAG_PATTERN = re.compile(
    r'([A-Za-z]+)[\s\-]*([0-9]+)(?:[\s\-]*([A-Za-z]))?',
    re.IGNORECASE
)

# 하이픈/공백 정규화
_HYPHENS = ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212", "–", "—", "-"]
_SPACES  = ["\u00A0", "\u2002", "\u2003", "\u2009"]
_TRANSLATE_TABLE = {ord(h): "-" for h in _HYPHENS}
_TRANSLATE_TABLE.update({ord(s): " " for s in _SPACES})

# OCR 흔오류 보정(보수적)
_CONFUSION_MAP: Iterable[Tuple[re.Pattern, str]] = [
    (re.compile(r"\bO(?=\d)\b"), "0"),
    (re.compile(r"(?<=\D)0(?=[A-Za-z])"), "O"),
    (re.compile(r"\bI(?=\d)\b"), "1"),
    (re.compile(r"(?<=\D)1(?=[A-Za-z])"), "I"),
]

def _normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = "" if pd.isna(s) else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.translate(_TRANSLATE_TABLE).replace("_", "-")
    s = re.sub(r"\s+", " ", s).strip().upper()
    for pat, repl in _CONFUSION_MAP:
        s = pat.sub(repl, s)
    return s

def _split_candidates(s: str) -> List[str]:
    if not s:
        return []
    parts = re.split(r"[,\;/]+", s)
    return [p.strip() for p in parts if p.strip()]

def _build_final_tag(prefix: str, number: str, suffix: str) -> str:
    return f"{(prefix or '').upper()}-{number or ''}{(suffix or '').upper()}".upper()

def extract_first_tag(raw_text: str) -> str:
    """
    정규화 → 쉼표/슬래시 분해 → **첫 번째 매칭 하나**만 반환.
    중복 제거 없음.
    """
    norm = _normalize_text(raw_text)
    chunks = _split_candidates(norm) or [norm]
    for c in chunks:
        # 첫 매칭 1개만 반환
        m = TAG_PATTERN.search(c)
        if m:
            pre, num, suf = m.group(1), m.group(2), (m.group(3) or "")
            return _build_final_tag(pre, num, suf)
    return ""

# --- 메인 ---
def process_excel_singlecol(
    input_xlsx: str,
    recognized_col: str = "Recognized Text",
    output_xlsx: str = None
) -> pd.DataFrame:
    if not os.path.exists(input_xlsx):
        raise FileNotFoundError(f"입력 파일이 없습니다: {input_xlsx}")

    df = pd.read_excel(input_xlsx, dtype=str)

    if recognized_col not in df.columns:
        raise KeyError(f"엑셀에 '{recognized_col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

    out_col = "Tag (Normalized)"
    # 기존에 있으면 덮어씀
    df[out_col] = ""

    for idx, val in df[recognized_col].items():
        df.at[idx, out_col] = extract_first_tag(val)

    base, _ = os.path.splitext(input_xlsx)
    output_xlsx = output_xlsx or f"{base}_tagclean.xlsx"
    os.makedirs(os.path.dirname(output_xlsx) or ".", exist_ok=True)
    df.to_excel(output_xlsx, index=False)
    print(f"[DONE] 저장: {output_xlsx}")
    return df

def _cli():
    ap = argparse.ArgumentParser(description="Recognized Text → 'Tag (Normalized)' (중복제거 없음)")
    ap.add_argument("-i", "--input", required=True, help="입력 엑셀 경로 (*.xlsx)")
    ap.add_argument("--col-name", default="Recognized Text", help="인식 텍스트 컬럼명 (기본: 'Recognized Text')")
    ap.add_argument("-o", "--output", help="출력 엑셀 경로 (기본: <input>_tagclean.xlsx)")
    args = ap.parse_args()
    process_excel_singlecol(args.input, args.col_name, args.output)

if __name__ == "__main__":
    _cli()
