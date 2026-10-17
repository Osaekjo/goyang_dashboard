# -*- coding: utf-8 -*-
"""API 키 읽기 — 환경변수 → .env → Streamlit Secrets 순.

키는 저장소에 올리지 않는다. 로컬은 .env, Streamlit Cloud 는 Secrets 다.
쓰는 사람마다 이름을 달리 적으므로 여러 이름을 받는다.
"""
from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
_PLACEHOLDER = ("여기에", "붙여넣", "your_", "xxx")


def _clean(v: str) -> str:
    v = str(v).strip().strip('"').strip("'").strip()
    low = v.lower()
    if not v or low in {"none", "null"} or any(p in low or p in v
                                               for p in _PLACEHOLDER):
        return ""
    return v


def _from_env_file(names: tuple[str, ...]) -> str:
    f = HERE / ".env"
    if not f.exists():
        return ""
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, val = line.partition("=")
        if name.strip() in names:
            v = _clean(val)
            if v:
                return v
    return ""


def get(*names: str) -> str | None:
    """주어진 이름들 가운데 먼저 찾아지는 값. 없으면 None."""
    for n in names:
        v = _clean(os.environ.get(n, ""))
        if v:
            return v
    v = _from_env_file(names)
    if v:
        return v
    try:
        import streamlit as st
        for n in names:
            v = _clean(st.secrets.get(n, ""))
            if v:
                return v
    except Exception:
        pass
    return None


KAKAO = ("KAKAO_REST_API_KEY", "KAKAO_REST_KEY")
VWORLD = ("VWORLD_API_KEY", "VWORLD_KEY")
