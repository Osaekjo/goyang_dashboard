# -*- coding: utf-8 -*-
"""카카오모빌리티 길찾기 — 실제 차량 경로의 거리·소요시간을 받아 온다.

왜 미리 받아 두는가
  대시보드는 차종·대수·승차장 수를 바꿀 때마다 수백 번 다시 계산한다. 그때마다
  API 를 부르면 쿼터가 순식간에 녹고 화면도 느려진다. 그래서 **승차장 ↔ 의원
  거리행렬을 한 번 받아 디스크에 쟁여 두고**, 화면은 그 캐시만 읽는다.

  여러 목적지(/v1/destinations)는 한 번에 최대 30곳까지 묻을 수 있어, 승차장
  하나에 1~2회면 끝난다. 경로 모양이 필요한 몇 개만 /v1/directions 로 따로
  받는다.

키
  .env 의 KAKAO_REST_API_KEY. Streamlit Cloud 에서는 Secrets 에 같은 이름으로
  넣으면 된다. 키가 없으면 None 을 돌려주고, 부르는 쪽이 도로망 계산으로
  물러선다 — 키 없이도 화면은 돈다.

좌표
  이 연구는 EPSG:5179 를 쓰고 카카오는 WGS84 경위도를 받는다. 넘기기 직전에만
  바꾼다.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import requests

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "kakao_cache.json"
MATRIX_URL = "https://apis-navi.kakaomobility.com/v1/destinations/directions"
ROUTE_URL = "https://apis-navi.kakaomobility.com/v1/directions"
MAX_DEST = 30               # /v1/destinations 한 번에 묻을 수 있는 목적지 수
PAUSE = 0.12                # 연속 호출 사이 간격 (초)


# 쓰는 사람마다 이름을 달리 적는다. 둘 다 받는다.
KEY_NAMES = ("KAKAO_REST_API_KEY", "KAKAO_REST_KEY")


def _clean(v: str) -> str:
    v = v.strip().strip('"').strip("'").strip()
    # 안 채운 자리표시자를 키로 착각하지 않는다
    return "" if (not v or "여기에" in v or v.lower() in {"none", "null"}) else v


def api_key() -> str | None:
    """환경변수 → .env → Streamlit Secrets 순으로 찾는다."""
    for n in KEY_NAMES:
        k = _clean(os.environ.get(n, ""))
        if k:
            return k
    env = HERE / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            name, _, val = line.partition("=")
            if name.strip() in KEY_NAMES:
                k = _clean(val)
                if k:
                    return k
    try:
        import streamlit as st
        for n in KEY_NAMES:
            k = _clean(str(st.secrets.get(n, "")))
            if k:
                return k
    except Exception:
        pass
    return None


def _load() -> dict:
    if CACHE.exists():
        try:
            return json.loads(CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save(d: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, ensure_ascii=False,
                                separators=(",", ":")), encoding="utf-8")


def _key(a: tuple[float, float], b: tuple[float, float]) -> str:
    """캐시 열쇠 — 5자리(약 1m)로 끊어 같은 지점이 다르게 들어가지 않게 한다."""
    return f"{a[0]:.5f},{a[1]:.5f}|{b[0]:.5f},{b[1]:.5f}"


class Kakao:
    """거리행렬과 경로를 받아 디스크에 쟁인다. 키가 없으면 ok 가 False 다."""

    def __init__(self, verbose: bool = True) -> None:
        self.key = api_key()
        self.ok = self.key is not None
        self.cache = _load()
        self.calls = 0
        self.hits = 0
        self.fail = 0
        self.verbose = verbose

    # ── 거리행렬 ──────────────────────────────────────────────────────
    def matrix(self, origin: tuple[float, float],
               dests: list[tuple[float, float]]) -> list[dict | None]:
        """한 출발지에서 여러 목적지까지. [{km, min}] 또는 None 을 돌려준다."""
        out: list[dict | None] = [None] * len(dests)
        todo: list[int] = []
        for i, d in enumerate(dests):
            c = self.cache.get(_key(origin, d))
            if c is not None:
                out[i] = c
                self.hits += 1
            else:
                todo.append(i)
        if not todo or not self.ok:
            return out

        for s in range(0, len(todo), MAX_DEST):
            chunk = todo[s:s + MAX_DEST]
            body = {
                "origin": {"x": origin[0], "y": origin[1]},
                "destinations": [
                    {"x": dests[i][0], "y": dests[i][1], "key": str(i)}
                    for i in chunk],
                # 다중 목적지 쪽은 priority 를 받지 않는다. 넣으면 400 이다.
                "radius": 10000,
            }
            try:
                r = requests.post(
                    MATRIX_URL, json=body, timeout=20,
                    headers={"Authorization": f"KakaoAK {self.key}",
                             "Content-Type": "application/json"})
                self.calls += 1
                r.raise_for_status()
                for row in r.json().get("routes", []):
                    i = int(row["key"])
                    if row.get("result_code") != 0:
                        continue
                    v = {"km": row["summary"]["distance"] / 1000,
                         "min": row["summary"]["duration"] / 60}
                    out[i] = v
                    self.cache[_key(origin, dests[i])] = v
            except Exception as e:                       # 네트워크·쿼터·권한
                self.fail += 1
                if self.verbose and self.fail <= 3:
                    print(f"  [카카오] 거리행렬 실패 — {type(e).__name__}: "
                          f"{str(e)[:120]}")
            time.sleep(PAUSE)
        return out

    # ── 경로 모양 ─────────────────────────────────────────────────────
    def route(self, origin: tuple[float, float],
              dest: tuple[float, float]) -> dict | None:
        """{km, min, path:[[lon,lat],...]} — 지도에 그릴 실제 주행 경로."""
        k = _key(origin, dest) + "|geom"
        if k in self.cache:
            self.hits += 1
            return self.cache[k]
        if not self.ok:
            return None
        try:
            r = requests.get(
                ROUTE_URL, timeout=20,
                params={"origin": f"{origin[0]},{origin[1]}",
                        "destination": f"{dest[0]},{dest[1]}",
                        "priority": "RECOMMEND", "road_details": "false"},
                headers={"Authorization": f"KakaoAK {self.key}"})
            self.calls += 1
            r.raise_for_status()
            js = r.json()
            rt = js["routes"][0]
            if rt.get("result_code") != 0:
                return None
            pts: list[list[float]] = []
            for sec in rt.get("sections", []):
                for road in sec.get("roads", []):
                    v = road.get("vertexes", [])
                    # vertexes 는 [lon, lat, lon, lat, ...] 로 평탄하게 온다
                    pts += [[round(v[i], 6), round(v[i + 1], 6)]
                            for i in range(0, len(v) - 1, 2)]
            val = {"km": rt["summary"]["distance"] / 1000,
                   "min": rt["summary"]["duration"] / 60, "path": pts}
            self.cache[k] = val
            return val
        except Exception as e:
            self.fail += 1
            if self.verbose and self.fail <= 3:
                print(f"  [카카오] 경로 실패 — {type(e).__name__}: "
                      f"{str(e)[:120]}")
            return None
        finally:
            time.sleep(PAUSE)

    def flush(self) -> None:
        _save(self.cache)

    def report(self) -> str:
        return (f"캐시 적중 {self.hits:,} · 호출 {self.calls:,} · "
                f"실패 {self.fail:,} · 보관 {len(self.cache):,}건")


def selftest() -> None:
    """키가 먹는지 한 번 확인한다 — 호출 2회만 쓴다."""
    k = Kakao()
    print(f"키 인식: {'예' if k.ok else '아니오 (.env 를 확인하세요)'}")
    if not k.ok:
        return
    # 일산 킨텍스 → 일산병원 (대략)
    a, b = (126.7420, 37.6680), (126.7760, 37.6560)
    m = k.matrix(a, [b])[0]
    print(f"거리행렬: {m}")
    r = k.route(a, b)
    if r:
        print(f"경로: {r['km']:.2f} km · {r['min']:.1f}분 · "
              f"꼭짓점 {len(r['path'])}개")
    k.flush()
    print(k.report())


if __name__ == "__main__":
    selftest()
