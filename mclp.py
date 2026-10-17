# -*- coding: utf-8 -*-
"""MCLP — 최대 커버 입지 문제로 지정 승차장을 고른다.

    Church & ReVelle (1974). 시설 p곳으로 기준 거리 S 안에 든 수요를 최대화한다.

        max  Σ aᵢyᵢ
        s.t. Σ_{j∈Nᵢ} xⱼ ≥ yᵢ     모든 수요점 i
             Σ xⱼ = p
             xⱼ, yᵢ ∈ {0,1}
        aᵢ = 수요점 i 의 고령인구,  Nᵢ = { j : 보행거리(i,j) ≤ S }

왜 k-평균이 아닌가
  k-평균은 거리 **제곱합**을 줄일 뿐이라 "도보 몇 분 안에 몇 명이 탈 수 있나"를
  말하지 못한다. 외딴 집 한 채 때문에 승차장이 끌려가 아무도 제대로 못 덮는
  자리에 놓이기도 한다. MCLP 는 서비스 기준 S 를 명시하고, 못 덮을 곳은
  포기하는 대신 덮을 수 있는 밀집지를 확실히 덮는다. 후보지도 실재하는 지점
  중에서만 고르므로 "계산한 자리"와 "놓이는 자리"가 어긋나지 않는다.

푸는 법
  `scipy.optimize.milp`(HiGHS). 수요 124 × 후보 1,024 에서 **k당 0.1초**다.
  새 의존성이 없다 — scipy 는 이미 requirements 에 있다.

  y(수요 피복 여부)는 **연속**으로 둔다. x 가 정해지면 y 는 상계에 붙으므로
  최적해에서 저절로 0/1 이고, 이진 변수를 124개 줄여 더 빨리 풀린다.

같은 값을 내는 해가 여럿일 때
  ε 사전식으로 결정한다. 1순위 커버인구, 2순위 실재 정류장 선호, 3순위 인구가중
  보행거리. ε 은 가장 작은 유의 인구차보다 훨씬 작게 둬서 **최적 커버값을
  바꾸지 않는다**. 이렇게 해야 슬라이더를 움직일 때 지도가 임의로 뒤집히지 않는다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, eye, hstack

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

WALK_SPEED = 0.8        # m/s, 고령 보행
WALK_DETOUR = 1.4       # 보행망에서 끊긴 쌍의 직선 보정 (실측 중앙 1.338)
FAR = 65535             # bake_walk 의 '멀다/끊김' 표시
EPS_EXIST = 2e-2        # 실재 정류장 선호
EPS_WALK = 1e-3         # 인구가중 보행거리(km) 선호


@dataclass
class Cands:
    walk: np.ndarray        # (수요 I × 후보 N) 보행거리 m, 끊김은 inf
    pop: np.ndarray         # (I,) 고령인구
    xy: np.ndarray          # (I, 2) 수요점 좌표
    cx: np.ndarray          # (N,) 후보 x
    cy: np.ndarray
    node: np.ndarray        # (N,) 차량 도로망 노드
    spd: np.ndarray         # (N,) 접한 링크 최저 제한속도
    exist: np.ndarray       # (N,) 실재 정류장인가
    stop_id: np.ndarray
    stop_name: np.ndarray
    meta: dict

    @property
    def n(self) -> int:
        return len(self.cx)


def load_cands(meta: dict, path: Path = DATA / "walk_cand.npz") -> Cands:
    """보행거리표를 읽고 data.json 과 맞는지 확인한다.

    격자가 바뀌었는데 옛 표를 쓰면 조용히 틀린 승차장이 나온다. 서명이
    어긋나면 바로 세운다.
    """
    import hashlib
    z = np.load(path, allow_pickle=False)
    cells = [c for c in meta["cells"] if c[4] and c[3] > 0]
    P = np.array([[c[0] + 125, c[1] + 125] for c in cells], float)
    pop = np.array([c[3] for c in cells], float)
    sig = hashlib.md5(
        np.ascontiguousarray(np.c_[P, pop]).tobytes()).hexdigest()[:12]
    got = str(z["cells_sig"][0])
    if got != sig:
        raise RuntimeError(
            f"walk_cand.npz 가 지금 data.json 과 맞지 않는다 "
            f"(표 {got} ≠ 격자 {sig}). bake_walk.py 를 다시 돌려라.")
    w = z["walk"].astype(np.float64)
    w[w >= FAR] = np.inf
    return Cands(w, pop, P, z["cand_x"].astype(float),
                 z["cand_y"].astype(float), z["cand_node"],
                 z["cand_spd"].astype(float), z["cand_exist"],
                 z["cand_stop_id"], z["cand_stop_name"],
                 json.loads(str(z["meta"][0])))


def coverage(C: Cands, S_m: float,
             exclude_fast_kmh: float | None = None) -> np.ndarray:
    """(I × N) bool — 수요 i 가 후보 j 로 덮이는가."""
    N = C.walk <= S_m
    if exclude_fast_kmh is not None:
        N = N & (C.spd < exclude_fast_kmh)[None, :]
    return N


def solve(C: Cands, k: int, S_m: float, *,
          exclude_fast_kmh: float | None = None,
          force: set[int] | None = None) -> dict:
    """MCLP 한 판. 고른 후보 번호와 커버인구를 돌려준다."""
    N = coverage(C, S_m, exclude_fast_kmh)
    live = N.any(0)                       # 아무도 못 덮는 후보는 뺀다
    if force:
        live[list(force)] = True
    cols = np.where(live)[0]
    rows = np.where(N.any(1))[0]          # 아무 후보로도 못 덮이는 수요는 뺀다
    A = csr_matrix(N[np.ix_(rows, cols)].astype(float))
    nN, nI = len(cols), len(rows)

    # 목적 — 1순위 커버인구(최대), 2순위 실재 정류장, 3순위 보행
    gw = np.where(np.isfinite(C.walk), np.minimum(C.walk, 3000), 3000)
    gw = (gw[rows][:, cols] * C.pop[rows, None]).sum(0) / C.pop.sum() / 1000
    c = np.r_[EPS_WALK * gw - EPS_EXIST * C.exist[cols].astype(float),
              -C.pop[rows]]

    lo = np.zeros(nN + nI)
    if force:
        for j in force:
            lo[int(np.searchsorted(cols, j))] = 1
    r = milp(
        c,
        constraints=[
            LinearConstraint(hstack([-A, eye(nI, format="csr")]).tocsr(),
                             -np.inf, 0),
            LinearConstraint(np.r_[np.ones(nN), np.zeros(nI)], k, k)],
        integrality=np.r_[np.ones(nN), np.zeros(nI)],
        bounds=Bounds(lo, 1),
        options={"time_limit": 120, "mip_rel_gap": 0.0})
    if not r.success:
        raise RuntimeError(f"MCLP k={k} S={S_m} 풀이 실패: {r.message}")
    cids = sorted(int(cols[j]) for j in np.where(r.x[:nN] > .5)[0])
    cov = N[:, cids].any(1)
    return {"cids": cids, "cov_pop": float(C.pop[cov].sum()),
            "cov_share": float(C.pop[cov].sum() / C.pop.sum()),
            "cov_cells": int(cov.sum()), "sec": float(r.mip_dual_bound or 0)}


def assign(C: Cands, cids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """수요점 → (배정된 승차장의 리스트 위치, 보행거리 m).

    커버 밖 사람도 가장 가까운 승차장까지 걸어서 탄다 — MCLP 는 어디에 놓을지만
    정한다. 보행망에서 끊긴 쌍은 직선 × 1.4 로 물러선다.
    """
    W = C.walk[:, cids].copy()
    S = np.hypot(C.xy[:, None, 0] - C.cx[cids][None, :],
                 C.xy[:, None, 1] - C.cy[cids][None, :]) * WALK_DETOUR
    W = np.where(np.isfinite(W), W, S)
    j = W.argmin(1)
    return j, W[np.arange(len(W)), j]


def stops(C: Cands, cids: list[int], S_m: float) -> list[dict]:
    """sim 이 먹는 승차장 목록.

    **리스트 위치 == i 불변식**을 지킨다. km_table 열쇠 ("S", i), reach 의
    cell_stop, hist 행 번호가 모두 여기에 걸려 있다.

    정렬은 후보번호(cid) 오름차순이다. 인구 순위로 매기면 k 를 1 늘릴 때마다
    번호가 통째로 밀려 "가좌·송포 03"이 다른 장소를 가리키게 된다.
    """
    from pyproj import Transformer
    T = Transformer.from_crs(5179, 4326, always_xy=True)
    cids = sorted(cids)
    j, wm = assign(C, cids)
    N = coverage(C, S_m)
    out = []
    for p, cid in enumerate(cids):
        m = j == p
        cov = m & N[:, cid]
        lon, lat = T.transform(C.cx[cid], C.cy[cid])
        out.append({
            "i": p, "cid": int(cid),
            "x": float(C.cx[cid]), "y": float(C.cy[cid]),
            "node": int(C.node[cid]),
            "lon": round(float(lon), 6), "lat": round(float(lat), 6),
            "pop": round(float(C.pop[m].sum()), 1),
            "pop_cov": round(float(C.pop[cov].sum()), 1),
            "cells": int(m.sum()),
            "walk_med": int(np.median(wm[m])) if m.any() else 0,
            "walk_max": int(wm[m].max()) if m.any() else 0,
            "exist": bool(C.exist[cid]),
            "stop_name": str(C.stop_name[cid]) if C.exist[cid] else "",
            "spd_min": round(float(C.spd[cid]), 0),
            "name": f"가좌·송포 {p + 1:02d}",
        })
    assert all(s["i"] == n for n, s in enumerate(out)), "리스트 위치 ≠ i"
    return out


def cids_of(C: Cands, stops: list[dict], tol: float = 1.0) -> list[int]:
    """승차장 좌표를 후보번호로 되찾는다.

    비교 기준선인 k-평균 승차장도 후보 풀에 넣어 뒀으므로, 좌표로 찾아
    **같은 보행망 자로** 보행거리를 잴 수 있다. 못 찾으면 바로 세운다 —
    조용히 직선으로 물러서면 두 선정법 비교가 공정하지 않게 된다.
    """
    from scipy.spatial import cKDTree
    d, j = cKDTree(np.c_[C.cx, C.cy]).query(
        np.array([[s["x"], s["y"]] for s in stops]), k=1)
    if d.max() > tol:
        raise RuntimeError(
            f"후보 풀에 없는 승차장이 있다 (최대 {d.max():.1f} m). "
            "bake_walk.py 를 다시 돌려라.")
    return [int(v) for v in j]


def solve_ladder(C: Cands, ks, S_m: float, *, nested: bool = False,
                 **kw) -> dict[int, dict]:
    """k 를 훑으며 푼다. nested 면 직전 해를 유지시킨다(슬라이더가 매끄럽다)."""
    out, prev = {}, None
    for k in ks:
        r = solve(C, k, S_m, force=set(prev) if (nested and prev) else None,
                  **kw)
        r["new"] = sorted(set(r["cids"]) - set(prev or []))
        r["dropped"] = sorted(set(prev or []) - set(r["cids"]))
        out[k] = r
        prev = r["cids"]
    return out


if __name__ == "__main__":
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=float, default=500.0)
    ap.add_argument("--kmin", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=16)
    ap.add_argument("--nested", action="store_true")
    a = ap.parse_args()
    meta = json.loads((DATA / "data.json").read_text(encoding="utf-8"))
    C = load_cands(meta)
    print(f"수요 {len(C.pop)} · 후보 {C.n:,} · 기준 S={a.S:.0f} m "
          f"({a.S / WALK_SPEED / 60:.1f}분)")
    t = time.perf_counter()
    L = solve_ladder(C, range(a.kmin, a.kmax + 1), a.S, nested=a.nested)
    print(f"{len(L)}개 k · {time.perf_counter() - t:.1f}초")
    print(f"{'k':>3}{'커버명':>10}{'커버%':>8}{'실재정류장':>10}"
          f"{'신규':>6}{'탈락':>6}{'보행중앙':>9}")
    for k, r in L.items():
        ss = stops(C, r["cids"], a.S)
        print(f"{k:>3}{r['cov_pop']:>9.1f}{100 * r['cov_share']:>7.1f}%"
              f"{sum(s['exist'] for s in ss):>9}곳{len(r['new']):>5}"
              f"{len(r['dropped']):>6}"
              f"{int(np.median([s['walk_med'] for s in ss])):>8}m")
    U = sorted({c for r in L.values() for c in r["cids"]})
    print(f"k 전체 합집합 {len(U)}곳 — 카카오는 이만큼만 받으면 된다")
