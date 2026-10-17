# -*- coding: utf-8 -*-
"""가좌·송포 똑버스 운행 모형 — 도로망 길찾기 + 수요응답 배차 + 탄소 수지.

이 파일이 숫자를 전부 쥔다. 화면(streamlit_app.py)은 여기서 나온 것만 그린다.

모형의 뼈대
  한 '통행'은 왕복이다. 집→의원, 의원→집 **두 구간**으로 쪼개 센다. 그래야
  차량 점유시간과 공차주행이 겹쳐 세어지지 않는다.

  수요   고령 946.5명 × 연 30회 ÷ 365일 = 하루 78통행 (고령 의료통행만,
         보수적 하한). 발생 시각은 오전에 치우치게 둔다.
  합승   호출 35분·출발 2km·도착 2.5km 안이면 한 차에 묶는다(최대 4명).
         고양 똑버스 실제 대기가 30~40분이라는 이용자 체감에 맞춘 창이다.
  배차   먼저 비는 차량에 붙인다. 직전 하차지에서 승차지까지는 **공차**다.
  주행   국가표준노드링크(2025-05-12) 실제 도로 최단경로. 직선 근사가 아니다.

배출계수 (환경부 2021년 승인 국가계수, g CO2/대·km, 65.4 km/h 미만)
  택시   승용 중형 LPG     1513.8104 x^-0.6075
  자가용 승용 소형 휘발유   1256.0382 x^-0.5914
  디젤승합은 **국가계수에 소형승합이 없어** 승용 대형 경유~화물 소형 경유로 범위
  전기   0.4747 kgCO2/kWh (소비단) × 전비
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

DATA = Path(__file__).resolve().parent / "data"

# ── 상수 ──────────────────────────────────────────────────────────────
SPEED_KMH = 30.0                  # 평균 운행속도
ELEC = 0.4747                     # kgCO2/kWh, 소비단
EF_TAXI = 1513.8104 * SPEED_KMH ** -0.6075       # g/대·km
EF_CAR = 1256.0382 * SPEED_KMH ** -0.5914
COST_DAY = 560_428                # 원/대·일, 수원시 2024 표준운송원가(안)
GWANGGYO_PROD = 329_501 / 10 / 365               # 90.3 통행/대·일

T0, T1 = 8 * 60 + 30, 17 * 60     # 운행 08:30–17:00 (분)
CARE = 90                         # 진료 소요 (분)
WIN, R_ORI, R_DST = 35, 2000, 2500               # 합승 창
BOARD = 2                         # 승하차 (분)
TRIPS_PER_YEAR = 30               # 고령 1인 연간 외래

VEHICLES = [
    {"id": "A", "name": "A 쏠라티 11인승 디젤", "cap": 11,
     "note": "고봉동 운행 중",
     "ef": 1149.2206 * SPEED_KMH ** -0.5313,
     "ef_hi": 1250.4831 * SPEED_KMH ** -0.4630},
    {"id": "B", "name": "B 11인승 전기 승합", "cap": 11,
     "note": "전비 0.25 kWh/km", "ef": 0.25 * ELEC * 1000, "ef_hi": None},
    {"id": "C", "name": "C 25인승 중형 전기버스", "cap": 25,
     "note": "향동동 운행 중 · 1.0 kWh/km",
     "ef": 1.0 * ELEC * 1000, "ef_hi": 1.2 * ELEC * 1000},
]


# ── 자료 ──────────────────────────────────────────────────────────────
def load_meta() -> dict:
    return json.loads((DATA / "data.json").read_text(encoding="utf-8"))


class Road:
    """도로망 — 유향 그래프와 링크 모양을 함께 쥔다."""

    def __init__(self) -> None:
        z = np.load(DATA / "v6_graph.npz")
        self.nx, self.ny = z["nx"].astype(float), z["ny"].astype(float)
        self.ef, self.et = z["ef"], z["et"]
        self.elen = z["elen"].astype(float)
        self.gi, self.gxy = z["gi"], z["gxy"].astype(float)
        n = len(self.nx)
        self.W = csr_matrix((self.elen, (self.ef, self.et)), shape=(n, n))
        self.tree = cKDTree(np.c_[self.nx, self.ny])
        # (출발노드, 도착노드) → 링크 번호. 중복이면 짧은 쪽을 남긴다.
        self.edge: dict[tuple[int, int], int] = {}
        for k, (a, b) in enumerate(zip(self.ef, self.et)):
            key = (int(a), int(b))
            old = self.edge.get(key)
            if old is None or self.elen[k] < self.elen[old]:
                self.edge[key] = k

    def snap(self, xy: np.ndarray) -> np.ndarray:
        return self.tree.query(np.asarray(xy, float))[1]

    def solve(self, sources: np.ndarray):
        """출발 노드 묶음에서 전체로 가는 거리와 선행노드."""
        src = np.unique(np.asarray(sources, int))
        dist, pred = dijkstra(self.W, directed=True, indices=src,
                              return_predecessors=True)
        return {int(v): i for i, v in enumerate(src)}, dist, pred

    def path_xy(self, pred_row: np.ndarray, a: int, b: int) -> np.ndarray:
        """선행노드 배열로 a→b 경로의 실제 도로 모양을 복원한다."""
        seq, cur, guard = [b], b, 0
        while cur != a and guard < 10_000:
            cur = int(pred_row[cur])
            if cur < 0:
                return np.empty((0, 2))
            seq.append(cur)
            guard += 1
        seq.reverse()
        out = []
        for u, v in zip(seq[:-1], seq[1:]):
            k = self.edge.get((u, v))
            if k is None:
                out.append(np.array([[self.nx[u], self.ny[u]],
                                     [self.nx[v], self.ny[v]]]))
            else:
                out.append(self.gxy[self.gi[k]:self.gi[k + 1]])
        if not out:
            return np.array([[self.nx[a], self.ny[a]]])
        return np.vstack(out)


# ── 시뮬레이션 ────────────────────────────────────────────────────────
@dataclass
class Ride:
    v: int            # 차량 번호
    n: int            # 탑승 인원
    call: float       # 호출 시각 (분)
    depart: float     # 공차 출발
    pick: float       # 승차
    drop: float       # 하차
    km_rev: float     # 수송거리
    km_dead: float    # 공차거리
    path: np.ndarray  # 수송 구간 도로 모양 (EPSG:5179)
    path_dead: np.ndarray  # 공차 구간(직전 하차지→승차지) 도로 모양
    back: int         # 0 집→의원 · 1 의원→집
    stop: int         # 지정 승차장 번호 (문앞 호출이면 -1)
    dest: int         # 의료기관 번호


def simulate(road: Road, meta: dict, veh: dict, fleet: int,
             seed: int = 20261016, stops: list[dict] | None = None,
             km_table: dict | None = None) -> dict:
    """지정 승차장에서 태워 의원까지 나른다.

    똑버스는 지정 승차장 운영이다. 고령자가 승차장까지 걷는 거리도 함께
    돌려준다 — 승차장을 줄이면 탄소는 좋아지고 보행은 나빠진다.

    km_table 을 주면 주행거리를 거기서 읽는다. 열쇠는 ("S", 승차장번호) 와
    ("D", 의료기관번호) 의 쌍이다. 카카오 길찾기로 받아 둔 값을 넣는 자리이고,
    빠진 쌍은 도로망(국가표준노드링크) 계산으로 물러선다.
    """
    rnd = np.random.default_rng(seed)
    cells = [c for c in meta["cells"] if c[4] and c[3] > 0]
    w = np.array([c[3] for c in cells], float)
    ox = np.array([c[0] + 125 for c in cells], float)
    oy = np.array([c[1] + 125 for c in cells], float)
    dest = np.array([[d[0], d[1]] for d in meta["dest"]], float)
    dw = np.array([1 + (4 if d[2] >= 10 else 0) + (1 if d[2] > 0 else 0)
                   for d in meta["dest"]], float)

    n_trip = round(meta["meta"]["고령"] * TRIPS_PER_YEAR / 365)
    oi = rnd.choice(len(cells), size=n_trip, p=w / w.sum())
    u = rnd.random(n_trip)
    t_call = T0 + (T1 - T0 - CARE - 20) * u ** 1.45        # 오전 가중

    # 목적지 — 가까울수록, 내과 전문의가 많을수록. 지수 6.0 은 평균 왕복이
    # v6 탄소장의 6.90~10.54 km(직선) 범위에 들도록 맞춘 값이다.
    di = np.empty(n_trip, int)
    for i in range(n_trip):
        km = np.hypot(dest[:, 0] - ox[oi[i]], dest[:, 1] - oy[oi[i]]) / 1000
        p = dw / (1 + km) ** 6.0
        di[i] = rnd.choice(len(dest), p=p / p.sum())

    # 승차 지점 — 가장 가까운 지정 승차장. 거기까지는 걸어간다.
    if not stops:
        raise ValueError("똑버스는 지정 승차장 운영이다 — stops 가 필요하다")
    sx = np.array([s["x"] for s in stops])
    sy = np.array([s["y"] for s in stops])
    walk = np.zeros(n_trip)
    stop_of = np.zeros(n_trip, int)
    for i in range(n_trip):
        d = np.hypot(sx - ox[oi[i]], sy - oy[oi[i]])
        stop_of[i] = int(np.argmin(d))
        walk[i] = float(d[stop_of[i]])
    px, py = sx[stop_of], sy[stop_of]
    o_node = np.array([stops[j]["node"] for j in stop_of])
    d_node = road.snap(dest[di])
    row, dist, pred = road.solve(np.r_[o_node, d_node])

    legs = []
    for i in range(n_trip):
        ks, kd = ("S", int(stop_of[i])), ("D", int(di[i]))
        legs.append({"t": t_call[i], "a": int(o_node[i]), "b": int(d_node[i]),
                     "ka": ks, "kb": kd,
                     "ax": px[i], "ay": py[i],
                     "bx": dest[di[i], 0], "by": dest[di[i], 1], "back": 0,
                     "stop": int(stop_of[i]), "dest": int(di[i])})
        legs.append({"t": t_call[i] + CARE, "a": int(d_node[i]),
                     "b": int(o_node[i]),
                     "ka": kd, "kb": ks,
                     "ax": dest[di[i], 0], "ay": dest[di[i], 1],
                     "bx": px[i], "by": py[i], "back": 1,
                     "stop": int(stop_of[i]), "dest": int(di[i])})
    legs.sort(key=lambda r: r["t"])

    # 합승
    groups, used = [], [False] * len(legs)
    for i, li in enumerate(legs):
        if used[i]:
            continue
        used[i] = True
        grp = [li]
        for j in range(i + 1, len(legs)):
            lj = legs[j]
            if lj["t"] - li["t"] > WIN:
                break
            if used[j] or len(grp) >= min(4, veh["cap"]):
                continue
            if (math.hypot(lj["ax"] - li["ax"], lj["ay"] - li["ay"]) <= R_ORI
                    and math.hypot(lj["bx"] - li["bx"],
                                   lj["by"] - li["by"]) <= R_DST):
                used[j] = True
                grp.append(lj)
        groups.append(grp)

    # 배차 — **가장 빨리 도착하는 차량**에 붙인다. 먼저 비는 차를 그냥 쓰면
    # 멀리 있는 차가 뽑혀 공차가 불어난다. 실제 배차는 도착시각으로 고른다.
    # 직전 하차지에서 승차지까지가 공차다.
    free = [float(T0)] * fleet
    at: list[tuple[int, tuple] | None] = [None] * fleet
    rides: list[Ride] = []
    T = km_table or {}
    src = 0           # 카카오에서 읽은 쌍 / 도로망으로 물러선 쌍

    def km_of(ka, kb, a: int, b: int, fallback_xy=None) -> tuple[float, bool]:
        v = T.get((ka, kb))
        if v is not None and np.isfinite(v):
            return float(v), True
        d = dist[row[a], b] if a in row else np.inf
        if np.isfinite(d):
            return float(d) / 1000, False
        return (math.hypot(*fallback_xy) / 1000 * 1.4, False) \
            if fallback_xy else (1.5, False)

    for grp in groups:
        g = grp[0]
        a, b = g["a"], g["b"]
        best, best_t, best_d, best_k = 0, math.inf, 1.0, False
        for v in range(fleet):
            if at[v] is None:
                dk, kk = 1.0, False
            else:
                dk, kk = km_of(at[v][1], g["ka"], at[v][0], a)
            arr = max(g["t"], free[v]) + dk / SPEED_KMH * 60 + BOARD
            if arr < best_t:
                best, best_t, best_d, best_k = v, arr, dk, kk
        v, km_dead, pick = best, best_d, best_t
        src += int(best_k)
        # 공차 구간도 도로로 그린다 — 안 그리면 배차 이동 중 차가 멈춰 보인다
        pd_ = (road.path_xy(pred[row[at[v][0]]], at[v][0], a)
               if at[v] is not None and at[v][0] in row else np.empty((0, 2)))
        km_rev, kk = km_of(g["ka"], g["kb"], a, b,
                           (g["bx"] - g["ax"], g["by"] - g["ay"]))
        src += int(kk)
        depart = max(g["t"], free[v])
        drop = pick + km_rev / SPEED_KMH * 60
        free[v] = drop + BOARD
        at[v] = (b, g["kb"])
        rides.append(Ride(v, len(grp), g["t"], depart, pick, drop,
                          km_rev, km_dead, road.path_xy(pred[row[a]], a, b),
                          pd_, g["back"], g["stop"], g["dest"]))

    pax = sum(r.n for r in rides)
    km_rev = sum(r.km_rev for r in rides)
    km_dead = sum(r.km_dead for r in rides)
    wait = sum((r.pick - r.call) * r.n for r in rides) / pax
    ww = np.repeat(walk, 2)          # 구간마다 같은 보행이 한 번씩 붙는다
    n_pair = 2 * len(rides)          # 배차마다 공차 1 + 수송 1
    return {"rides": rides, "trips": n_trip, "pax": pax, "fleet": fleet,
            "occ": pax / len(rides), "km_rev": km_rev, "km_dead": km_dead,
            "km": km_rev + km_dead, "wait": wait,
            "end": max(r.drop for r in rides), "veh": veh,
            "dead_share": km_dead / (km_rev + km_dead),
            "stops": True, "n_stops": len(stops),
            "kakao_share": src / max(1, n_pair),
            "walk_med": float(np.median(ww)),
            "walk_mean": float(ww.mean()),
            "walk_max": float(ww.max()),
            "walk_p90": float(np.percentile(ww, 90)),
            "walk_min_med": float(np.median(ww)) / 0.8 / 60}


# ── 탄소 ──────────────────────────────────────────────────────────────
def carbon(sim: dict, ef: float | None = None) -> dict:
    """현행(택시 2인 합승)과 견준 하루 탄소 수지.

    기준선은 **같은 도로거리**를 택시로 달렸을 때다. 거리 가정이 양쪽에 똑같이
    들어가므로 손익분기 재차인원은 거리에 흔들리지 않는다.
    """
    ef = sim["veh"]["ef"] if ef is None else ef
    base = sim["km_rev"] * EF_TAXI / 2 / 1000        # 택시 2인 합승 kg
    return {
        "co2_rev": sim["km_rev"] * ef / 1000,
        "co2_tot": sim["km"] * ef / 1000,
        "base": base,
        "cut_rev": base - sim["km_rev"] * ef / 1000,
        "cut_tot": base - sim["km"] * ef / 1000,
        "breakeven": ef / (EF_TAXI / 2),
        "breakeven_dead": ef / (EF_TAXI / 2) / (1 - sim["dead_share"]),
    }


# ── 지정 승차장 ───────────────────────────────────────────────────────
def designated_stops(road: Road, meta: dict, k: int = 10,
                     seed: int = 20261016) -> list[dict]:
    """고령인구 밀집에 맞춰 승차장 후보를 뽑는다 — **분석 전 임시 가정**.

    실제 승차장 선정은 별도 분석(보행 접근권, 정차 가능 지점, 소유관계)이
    필요하다. 여기서는 고령인구를 가중치로 둔 k-평균 중심을 도로 노드에 붙여
    자리만 잡아 둔다. 나중에 실제 목록이 나오면 이 함수만 갈아끼우면 된다.

    되돌려 주는 것
      x, y        EPSG:5179 (도로 노드로 스냅한 자리)
      node        도로망 노드 번호
      pop         이 승차장이 맡는 고령인구
      walk_med    맡은 격자에서 승차장까지 보행거리 중앙 (m)
      walk_max    같은 것의 최대 (m)
    """
    cells = [c for c in meta["cells"] if c[4] and c[3] > 0]
    P = np.array([[c[0] + 125, c[1] + 125] for c in cells], float)
    w = np.array([c[3] for c in cells], float)

    rnd = np.random.default_rng(seed)
    # k-평균++ 로 씨앗을 고른다 — 인구가 많은 쪽이 뽑히게 가중한다
    cen = [P[rnd.choice(len(P), p=w / w.sum())]]
    for _ in range(k - 1):
        d2 = np.min([((P - c) ** 2).sum(1) for c in cen], axis=0) * w
        cen.append(P[rnd.choice(len(P), p=d2 / d2.sum())])
    cen = np.array(cen)
    for _ in range(80):
        lab = np.argmin(((P[:, None, :] - cen[None]) ** 2).sum(2), axis=1)
        new = cen.copy()
        for j in range(k):
            m = lab == j
            if m.any():
                new[j] = (P[m] * w[m, None]).sum(0) / w[m].sum()
        if np.allclose(new, cen, atol=.5):
            cen = new
            break
        cen = new

    node = road.snap(cen)
    sx, sy = road.nx[node], road.ny[node]            # 도로에 붙인 실제 자리
    out = []
    for j in range(k):
        m = lab == j
        if not m.any():
            continue
        d = np.hypot(P[m, 0] - sx[j], P[m, 1] - sy[j])
        out.append({"i": len(out), "x": float(sx[j]), "y": float(sy[j]),
                    "node": int(node[j]), "pop": float(w[m].sum()),
                    "cells": int(m.sum()),
                    "walk_med": float(np.median(d)),
                    "walk_max": float(d.max())})
    out.sort(key=lambda s: -s["pop"])
    for i, s in enumerate(out):
        s["i"] = i
        s["name"] = f"가좌·송포 {i + 1:02d}"
    return out


def stop_routes(road: Road, stops: list[dict], meta: dict,
                per_stop: int = 3) -> list[dict]:
    """승차장마다 가까운 의원 몇 곳까지의 실제 도로 경로."""
    dest = np.array([[d[0], d[1]] for d in meta["dest"]], float)
    dn = road.snap(dest)
    row, dist, pred = road.solve(np.array([s["node"] for s in stops]))
    out = []
    for s in stops:
        dv = dist[row[s["node"]], dn]
        order = np.argsort(dv)[:per_stop]
        for rank, j in enumerate(order):
            if not np.isfinite(dv[j]):
                continue
            d = meta["dest"][int(j)]
            out.append({
                "stop": s["i"], "stop_name": s["name"], "dest": int(j),
                "name": d[5], "kind": d[6], "doctors": d[7],
                "internal": d[2], "km": float(dv[j]) / 1000,
                "min": float(dv[j]) / 1000 / SPEED_KMH * 60, "rank": rank,
                "path": road.path_xy(pred[row[s["node"]]], s["node"], int(dn[j])),
            })
    return out


# ── 도달 ──────────────────────────────────────────────────────────────
def reach_tables(road: Road, meta: dict, stops: list[dict],
                 km_table: dict | None = None,
                 max_min: int = 90) -> dict:
    """도달 계산에 필요한 것만 추려 둔다.

    도입 후 한 사람의 문앞-의원 시간은
        집→승차장 보행  +  대기  +  승차장→의원 차내
    이고, 예산 안에 드는 의료기관 수가 그 사람의 도달 기관수다.

    대기는 대수에 따라 달라지고 보행·차내는 승차장 배치에 따라 달라지므로,
    **승차장별 차내시간 누적분포**와 **출발점별 보행시간**만 구워 두면 화면에서
    (예산, 대기) 를 바꿔도 즉시 다시 셀 수 있다.

    되돌려 주는 것
      cell_stop   출발점마다 배정된 승차장 번호
      cell_walk   그 승차장까지 보행시간 (분, 0.8 m/s)
      cell_pop    출발점 고령인구
      cell_now    현행 30분 주점수 A_r (곳)
      hist        승차장 × (0..max_min) — 차내시간이 t분 이하인 의료기관 수
    """
    cells = [c for c in meta["cells"] if c[4] and c[3] > 0]
    P = np.array([[c[0] + 125, c[1] + 125] for c in cells], float)
    pop = np.array([c[3] for c in cells], float)
    now = np.array([c[2] for c in cells], float)

    sx = np.array([s["x"] for s in stops])
    sy = np.array([s["y"] for s in stops])
    d = np.hypot(P[:, None, 0] - sx[None], P[:, None, 1] - sy[None])
    cs = d.argmin(1)
    walk = d[np.arange(len(P)), cs] / 0.8 / 60          # 분

    dest = np.array([[x[0], x[1]] for x in meta["dest"]], float)
    dn = road.snap(dest)
    row, dist, _ = road.solve(np.array([s["node"] for s in stops]))
    T = km_table or {}
    hist = np.zeros((len(stops), max_min + 1), np.int32)
    for s in stops:
        km = dist[row[s["node"]], dn] / 1000
        for j in range(len(dest)):                      # 카카오 값이 있으면 쓴다
            v = T.get((("S", s["i"]), ("D", j)))
            if v is not None:
                km[j] = v
        t = km / SPEED_KMH * 60
        t = t[np.isfinite(t)]
        hist[s["i"]] = np.bincount(
            np.clip(np.ceil(t).astype(int), 0, max_min),
            minlength=max_min + 1).cumsum()
    return {"cell_stop": cs.tolist(), "cell_walk": np.round(walk, 2).tolist(),
            "cell_pop": pop.tolist(), "cell_now": now.tolist(),
            "cell_xy": [[int(c[0]), int(c[1])] for c in cells],
            "hist": hist.tolist(), "max_min": max_min}


def reach(tab: dict, budget: float, wait: float) -> dict:
    """예산과 대기를 주면 도달 결과를 센다. 화면에서 매번 부른다."""
    cs = np.asarray(tab["cell_stop"])
    walk = np.asarray(tab["cell_walk"], float)
    pop = np.asarray(tab["cell_pop"], float)
    now = np.asarray(tab["cell_now"], float)
    H = np.asarray(tab["hist"])
    avail = np.clip(np.floor(budget - wait - walk).astype(int),
                    -1, tab["max_min"])
    cnt = np.where(avail < 0, 0, H[cs, np.clip(avail, 0, None)])
    tot = pop.sum()
    # 똑버스가 들어와도 기존 대중교통이 없어지지 않는다. 사람은 **둘 중 나은
    # 쪽**을 쓰므로 도입 후 도달은 현행과 똑버스의 합집합이다.
    both = (now > 0) | (cnt > 0)
    return {
        "per_cell": cnt,
        "per_cell_now": now,
        "pop": tot,
        "now_pop": float(pop[now > 0].sum()),
        "now_share": float(pop[now > 0].sum() / tot),
        "now_A": float((now * pop).sum() / tot),
        "drt_pop": float(pop[cnt > 0].sum()),          # 똑버스만으로
        "drt_share": float(pop[cnt > 0].sum() / tot),
        "new_pop": float(pop[both].sum()),             # 도입 후 (합집합)
        "new_share": float(pop[both].sum() / tot),
        "new_A": float((np.maximum(now, cnt) * pop).sum() / tot),
        "gain_pop": float(pop[(cnt > 0) & (now <= 0)].sum()),
        "still_pop": float(pop[~both].sum()),          # 그래도 못 닿는
        "walk_min": float(np.average(walk, weights=pop)),
        "ride_min": float(np.average(
            np.where(cnt > 0, budget - wait - walk, np.nan)[cnt > 0],
            weights=pop[cnt > 0])) if (cnt > 0).any() else 0.0,
    }


# ── 의원 실시간 현황 (가상) ───────────────────────────────────────────
def clinic_status(meta: dict, idx: list[int], clock: float,
                  seed: int = 20261016) -> list[dict]:
    """의원별 현재 진료 의사 수와 대기 환자 — **가상 자료**.

    건강보험심사평가원은 기관별 인력 현황을 공개하지만 **실시간 진료 인원이나
    대기 환자를 주는 공개 API 는 없다.** 아래 값은 그런 API 가 있다고 치고
    만든 것이며 실제가 아니다. 뒷날 연동한다면 이 함수만 갈아끼우면 된다.

    그럴듯하게 두려고 세 가지를 지킨다.
      · 점심(12:30~14:00)과 18시 이후에는 진료 인원이 줄거나 0 이 된다
      · 대기 환자는 오전 9~11시와 오후 2~4시에 몰린다
      · 같은 의원·같은 분이면 언제나 같은 값이 나온다(시드 고정)
    """
    out = []
    for j in idx:
        d = meta["dest"][j]
        staff = max(1, int(d[7]))
        rnd = np.random.default_rng(seed + j * 7919)
        h = clock / 60
        if h < 9 or h >= 18:
            open_ratio = 0.0
        elif 12.5 <= h < 14:
            open_ratio = 0.35
        elif h >= 17:
            open_ratio = 0.6
        else:
            open_ratio = 1.0
        on = int(round(staff * open_ratio * (0.75 + 0.5 * rnd.random())))
        on = min(staff, max(0 if open_ratio == 0 else 1, on))
        # 혼잡도 — 오전·오후 봉우리 두 개
        peak = (np.exp(-((h - 10.0) / 1.5) ** 2)
                + 0.8 * np.exp(-((h - 15.0) / 1.6) ** 2))
        waiting = 0 if on == 0 else int(round(
            on * 4.5 * peak * (0.6 + 0.8 * rnd.random())))
        wait_min = 0 if on == 0 else int(round(waiting / on * 6.5))
        out.append({"dest": j, "name": d[5], "kind": d[6],
                    "staff": staff, "on": on, "waiting": waiting,
                    "wait_min": wait_min, "internal": d[2],
                    "open": on > 0})
    return out


def now_minutes(now) -> float:
    """실제 시각을 운행 시각(분)으로. 운행시간 밖이면 끝 시각에 붙인다."""
    m = now.hour * 60 + now.minute + now.second / 60
    return min(max(m, T0), T1 + 180)
