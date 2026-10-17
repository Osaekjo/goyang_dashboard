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
    back: int         # 0 집→의원 · 1 의원→집


def simulate(road: Road, meta: dict, veh: dict, fleet: int,
             seed: int = 20261016) -> dict:
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

    o_node = road.snap(np.c_[ox[oi], oy[oi]])
    d_node = road.snap(dest[di])
    row, dist, pred = road.solve(np.r_[o_node, d_node])

    legs = []
    for i in range(n_trip):
        legs.append({"t": t_call[i], "a": int(o_node[i]), "b": int(d_node[i]),
                     "ax": ox[oi[i]], "ay": oy[oi[i]],
                     "bx": dest[di[i], 0], "by": dest[di[i], 1], "back": 0})
        legs.append({"t": t_call[i] + CARE, "a": int(d_node[i]),
                     "b": int(o_node[i]),
                     "ax": dest[di[i], 0], "ay": dest[di[i], 1],
                     "bx": ox[oi[i]], "by": oy[oi[i]], "back": 1})
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
    at: list[int | None] = [None] * fleet
    rides: list[Ride] = []

    def dead_km(v: int, a: int) -> float:
        if at[v] is None:
            return 1.0
        d = dist[row[at[v]], a] if at[v] in row else np.inf
        return float(d) / 1000 if np.isfinite(d) else 1.5

    for grp in groups:
        g = grp[0]
        a, b = g["a"], g["b"]
        best, best_t, best_d = 0, math.inf, 1.0
        for v in range(fleet):
            dk = dead_km(v, a)
            arr = max(g["t"], free[v]) + dk / SPEED_KMH * 60 + BOARD
            if arr < best_t:
                best, best_t, best_d = v, arr, dk
        v, km_dead, pick = best, best_d, best_t
        km_rev = dist[row[a], b] / 1000
        if not np.isfinite(km_rev):
            km_rev = math.hypot(g["bx"] - g["ax"], g["by"] - g["ay"]) / 1000 * 1.4
        depart = max(g["t"], free[v])
        drop = pick + km_rev / SPEED_KMH * 60
        free[v] = drop + BOARD
        at[v] = b
        rides.append(Ride(v, len(grp), g["t"], depart, pick, drop,
                          km_rev, km_dead, road.path_xy(pred[row[a]], a, b),
                          g["back"]))

    pax = sum(r.n for r in rides)
    km_rev = sum(r.km_rev for r in rides)
    km_dead = sum(r.km_dead for r in rides)
    wait = sum((r.pick - r.call) * r.n for r in rides) / pax
    return {"rides": rides, "trips": n_trip, "pax": pax, "fleet": fleet,
            "occ": pax / len(rides), "km_rev": km_rev, "km_dead": km_dead,
            "km": km_rev + km_dead, "wait": wait,
            "end": max(r.drop for r in rides), "veh": veh,
            "dead_share": km_dead / (km_rev + km_dead)}


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


def now_minutes(now) -> float:
    """실제 시각을 운행 시각(분)으로. 운행시간 밖이면 끝 시각에 붙인다."""
    m = now.hour * 60 + now.minute + now.second / 60
    return min(max(m, T0), T1 + 180)
