# -*- coding: utf-8 -*-
"""보행거리표를 굽는다 — 빌드 전용, 원 프로젝트 자료가 필요하다.

왜 굽는가
  승차장을 MCLP 로 고르려면 **출발점에서 후보지까지 실제 보행거리**가 있어야
  한다. 그 보행망(OSM, 노드 184만·연장 53,046 km)은 원 분석 프로젝트에만 있고
  44 MB 라 대시보드 저장소에 올리지 않는다. 대신 여기서 한 번 돌려
  (출발점 124 × 후보 N) 거리표만 작은 npz 로 구워 넣는다.

  이 스크립트는 `data.json` 의 격자가 바뀔 때만 다시 돌리면 된다.

왜 직선이 아닌가
  이 지역에서 보행망 거리 ÷ 직선거리는 중앙 1.38, 상위 10%는 1.98 이다.
  논·하천·자유로가 가로막는데 직선으로 재면 보행이 절반으로 줄어 보인다.

후보지
  실재 버스정류장(보행망 노드를 직접 들고 있다) + 차량 도로망 노드를 격자로
  솎은 것. 둘을 섞는 이유는 도로 노드가 교차점이라 집이 몰린 이면도로 중간에
  점이 없는 반면, 정류장은 노변 실제 지점이기 때문이다.

실행 (원 프로젝트 venv 로)
  C:\\jhj\\goyang\\.venv\\Scripts\\python.exe -u bake_walk.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = DATA / "walk_cand.npz"

WALK_GRAPH = Path(r"C:\jhj\goyang\data\interim\walk_graph.npz")
STOPS_PARQUET = Path(r"C:\jhj\goyang\data\interim\v6_walk\stops.parquet")

PAD_M = 800.0          # 군집 bbox 바깥 여유 — 경계 밖 승차장도 후보로
THIN_M = 100.0         # 도로 노드 솎는 격자. 75m 로 줄여도 커버가 거의 같다
STOP_BUFFER_M = 1000.0
WSNAP_MAX_M = 60.0     # 후보→보행망 스냅 상한. 없으면 '순간이동'이 생긴다
LIMIT_M = 3000.0       # 다익스트라 반경
CHUNK = 8              # scipy 는 반경을 줘도 출력을 전량 할당한다. 8 ≈ 118 MB
FAR = 65535            # uint16 로 저장할 때의 '멀다/끊김' 표시


# ── 보행망 ────────────────────────────────────────────────────────────
# 원 프로젝트 goyang/src/goyang/walklib.py 의 WalkGraph.save 포맷을 그대로
# 읽는다. 저장소를 엮지 않으려고 적재부만 옮겨 적었다.
@dataclass
class WalkGraph:
    csr: csr_matrix
    xy: np.ndarray
    comp: np.ndarray
    main: int

    @classmethod
    def load(cls, path: Path) -> "WalkGraph":
        z = np.load(path)
        csr = csr_matrix((z["data"], z["indices"], z["indptr"]),
                         shape=tuple(z["shape"]))
        return cls(csr, z["xy"], z["comp"], int(z["main"][0]))

    @property
    def main_mask(self) -> np.ndarray:
        return self.comp == self.main

    def snap(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """최대 연결요소에만 붙인다 — 고립 섬에 갇히지 않게."""
        idx = np.where(self.main_mask)[0]
        d, j = cKDTree(self.xy[idx]).query(np.c_[np.asarray(x, float),
                                                 np.asarray(y, float)], k=1)
        return idx[j], d


def walk_rows(wg: WalkGraph, src: np.ndarray, dst: np.ndarray,
              limit: float, chunk: int) -> np.ndarray:
    """출발 노드 × 목적 노드 보행거리 (m). 못 닿으면 inf."""
    out = np.empty((len(src), len(dst)), np.float32)
    for s in range(0, len(src), chunk):
        part = dijkstra(wg.csr, directed=False, indices=src[s:s + chunk],
                        limit=limit)
        out[s:s + chunk] = part[:, dst]
    return out


# ── 후보지 ────────────────────────────────────────────────────────────
def build_candidates(P: np.ndarray, stops_parquet: Path) -> pd.DataFrame:
    g = np.load(DATA / "v6_graph.npz")
    rn = np.c_[g["nx"], g["ny"]].astype(float)
    spd = np.full(len(rn), np.inf)
    # 노드에 접한 링크의 최저 제한속도 — 간선 노변 정차는 따로 확인해야 한다
    v = g["elen"] / np.maximum(g["esec"], 1e-6) * 3.6
    for arr in (g["ef"], g["et"]):
        np.minimum.at(spd, arr, v)

    lo, hi = P.min(0) - PAD_M, P.max(0) + PAD_M
    m = ((rn[:, 0] >= lo[0]) & (rn[:, 0] <= hi[0]) &
         (rn[:, 1] >= lo[1]) & (rn[:, 1] <= hi[1]))
    rn, spd_r, nid = rn[m], spd[m], np.where(m)[0]
    keep = np.unique(np.floor(rn / THIN_M).astype(int), axis=0,
                     return_index=True)[1]
    road = pd.DataFrame({"x": rn[keep, 0], "y": rn[keep, 1],
                         "node": nid[keep], "spd_min": spd_r[keep],
                         "exist": False, "stop_id": "", "stop_name": "",
                         "src": "road"})

    lo, hi = P.min(0) - STOP_BUFFER_M, P.max(0) + STOP_BUFFER_M
    st = pd.read_parquet(stops_parquet)
    st = st[(st.x >= lo[0]) & (st.x <= hi[0]) &
            (st.y >= lo[1]) & (st.y <= hi[1])]
    st = st.drop_duplicates(subset=["x", "y"])
    tree = cKDTree(np.c_[g["nx"], g["ny"]])
    _, sn = tree.query(st[["x", "y"]].values, k=1)
    pole = pd.DataFrame({"x": st.x.values, "y": st.y.values, "node": sn,
                         "spd_min": spd[sn], "exist": True,
                         "stop_id": st.stop_id.astype(str).values,
                         "stop_name": st.stop_name.astype(str).values,
                         "src": "pole"})

    # 비교 기준선인 가중 k-평균 승차장도 후보로 넣는다. 안 넣으면 그쪽만
    # 직선거리로 재게 돼 "보행망으로 바꾼 몫"과 "MCLP 로 바꾼 몫"을 가를 수 없다.
    import sim
    rd = sim.Road()
    meta = json.loads((DATA / "data.json").read_text(encoding="utf-8"))
    km = []
    for k in range(4, 17):
        for s in sim.designated_stops(rd, meta, k):
            km.append((s["x"], s["y"], s["node"]))
    km = pd.DataFrame(km, columns=["x", "y", "node"]).drop_duplicates(
        subset=["x", "y"])
    km["spd_min"] = spd[km.node.values]
    km["exist"] = False
    km["stop_id"] = ""
    km["stop_name"] = ""
    km["src"] = "kmeans"
    print(f"  k-평균 승차장 후보 {len(km)}개 (k=4..16 서로 다른 좌표)")

    # km 을 앞에 둬야 중복 제거에서 살아남아 'kmeans' 표시를 지킨다.
    # 뒤에 두면 같은 좌표의 도로 노드 행에 흡수된 뒤 스냅 필터에 걸려 사라진다.
    c = pd.concat([pole, km, road], ignore_index=True)
    c = c.drop_duplicates(subset=["x", "y"]).reset_index(drop=True)
    c["cid"] = np.arange(len(c))
    return c


def main(write: bool, walk_graph: Path, stops_parquet: Path) -> None:
    t0 = time.perf_counter()
    meta = json.loads((DATA / "data.json").read_text(encoding="utf-8"))
    cells = [c for c in meta["cells"] if c[4] and c[3] > 0]
    P = np.array([[c[0] + 125, c[1] + 125] for c in cells], float)
    pop = np.array([c[3] for c in cells], float)
    print(f"[출발점] {len(P)}개 · 고령 {pop.sum():,.1f}명")

    cand = build_candidates(P, stops_parquet)
    print(f"[후보지] {len(cand):,}개 = 실재 정류장 {int(cand.exist.sum()):,}"
          f" + 도로 노드 {int((~cand.exist).sum()):,} ({THIN_M:.0f}m 솎음)")

    wg = WalkGraph.load(walk_graph)
    print(f"[보행망] 노드 {len(wg.xy):,} · 최대 연결요소 "
          f"{int(wg.main_mask.sum()):,} ({100 * wg.main_mask.mean():.1f}%)")

    on, od = wg.snap(P[:, 0], P[:, 1])
    qn, qd = wg.snap(cand.x.values, cand.y.values)
    # 스냅거리는 보행거리에 더해 세므로 '공짜 이동'은 없다. 이 상한은
    # 보행로에서 한참 떨어진 도로 노드를 승차장 후보로 삼지 않으려는 것이다.
    # 비교 기준선(k-평균)과 실재 정류장은 그대로 둬야 같은 자로 잴 수 있다.
    keep = (qd <= WSNAP_MAX_M) | (cand.src.values != "road")
    print(f"[스냅] 출발점 중앙 {np.median(od):.0f} m · 최대 {od.max():.0f} m")
    print(f"       후보 중앙 {np.median(qd):.0f} m · "
          f"{WSNAP_MAX_M:.0f}m 초과로 버린 후보 {int((~keep).sum()):,}개"
          f" (남은 {int(keep.sum()):,})")
    nk0 = int((cand.src == "kmeans").sum())
    cand, qn, qd = cand[keep].reset_index(drop=True), qn[keep], qd[keep]
    print(f"       비교 기준선(k-평균) 후보 {int((cand.src == 'kmeans').sum())}"
          f"/{nk0}개 생존")
    cand["cid"] = np.arange(len(cand))

    t1 = time.perf_counter()
    U, V = np.unique(on), np.unique(qn)
    M = walk_rows(wg, U, V, LIMIT_M, CHUNK)
    ui = {v: i for i, v in enumerate(U)}
    vi = np.array([np.searchsorted(V, v) for v in qn])
    W = M[[ui[v] for v in on]][:, vi]
    W += od[:, None] + qd[None, :]          # 양 끝 스냅거리를 포함한다
    print(f"[보행거리] {W.shape[0]}×{W.shape[1]} · "
          f"{time.perf_counter() - t1:.1f}초")

    S = np.hypot(P[:, None, 0] - cand.x.values[None],
                 P[:, None, 1] - cand.y.values[None])
    ok = np.isfinite(W)
    print(f"  보행÷직선 중앙 {np.median(W[ok] / np.maximum(S[ok], 1)):.3f}"
          f" · 90% {np.percentile(W[ok] / np.maximum(S[ok], 1), 90):.3f}")
    bad = int((W < S - 1).sum())
    assert bad == 0, f"보행거리가 직선보다 짧은 쌍 {bad}개 — 스냅이 틀렸다"
    near = np.nanmin(np.where(ok, W, np.nan), axis=1)
    print(f"  출발점별 최근접 후보 중앙 {np.median(near):.0f} m · "
          f"{LIMIT_M:.0f}m 안에 후보가 없는 출발점 "
          f"{int(np.isnan(near).sum())}개")
    for s in (400, 500, 600, 800):
        c = pop[(W <= s).any(1)].sum()
        print(f"  후보 전부 쓸 때 {s:>3}m 커버 천장 {c:6.1f}명"
              f" ({100 * c / pop.sum():4.1f}%)")

    sig = hashlib.md5(
        np.ascontiguousarray(np.c_[P, pop]).tobytes()).hexdigest()[:12]
    if write:
        np.savez_compressed(
            OUT,
            walk=np.where(np.isfinite(W), np.minimum(W, FAR - 1),
                          FAR).astype(np.uint16),
            cand_x=cand.x.values.astype(np.float32),
            cand_y=cand.y.values.astype(np.float32),
            cand_node=cand.node.values.astype(np.int32),
            cand_spd=cand.spd_min.values.astype(np.float32),
            cand_exist=cand.exist.values.astype(bool),
            cand_stop_id=cand.stop_id.values.astype("U24"),
            cand_stop_name=cand.stop_name.values.astype("U40"),
            o_snap=od.astype(np.float32),
            cells_sig=np.array([sig]),
            meta=np.array([json.dumps({
                "radius_candidates": len(cand), "thin_m": THIN_M,
                "pad_m": PAD_M, "wsnap_max_m": WSNAP_MAX_M,
                "limit_m": LIMIT_M, "walk_nodes": int(len(wg.xy)),
                "walk_src": str(walk_graph),
                "built": time.strftime("%Y-%m-%d")}, ensure_ascii=False)]))
        print(f"\n  저장 {OUT.relative_to(HERE)}  "
              f"{OUT.stat().st_size / 1024:,.0f} KB · 격자서명 {sig}")
    print(f"총 {time.perf_counter() - t0:.0f}초")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk-graph", type=Path, default=WALK_GRAPH)
    ap.add_argument("--stops", type=Path, default=STOPS_PARQUET)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(not a.dry_run, a.walk_graph, a.stops)
