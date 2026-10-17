# -*- coding: utf-8 -*-
"""시나리오 격자를 미리 굽는다 — 승차장 선정법 × 차종 × 대수 × 승차장 수.

왜 미리 굽는가
  정책 판단용 화면에서 슬라이더를 움직일 때마다 몇 초씩 기다리면 비교가 안
  된다. 조합이 2 × 3 × 12 × 13 = 936개뿐이므로 전부 미리 돌려 표로 만든다.

승차장 선정법 둘을 모두 굽는다
  mclp    최대 커버 입지(권장). 기준 거리 S 안 고령인구를 최대화한다.
  kmeans  고령인구 가중 k-평균(비교 기준선). 예전 방식이다.
  보행거리도 선정법에 따라 다르다 — mclp 는 **실제 보행망**, kmeans 는 직선.
  둘을 나란히 둬야 "보행망으로 바꾼 몫"과 "MCLP 로 바꾼 몫"을 나눠 볼 수 있다.

카카오 호출을 줄이는 2패스
  패스 1 에서 카카오 없이 MCLP 를 k=4..16 풀어 **선택된 후보의 합집합**을
  얻는다(실측 17곳). 패스 2 에서 그 17곳에 대해서만 거리행렬을 받는다.
  열쇠를 승차장 번호가 아니라 **후보번호(cid)** 로 두므로 13개 k 가 한 표를
  나눠 쓰고, S 를 바꿔 다시 돌려도 이미 받은 후보는 재사용된다.

실행
  .venv/Scripts/python.exe -u build_scenarios.py              # 기본 S=600
  .venv/Scripts/python.exe -u build_scenarios.py --no-kakao
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pyproj import Transformer

import mclp
import sim

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "scenarios.json"
TO_WGS = Transformer.from_crs(5179, 4326, always_xy=True)

K_RANGE = range(4, 17)
FLEET_RANGE = range(1, 13)
NEAR_CLINICS = 25
SENS_S = (400.0, 500.0, 600.0, 800.0)


def lonlat(x, y) -> tuple[float, float]:
    a, b = TO_WGS.transform(x, y)
    return round(float(a), 6), round(float(b), 6)


def kakao_cids(road: sim.Road, meta: dict, C: mclp.Cands,
               cids: list[int], kk) -> tuple[dict, int]:
    """후보 합집합에 대해서만 카카오 거리행렬을 받는다.

    열쇠는 ("C", cid) ↔ ("D", 의원). 승차장 번호가 아니라 후보번호라서
    k 가 달라도, S 가 달라도 같은 표를 다시 쓴다.
    """
    dest = np.array([[d[0], d[1]] for d in meta["dest"]], float)
    dn = road.snap(dest)
    nodes = np.array([int(C.node[c]) for c in cids])
    row, dist, _ = road.solve(nodes)

    table, got = {}, 0
    near_of: dict[int, list[int]] = {}
    for c in cids:
        dv = dist[row[int(C.node[c])], dn]
        near_of[c] = [int(j) for j in np.argsort(dv)[:NEAR_CLINICS]
                      if np.isfinite(dv[j])]

    cl = [lonlat(C.cx[c], C.cy[c]) for c in cids]
    for c, o in zip(cids, cl):                       # 후보 → 의원
        js = near_of[c]
        for j, v in zip(js, kk.matrix(o, [lonlat(*dest[j]) for j in js])):
            if v:
                table[(("C", c), ("D", j))] = v["km"]
                got += 1
    used = sorted({j for js in near_of.values() for j in js})
    for j in used:                                   # 의원 → 후보
        for c, v in zip(cids, kk.matrix(lonlat(*dest[j]), cl)):
            if v:
                table[(("D", j), ("C", c))] = v["km"]
                got += 1
    for c, o in zip(cids, cl):                       # 후보 → 후보 (공차)
        for c2, v in zip(cids, kk.matrix(o, cl)):
            if v and c2 != c:
                table[(("C", c), ("C", c2))] = v["km"]
                got += 1
    return table, got


def rekey(T: dict, stops: list[dict]) -> dict:
    """("C", cid) 표를 sim 이 쓰는 ("S", i) 표로 옮긴다.

    sim 의 열쇠 규약은 건드리지 않는다 — 변경 범위를 좁게 둔다.
    """
    m = {s["cid"]: s["i"] for s in stops}
    out = {}
    for (a, b), v in T.items():
        ka = ("S", m[a[1]]) if a[0] == "C" and a[1] in m else a
        kb = ("S", m[b[1]]) if b[0] == "C" and b[1] in m else b
        if ka[0] == "C" or kb[0] == "C":
            continue
        out[(ka, kb)] = v
    return out


def serialize(stops: list[dict]) -> list[dict]:
    """화면이 다시 풀지 않아도 되게 x·y·node 까지 전부 내보낸다."""
    return [{k: s[k] for k in (
        "i", "cid", "x", "y", "node", "lon", "lat", "pop", "pop_cov",
        "cells", "walk_med", "walk_max", "exist", "stop_name", "spd_min",
        "name") if k in s} for s in stops]


def main(use_kakao: bool, write: bool, S_m: float,
         methods: tuple[str, ...]) -> None:
    t0 = time.perf_counter()
    road, meta = sim.Road(), sim.load_meta()
    C = mclp.load_cands(meta)
    kk = None
    if use_kakao:
        import kakao
        kk = kakao.Kakao()
        print(f"[카카오] 키 {'있음' if kk.ok else '없음 → 도로망으로 진행'}")

    # ── 패스 1 — 카카오 없이 풀어 쓰이는 후보의 합집합을 얻는다 ───────
    # 두 선정법이 쓰는 후보를 **모두** 모아 한 번에 받는다. 한쪽만 카카오
    # 거리를 쓰면 비교가 또 어긋난다.
    ladder, T_cid, km_cids = {}, {}, {}
    U: set[int] = set()
    if "mclp" in methods:
        t = time.perf_counter()
        ladder = mclp.solve_ladder(C, K_RANGE, S_m)
        U |= {c for r in ladder.values() for c in r["cids"]}
        print(f"[MCLP] S={S_m:.0f}m · k={K_RANGE.start}~{K_RANGE.stop - 1} "
              f"{time.perf_counter() - t:.1f}초 · 후보 합집합 {len(U)}곳 "
              f"(실재 정류장 {int(C.exist[sorted(U)].sum())})")
    if "kmeans" in methods:
        for k in K_RANGE:
            km_cids[k] = mclp.cids_of(C, sim.designated_stops(road, meta, k))
            U |= set(km_cids[k])
        print(f"[k-평균] 비교 기준선 후보 포함 → 합집합 {len(U)}곳")
    if kk and kk.ok and U:
        t = time.perf_counter()
        T_cid, got = kakao_cids(road, meta, C, sorted(U), kk)
        kk.flush()
        print(f"[카카오] 거리 {got:,}쌍 · {kk.report()} · "
              f"{time.perf_counter() - t:.0f}초")

    rows, stops_out, reach_out, cover = [], {}, {}, {}
    for sel in methods:
        stops_out[sel], reach_out[sel], cover[sel] = {}, {}, {}
        for k in K_RANGE:
            if sel == "mclp":
                st = mclp.stops(C, ladder[k]["cids"], S_m)
                wt = mclp.assign(C, sorted(ladder[k]["cids"]))
                tab = rekey(T_cid, st) if T_cid else None
                cov = ladder[k]["cov_share"]
            else:
                st = sim.designated_stops(road, meta, k)
                for s in st:
                    s.setdefault("cid", -1)
                    s.setdefault("exist", False)
                    s.setdefault("pop_cov", 0.0)
                    s.setdefault("spd_min", 0.0)
                    s.setdefault("stop_name", "")
                    s["lon"], s["lat"] = lonlat(s["x"], s["y"])
                # k-평균 승차장도 후보 풀에 넣어 뒀으므로 **같은 보행망 자로**
                # 잰다. 직선으로 재면 옛 방식이 유리해져 비교가 깨진다.
                kc = km_cids[k]
                for s, c in zip(st, kc):
                    s["cid"] = int(c)
                wt = mclp.assign(C, kc)
                tab = rekey(T_cid, st) if T_cid else None
                cov = float(C.pop[(C.walk[:, kc] <= S_m).any(1)].sum()
                            / C.pop.sum())
            reach_out[sel][str(k)] = sim.reach_tables(
                road, meta, st, tab, walk_tab=wt)
            stops_out[sel][str(k)] = serialize(st)
            cover[sel][str(k)] = round(cov, 4)

            for vi, veh in enumerate(sim.VEHICLES):
                for f in FLEET_RANGE:
                    S = sim.simulate(road, meta, veh, f, stops=st,
                                     km_table=tab, walk_tab=wt)
                    c = sim.carbon(S)
                    w = np.array([r.pick - r.call for r in S["rides"]])
                    rows.append({
                        "sel": sel, "vi": vi, "fleet": f, "k": k,
                        "cover_share": round(cov, 4),
                        "occ": round(S["occ"], 3),
                        "wait": round(S["wait"], 1),
                        "wait_p90": round(float(np.percentile(w, 90)), 1),
                        "km_rev": round(S["km_rev"], 1),
                        "km_dead": round(S["km_dead"], 1),
                        "km": round(S["km"], 1),
                        "dead_share": round(S["dead_share"], 4),
                        "walk_med": round(S["walk_med"]),
                        "walk_mean": round(S["walk_mean"]),
                        "walk_p90": round(S["walk_p90"]),
                        "walk_max": round(S["walk_max"]),
                        "walk_min": round(S["walk_min_med"], 1),
                        "co2": round(c["co2_tot"], 2),
                        "co2_rev": round(c["co2_rev"], 2),
                        "base": round(c["base"], 2),
                        "cut": round(c["cut_tot"], 2),
                        "co2_year_t": round(c["co2_tot"] * 365 / 1000, 2),
                        "cut_year_t": round(c["cut_tot"] * 365 / 1000, 2),
                        "breakeven": round(c["breakeven"], 3),
                        "breakeven_dead": round(c["breakeven_dead"], 3),
                        "cost_year": f * sim.COST_DAY * 365,
                        "end": round(S["end"], 1),
                        "in_hours": bool(S["end"] <= sim.T1),
                        "kakao": round(S["kakao_share"], 3),
                    })
        print(f"[{sel}] {len([r for r in rows if r['sel'] == sel]):,}행 "
              f"({time.perf_counter() - t0:.0f}초)")

    # ── 기준 거리 민감도 — 커버만, 시뮬레이션 없음 (2초) ──────────────
    sens = {}
    if "mclp" in methods:
        for s in SENS_S:
            L = mclp.solve_ladder(C, K_RANGE, s)
            sens[str(int(s))] = {str(k): round(L[k]["cov_share"], 4)
                                 for k in K_RANGE}
        print(f"[민감도] 기준 거리 {len(SENS_S)}종 × k {len(list(K_RANGE))}개")

    if kk:
        kk.flush()
    out = {
        "rows": rows, "stops": stops_out, "reach": reach_out,
        "cover": cover, "sens_S": sens,
        "vehicles": [{"id": v["id"], "name": v["name"], "ef": round(v["ef"], 1),
                      "ef_hi": round(v["ef_hi"], 1) if v["ef_hi"] else None,
                      "cap": v["cap"]} for v in sim.VEHICLES],
        "meta": {"생성": time.strftime("%Y-%m-%d"),
                 "거리출처": "카카오모빌리티" if (kk and kk.ok)
                          else "국가표준노드링크 2025-05-12",
                 "S_m": S_m, "methods": list(methods),
                 "보행": "OSM 보행망 최단경로 (MCLP) / 직선 (k-평균)",
                 "후보지": C.meta, "k": list(K_RANGE),
                 "fleet": list(FLEET_RANGE)},
    }
    print(f"\n[표] {len(rows):,}행 · 선정법 {len(methods)}종")
    for sel in methods:
        r = next(x for x in rows if x["sel"] == sel and x["k"] == 10
                 and x["fleet"] == 10 and x["vi"] == 1)
        rc = sim.reach(reach_out[sel]["10"], 30, r["wait"])
        print(f"  {sel:<7} k=10·10대 · 커버 {100 * r['cover_share']:.1f}% · "
              f"보행중앙 {r['walk_med']}m · 30분 도달 "
              f"{100 * rc['now_share']:.1f}% → {100 * rc['new_share']:.1f}%")
    if write:
        OUT.write_text(json.dumps(out, ensure_ascii=False,
                                  separators=(",", ":")), encoding="utf-8")
        print(f"\n  저장 {OUT.relative_to(HERE)}  "
              f"{OUT.stat().st_size / 1024:,.0f} KB")
    print(f"총 {time.perf_counter() - t0:.0f}초")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=float, default=600.0)
    ap.add_argument("--methods", default="mclp,kmeans")
    ap.add_argument("--no-kakao", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(not a.no_kakao, not a.dry_run, a.S,
         tuple(m.strip() for m in a.methods.split(",") if m.strip()))
