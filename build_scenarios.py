# -*- coding: utf-8 -*-
"""시나리오 격자를 미리 굽는다 — 차종 × 대수 × 승차장 수.

왜 미리 굽는가
  정책 판단용 화면에서 슬라이더를 움직일 때마다 몇 초씩 기다리면 비교가 안
  된다. 조합이 3 × 12 × 13 = 468개뿐이므로 전부 미리 돌려 표로 만든다.
  화면은 그 표만 읽어 즉시 그린다.

카카오 길찾기
  .env 에 KAKAO_REST_API_KEY 가 있으면 **승차장 ↔ 의원** 주행거리를 카카오에서
  받아 쓴다. 승차장별로 가까운 의원 25곳까지만 묻고(한 번에 30곳까지 가능)
  응답은 data/kakao_cache.json 에 쟁여 둔다. 다시 돌리면 호출이 거의 없다.
  키가 없으면 국가표준노드링크 최단경로로 물러선다 — 화면은 똑같이 돈다.

실행
  .venv/Scripts/python.exe -u build_scenarios.py            # 전체
  .venv/Scripts/python.exe -u build_scenarios.py --no-kakao # 도로망만
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pyproj import Transformer

import sim

HERE = Path(__file__).resolve().parent
OUT = HERE / "data" / "scenarios.json"
TO_WGS = Transformer.from_crs(5179, 4326, always_xy=True)

K_RANGE = range(4, 17)          # 지정 승차장 수
FLEET_RANGE = range(1, 13)      # 도입 대수
NEAR_CLINICS = 25               # 승차장마다 카카오로 물어 둘 의원 수


def lonlat(x, y) -> tuple[float, float]:
    a, b = TO_WGS.transform(x, y)
    return round(float(a), 6), round(float(b), 6)


def kakao_table(road: sim.Road, meta: dict, stops: list[dict],
                kk) -> tuple[dict, int]:
    """승차장 ↔ 의원 주행거리를 카카오에서 받아 표로. (표, 받은 쌍 수)"""
    dest = np.array([[d[0], d[1]] for d in meta["dest"]], float)
    dn = road.snap(dest)
    row, dist, _ = road.solve(np.array([s["node"] for s in stops]))

    table, got = {}, 0
    near_of: dict[int, list[int]] = {}
    for s in stops:
        dv = dist[row[s["node"]], dn]
        near_of[s["i"]] = [int(j) for j in np.argsort(dv)[:NEAR_CLINICS]
                           if np.isfinite(dv[j])]

    # 승차장 → 의원
    for s in stops:
        js = near_of[s["i"]]
        o = lonlat(s["x"], s["y"])
        res = kk.matrix(o, [lonlat(*dest[j]) for j in js])
        for j, v in zip(js, res):
            if v:
                table[(("S", s["i"]), ("D", j))] = v["km"]
                got += 1

    # 의원 → 승차장 (귀가 구간과 공차)
    used = sorted({j for js in near_of.values() for j in js})
    sl = [lonlat(s["x"], s["y"]) for s in stops]
    for j in used:
        res = kk.matrix(lonlat(*dest[j]), sl)
        for s, v in zip(stops, res):
            if v:
                table[(("D", j), ("S", s["i"]))] = v["km"]
                got += 1
    return table, got


def main(use_kakao: bool, write: bool) -> None:
    t0 = time.perf_counter()
    road, meta = sim.Road(), sim.load_meta()
    kk = None
    if use_kakao:
        import kakao
        kk = kakao.Kakao()
        print(f"[카카오] 키 {'있음' if kk.ok else '없음 → 도로망으로 진행'}")

    rows, stops_out = [], {}
    for k in K_RANGE:
        stops = sim.designated_stops(road, meta, k)
        table = {}
        if kk and kk.ok:
            table, got = kakao_table(road, meta, stops, kk)
            kk.flush()
            print(f"  승차장 {k:>2}곳 · 카카오 거리 {got:,}쌍 · {kk.report()}")
        stops_out[str(k)] = [{
            "i": s["i"], "name": s["name"],
            "lon": lonlat(s["x"], s["y"])[0], "lat": lonlat(s["x"], s["y"])[1],
            "pop": round(s["pop"], 1), "cells": s["cells"],
            "walk_med": round(s["walk_med"]), "walk_max": round(s["walk_max"]),
        } for s in stops]

        for vi, veh in enumerate(sim.VEHICLES):
            for f in FLEET_RANGE:
                S = sim.simulate(road, meta, veh, f, stops=stops,
                                 km_table=table or None)
                c = sim.carbon(S)
                w = np.array([r.pick - r.call for r in S["rides"]])
                rows.append({
                    "vi": vi, "fleet": f, "k": k,
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
                    "cut_rev": round(c["cut_rev"], 2),
                    "co2_year_t": round(c["co2_tot"] * 365 / 1000, 2),
                    "cut_year_t": round(c["cut_tot"] * 365 / 1000, 2),
                    "breakeven": round(c["breakeven"], 3),
                    "breakeven_dead": round(c["breakeven_dead"], 3),
                    "cost_year": f * sim.COST_DAY * 365,
                    "end": round(S["end"], 1),
                    "in_hours": bool(S["end"] <= sim.T1),
                    "kakao": round(S["kakao_share"], 3),
                })
        print(f"  승차장 {k:>2}곳 끝 — 누적 {len(rows):,}행 "
              f"({time.perf_counter() - t0:.0f}초)")

    if kk:
        kk.flush()
    out = {
        "rows": rows, "stops": stops_out,
        "vehicles": [{"id": v["id"], "name": v["name"], "ef": round(v["ef"], 1),
                      "ef_hi": round(v["ef_hi"], 1) if v["ef_hi"] else None,
                      "cap": v["cap"]} for v in sim.VEHICLES],
        "meta": {"생성": time.strftime("%Y-%m-%d"),
                 "거리출처": "카카오모빌리티" if (kk and kk.ok)
                          else "국가표준노드링크 2025-05-12",
                 "k": list(K_RANGE), "fleet": list(FLEET_RANGE),
                 "trips": rows[0]["km"] and sim.TRIPS_PER_YEAR},
    }
    print(f"\n[표] {len(rows):,}행 · 승차장 집합 {len(stops_out)}개")
    best = min((r for r in rows if r["in_hours"] and r["wait"] <= 40),
               key=lambda r: (-r["cut_year_t"], r["walk_med"]), default=None)
    if best:
        print(f"  대기 40분 이하·운행시간 안에서 탄소가 가장 나은 조합: "
              f"{sim.VEHICLES[best['vi']]['name']} · {best['fleet']}대 · "
              f"승차장 {best['k']}곳 → 연 {best['cut_year_t']:+.1f} t · "
              f"보행 중앙 {best['walk_med']} m · 대기 {best['wait']:.0f}분")
    if write:
        OUT.write_text(json.dumps(out, ensure_ascii=False,
                                  separators=(",", ":")), encoding="utf-8")
        print(f"\n  저장 {OUT.relative_to(HERE)}  "
              f"{OUT.stat().st_size / 1024:,.0f} KB")
    print(f"총 {time.perf_counter() - t0:.0f}초")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-kakao", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(not a.no_kakao, not a.dry_run)
