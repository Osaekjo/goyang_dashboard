# -*- coding: utf-8 -*-
"""거점에서 의료기관까지 — 거점 연계형의 '그 다음'을 굽는다.

왜 필요한가
  똑버스가 거점까지만 태워주면, 거기서 의료기관까지 얼마나 더 걸리는지를
  알아야 "30분 안에 병원에 닿는가"를 셀 수 있다. 처음에는 협업자 수치를 빌려
  **일산백병원 하나**만 봤는데, 그러면 대화역 앞 39 m 에 있는 의원을 못 센다.
  이 자리에서 우리 v6 분석으로 일반화한다.

두 갈래를 합친다
  1) 도보  — 거점에서 걸어갈 수 있는 의료기관. 대화역은 300 m 안에 8곳이다.
  2) 환승  — 거점에서 대중교통으로 더 가는 것. 원 분석(step05)이 예산별
             주점수 A(b) 를 이미 갖고 있어, 거점에 가장 가까운 출발점의 값을
             쓴다. 대화역 48 m, 일산서구청 96 m 로 충분히 가깝다.

  남은 예산 r 에서 닿는 기관 수 = max(도보로 r 안, A(r) 보간)
  A 는 b = 20·30·45·60·80 분에서만 있으므로 그 사이는 선형보간하고, 20분
  아래는 도보 쪽이 맡는다.

실행 (원 프로젝트 자료가 필요하다)
  C:\\jhj\\goyang\\.venv\\Scripts\\python.exe -u bake_hub.py
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = DATA / "hub_access.json"
V6 = Path(r"C:\jhj\goyang")

BUDGETS = (20, 30, 45, 60, 80)
WALK_SPEED = 0.8
WALK_DETOUR = 1.34        # 보행망÷직선 실측 중앙값
HUBS = {
    "대화역": (933643.0, 1964360.0),
    "일산서구청": (935203.0, 1964889.0),
}


def main(write: bool, v6: Path) -> None:
    t0 = time.perf_counter()
    meta = json.loads((DATA / "data.json").read_text(encoding="utf-8"))
    dest = np.array([[d[0], d[1]] for d in meta["dest"]], float)
    names = [d[5] for d in meta["dest"]]

    o2 = pd.read_csv(v6 / "outputs/v6/tables/step02_origins.csv")
    out = {}
    for hub, (hx, hy) in HUBS.items():
        d = np.hypot(o2.x - hx, o2.y - hy)
        i = int(d.idxmin())
        gid, snap = o2.loc[i, "gid"], float(d[i])

        A = {}
        for b in BUDGETS:
            s = pd.read_csv(
                v6 / f"outputs/v6/tables/step05_origin_score_b3_t{b}_v0.8.csv")
            m = s[s.gid == gid]
            A[b] = float(m.A_r.iloc[0]) if len(m) else float("nan")

        # 도보로 닿는 의료기관 — 분 단위로 누적해 둔다
        km = np.hypot(dest[:, 0] - hx, dest[:, 1] - hy) * WALK_DETOUR
        mins = km / WALK_SPEED / 60
        walk_cum = [int((mins <= t).sum()) for t in range(0, 61)]
        near = int(np.argmin(mins))

        out[hub] = {
            "xy": [hx, hy], "origin_gid": gid, "origin_snap_m": round(snap, 1),
            "A": {str(b): round(A[b], 2) for b in BUDGETS},
            "walk_cum": walk_cum,
            "nearest_min": round(float(mins[near]), 2),
            "nearest_name": names[near],
        }
        print(f"[{hub}] 최근접 출발점 {gid} ({snap:.0f} m)")
        print(f"   예산별 주점수 " + " · ".join(
            f"{b}분 {A[b]:.0f}곳" for b in BUDGETS))
        print(f"   도보권 의료기관 — 가장 가까운 곳 {mins[near]:.1f}분"
              f"({names[near][:18]}) · 5분 내 {walk_cum[5]}곳 · "
              f"10분 내 {walk_cum[10]}곳")

    if write:
        OUT.write_text(json.dumps(out, ensure_ascii=False,
                                  separators=(",", ":")), encoding="utf-8")
        print(f"\n  저장 {OUT.relative_to(HERE)}  "
              f"{OUT.stat().st_size / 1024:,.1f} KB")
    print(f"총 {time.perf_counter() - t0:.0f}초")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--v6", type=Path, default=V6)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(not a.dry_run, a.v6)
