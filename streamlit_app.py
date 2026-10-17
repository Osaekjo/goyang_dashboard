# -*- coding: utf-8 -*-
"""가좌·송포 똑버스 운행 대시보드.

시계를 **실제 현재 시각**에 묶는다. 지금이 14:23이면 화면도 14:23 상태를 보이고
10초마다 스스로 갱신한다. 다만 가좌·송포에는 아직 똑버스가 없으므로, 이것은
"지금 이 시각 똑버스가 돌고 있다면" 을 보여주는 **가상 운행**이다.

지도는 인구 분포가 아니라 **경로**를 보인다. 지정 승차장에서 가까운 의원까지
실제 도로 경로를 깔고, 그 위로 차량이 움직인다.

숫자는 전부 sim.py 가 쥔다. 이 파일은 그린다.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from pyproj import Transformer

import sim

KST = timezone(timedelta(hours=9))
TO_WGS = Transformer.from_crs(5179, 4326, always_xy=True)
C1, C2, C3, C4 = "#16707f", "#b8540f", "#6b4a9e", "#8c9499"

st.set_page_config(page_title="가좌·송포 똑버스 운행 대시보드",
                   page_icon="🚐", layout="wide")


# ── 자료 ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="도로망을 읽는 중")
def get_road() -> sim.Road:
    return sim.Road()


@st.cache_data(show_spinner=False)
def get_meta() -> dict:
    return sim.load_meta()


def ll(xy) -> list[list[float]]:
    xy = np.asarray(xy, float)
    if len(xy) == 0:
        return []
    lon, lat = TO_WGS.transform(xy[:, 0], xy[:, 1])
    return [[round(a, 6), round(b, 6)] for a, b in zip(lon, lat)]


def one(x: float, y: float) -> list[float]:
    a, b = TO_WGS.transform(x, y)
    return [round(a, 6), round(b, 6)]


def _cum(xy) -> list[float]:
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return [0.0]
    d = np.r_[0, np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))]
    return (d / d[-1]).tolist() if d[-1] > 0 else [0.0] * len(xy)


@st.cache_data(show_spinner="승차장을 고르는 중", max_entries=12)
def get_stops(k: int) -> list[dict]:
    S = sim.designated_stops(get_road(), get_meta(), k)
    for s in S:
        s["ll"] = one(s["x"], s["y"])
    return S


@st.cache_data(show_spinner="경로를 찾는 중", max_entries=12)
def get_stop_routes(k: int, per: int) -> list[dict]:
    R = sim.stop_routes(get_road(), get_stops(k), get_meta(), per)
    for r in R:
        r["ll"] = ll(r.pop("path"))
    return R


@st.cache_data(show_spinner="운행을 계산하는 중", max_entries=60)
def run(vi: int, fleet: int, k: int | None) -> dict:
    """차량·대수·승차방식별 하루 운행. 시드가 고정이라 결과가 늘 같다."""
    s = sim.simulate(get_road(), get_meta(), sim.VEHICLES[vi], fleet,
                     stops=get_stops(k) if k else None)
    out = {kk: v for kk, v in s.items() if kk != "rides"}
    out["rides"] = [{
        "v": r.v, "n": r.n, "call": r.call, "depart": r.depart,
        "pick": r.pick, "drop": r.drop, "km_rev": r.km_rev,
        "km_dead": r.km_dead, "back": r.back, "stop": r.stop, "dest": r.dest,
        "ll": ll(r.path), "cum": _cum(r.path),
    } for r in s["rides"]]
    return out


def at_frac(path: list, cum: list[float], f: float) -> list[float]:
    if not path:
        return [0.0, 0.0]
    i = max(1, min(int(np.searchsorted(cum, f)), len(path) - 1))
    a, b = path[i - 1], path[i]
    span = cum[i] - cum[i - 1]
    t = 0.0 if span <= 0 else (f - cum[i - 1]) / span
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


META = get_meta()
M = META["meta"]

# ── 조작 ──────────────────────────────────────────────────────────────
st.markdown(
    "<div style='display:flex;gap:12px;align-items:baseline;flex-wrap:wrap'>"
    "<h2 style='margin:0;font-size:1.35rem'>가좌·송포 똑버스 운행 대시보드</h2>"
    "<span style='font-family:monospace;font-size:.68rem;letter-spacing:.08em;"
    "background:#fdf3e4;color:#7a4a09;border:1px dashed #dfb87e;"
    "border-radius:3px;padding:3px 8px;font-weight:600'>"
    "시뮬레이션 · 실제 운행 데이터 아님</span></div>", unsafe_allow_html=True)

c = st.columns([2, 1.1, 1.5, 1.1, 1.5, 1.2])
vi = c[0].selectbox("차량", range(len(sim.VEHICLES)),
                    format_func=lambda i: sim.VEHICLES[i]["name"], index=1)
fleet = c[1].slider("도입 대수", 1, 12, 7)
board_mode = c[2].radio("승차 방식", ["지정 승차장", "문 앞 호출"],
                        horizontal=True)
K = c[3].slider("승차장 수", 4, 16, 10,
                disabled=(board_mode == "문 앞 호출"))
mode = c[4].radio("시각", ["지금", "직접 고르기"], horizontal=True)

SIM = run(vi, fleet, K if board_mode == "지정 승차장" else None)
END = max(sim.T1, SIM["end"])
manual = c[5].slider("운행 시각", sim.T0, int(END), int(sim.T0 + 180),
                     disabled=(mode == "지금")) if mode != "지금" else None

STOPS = get_stops(K) if board_mode == "지정 승차장" else []
SROUTES = get_stop_routes(K, 3) if board_mode == "지정 승차장" else []


@st.fragment(run_every=10 if mode == "지금" else None)
def board() -> None:
    now = datetime.now(KST)
    if mode == "지금":
        raw = now.hour * 60 + now.minute + now.second / 60
        clock = min(max(raw, sim.T0), END)
        phase = ("운행 시작 전 — 오늘 08:30 개시" if raw < sim.T0
                 else "오늘 운행 종료" if raw > END else "운행 중")
    else:
        clock, phase = float(manual), "직접 고른 시각"
    draw(clock, phase, now)


def draw(clock: float, phase: str, now: datetime) -> None:
    pax = kmR = kmD = 0.0
    moving = onboard = 0
    per = [{"pax": 0, "km": 0.0, "st": "대기"} for _ in range(fleet)]
    vehs, waits_pt, live, waits = [], [], [], []
    busy_dest: dict[int, int] = {}
    for r in SIM["rides"]:
        P = per[r["v"]]
        if clock >= r["drop"]:
            pax += r["n"]; kmR += r["km_rev"]; kmD += r["km_dead"]
            P["pax"] += r["n"]; P["km"] += r["km_rev"] + r["km_dead"]
        elif clock >= r["pick"]:
            f = (clock - r["pick"]) / max(.01, r["drop"] - r["pick"])
            onboard += r["n"]; moving += 1
            kmR += r["km_rev"] * f; kmD += r["km_dead"]
            P["km"] += r["km_rev"] * f + r["km_dead"]
            P["st"] = f"승객 {r['n']}명 수송"
            vehs.append({"lon": at_frac(r["ll"], r["cum"], f)[0],
                         "lat": at_frac(r["ll"], r["cum"], f)[1],
                         "n": r["n"], "v": r["v"] + 1,
                         "lab": f"{r['n']}"})
            live.append({"path": r["ll"]})
        elif clock >= r["depart"]:
            moving += 1
            kmD += r["km_dead"] * min(
                1, (clock - r["depart"]) / max(.01, r["pick"] - r["depart"]))
            P["st"] = "배차 이동 (공차)"
        if r["call"] <= clock < r["pick"]:
            waits.append(clock - r["call"])
            if r["ll"]:
                waits_pt.append({"lon": r["ll"][0][0], "lat": r["ll"][0][1]})
        if r["call"] <= clock < r["drop"]:
            busy_dest[r["dest"]] = busy_dest.get(r["dest"], 0) + r["n"]

    ef = sim.VEHICLES[vi]["ef"]
    km, co2 = kmR + kmD, (kmR + kmD) * ef / 1000
    base = kmR * sim.EF_TAXI / 2 / 1000
    cut = base - co2

    hh, mm = int(clock // 60), int(clock % 60)
    st.markdown(
        f"<div style='font-family:monospace;font-size:1.5rem;font-weight:600'>"
        f"<span style='color:{C1}'>●</span> {hh:02d}:{mm:02d}"
        f"<span style='font-size:.5em;font-weight:400;color:#777;"
        f"margin-left:10px'>{phase} · 실제 시각 {now:%Y-%m-%d %H:%M:%S} KST"
        f"</span></div>", unsafe_allow_html=True)

    k = st.columns(6)
    k[0].metric("누적 탑승 (연인원)", f"{pax:,.0f}명",
                f"하루 {SIM['trips']}통행 × 2구간 = {SIM['pax']}명 중 "
                f"{100 * pax / SIM['pax']:.0f}%", delta_color="off")
    k[1].metric("운행 중 차량", f"{moving}/{fleet}대",
                f"탑승 {onboard}명 · 대기 호출 {len(waits)}건",
                delta_color="off")
    k[2].metric("누적 주행거리", f"{km:,.0f} km",
                f"수송 {kmR:,.0f} + 공차 {kmD:,.0f}"
                f" ({100 * kmD / max(1, km):.0f}%)", delta_color="off")
    k[3].metric("누적 탄소배출", f"{co2:,.1f} kg",
                f"현행 택시 2인 {base:,.1f} kg", delta_color="off")
    k[4].metric("현행 대비",
                f"{'▲ +' if cut >= 0 else '▼ −'}{abs(cut):,.1f} kg",
                f"연 {'+' if cut >= 0 else '−'}{abs(cut) * 365 / 1000:,.1f} t",
                delta_color="normal" if cut >= 0 else "inverse")
    if SIM["stops"]:
        k[5].metric("승차장까지 보행", f"{SIM['walk_med']:,.0f} m",
                    f"0.8 m/s 로 {SIM['walk_min_med']:.1f}분 · "
                    f"최대 {SIM['walk_max']:,.0f} m", delta_color="off")
    else:
        k[5].metric("승차장까지 보행", "0 m", "문 앞에서 탄다",
                    delta_color="off")

    left, right = st.columns([2.05, 1])
    with left:
        st.pydeck_chart(deck(live, vehs, waits_pt, clock, busy_dest),
                        width="stretch")
        st.caption(
            "옅은 선 = 지정 승차장에서 가까운 의원 3곳까지의 **실제 도로 경로**"
            "(국가표준노드링크 2025-05-12) · 굵은 주황 = 지금 수송 중 · "
            "청록 큰 점 = 지정 승차장(크기는 맡은 고령인구) · "
            "보라 = 의원(크기는 의사 수, 붉을수록 혼잡) · 사각 = 차량")
    with right:
        st.markdown("**의원 실시간 현황** &nbsp;"
                    "<span style='font-family:monospace;font-size:.62rem;"
                    "background:#fdf3e4;color:#7a4a09;border:1px dashed "
                    "#dfb87e;border-radius:3px;padding:2px 6px'>가상 자료"
                    "</span>", unsafe_allow_html=True)
        st.dataframe(clinic_table(clock, busy_dest), hide_index=True,
                     width="stretch", height=232)
        st.caption(
            "심평원은 기관별 **인력 현황**은 공개하지만 실시간 진료 인원·대기"
            " 환자를 주는 공개 API 는 없다. 위 값은 그런 API 가 있다고 치고"
            " 만든 것이다 — 총의사수만 실제 자료다.")
        st.markdown("**차량 현황**")
        st.dataframe(pd.DataFrame([{
            "차량": f"{sim.VEHICLES[vi]['id']}-{i + 1:02d}", "상태": p["st"],
            "탑승": p["pax"], "주행 km": round(p["km"])}
            for i, p in enumerate(per)]), hide_index=True, width="stretch",
            height=min(250, 36 * fleet + 38))

    a, b, cc = st.columns(3)
    with a:
        st.markdown("**승차장별 수요와 보행거리**")
        st.altair_chart(chart_stops(), width="stretch")
    with b:
        st.markdown("**탄소 수지** — 현행 택시 2인 합승 대비")
        st.altair_chart(chart_co2(clock, ef), width="stretch")
    with cc:
        st.markdown("**시간대별 탑승**")
        st.altair_chart(chart_hourly(clock), width="stretch")


# ── 의원 현황 ─────────────────────────────────────────────────────────
def clinic_table(clock: float, busy: dict[int, int]) -> pd.DataFrame:
    if SROUTES:
        idx, seen = [], set()
        for r in sorted(SROUTES, key=lambda r: (r["rank"], r["km"])):
            if r["dest"] not in seen:
                seen.add(r["dest"]); idx.append(r["dest"])
        near = {r["dest"]: min(
            x["km"] for x in SROUTES if x["dest"] == r["dest"])
            for r in SROUTES}
    else:
        cnt: dict[int, int] = {}
        for r in SIM["rides"]:
            cnt[r["dest"]] = cnt.get(r["dest"], 0) + r["n"]
        idx = sorted(cnt, key=lambda j: -cnt[j])[:14]
        near = {j: 0.0 for j in idx}
    rows = []
    for s in sim.clinic_status(META, idx[:14], clock):
        rows.append({
            "의원": s["name"][:14], "종별": s["kind"],
            "거리 km": round(near.get(s["dest"], 0), 1) or None,
            "의사": f"{s['on']}/{s['staff']}",
            "대기": s["waiting"],
            "예상 대기": "휴진" if not s["open"] else f"{s['wait_min']}분",
            "이동 중": busy.get(s["dest"], 0) or None,
        })
    return pd.DataFrame(rows)


# ── 지도 ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def base_layers() -> dict:
    dl, dt = TO_WGS.transform([d[0] for d in META["dest"]],
                              [d[1] for d in META["dest"]])
    dest = pd.DataFrame({
        "lon": dl, "lat": dt,
        "name": [d[5] for d in META["dest"]],
        "kind": [d[6] for d in META["dest"]],
        "doctors": [max(1, d[7]) for d in META["dest"]]})
    dong = []
    for d in META["dong"]:
        for ring in d["rings"]:
            a, b = TO_WGS.transform([p[0] for p in ring], [p[1] for p in ring])
            dong.append({"polygon": [[round(p, 6), round(q, 6)]
                                     for p, q in zip(a, b)]})
    gz = [c for c in META["cells"] if c[4]]
    lon, lat = TO_WGS.transform([c[0] + 125 for c in gz],
                                [c[1] + 125 for c in gz])
    return {"dest": dest, "dong": dong,
            "view": (float(np.mean(lon)), float(np.mean(lat)))}


def deck(live, vehs, waits, clock: float, busy) -> pdk.Deck:
    B = base_layers()
    L = [pdk.Layer("PolygonLayer", B["dong"], get_polygon="polygon",
                   stroked=True, filled=False,
                   get_line_color=[184, 84, 15, 130],
                   line_width_min_pixels=2)]

    # 승차장 → 의원 경로. 가장 가까운 한 곳은 진하게, 나머지는 옅게.
    if SROUTES:
        L.append(pdk.Layer(
            "PathLayer",
            [{"path": r["ll"], "w": 14 if r["rank"] == 0 else 7,
              "c": [22, 112, 127, 150] if r["rank"] == 0
                   else [22, 112, 127, 55],
              "name": f"{r['stop_name']} → {r['name']}",
              "tip": f"{r['km']:.1f} km · {r['min']:.0f}분"}
             for r in SROUTES],
            get_path="path", get_color="c", get_width="w",
            width_min_pixels=1, pickable=True,
            cap_rounded=True, joint_rounded=True))

    # 의원 — 크기는 의사 수, 색은 가상 혼잡도
    st_now = {s["dest"]: s for s in sim.clinic_status(
        META, list(range(len(META["dest"]))), clock)}
    d = B["dest"].copy()
    d["on"] = [st_now[i]["on"] for i in range(len(d))]
    d["waiting"] = [st_now[i]["waiting"] for i in range(len(d))]
    d["wait_min"] = [st_now[i]["wait_min"] for i in range(len(d))]
    d["col"] = [
        [150, 155, 160, 70] if r.on == 0 else
        [184, 84, 15, 215] if r.wait_min >= 30 else
        [196, 140, 60, 200] if r.wait_min >= 15 else
        [107, 74, 158, 190] for r in d.itertuples()]
    L.append(pdk.Layer("ScatterplotLayer", d, get_position=["lon", "lat"],
                       get_radius="60 + doctors * 22", radius_min_pixels=3,
                       radius_max_pixels=22, get_fill_color="col",
                       pickable=True, stroked=False))

    if live:
        L.append(pdk.Layer("PathLayer", live, get_path="path",
                           get_color=[184, 84, 15, 225], get_width=28,
                           width_min_pixels=3, cap_rounded=True,
                           joint_rounded=True))
    if waits:
        L.append(pdk.Layer("ScatterplotLayer", waits,
                           get_position=["lon", "lat"], get_radius=300,
                           radius_min_pixels=6, radius_max_pixels=18,
                           stroked=True, filled=False,
                           get_line_color=[22, 112, 127, 235],
                           line_width_min_pixels=2))
    if STOPS:
        sdf = pd.DataFrame([{
            "lon": s["ll"][0], "lat": s["ll"][1], "name": s["name"],
            "pop": s["pop"], "walk": round(s["walk_med"]),
            "lab": s["name"].split()[-1],
            "tip": f"고령 {s['pop']:.0f}명 · 보행 중앙 {s['walk_med']:.0f} m"}
            for s in STOPS])
        L.append(pdk.Layer("ScatterplotLayer", sdf,
                           get_position=["lon", "lat"],
                           get_radius="120 + pop * 2.4", radius_min_pixels=7,
                           radius_max_pixels=28,
                           get_fill_color=[22, 112, 127, 205],
                           stroked=True, get_line_color=[255, 255, 255, 230],
                           line_width_min_pixels=1.5, pickable=True))
        L.append(pdk.Layer("TextLayer", sdf, get_position=["lon", "lat"],
                           get_text="lab", get_size=11, get_color=[255] * 3,
                           get_alignment_baseline="'center'"))
    if vehs:
        L.append(pdk.Layer("ScatterplotLayer", vehs,
                           get_position=["lon", "lat"], get_radius=320,
                           radius_min_pixels=8, radius_max_pixels=18,
                           get_fill_color=[70, 40, 120, 240], pickable=True))
        L.append(pdk.Layer("TextLayer", vehs, get_position=["lon", "lat"],
                           get_text="lab", get_size=11, get_color=[255] * 3,
                           get_alignment_baseline="'center'"))
    return pdk.Deck(
        layers=L, map_provider="carto", map_style="light",
        initial_view_state=pdk.ViewState(
            longitude=B["view"][0], latitude=B["view"][1], zoom=11.3),
        tooltip={"html": "<b>{name}</b><br/>{tip}{kind} · 의사 {on}/{doctors}"
                         "명 · 대기 {waiting}명 ({wait_min}분)"},
        height=486)


# ── 그래프 ────────────────────────────────────────────────────────────
def chart_stops():
    if not STOPS:
        return alt.Chart(pd.DataFrame({"x": [0], "y": [0]})).mark_text(
            text="문 앞 호출 — 승차장이 없다", color="#888").encode()
    d = pd.DataFrame([{"승차장": s["name"].split()[-1], "고령인구": s["pop"],
                       "보행 중앙(m)": s["walk_med"]} for s in STOPS])
    bar = alt.Chart(d).mark_bar(color=C1, cornerRadius=2).encode(
        x=alt.X("승차장:N", sort=None, title=None),
        y=alt.Y("고령인구:Q", title="맡은 고령인구 (명)"),
        tooltip=["승차장", "고령인구", "보행 중앙(m)"])
    line = alt.Chart(d).mark_line(color=C2, point=True, strokeWidth=2).encode(
        x=alt.X("승차장:N", sort=None),
        y=alt.Y("보행 중앙(m):Q", title="보행 중앙 (m)"))
    return alt.layer(bar, line).resolve_scale(y="independent").properties(
        height=186)


def chart_co2(clock: float, ef: float):
    R = sorted(SIM["rides"], key=lambda r: r["drop"])
    rows, aR, aT = [], 0.0, 0.0
    for r in R:
        b = r["km_rev"] * sim.EF_TAXI / 2 / 1000
        aR += b - r["km_rev"] * ef / 1000
        aT += b - (r["km_rev"] + r["km_dead"]) * ef / 1000
        if r["drop"] <= clock:
            rows += [{"t": r["drop"] / 60, "kg": aR, "구분": "수송거리만"},
                     {"t": r["drop"] / 60, "kg": aT, "구분": "공차 포함"}]
    if not rows:
        rows = [{"t": sim.T0 / 60, "kg": 0.0, "구분": "수송거리만"},
                {"t": sim.T0 / 60, "kg": 0.0, "구분": "공차 포함"}]
    d = pd.DataFrame(rows)
    line = alt.Chart(d).mark_line(strokeWidth=2.4).encode(
        x=alt.X("t:Q", title="시각",
                scale=alt.Scale(domain=[sim.T0 / 60, END / 60])),
        y=alt.Y("kg:Q", title="현행 대비 누적 (kg CO₂)"),
        color=alt.Color("구분:N", scale=alt.Scale(
            domain=["수송거리만", "공차 포함"], range=[C1, C2]),
            legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("구분:N", scale=alt.Scale(
            domain=["수송거리만", "공차 포함"], range=[[1, 0], [5, 3]]),
            legend=None))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color="#888", strokeWidth=1.2).encode(y="y:Q")
    return (zero + line).properties(height=186)


def chart_hourly(clock: float):
    h0, h1 = 8, int(np.ceil(END / 60))
    d = pd.DataFrame([{
        "시": f"{h}시",
        "명": sum(r["n"] for r in SIM["rides"]
                 if h * 60 <= r["drop"] < (h + 1) * 60),
        "구분": "운행시간 후" if h >= 17 else "운행시간",
        "지남": clock >= (h + 1) * 60} for h in range(h0, h1)])
    return alt.Chart(d).mark_bar(cornerRadius=2).encode(
        x=alt.X("시:N", sort=None, title=None),
        y=alt.Y("명:Q", title="수송 인원"),
        color=alt.Color("구분:N", scale=alt.Scale(
            domain=["운행시간", "운행시간 후"], range=[C1, C2]),
            legend=alt.Legend(orient="top", title=None)),
        opacity=alt.Opacity("지남:N", scale=alt.Scale(
            domain=[True, False], range=[1.0, 0.25]), legend=None),
        tooltip=["시", "명"]).properties(height=186)


board()

# ── 실측 대조와 한계 ──────────────────────────────────────────────────
st.divider()
rt = SIM["km_rev"] / SIM["trips"]
chk = [
    ("일 통행수", f"{SIM['trips']}회", "78 (고령 946.5명 × 연 30회 ÷ 365)",
     73 <= SIM["trips"] <= 83),
    ("평균 재차인원", f"{SIM['occ']:.2f}명", "광교 실측 2.43–3.24",
     SIM["occ"] >= 1.5),
    ("평균 대기", f"{SIM['wait']:.0f}분", "수원 실측 14.5 / 고양 체감 30–40",
     10 <= SIM["wait"] <= 45),
    ("왕복 수송거리", f"{rt:.2f} km", "직선 6.90 × 실측 우회 1.38 = 9.52",
     7.0 <= rt <= 12.0),
    ("공차 비율", f"{100 * SIM['dead_share']:.0f}%", "DRT 통상 30–50",
     20 <= 100 * SIM["dead_share"] <= 55),
]
l, r = st.columns(2)
with l:
    st.markdown("**실측 대조** — 시뮬레이션이 범위 안에서 도는가")
    st.dataframe(pd.DataFrame(
        [[a, b, c, "○" if d else "△"] for a, b, c, d in chk],
        columns=["지표", "시뮬레이션", "기준", "판정"]),
        hide_index=True, width="stretch")
    st.markdown("**대상 지역 진단** — 실측·계산")
    s = META["split"]["유형"]
    st.dataframe(pd.DataFrame([
        ["고령인구", f"{M['고령']:,.1f}명", "44격자 · 출발점 124"],
        ["30분 내 의료기관", f"{M['A']:.2f}곳",
         f"시 평균 {M['A_시평균']:.1f}곳의 {100 * M['A'] / M['A_시평균']:.2f}%"],
        ["완전단절 격자", f"{M['완전단절_격자']}개",
         f"80분에도 0곳 · 고령 {M['완전단절_고령']:,.1f}명"],
        ["45분에도 미도달", f"{s['미도달']:,.1f}명",
         f"{100 * s['미도달'] / M['고령']:.1f}% · 길이 없다"],
        ["현행 똑버스까지", f"{M['똑버스_최단m'] / 1000:.1f} km",
         f"최단 직선 · 중앙 {M['똑버스_중앙m'] / 1000:.1f} km"]],
        columns=["항목", "값", "설명"]), hide_index=True, width="stretch")
with r:
    st.markdown("**차량 선택** — 손익분기 재차인원 (택시 2인 합승 대체)")
    rows = []
    for vv in sim.VEHICLES:
        lo = vv["ef"] / (sim.EF_TAXI / 2)
        hi = (vv["ef_hi"] or vv["ef"]) / (sim.EF_TAXI / 2)
        dh = 1 - SIM["dead_share"]
        rows.append({
            "차량": vv["name"],
            "수송거리만": f"{lo:.2f}" + (f"–{hi:.2f}" if hi > lo else ""),
            "공차 포함": f"{lo / dh:.2f}" + (f"–{hi / dh:.2f}"
                                       if hi > lo else ""),
            "판정": "▲ 감축" if hi / dh <= SIM["occ"] else "▼ 증가"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        f"이 설정의 실제 재차인원은 **{SIM['occ']:.2f}명**. 고령 의료통행만으로는"
        " 합승이 더 생기지 않는다 — 광교의 2.43~3.24는 전 연령·전 목적 수요에서"
        " 나온 값이다. 손익분기 재차인원은 배출계수의 비라 거리 가정에 흔들리지"
        " 않는다.")
    if SIM["stops"]:
        st.info(
            f"**지정 승차장 {K}곳** — 같은 자리에서 타는 호출이 늘어 합승이"
            f" 좋아지고 공차가 준다. 대신 고령자가 중앙 "
            f"{SIM['walk_med']:.0f} m(0.8 m/s 로 {SIM['walk_min_med']:.1f}분),"
            f" 멀면 {SIM['walk_max']:.0f} m 를 걸어야 한다. 승차장을 줄이면"
            f" 탄소는 좋아지고 보행은 나빠진다.", icon="🚏")

st.caption(
    "**실시간 똑버스 운행 데이터는 공개되지 않는다.** 차량 움직임·탑승객·탄소"
    " 누적은 광교 똑버스 실측 지표(호출 329,501건, 수원시정연구원 2024)로 만든"
    " 가상 운행이며 고양시의 실제 운행이 아니다. 가좌·송포에는 아직 똑버스가"
    " 없다. · **의원 실시간 현황도 가상이다** — 심평원이 주는 것은 기관별 인력"
    " 현황이고, 실시간 진료 인원·대기 환자를 주는 공개 API 는 없다. 표의"
    " 총의사수만 실제 자료다. · **지정 승차장도 임시 가정이다** — 고령인구를"
    " 가중치로 둔 군집 중심을 도로에 붙인 것이며, 실제 선정에는 보행 접근권과"
    " 정차 가능 지점 분석이 따로 필요하다. · **가정** — 고령 의료통행만 센다"
    "(보수적 하한). 한 통행은 집→의원·의원→집 두 구간. 주행은 국가표준노드링크"
    " 실제 도로 최단경로, 30 km/h. 배출계수는 환경부 2021년 승인 국가계수, 전력"
    " 0.4747 kgCO₂/kWh(소비단). 운송원가 560,428원/대·일."
    f" · 자료 생성 {M['생성']} · 고양시 고령자 의료접근성 연구 v6")
