# -*- coding: utf-8 -*-
"""가좌·송포 똑버스 운행 대시보드.

시계를 **실제 현재 시각**에 묶는다. 지금이 14:23이면 화면도 14:23 상태를 보이고
10초마다 스스로 갱신한다. 다만 가좌·송포에는 아직 똑버스가 없으므로, 이것은
"지금 이 시각 똑버스가 돌고 있다면" 을 보여주는 **가상 운행**이다.

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

st.set_page_config(page_title="가좌·송포 똑버스 운행 대시보드",
                   page_icon="🚐", layout="wide")


# ── 자료 ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="도로망을 읽는 중")
def get_road() -> sim.Road:
    return sim.Road()


@st.cache_data(show_spinner=False)
def get_meta() -> dict:
    return sim.load_meta()


@st.cache_data(show_spinner="운행을 계산하는 중", max_entries=40)
def run(vi: int, fleet: int) -> dict:
    """차량·대수별 하루 운행. 시드가 고정이라 같은 설정이면 같은 결과다."""
    s = sim.simulate(get_road(), get_meta(), sim.VEHICLES[vi], fleet)
    out = {k: v for k, v in s.items() if k != "rides"}
    out["rides"] = [{
        "v": r.v, "n": r.n, "call": r.call, "depart": r.depart,
        "pick": r.pick, "drop": r.drop, "km_rev": r.km_rev,
        "km_dead": r.km_dead, "back": r.back,
        "path": ll(r.path), "len": _cum(r.path),
    } for r in s["rides"]]
    return out


def ll(xy: np.ndarray) -> list[list[float]]:
    if len(xy) == 0:
        return []
    lon, lat = TO_WGS.transform(xy[:, 0], xy[:, 1])
    return [[round(a, 6), round(b, 6)] for a, b in zip(lon, lat)]


def _cum(xy: np.ndarray) -> list[float]:
    """경로 위 누적 길이 비율 — 차량 위치를 도로를 따라 찍는다."""
    if len(xy) < 2:
        return [0.0]
    d = np.r_[0, np.cumsum(np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1])))]
    return (d / d[-1]).tolist() if d[-1] > 0 else [0.0] * len(xy)


def at_frac(path: list, cum: list[float], f: float) -> list[float]:
    if not path:
        return [0.0, 0.0]
    i = int(np.searchsorted(cum, f))
    i = max(1, min(i, len(path) - 1))
    a, b = path[i - 1], path[i]
    span = cum[i] - cum[i - 1]
    t = 0.0 if span <= 0 else (f - cum[i - 1]) / span
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


META = get_meta()
M = META["meta"]
C1, C2, C3, C4 = "#16707f", "#b8540f", "#6b4a9e", "#8c9499"


# ── 조작 ──────────────────────────────────────────────────────────────
st.markdown(
    "<div style='display:flex;gap:12px;align-items:baseline;flex-wrap:wrap'>"
    "<h2 style='margin:0;font-size:1.35rem'>가좌·송포 똑버스 운행 대시보드</h2>"
    "<span style='font-family:monospace;font-size:.68rem;letter-spacing:.08em;"
    "background:#fdf3e4;color:#7a4a09;border:1px dashed #dfb87e;"
    "border-radius:3px;padding:3px 8px;font-weight:600'>"
    "시뮬레이션 · 실제 운행 데이터 아님</span></div>", unsafe_allow_html=True)

c = st.columns([2.2, 1.5, 1.6, 1.2])
vi = c[0].selectbox("차량", range(len(sim.VEHICLES)),
                    format_func=lambda i: sim.VEHICLES[i]["name"], index=1)
fleet = c[1].slider("도입 대수", 1, 12, 7)
mode = c[2].radio("시각", ["지금", "직접 고르기"], horizontal=True,
                  label_visibility="visible")
SIM = run(vi, fleet)
END = max(sim.T1, SIM["end"])
manual = c[3].slider("운행 시각", sim.T0, int(END), int(sim.T0 + 300),
                     disabled=(mode == "지금"),
                     format="%d") if mode != "지금" else None


@st.fragment(run_every=10 if mode == "지금" else None)
def board() -> None:
    now = datetime.now(KST)
    if mode == "지금":
        raw = now.hour * 60 + now.minute + now.second / 60
        clock = min(max(raw, sim.T0), END)
        if raw < sim.T0:
            phase = f"운행 시작 전 — 오늘 {sim.T0 // 60}:{sim.T0 % 60:02d} 개시"
        elif raw > END:
            phase = "오늘 운행 종료"
        else:
            phase = "운행 중"
    else:
        clock = float(manual)
        phase = "직접 고른 시각"
    draw(clock, phase, now)


def draw(clock: float, phase: str, now: datetime) -> None:
    R = SIM["rides"]
    pax = kmR = kmD = 0.0
    moving = onboard = 0
    per = [{"pax": 0, "km": 0.0, "st": "대기"} for _ in range(fleet)]
    veh_pts, wait_pts, done_pts, live_paths = [], [], [], []
    waits = []
    for r in R:
        P = per[r["v"]]
        if clock >= r["drop"]:
            pax += r["n"]; kmR += r["km_rev"]; kmD += r["km_dead"]
            P["pax"] += r["n"]; P["km"] += r["km_rev"] + r["km_dead"]
            if r["path"]:
                done_pts.append({"p": r["path"][-1], "n": r["n"]})
        elif clock >= r["pick"]:
            f = (clock - r["pick"]) / max(.01, r["drop"] - r["pick"])
            onboard += r["n"]; moving += 1
            kmR += r["km_rev"] * f; kmD += r["km_dead"]
            P["km"] += r["km_rev"] * f + r["km_dead"]
            P["st"] = f"승객 {r['n']}명 수송"
            veh_pts.append({"p": at_frac(r["path"], r["len"], f),
                            "n": r["n"], "v": r["v"] + 1})
            live_paths.append({"path": r["path"], "n": r["n"]})
        elif clock >= r["depart"]:
            moving += 1
            kmD += r["km_dead"] * min(
                1, (clock - r["depart"]) / max(.01, r["pick"] - r["depart"]))
            P["st"] = "배차 이동 (공차)"
        if r["call"] <= clock < r["pick"]:
            waits.append(clock - r["call"])
            if r["path"]:
                wait_pts.append({"p": r["path"][0], "w": clock - r["call"]})

    ef = sim.VEHICLES[vi]["ef"]
    km = kmR + kmD
    co2 = km * ef / 1000
    base = kmR * sim.EF_TAXI / 2 / 1000
    cut = base - co2

    hh, mm = int(clock // 60), int(clock % 60)
    st.markdown(
        f"<div style='font-family:monospace;font-size:1.5rem;font-weight:600'>"
        f"<span style='color:{C1}'>●</span> {hh:02d}:{mm:02d}"
        f"<span style='font-size:.5em;font-weight:400;color:#777;"
        f"margin-left:10px'>{phase} · 실제 시각 "
        f"{now:%Y-%m-%d %H:%M:%S} KST</span></div>",
        unsafe_allow_html=True)

    k = st.columns(5)
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
    k[4].metric("현행 대비", f"{'▲ +' if cut >= 0 else '▼ −'}{abs(cut):,.1f} kg",
                f"연 환산 {'+' if cut >= 0 else '−'}"
                f"{abs(cut) * 365 / 1000:,.1f} t",
                delta_color="normal" if cut >= 0 else "inverse")

    left, right = st.columns([2.1, 1])
    with left:
        st.pydeck_chart(deck(live_paths, veh_pts, wait_pts, done_pts),
                        width="stretch")
        st.caption(
            "실선 = 지금 수송 중인 구간의 **실제 도로 경로**(국가표준노드링크"
            " 2025-05-12) · 원 = 대기 중 호출 · 사각 = 차량(숫자는 탑승) · "
            "옅은 점 = 고령인구 250m 격자 · 보라 점 = 의료기관")
    with right:
        st.markdown("**차량 현황**")
        st.dataframe(pd.DataFrame([{
            "차량": f"{sim.VEHICLES[vi]['id']}-{i + 1:02d}", "상태": p["st"],
            "탑승": p["pax"], "주행 km": round(p["km"])}
            for i, p in enumerate(per)]), hide_index=True,
            width="stretch", height=min(390, 36 * fleet + 40))
        if waits:
            w = np.array(waits)
            st.markdown(
                f"**대기 중 호출 {len(w)}건** — 최장 {w.max():.0f}분 · "
                f"40분 초과 {int((w > 40).sum())}건")
        else:
            st.markdown("**대기 중 호출 없음**")
        st.caption(
            f"하루 평균 대기 **{SIM['wait']:.0f}분** · 평균 재차 "
            f"**{SIM['occ']:.2f}명** · 마지막 하차 "
            f"{int(SIM['end'] // 60):02d}:{int(SIM['end'] % 60):02d}")

    a, b, cc = st.columns(3)
    with a:
        st.markdown("**시간대별 탑승**")
        st.altair_chart(chart_hourly(clock), width="stretch")
    with b:
        st.markdown("**탄소 수지** — 현행 택시 2인 합승 대비")
        st.altair_chart(chart_co2(clock, ef), width="stretch")
    with cc:
        st.markdown("**대상 지역 진단** — 실측·계산")
        st.dataframe(diag(), hide_index=True, width="stretch")


# ── 지도 ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def static_layers() -> dict:
    cells = [c for c in META["cells"] if c[3] > 0]
    lon, lat = TO_WGS.transform([c[0] + 125 for c in cells],
                                [c[1] + 125 for c in cells])
    pop = pd.DataFrame({"lon": lon, "lat": lat,
                        "E": [c[3] for c in cells],
                        "cl": [c[4] for c in cells]})
    dl, dt = TO_WGS.transform([d[0] for d in META["dest"]],
                              [d[1] for d in META["dest"]])
    dest = pd.DataFrame({"lon": dl, "lat": dt,
                         "big": [1 if d[2] >= 10 else 0 for d in META["dest"]]})
    dong = []
    for d in META["dong"]:
        for ring in d["rings"]:
            x = [p[0] for p in ring]; y = [p[1] for p in ring]
            a, bb = TO_WGS.transform(x, y)
            dong.append({"polygon": [[round(p, 6), round(q, 6)]
                                     for p, q in zip(a, bb)]})
    gz = pop[pop.cl == 1]
    return {"pop": pop, "dest": dest, "dong": dong,
            "view": {"lon": float(gz.lon.mean()), "lat": float(gz.lat.mean())}}


def deck(paths, vehs, waits, dones) -> pdk.Deck:
    S = static_layers()
    L = [
        pdk.Layer("PolygonLayer", S["dong"], get_polygon="polygon",
                  stroked=True, filled=False, get_line_color=[184, 84, 15, 150],
                  line_width_min_pixels=2),
        pdk.Layer("ScatterplotLayer", S["pop"],
                  get_position=["lon", "lat"], get_radius="30 + E * 6",
                  radius_min_pixels=2, radius_max_pixels=26,
                  get_fill_color="cl == 1 ? [22, 112, 127, 70]"
                                 " : [140, 148, 153, 35]",
                  stroked=False, pickable=False),
        pdk.Layer("ScatterplotLayer", S["dest"], get_position=["lon", "lat"],
                  get_radius="big == 1 ? 180 : 70", radius_min_pixels=2,
                  radius_max_pixels=9,
                  get_fill_color="big == 1 ? [107, 74, 158, 210]"
                                 " : [107, 74, 158, 90]"),
    ]
    if dones:
        L.append(pdk.Layer("ScatterplotLayer",
                           [{"lon": d["p"][0], "lat": d["p"][1]} for d in dones],
                           get_position=["lon", "lat"], get_radius=90,
                           radius_min_pixels=2, radius_max_pixels=7,
                           get_fill_color=[184, 84, 15, 90]))
    if paths:
        L.append(pdk.Layer("PathLayer", paths, get_path="path",
                           get_color=[184, 84, 15, 190], width_min_pixels=3,
                           get_width=20, cap_rounded=True, joint_rounded=True))
    if waits:
        L.append(pdk.Layer("ScatterplotLayer",
                           [{"lon": w["p"][0], "lat": w["p"][1],
                             "w": w["w"]} for w in waits],
                           get_position=["lon", "lat"], get_radius=260,
                           radius_min_pixels=5, radius_max_pixels=16,
                           stroked=True, filled=False,
                           get_line_color=[22, 112, 127, 230],
                           line_width_min_pixels=2))
    if vehs:
        L.append(pdk.Layer("ScatterplotLayer",
                           [{"lon": v["p"][0], "lat": v["p"][1],
                             "n": v["n"], "v": v["v"]} for v in vehs],
                           get_position=["lon", "lat"], get_radius=300,
                           radius_min_pixels=7, radius_max_pixels=18,
                           get_fill_color=[107, 74, 158, 235], pickable=True))
    return pdk.Deck(
        layers=L, map_provider="carto", map_style="light",
        initial_view_state=pdk.ViewState(
            latitude=S["view"]["lat"], longitude=S["view"]["lon"],
            zoom=11.2, pitch=0),
        tooltip={"text": "{v}호차 · 탑승 {n}명"}, height=470)


# ── 그래프 ────────────────────────────────────────────────────────────
def chart_hourly(clock: float):
    R = SIM["rides"]
    h0, h1 = 8, int(np.ceil(max(sim.T1, SIM["end"]) / 60))
    rows = []
    for h in range(h0, h1):
        n = sum(r["n"] for r in R if h * 60 <= r["drop"] < (h + 1) * 60)
        rows.append({"시": f"{h}시", "명": n,
                     "구분": "운행시간 후" if h >= 17 else "운행시간",
                     "지남": clock >= (h + 1) * 60})
    d = pd.DataFrame(rows)
    return (alt.Chart(d).mark_bar(cornerRadius=2).encode(
        x=alt.X("시:N", sort=None, title=None),
        y=alt.Y("명:Q", title="수송 인원"),
        color=alt.Color("구분:N", scale=alt.Scale(
            domain=["운행시간", "운행시간 후"], range=[C1, C2]),
            legend=alt.Legend(orient="top", title=None)),
        opacity=alt.Opacity("지남:N", scale=alt.Scale(
            domain=[True, False], range=[1.0, 0.25]), legend=None),
        tooltip=["시", "명"]).properties(height=190))


def chart_co2(clock: float, ef: float):
    R = sorted(SIM["rides"], key=lambda r: r["drop"])
    rows, aR, aT = [], 0.0, 0.0
    for r in R:
        b = r["km_rev"] * sim.EF_TAXI / 2 / 1000
        aR += b - r["km_rev"] * ef / 1000
        aT += b - (r["km_rev"] + r["km_dead"]) * ef / 1000
        if r["drop"] <= clock:
            rows.append({"t": r["drop"] / 60, "kg": aR, "구분": "수송거리만"})
            rows.append({"t": r["drop"] / 60, "kg": aT, "구분": "공차 포함"})
    if not rows:
        rows = [{"t": sim.T0 / 60, "kg": 0, "구분": "수송거리만"},
                {"t": sim.T0 / 60, "kg": 0, "구분": "공차 포함"}]
    d = pd.DataFrame(rows)
    line = alt.Chart(d).mark_line(strokeWidth=2.4).encode(
        x=alt.X("t:Q", title="시각", scale=alt.Scale(
            domain=[sim.T0 / 60, max(sim.T1, SIM["end"]) / 60])),
        y=alt.Y("kg:Q", title="현행 대비 누적 (kg CO₂)"),
        color=alt.Color("구분:N", scale=alt.Scale(
            domain=["수송거리만", "공차 포함"], range=[C1, C2]),
            legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("구분:N", scale=alt.Scale(
            domain=["수송거리만", "공차 포함"], range=[[1, 0], [5, 3]]),
            legend=None))
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
        color="#888", strokeWidth=1.2).encode(y="y:Q")
    return (zero + line).properties(height=190)


def diag() -> pd.DataFrame:
    s = META["split"]["유형"]
    return pd.DataFrame([
        ["고령인구", f"{M['고령']:,.1f}명", "44격자 · 출발점 124"],
        ["30분 내 의료기관", f"{M['A']:.2f}곳",
         f"시 평균 {M['A_시평균']:.1f}곳의 {100 * M['A'] / M['A_시평균']:.2f}%"],
        ["완전단절 격자", f"{M['완전단절_격자']}개",
         f"80분에도 0곳 · 고령 {M['완전단절_고령']:,.1f}명"],
        ["45분에도 미도달", f"{s['미도달']:,.1f}명",
         f"{100 * s['미도달'] / M['고령']:.1f}% · 길이 없다"],
        ["현행 똑버스까지", f"{M['똑버스_최단m'] / 1000:.1f} km",
         f"최단 직선 · 중앙 {M['똑버스_중앙m'] / 1000:.1f} km"],
    ], columns=["항목", "값", "설명"])


board()

# ── 실측 대조와 한계 ──────────────────────────────────────────────────
st.divider()
v = sim.VEHICLES[vi]
cb = sim.carbon({**SIM, "veh": v})
rt = SIM["km_rev"] / SIM["trips"]
chk = [
    ("일 통행수", f"{SIM['trips']}회", "78 (고령 946.5명 × 연 30회 ÷ 365)",
     73 <= SIM["trips"] <= 83),
    ("평균 재차인원", f"{SIM['occ']:.2f}명", "광교 실측 2.43–3.24",
     SIM["occ"] >= 1.5),
    ("평균 대기", f"{SIM['wait']:.0f}분", "수원 실측 14.5 / 고양 체감 30–40",
     10 <= SIM["wait"] <= 45),
    ("왕복 수송거리", f"{rt:.2f} km", "직선 6.90 × 실측 우회 1.38 = 9.52",
     7.5 <= rt <= 12.0),
    ("공차 비율", f"{100 * SIM['dead_share']:.0f}%", "DRT 통상 30–50",
     20 <= 100 * SIM["dead_share"] <= 55),
]
l, r = st.columns([1, 1])
with l:
    st.markdown("**실측 대조** — 시뮬레이션이 범위 안에서 도는가")
    st.dataframe(pd.DataFrame(
        [[a, b, c, "○" if d else "△"] for a, b, c, d in chk],
        columns=["지표", "시뮬레이션", "기준", "판정"]),
        hide_index=True, width="stretch")
with r:
    st.markdown("**차량 선택** — 손익분기 재차인원 (택시 2인 합승 대체)")
    rows = []
    for k, vv in enumerate(sim.VEHICLES):
        lo = vv["ef"] / (sim.EF_TAXI / 2)
        hi = (vv["ef_hi"] or vv["ef"]) / (sim.EF_TAXI / 2)
        rows.append({
            "차량": vv["name"],
            "수송거리만": f"{lo:.2f}" + (f"–{hi:.2f}" if hi > lo else ""),
            "공차 포함": f"{lo / (1 - SIM['dead_share']):.2f}"
                      + (f"–{hi / (1 - SIM['dead_share']):.2f}"
                         if hi > lo else ""),
            "판정": "▲ 감축" if hi / (1 - SIM["dead_share"]) <= SIM["occ"]
                  else "▼ 증가"})
    st.dataframe(pd.DataFrame(rows), hide_index=True,
                 width="stretch")
    st.caption(
        f"이 설정의 실제 재차인원은 **{SIM['occ']:.2f}명**. 고령 의료통행만으로는"
        " 합승이 더 생기지 않는다 — 광교의 2.43~3.24는 전 연령·전 목적 수요에서"
        " 나온 값이다.")

st.caption(
    "**실시간 똑버스 운행 데이터는 공개되지 않는다.** 차량 움직임·탑승객·탄소"
    " 누적은 광교 똑버스 실측 지표(호출 329,501건, 수원시정연구원 2024)로 만든"
    " 가상 운행이며 고양시의 실제 운행이 아니다. 가좌·송포에는 아직 똑버스가"
    " 없다. 시드를 고정해 같은 설정이면 언제나 같은 결과가 나온다. · "
    "**가정** — 고령 의료통행만 센다(보수적 하한). 한 통행은 집→의원·의원→집 두"
    " 구간. 주행은 국가표준노드링크 실제 도로 최단경로, 30 km/h. 배출계수는"
    " 환경부 2021년 승인 국가계수, 전력 0.4747 kgCO₂/kWh(소비단). 운송원가"
    " 560,428원/대·일. 광교의 90.3 통행/대·일은 신도시 내부 짧은 통행 값이라,"
    " 문 앞에서 의원까지 직결이라 통행거리가 2~3배인 가좌·송포에 그대로"
    f" 옮겨지지 않는다. · 자료 생성 {M['생성']} · 고양시 고령자 의료접근성"
    " 연구 v6")
