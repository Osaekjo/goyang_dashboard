# -*- coding: utf-8 -*-
"""가좌·송포 똑버스 도입 검토판 — 정책 판단용.

운영 모니터가 아니다. **무엇을 정하면 무엇이 달라지는가**를 보여준다.

  정하는 것   차종 · 도입 대수 · 지정 승차장 수
  달라지는 것 탄소배출 · 승차장까지 보행거리 · 대기시간 · 예산

조합 3 × 12 × 13 = 468가지를 미리 돌려 data/scenarios.json 에 담아 두었다
(build_scenarios.py). 화면은 그 표만 읽으므로 조작이 즉시 반영된다.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

import sim

HERE = Path(__file__).resolve().parent
C1, C2, C3, C4 = "#16707f", "#b8540f", "#6b4a9e", "#8c9499"

st.set_page_config(page_title="가좌·송포 똑버스 도입 검토판",
                   page_icon="🚐", layout="wide")


@st.cache_data(show_spinner=False)
def load() -> dict:
    return json.loads((HERE / "data" / "scenarios.json").read_text("utf-8"))


@st.cache_data(show_spinner=False)
def meta() -> dict:
    return sim.load_meta()


@st.cache_resource(show_spinner="도로망을 읽는 중")
def road() -> sim.Road:
    return sim.Road()


@st.cache_data(show_spinner="경로를 그리는 중", max_entries=16)
def routes(k: int) -> list[dict]:
    """승차장에서 가까운 의원 3곳까지의 경로 — 지도에만 쓴다."""
    from pyproj import Transformer
    T = Transformer.from_crs(5179, 4326, always_xy=True)
    R = sim.stop_routes(road(), sim.designated_stops(road(), meta(), k),
                        meta(), 3)
    out = []
    for r in R:
        xy = np.asarray(r.pop("path"), float)
        if len(xy) > 2:
            xy = _thin(xy)
        lon, lat = T.transform(xy[:, 0], xy[:, 1])
        r["path"] = [[round(a, 6), round(b, 6)] for a, b in zip(lon, lat)]
        out.append(r)
    return out


def _thin(xy: np.ndarray, tol: float = 12.0) -> np.ndarray:
    """경로 꼭짓점 솎기 — 12 m 면 화면에서 차이가 나지 않는다."""
    n = len(xy)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = xy[i], xy[j]
        ab = b - a
        L = float(np.hypot(*ab))
        seg = xy[i + 1:j]
        d = (np.hypot(seg[:, 0] - a[0], seg[:, 1] - a[1]) if L < 1e-9 else
             np.abs(ab[0] * (a[1] - seg[:, 1])
                    - ab[1] * (a[0] - seg[:, 0])) / L)
        m = int(np.argmax(d))
        if d[m] > tol:
            keep[i + 1 + m] = True
            stack += [(i, i + 1 + m), (i + 1 + m, j)]
    return xy[keep]


D = load()
M = meta()["meta"]
ROWS = pd.DataFrame(D["rows"])
VEH = D["vehicles"]

# ── 현행(기준선) ──────────────────────────────────────────────────────
# 가좌·송포에는 갈 수 있는 대중교통이 없다. 지금 이 통행은 택시나 자가용으로
# 이뤄지고 있고, 그 배출이 비교 기준이다. 무엇으로 보느냐에 따라 결론이 갈려
# 고르게 해 둔다.
BASELINE = {
    "택시 단독": ("승용 중형 LPG · 1인 탑승", sim.EF_TAXI, 1.0),
    "택시 2인 합승": ("승용 중형 LPG · 2인 분담", sim.EF_TAXI, 2.0),
    "자가용 단독": ("승용 소형 휘발유 · 1인 탑승", sim.EF_CAR, 1.0),
    "자가용 2인": ("승용 소형 휘발유 · 보호자 동반", sim.EF_CAR, 2.0),
}

st.markdown(
    "<div style='display:flex;gap:12px;align-items:baseline;flex-wrap:wrap'>"
    "<h2 style='margin:0;font-size:1.4rem'>가좌·송포 똑버스 도입 검토판</h2>"
    "<span style='color:#6b7a81;font-size:.88rem'>차종 · 대수 · 승차장 수를"
    " 정하면 탄소와 보행거리가 어떻게 달라지는가</span>"
    "<span style='font-family:monospace;font-size:.66rem;letter-spacing:.07em;"
    "background:#fdf3e4;color:#7a4a09;border:1px dashed #dfb87e;"
    "border-radius:3px;padding:3px 8px;font-weight:600'>수요·운행은 추정치"
    "</span></div>", unsafe_allow_html=True)

c = st.columns([2.2, 1.5, 1.5, 1.9])
vi = c[0].selectbox("차종", range(len(VEH)),
                    format_func=lambda i: VEH[i]["name"], index=1)
fleet = c[1].slider("도입 대수", 1, 12, 7)
K = c[2].slider("지정 승차장 수", 4, 16, 10)
bl = c[3].selectbox("현행(비교 기준)", list(BASELINE), index=0,
                    help="가좌·송포에는 갈 수 있는 대중교통이 없다. "
                         "지금 이 통행이 무엇으로 이뤄지고 있다고 보느냐다.")
BL_NOTE, BL_EF, BL_OCC = BASELINE[bl]


def cut_year(df: pd.DataFrame) -> pd.Series:
    """고른 기준선으로 다시 잰 연간 탄소 증감 (톤). 양수면 감축."""
    base = df["km_rev"] * BL_EF / BL_OCC / 1000
    return (base - df["co2"]) * 365 / 1000


ROWS = ROWS.assign(cut_y=cut_year(ROWS))
cur = ROWS[(ROWS.vi == vi) & (ROWS.fleet == fleet) & (ROWS.k == K)].iloc[0]

# ── 핵심 수치 ─────────────────────────────────────────────────────────
k = st.columns(5)
flat = abs(cur.cut_y) < 0.05          # 반올림하면 0 인 구간은 '비긴다'로 적는다
k[0].metric("연간 탄소",
            "± 0.0 t" if flat else
            f"{'▲ +' if cur.cut_y >= 0 else '▼ −'}{abs(cur.cut_y):,.1f} t",
            ("현행과 비긴다" if flat else
             f"{'감축' if cur.cut_y >= 0 else '증가'}") + f" · 현행 {bl} 대비",
            delta_color="off" if flat
            else "normal" if cur.cut_y >= 0 else "inverse")
k[1].metric("승차장까지 보행", f"{cur.walk_med:,.0f} m",
            f"0.8 m/s 로 {cur.walk_min:.1f}분 · 10명 중 1명은 "
            f"{cur.walk_p90:,.0f} m 이상", delta_color="off")
k[2].metric("평균 대기", f"{cur.wait:,.0f}분",
            f"10번 중 1번은 {cur.wait_p90:,.0f}분 이상"
            + ("" if cur.in_hours else " · 운행시간 초과"),
            delta_color="off")
k[3].metric("연간 운영비", f"{cur.cost_year / 1e8:,.1f}억원",
            f"고령 1인당 {cur.cost_year / M['고령']:,.0f}원/년",
            delta_color="off")
k[4].metric("평균 재차인원", f"{cur.occ:.2f}명",
            f"손익분기 {cur.breakeven_dead:.2f}명 (공차 포함)",
            delta_color="off")

if not cur.in_hours:
    st.warning(
        f"**{fleet}대로는 하루 78통행을 운행시간(08:30–17:00) 안에 처리하지"
        f" 못한다.** 마지막 하차가 {int(cur.end // 60):02d}:"
        f"{int(cur.end % 60):02d}이고 평균 대기가 {cur.wait:,.0f}분이다.",
        icon="⚠️")

# ── 트레이드오프 ──────────────────────────────────────────────────────
st.markdown("### 무엇을 정하면 무엇이 달라지는가")
t1, t2, t3 = st.columns(3)


def dual(df, x, xt, left, lt, lc, right, rt, rc, mark):
    """왼쪽 축 하나, 오른쪽 축 하나. 고른 값에 세로선을 긋는다."""
    base = alt.Chart(df)
    a = base.mark_line(color=lc, point=True, strokeWidth=2.4).encode(
        x=alt.X(f"{x}:Q", title=xt, axis=alt.Axis(tickMinStep=1)),
        y=alt.Y(f"{left}:Q", title=lt, axis=alt.Axis(titleColor=lc)),
        tooltip=[x, left, right])
    b = base.mark_line(color=rc, point=True, strokeWidth=2.4,
                       strokeDash=[5, 3]).encode(
        x=alt.X(f"{x}:Q"),
        y=alt.Y(f"{right}:Q", title=rt, axis=alt.Axis(titleColor=rc)))
    rule = alt.Chart(pd.DataFrame({x: [mark]})).mark_rule(
        color="#555", strokeWidth=1.4, strokeDash=[2, 2]).encode(x=f"{x}:Q")
    return alt.layer(a, b, rule).resolve_scale(y="independent").properties(
        height=236)


with t1:
    st.markdown(f"**승차장 수** — 보행과 탄소는 반대로 움직인다"
                f" &nbsp;<span style='color:#8699a1;font-size:.8rem'>"
                f"{VEH[vi]['name'].split()[0]} · {fleet}대</span>",
                unsafe_allow_html=True)
    d = ROWS[(ROWS.vi == vi) & (ROWS.fleet == fleet)].sort_values("k")
    d = d.rename(columns={"walk_med": "보행 중앙(m)", "cut_y": "연간 탄소(t)"})
    st.altair_chart(dual(d, "k", "지정 승차장 수",
                         "보행 중앙(m)", "보행 중앙 (m)", C1,
                         "연간 탄소(t)", "연간 탄소 (t)", C2, K),
                    width="stretch")
    lo, hi = d["보행 중앙(m)"].max(), d["보행 중앙(m)"].min()
    st.caption(
        f"승차장을 {d.k.min()}곳에서 {d.k.max()}곳으로 늘리면 보행 중앙이"
        f" {lo:,.0f} → {hi:,.0f} m 로 줄지만, 차가 더 많이 돌아 탄소는"
        f" {d['연간 탄소(t)'].iloc[0]:+.1f} → {d['연간 탄소(t)'].iloc[-1]:+.1f} t"
        f" 로 나빠진다.")

with t2:
    st.markdown(f"**도입 대수** — 대기와 예산은 반대로 움직인다"
                f" &nbsp;<span style='color:#8699a1;font-size:.8rem'>"
                f"{VEH[vi]['name'].split()[0]} · 승차장 {K}곳</span>",
                unsafe_allow_html=True)
    d = ROWS[(ROWS.vi == vi) & (ROWS.k == K)].sort_values("fleet").copy()
    d["연간 비용(억)"] = d.cost_year / 1e8
    d = d.rename(columns={"wait": "평균 대기(분)"})
    st.altair_chart(dual(d, "fleet", "도입 대수",
                         "평균 대기(분)", "평균 대기 (분)", C1,
                         "연간 비용(억)", "연간 비용 (억원)", C3, fleet),
                    width="stretch")
    need = d[d.in_hours].fleet.min() if d.in_hours.any() else None
    st.caption(
        f"운행시간 안에 하루 78통행을 처리하려면 **최소 {need}대**가 필요하다."
        if need else "어떤 대수로도 운행시간 안에 다 처리하지 못한다.")

with t3:
    st.markdown("**차종** — 같은 조건에서 배출만 바꿔 본다"
                f" &nbsp;<span style='color:#8699a1;font-size:.8rem'>"
                f"{fleet}대 · 승차장 {K}곳</span>", unsafe_allow_html=True)
    d = ROWS[(ROWS.fleet == fleet) & (ROWS.k == K)].copy()
    d["차종"] = [VEH[i]["name"] for i in d.vi]
    bar = alt.Chart(d).mark_bar(cornerRadius=3).encode(
        y=alt.Y("차종:N", sort=None, title=None,
                axis=alt.Axis(labelLimit=180)),
        x=alt.X("cut_y:Q", title="연간 탄소 (t) — 오른쪽이 좋다"),
        color=alt.condition(alt.datum.cut_y >= 0, alt.value(C1), alt.value(C2)),
        tooltip=["차종", "cut_y", "breakeven_dead", "occ"])
    lab = alt.Chart(d).mark_text(align="left", dx=5, fontSize=11,
                                 fontWeight=600).encode(
        y=alt.Y("차종:N", sort=None), x="cut_y:Q",
        text=alt.Text("cut_y:Q", format="+.1f"),
        color=alt.condition(alt.datum.cut_y >= 0, alt.value(C1), alt.value(C2)))
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color="#555", strokeWidth=1.4).encode(x="x:Q")
    st.altair_chart((bar + zero + lab).properties(height=236),
                    width="stretch")
    best = d.loc[d.cut_y.idxmax()]
    st.caption(
        f"**{best['차종']}** 가 가장 낫다. 손익분기 재차인원"
        f" {best.breakeven_dead:.2f}명(공차 포함)인데 실제 재차는"
        f" {best.occ:.2f}명이다.")

# ── 지도 ──────────────────────────────────────────────────────────────
st.markdown("### 지정 승차장과 의원까지의 경로")
mcol, scol = st.columns([2.05, 1])
S = D["stops"][str(K)]
RT = routes(K)


def deck_map() -> pdk.Deck:
    me = meta()
    dest = me["dest"]
    from pyproj import Transformer
    T = Transformer.from_crs(5179, 4326, always_xy=True)
    dl, dt = T.transform([d[0] for d in dest], [d[1] for d in dest])
    cl = pd.DataFrame({"lon": dl, "lat": dt,
                       "name": [d[5] for d in dest],
                       "kind": [d[6] for d in dest],
                       "doctors": [max(1, d[7]) for d in dest]})
    dong = []
    for d in me["dong"]:
        for ring in d["rings"]:
            a, b = T.transform([p[0] for p in ring], [p[1] for p in ring])
            dong.append({"polygon": [[round(p, 6), round(q, 6)]
                                     for p, q in zip(a, b)]})
    sdf = pd.DataFrame(S)
    sdf["tip"] = [f"고령 {r['pop']:.0f}명 · 보행 중앙 {r['walk_med']} m "
                  f"(최대 {r['walk_max']} m)" for r in S]
    sdf["lab"] = [r["name"].split()[-1] for r in S]
    used = sorted({r["dest"] for r in RT})
    lab = pd.DataFrame([{
        "lon": float(cl.lon.iloc[j]), "lat": float(cl.lat.iloc[j]),
        "lab": (cl["name"].iloc[j][:12]
                + f"  의사 {int(cl.doctors.iloc[j])}명")} for j in used])
    return pdk.Deck(layers=[
        pdk.Layer("PolygonLayer", dong, get_polygon="polygon", stroked=True,
                  filled=False, get_line_color=[184, 84, 15, 130],
                  line_width_min_pixels=2),
        pdk.Layer("PathLayer",
                  [{"path": r["path"], "w": 16 if r["rank"] == 0 else 7,
                    "c": [22, 112, 127, 170] if r["rank"] == 0
                         else [22, 112, 127, 60],
                    "name": f"{r['stop_name']} → {r['name']}",
                    "tip": f"{r['km']:.1f} km · {r['min']:.0f}분"} for r in RT],
                  get_path="path", get_color="c", get_width="w",
                  width_min_pixels=1, pickable=True, cap_rounded=True,
                  joint_rounded=True),
        pdk.Layer("ScatterplotLayer", cl, get_position=["lon", "lat"],
                  get_radius="60 + doctors * 22", radius_min_pixels=3,
                  radius_max_pixels=22, get_fill_color=[107, 74, 158, 170],
                  pickable=True),
        pdk.Layer("TextLayer", lab, get_position=["lon", "lat"],
                  get_text="lab", get_size=12, get_color=[40, 52, 60],
                  get_text_anchor="'start'", get_pixel_offset=[13, 0],
                  get_alignment_baseline="'center'", background=True,
                  get_background_color=[255, 255, 255, 226],
                  background_padding=[5, 3, 5, 3], font_weight=600,
                  font_family="'Malgun Gothic, Apple SD Gothic Neo, sans-serif'",
                  character_set="'auto'"),
        pdk.Layer("ScatterplotLayer", sdf, get_position=["lon", "lat"],
                  get_radius="130 + pop * 2.4", radius_min_pixels=8,
                  radius_max_pixels=30, get_fill_color=[22, 112, 127, 205],
                  stroked=True, get_line_color=[255, 255, 255, 230],
                  line_width_min_pixels=1.5, pickable=True),
        pdk.Layer("TextLayer", sdf, get_position=["lon", "lat"],
                  get_text="lab", get_size=11, get_color=[255, 255, 255],
                  get_alignment_baseline="'center'", font_weight=700,
                  character_set="'auto'"),
    ], map_provider="carto", map_style="light",
        initial_view_state=pdk.ViewState(
            longitude=float(sdf.lon.mean()), latitude=float(sdf.lat.mean()),
            zoom=11.4),
        tooltip={"html": "<b>{name}</b><br/>{tip}{kind} · 의사 {doctors}명"},
        height=470)


with mcol:
    st.pydeck_chart(deck_map(), width="stretch")
    st.caption(
        "청록 큰 점 = 지정 승차장(크기는 맡은 고령인구, 번호는 인구 순위) · "
        "청록 선 = 가까운 의원 3곳까지의 실제 도로 경로(진한 선이 최근접) · "
        "보라 점 = 의료기관(크기는 의사 수) · 주황 테두리 = 가좌동·송포동 경계"
        f" · 주행거리 출처 **{D['meta']['거리출처']}**"
        + (f" (이 조합의 {100 * cur.kakao:.0f}%, 나머지는 국가표준노드링크"
           f" 2025-05-12 최단경로 — 두 값의 비는 표본 50쌍에서 1.011이었다)"
           if cur.kakao > 0 else ""))

with scol:
    st.markdown(f"**승차장 {K}곳** — 어디가 얼마를 맡는가")
    sd = pd.DataFrame([{
        "승차장": s["name"].split()[-1], "고령인구": s["pop"],
        "격자": s["cells"], "보행 중앙": s["walk_med"],
        "보행 최대": s["walk_max"]} for s in S])
    st.dataframe(sd, hide_index=True, width="stretch", height=330)
    far = sd[sd["보행 최대"] > 1000]
    if len(far):
        st.warning(
            f"**{len(far)}곳은 가장 먼 사람이 1 km 넘게 걸어야 한다** "
            f"(최대 {sd['보행 최대'].max():,.0f} m · 0.8 m/s 로 "
            f"{sd['보행 최대'].max() / 0.8 / 60:.0f}분). 실제 선정에서는 그"
            f" 덩어리를 쪼개야 한다.", icon="🚶")

# ── 추천과 한계 ───────────────────────────────────────────────────────
st.markdown("### 조건을 걸고 추려 보기")
f1, f2, f3 = st.columns([1, 1, 2])
w_max = f1.slider("보행 중앙 상한 (m)", 200, 600, 350, 25)
t_max = f2.slider("평균 대기 상한 (분)", 10, 60, 30, 5)
cand = ROWS[(ROWS.walk_med <= w_max) & (ROWS.wait <= t_max) & ROWS.in_hours]
f3.caption(f"468개 조합 가운데 **{len(cand)}개**가 조건을 만족한다. "
           "탄소가 나은 순으로 위 10개를 보인다.")
if len(cand):
    top = cand.nlargest(10, "cut_y").copy()
    top["차종"] = [VEH[i]["name"] for i in top.vi]
    st.dataframe(pd.DataFrame({
        "차종": top["차종"], "대수": top.fleet, "승차장": top.k,
        "연간 탄소 (t)": top.cut_y.round(1),
        "보행 중앙 (m)": top.walk_med,
        "평균 대기 (분)": top.wait.round(0),
        "연간 비용 (억)": (top.cost_year / 1e8).round(1),
        "재차 (명)": top.occ.round(2),
    }), hide_index=True, width="stretch")
else:
    st.info("조건을 만족하는 조합이 없다. 상한을 늘려 보라.", icon="🔎")

with st.expander("이 숫자들은 어디서 왔나 — 방법과 한계", expanded=False):
    s = meta()["split"]["유형"]
    a, b = st.columns(2)
    a.markdown(f"""
**대상** — 고양시 저접근 하위 25% 격자의 연결성분 가운데 1순위 군집
(대표동 가좌동). 44격자 · 출발점 124 · **고령 {M['고령']:,.1f}명**.
30분 안에 닿는 의료기관은 평균 **{M['A']:.2f}곳**으로 시 평균
{M['A_시평균']:.1f}곳의 {100 * M['A'] / M['A_시평균']:.2f}%다.
**{M['완전단절_격자']}격자**는 80분을 줘도 0곳이고, 45분을 줘도
**{s['미도달']:,.0f}명({100 * s['미도달'] / M['고령']:.0f}%)**이 닿지 못한다.
느린 것이 아니라 길이 없다.

**수요** — 고령 의료통행만 센다. 실제 똑버스는 전 연령·전 목적이므로 여기서
나오는 소요 자원은 **보수적 하한**이다. 고령 1인 연 {sim.TRIPS_PER_YEAR}회
외래를 가정해 하루 78통행, 한 통행은 집→의원·의원→집 두 구간이다.

**현행 기준** — 가좌·송포에는 갈 수 있는 대중교통이 없다. 지금 이 통행은
택시나 자가용으로 이뤄지고 있다고 본다. 지금 고른 기준은 **{bl}**
({BL_NOTE})이다. 기준을 바꾸면 부호가 바뀔 수 있어 고르게 해 두었다.
""")
    b.markdown(f"""
**주행거리** — {D['meta']['거리출처']} 기준 실제 도로 경로. 운행 30 km/h —
카카오가 돌려준 소요시간에서 역산한 실질 평균속도가 31.0 km/h 로 이 가정과
맞았다. 카카오 거리와 국가표준노드링크 최단경로의 비는 표본 50쌍에서
**1.011**로, 어느 쪽을 써도 결론이 흔들리지 않는다.
배차하러 빈 차로 가는 **공차**를 따로 세며, 이 운영에서 공차는 전체 주행의
{100 * cur.dead_share:.0f}%다. 손익분기 재차인원은 배출계수의 비이므로 거리
가정에 흔들리지 않는다.

**배출계수** — 환경부 2021년 승인 국가계수. 택시 승용 중형 LPG
{sim.EF_TAXI:.1f}, 자가용 승용 소형 휘발유 {sim.EF_CAR:.1f} g/대·km(30 km/h).
전력 0.4747 kgCO₂/kWh(소비단). **국가계수에 소형승합이 없어** 쏠라티 디젤은
승용 대형 경유~화물 소형 경유로 범위를 잡았다.

**예산** — 표준운송원가 {sim.COST_DAY:,}원/대·일(수원시 2024 원가안).
인건비와 차량 감가상각비를 포함하므로 차량 구입비를 따로 세지 않는다.

**지정 승차장은 임시 가정이다** — 고령인구를 가중치로 둔 군집 중심을 도로에
붙인 것이다. 실제 선정에는 보행 접근권, 정차 가능 지점, 소유관계 분석이 따로
필요하다. 목록이 나오면 `sim.py` 의 `designated_stops()` 만 갈아끼우면 된다.

**재차인원** — 고령 의료통행만으로는 합승이 {cur.occ:.2f}명에서 더 늘지
않는다. 광교 실측 2.43~3.24명은 전 연령·전 목적 수요에서 나온 값이다.
""")

st.caption(
    f"고양시 고령자 의료접근성 연구 v6 · 시나리오 {len(ROWS):,}개 "
    f"(차종 {len(VEH)} × 대수 12 × 승차장 13) · 자료 생성 "
    f"{D['meta']['생성']} · 운송원가·재차인원 수원시정연구원 2024"
    f"(광교 똑버스 호출 329,501건)")
