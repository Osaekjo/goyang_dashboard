# -*- coding: utf-8 -*-
"""가좌·송포 똑버스 도입 검토판 — 정책 판단용.

묻는 것은 하나다. **똑버스를 넣으면 30분 안에 병원에 닿는 고령인구가
얼마나 늘어나는가.** 탄소와 예산은 그 다음이다.

  정하는 것   차종 · 도입 대수 · 지정 승차장 수 · 시간 예산
  달라지는 것 도달 고령인구 · 문앞-의원 시간 구성 · 탄소 · 예산

조합 3 × 12 × 13 = 468가지를 미리 돌려 data/scenarios.json 에 담아 두었다
(build_scenarios.py). 도달은 (예산, 대기) 만 주면 그 자리에서 다시 센다.
"""
from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from pyproj import Transformer

import kakao_map
import keys
import sim

HERE = Path(__file__).resolve().parent
TO_WGS = Transformer.from_crs(5179, 4326, always_xy=True)
C1, C2, C3, C4 = "#16707f", "#b8540f", "#6b4a9e", "#8c9499"
GOOD, BAD, NONE_ = "#16707f", "#b8540f", "#b9c3c7"

st.set_page_config(page_title="가좌·송포 똑버스 도입 검토판",
                   page_icon="🚐", layout="wide")


@st.cache_data(show_spinner=False)
def load() -> dict:
    return json.loads((HERE / "data" / "scenarios.json").read_text("utf-8"))


@st.cache_data(show_spinner=False)
def base() -> dict:
    return sim.load_meta()


@st.cache_resource(show_spinner="도로망을 읽는 중")
def road() -> sim.Road:
    return sim.Road()


@st.cache_data(show_spinner="경로를 그리는 중", max_entries=16)
def routes(k: int) -> list[dict]:
    R = sim.stop_routes(road(), sim.designated_stops(road(), base(), k),
                        base(), 2)
    out = []
    for r in R:
        xy = np.asarray(r.pop("path"), float)
        if len(xy) > 2:
            xy = _thin(xy)
        lon, lat = TO_WGS.transform(xy[:, 0], xy[:, 1])
        r["ll"] = [[round(b, 6), round(a, 6)] for a, b in zip(lon, lat)]
        r["lonlat"] = [[round(a, 6), round(b, 6)] for a, b in zip(lon, lat)]
        out.append(r)
    return out


def _thin(xy: np.ndarray, tol: float = 15.0) -> np.ndarray:
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
B = base()
M = B["meta"]
ROWS = pd.DataFrame(D["rows"])
VEH = D["vehicles"]

BASELINE = {
    "택시 단독": ("승용 중형 LPG · 1인 탑승", sim.EF_TAXI, 1.0),
    "택시 2인 합승": ("승용 중형 LPG · 2인 분담", sim.EF_TAXI, 2.0),
    "자가용 단독": ("승용 소형 휘발유 · 1인 탑승", sim.EF_CAR, 1.0),
    "자가용 2인": ("승용 소형 휘발유 · 보호자 동반", sim.EF_CAR, 2.0),
}

# ── 조작 ──────────────────────────────────────────────────────────────
st.markdown(
    "<div style='display:flex;gap:11px;align-items:baseline;flex-wrap:wrap'>"
    "<h2 style='margin:0;font-size:1.4rem'>가좌·송포 똑버스 도입 검토판</h2>"
    "<span style='color:#6b7a81;font-size:.88rem'>똑버스를 넣으면 30분 안에"
    " 병원에 닿는 고령인구가 얼마나 늘어나는가</span>"
    "<span style='font-family:monospace;font-size:.65rem;letter-spacing:.07em;"
    "background:#fdf3e4;color:#7a4a09;border:1px dashed #dfb87e;"
    "border-radius:3px;padding:3px 8px;font-weight:600'>수요·운행은 추정치"
    "</span></div>", unsafe_allow_html=True)

c = st.columns([1.9, 1.3, 1.3, 1.3, 1.5])
vi = c[0].selectbox("차종", range(len(VEH)),
                    format_func=lambda i: VEH[i]["name"], index=1)
fleet = c[1].slider("도입 대수", 1, 12, 10)
K = c[2].slider("지정 승차장 수", 4, 16, 10)
BUD = c[3].select_slider("시간 예산", [30, 45, 60], value=30,
                         format_func=lambda v: f"{v}분")
bl = c[4].selectbox("현행(탄소 비교 기준)", list(BASELINE), index=0,
                    help="가좌·송포에는 갈 수 있는 대중교통이 없다. "
                         "이 통행은 지금 택시나 자가용으로 이뤄지고 있다.")
BL_NOTE, BL_EF, BL_OCC = BASELINE[bl]

cur = ROWS[(ROWS.vi == vi) & (ROWS.fleet == fleet) & (ROWS.k == K)].iloc[0]
TAB = D["reach"][str(K)]
R0 = sim.reach(TAB, BUD, float(cur.wait))


def carbon_of(df: pd.DataFrame) -> pd.Series:
    return (df["km_rev"] * BL_EF / BL_OCC / 1000 - df["co2"]) * 365 / 1000


def reach_of(k: int, fleet_: int, budget: int) -> dict:
    w = float(ROWS[(ROWS.k == k) & (ROWS.fleet == fleet_)
                   & (ROWS.vi == 0)].iloc[0].wait)
    return sim.reach(D["reach"][str(k)], budget, w)


# ── 핵심 수치 ─────────────────────────────────────────────────────────
k = st.columns([1.5, 1, 1, 1, 1])
delta = 100 * (R0["new_share"] - R0["now_share"])
k[0].metric(f"{BUD}분 안에 병원에 닿는 고령인구",
            f"{100 * R0['new_share']:.1f}%",
            f"현행 {100 * R0['now_share']:.1f}% 에서 "
            f"{delta:+.1f}%포인트",
            delta_color="normal" if delta > 0 else "off")
k[1].metric("새로 닿게 되는 사람", f"{R0['gain_pop']:,.0f}명",
            f"고령 {M['고령']:,.0f}명 가운데", delta_color="off")
k[2].metric("그래도 못 닿는 사람", f"{R0['still_pop']:,.0f}명",
            f"현행 {M['고령'] - R0['now_pop']:,.0f}명 에서",
            delta_color="off")
cut = float(carbon_of(pd.DataFrame([cur])).iloc[0])
k[3].metric("연간 탄소", ("± 0.0 t" if abs(cut) < .05 else
                       f"{'▲ +' if cut >= 0 else '▼ −'}{abs(cut):,.1f} t"),
            f"현행 {bl} 대비", delta_color="off" if abs(cut) < .05
            else "normal" if cut >= 0 else "inverse")
k[4].metric("연간 운영비", f"{cur.cost_year / 1e8:,.1f}억원",
            f"새로 닿는 1인당 "
            + (f"{cur.cost_year / R0['gain_pop'] / 1e4:,.0f}만원"
               if R0["gain_pop"] > 0 else "—"), delta_color="off")

if cur.wait > BUD:
    st.error(
        f"**{fleet}대로는 평균 대기가 {cur.wait:,.0f}분이라 {BUD}분 예산 안에"
        f" 아무도 닿지 못한다.** 대수를 늘리거나 예산을 늘려야 한다.", icon="⛔")
elif not cur.in_hours:
    st.warning(
        f"{fleet}대로는 하루 78통행을 운행시간(08:30–17:00) 안에 다 처리하지"
        f" 못한다 — 마지막 하차 {int(cur.end // 60):02d}:"
        f"{int(cur.end % 60):02d}.", icon="⚠️")

# ── 도달 ──────────────────────────────────────────────────────────────
st.markdown("### 도달 — 대수가 가른다")
a1, a2, a3 = st.columns([1.25, 1, 1])

with a1:
    d = pd.DataFrame([{
        "대수": f, "도입 후": 100 * reach_of(K, f, BUD)["new_share"],
        "똑버스만": 100 * reach_of(K, f, BUD)["drt_share"],
        "대기": float(ROWS[(ROWS.k == K) & (ROWS.fleet == f)
                         & (ROWS.vi == 0)].iloc[0].wait),
    } for f in range(1, 13)])
    now = 100 * R0["now_share"]
    line = alt.Chart(d).transform_fold(
        ["도입 후", "똑버스만"], as_=["구분", "%"]).mark_line(
        point=True, strokeWidth=2.6).encode(
        x=alt.X("대수:Q", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("%:Q", title=f"{BUD}분 도달 고령인구 (%)",
                scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("구분:N", scale=alt.Scale(
            domain=["도입 후", "똑버스만"], range=[C1, C3]),
            legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("구분:N", scale=alt.Scale(
            domain=["도입 후", "똑버스만"], range=[[1, 0], [4, 3]]),
            legend=None),
        tooltip=["대수", "%:Q", "대기"])
    rule = alt.Chart(pd.DataFrame({"y": [now]})).mark_rule(
        color=C2, strokeWidth=2).encode(y="y:Q")
    txt = alt.Chart(pd.DataFrame({"y": [now], "t": [f"현행 {now:.1f}%"]})
                    ).mark_text(align="left", dx=6, dy=-7, color=C2,
                                fontWeight=600).encode(y="y:Q", text="t:N")
    pick = alt.Chart(pd.DataFrame({"x": [fleet]})).mark_rule(
        color="#555", strokeDash=[2, 2]).encode(x="x:Q")
    st.altair_chart((line + rule + txt + pick).properties(height=248),
                    width="stretch")
    st.caption(
        f"주황 선이 현행이다. **도입 후**는 기존 대중교통과 똑버스 가운데 나은"
        f" 쪽을 쓴 결과이고, **똑버스만**은 똑버스로만 갈 때다. 대수가 적으면"
        f" 대기가 길어 {BUD}분 안에 못 든다.")

with a2:
    d = pd.DataFrame([{
        "승차장": kk, "도달 (%)": 100 * reach_of(kk, fleet, BUD)["new_share"],
        "보행 중앙 (m)": int(ROWS[(ROWS.k == kk) & (ROWS.fleet == fleet)
                               & (ROWS.vi == 0)].iloc[0].walk_med),
    } for kk in D["meta"]["k"]])
    la = alt.Chart(d).mark_line(color=C1, point=True, strokeWidth=2.4).encode(
        x=alt.X("승차장:Q", axis=alt.Axis(tickMinStep=2)),
        y=alt.Y("도달 (%):Q", title=f"{BUD}분 도달 (%)",
                axis=alt.Axis(titleColor=C1)),
        tooltip=["승차장", "도달 (%)", "보행 중앙 (m)"])
    lb = alt.Chart(d).mark_line(color=C2, point=True, strokeWidth=2.2,
                                strokeDash=[5, 3]).encode(
        x="승차장:Q",
        y=alt.Y("보행 중앙 (m):Q", axis=alt.Axis(titleColor=C2)))
    pk = alt.Chart(pd.DataFrame({"x": [K]})).mark_rule(
        color="#555", strokeDash=[2, 2]).encode(x="x:Q")
    st.markdown("**승차장 수** — 많을수록 덜 걷는다")
    st.altair_chart(alt.layer(la, lb, pk).resolve_scale(
        y="independent").properties(height=226), width="stretch")
    st.caption("승차장을 늘리면 보행이 줄어 도달이 는다. 다만 차가 더 많이"
               " 돌아 탄소는 조금 나빠진다.")

with a3:
    walk = R0["walk_min"]
    wait = float(cur.wait)
    ride = max(0.0, BUD - walk - wait)
    parts = pd.DataFrame([
        {"구분": "집→승차장 보행", "분": walk, "ord": 1},
        {"구분": "대기", "분": wait, "ord": 2},
        {"구분": "승차장→의원 차내", "분": ride, "ord": 3}])
    st.markdown(f"**문앞-의원 {BUD}분은 어디에 쓰이나**")
    ch = alt.Chart(parts).mark_bar(cornerRadius=3).encode(
        x=alt.X("분:Q", stack="zero", title=f"분 (예산 {BUD}분)"),
        color=alt.Color("구분:N", sort=["집→승차장 보행", "대기",
                                      "승차장→의원 차내"],
                        scale=alt.Scale(range=[C1, C2, C3]),
                        legend=alt.Legend(orient="top", title=None,
                                          columns=1)),
        order="ord:Q",
        tooltip=["구분", alt.Tooltip("분:Q", format=".1f")])
    st.altair_chart(ch.properties(height=120), width="stretch")
    st.caption(
        f"보행 {walk:.1f}분 + 대기 {wait:.0f}분을 빼고 나면 차에 탈 수 있는"
        f" 시간은 **{ride:.1f}분**이다. 30 km/h 로 {ride / 2:.1f} km 안의"
        f" 의원만 닿는다. **대기를 줄이는 것이 가장 크게 듣는다.**")

# ── 지도 ──────────────────────────────────────────────────────────────
st.markdown("### 어디가 달라지는가")
mc, sc = st.columns([2.05, 1])

GRID = B["grid"]
GX0 = min(g[0] for g in GRID)
GY0 = min(g[1] for g in GRID)
GKEY = {(g[0], g[1]): i for i, g in enumerate(GRID)}


@st.cache_data(show_spinner=False)
def grid_cells(k_: int) -> list[int]:
    """출발점마다 속한 500m 격자 번호. 격자는 500m 격자망에 맞춰져 있다."""
    out = []
    for x, y in D["reach"][str(k_)]["cell_xy"]:
        gx = GX0 + ((x - GX0) // 500) * 500
        gy = GY0 + ((y - GY0) // 500) * 500
        out.append(GKEY.get((gx, gy), -1))
    return out


def grid_status() -> pd.DataFrame:
    """격자별 전후 도달 — 지도 색을 정한다."""
    gi = np.asarray(grid_cells(K))
    pop = np.asarray(TAB["cell_pop"], float)
    now = np.asarray(R0["per_cell_now"], float) > 0
    new = np.asarray(R0["per_cell"]) > 0
    rows = []
    for g in sorted(set(gi) - {-1}):
        m = gi == g
        p = pop[m].sum()
        pn, pa = pop[m & now].sum(), pop[m & (now | new)].sum()
        rows.append({"g": g, "pop": p, "now": pn, "after": pa,
                     "gain": pa - pn,
                     "stat": ("원래 닿던 곳" if pn >= p - .01 else
                              "새로 닿게 됨" if pa > pn + .01 else
                              "그래도 못 닿음" if pa < .01 else "일부만")})
    return pd.DataFrame(rows)


GS = grid_status()
COLORS = {"원래 닿던 곳": C4, "새로 닿게 됨": GOOD, "일부만": "#8aa8b0",
          "그래도 못 닿음": BAD}
RGB = {"원래 닿던 곳": [140, 148, 153], "새로 닿게 됨": [22, 112, 127],
       "일부만": [120, 160, 170], "그래도 못 닿음": [184, 84, 15]}
STOPS = D["stops"][str(K)]
RT = routes(K)

VWORLD = {"VWorld 일반": "Base", "VWorld 백지도": "white",
          "VWorld 위성혼합": "Hybrid", "VWorld 야간": "midnight"}


def payload_kakao() -> dict:
    grids = []
    for r in GS.itertuples():
        g = GRID[int(r.g)]
        corners = [(g[0], g[1]), (g[0] + 500, g[1]), (g[0] + 500, g[1] + 500),
                   (g[0], g[1] + 500)]
        poly = []
        for x, y in corners:
            lon, lat = TO_WGS.transform(x, y)
            poly.append([round(lat, 6), round(lon, 6)])
        grids.append({
            "poly": poly, "color": COLORS[r.stat],
            "opacity": 0.22 if r.stat == "원래 닿던 곳" else 0.5,
            "title": f"500m 격자 · {r.stat}",
            "body": f"고령 {r.pop:.0f}명 · {BUD}분 도달 {r.now:.0f} → "
                    f"{r.after:.0f}명"})
    stops = [{"lat": s["lat"], "lng": s["lon"],
              "label": s["name"].split()[-1],
              "r": 120 + s["pop"] * 1.9,
              "title": f"지정 승차장 {s['name']}",
              "body": f"고령 {s['pop']:.0f}명 · 보행 중앙 {s['walk_med']} m "
                      f"(최대 {s['walk_max']} m)"} for s in STOPS]
    dests, seen = [], set()
    for r in RT:
        if r["dest"] in seen:
            continue
        seen.add(r["dest"])
        d = B["dest"][r["dest"]]
        lon, lat = TO_WGS.transform(d[0], d[1])
        dests.append({"lat": round(lat, 6), "lng": round(lon, 6),
                      "r": 70 + max(1, d[7]) * 16,
                      "title": d[5],
                      "body": f"{d[6]} · 의사 {d[7]}명 · 내과 전문의 {d[2]}명"})
    paths = [{"line": r["ll"], "color": C1 if r["rank"] == 0 else "#7fb3bc",
              "weight": 4 if r["rank"] == 0 else 2,
              "opacity": 0.75 if r["rank"] == 0 else 0.45} for r in RT]
    dongs = []
    for d in B["dong"]:
        for ring in d["rings"]:
            lon, lat = TO_WGS.transform([p[0] for p in ring],
                                        [p[1] for p in ring])
            dongs.append([[round(b, 6), round(a, 6)]
                          for a, b in zip(lon, lat)])
    return {"center": [float(np.mean([s["lat"] for s in stops])),
                       float(np.mean([s["lng"] for s in stops]))],
            "level": 7, "grids": grids, "stops": stops, "dests": dests,
            "paths": paths, "dongs": dongs,
            "legend": {"title": f"{BUD}분 도달 변화",
                       "body": "회색 = 원래 닿던 곳 · 청록 = 새로 닿게 됨 · "
                               "주황 = 그래도 못 닿음"}}


def deck_fallback(bm: str) -> pdk.Deck:
    prov, style = "carto", "light"
    if bm in VWORLD:
        kk = keys.get(*keys.VWORLD)
        if kk:
            prov, style = "mapbox", {
                "version": 8,
                "sources": {"vw": {"type": "raster", "tileSize": 256,
                                   "maxzoom": 18, "attribution": "© VWorld",
                                   "tiles": [f"https://api.vworld.kr/req/wmts/"
                                             f"1.0.0/{kk}/{VWORLD[bm]}"
                                             "/{z}/{y}/{x}.png"]}},
                "layers": [{"id": "vw", "type": "raster", "source": "vw"}]}
    elif bm == "배경 없음":
        style = None
    polys = []
    for r in GS.itertuples():
        g = GRID[int(r.g)]
        ring = []
        for x, y in [(g[0], g[1]), (g[0] + 500, g[1]),
                     (g[0] + 500, g[1] + 500), (g[0], g[1] + 500)]:
            lon, lat = TO_WGS.transform(x, y)
            ring.append([round(lon, 6), round(lat, 6)])
        polys.append({"polygon": ring, "c": RGB[r.stat] + [
            60 if r.stat == "원래 닿던 곳" else 135],
            "name": r.stat,
            "tip": f"고령 {r.pop:.0f}명 · 도달 {r.now:.0f} → {r.after:.0f}명"})
    sdf = pd.DataFrame([{"lon": s["lon"], "lat": s["lat"],
                         "pop": s["pop"], "name": s["name"],
                         "lab": s["name"].split()[-1],
                         "tip": f"고령 {s['pop']:.0f}명 · 보행 중앙 "
                                f"{s['walk_med']} m"} for s in STOPS])
    return pdk.Deck(layers=[
        pdk.Layer("PolygonLayer", polys, get_polygon="polygon",
                  get_fill_color="c", stroked=True,
                  get_line_color=[255, 255, 255, 170],
                  line_width_min_pixels=1, pickable=True),
        pdk.Layer("PathLayer",
                  [{"path": r["lonlat"], "w": 14 if r["rank"] == 0 else 6}
                   for r in RT], get_path="path", get_width="w",
                  get_color=[22, 112, 127, 150], width_min_pixels=1),
        pdk.Layer("ScatterplotLayer", sdf, get_position=["lon", "lat"],
                  get_radius="130 + pop * 2.0", radius_min_pixels=8,
                  radius_max_pixels=28, get_fill_color=[22, 112, 127, 210],
                  stroked=True, get_line_color=[255, 255, 255, 230],
                  line_width_min_pixels=1.5, pickable=True),
        pdk.Layer("TextLayer", sdf, get_position=["lon", "lat"],
                  get_text="lab", get_size=11, get_color=[255, 255, 255],
                  get_alignment_baseline="'center'", font_weight=700,
                  character_set="'auto'"),
    ], map_provider=prov, map_style=style,
        initial_view_state=pdk.ViewState(
            longitude=float(sdf.lon.mean()), latitude=float(sdf.lat.mean()),
            zoom=11.4),
        tooltip={"html": "<b>{name}</b><br/>{tip}"}, height=500)


with mc:
    has_js = kakao_map.js_key() is not None
    opts = ["카카오맵"] + list(VWORLD) + ["Carto", "배경 없음"]
    bm = st.radio("배경지도", opts, horizontal=True,
                  index=0 if has_js else 1, label_visibility="collapsed")
    if bm == "카카오맵":
        kakao_map.render(payload_kakao(), height=500)
    else:
        st.pydeck_chart(deck_fallback(bm), width="stretch")
    st.caption(
        "격자 색 — **회색** 원래 닿던 곳 · **청록** 새로 닿게 됨 · "
        "**주황** 그래도 못 닿음. 큰 청록 점이 지정 승차장(크기는 맡은"
        " 고령인구), 선은 가까운 의원까지의 실제 도로 경로다. "
        f"주행거리 출처 {D['meta']['거리출처']}"
        + (f" (이 조합의 {100 * cur.kakao:.0f}%, 나머지는 국가표준노드링크 —"
           f" 두 값의 비는 표본 50쌍에서 1.011)" if cur.kakao > 0 else ""))

with sc:
    st.markdown(f"**{BUD}분 도달, 격자 {len(GS)}개**")
    agg = (GS.groupby("stat", as_index=False)
             .agg(격자=("g", "count"), 고령=("pop", "sum")))
    agg["고령"] = agg["고령"].round(0)
    st.dataframe(agg.rename(columns={"stat": "구분"}), hide_index=True,
                 width="stretch")
    st.markdown("**지정 승차장**")
    st.dataframe(pd.DataFrame([{
        "승차장": s["name"].split()[-1], "고령인구": s["pop"],
        "보행 중앙": s["walk_med"], "보행 최대": s["walk_max"]}
        for s in STOPS]), hide_index=True, width="stretch", height=270)

# ── 탄소·예산 ─────────────────────────────────────────────────────────
with st.expander("탄소와 예산 — 도달 다음의 이야기", expanded=False):
    e1, e2 = st.columns(2)
    with e1:
        d = ROWS[(ROWS.fleet == fleet) & (ROWS.k == K)].copy()
        d["차종"] = [VEH[i]["name"] for i in d.vi]
        d["연간 탄소(t)"] = carbon_of(d)
        bar = alt.Chart(d).mark_bar(cornerRadius=3).encode(
            y=alt.Y("차종:N", sort=None, title=None,
                    axis=alt.Axis(labelLimit=190)),
            x=alt.X("연간 탄소(t):Q", title="연간 탄소 (t) — 오른쪽이 좋다"),
            color=alt.condition(alt.datum["연간 탄소(t)"] >= 0,
                                alt.value(C1), alt.value(C2)),
            tooltip=["차종", "연간 탄소(t)", "breakeven_dead", "occ"])
        zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
            color="#555", strokeWidth=1.4).encode(x="x:Q")
        st.markdown(f"**차종별 연간 탄소** — {fleet}대 · 승차장 {K}곳 · "
                    f"현행 {bl} 대비")
        st.altair_chart((bar + zero).properties(height=190), width="stretch")
        st.caption(
            f"이 설정의 실제 재차인원은 {cur.occ:.2f}명, 손익분기는 공차까지"
            f" 세면 {cur.breakeven_dead:.2f}명이다. 고령 의료통행만으로는"
            f" 합승이 더 생기지 않는다 — 광교의 2.43~3.24는 전 연령·전 목적"
            f" 수요에서 나온 값이다. **탄소는 똑버스 도입의 근거가 아니다."
            f" 근거는 접근성이고, 차종 선택으로 손해를 줄이는 것이다.**")
    with e2:
        d = ROWS[(ROWS.vi == vi) & (ROWS.k == K)].sort_values("fleet").copy()
        d["연간 비용(억)"] = d.cost_year / 1e8
        d["도달(%)"] = [100 * reach_of(K, f, BUD)["new_share"] for f in d.fleet]
        la = alt.Chart(d).mark_line(color=C1, point=True,
                                    strokeWidth=2.4).encode(
            x=alt.X("fleet:Q", title="도입 대수",
                    axis=alt.Axis(tickMinStep=1)),
            y=alt.Y("도달(%):Q", axis=alt.Axis(titleColor=C1)),
            tooltip=["fleet", "도달(%)", "연간 비용(억)"])
        lb = alt.Chart(d).mark_line(color=C3, point=True, strokeWidth=2.2,
                                    strokeDash=[5, 3]).encode(
            x="fleet:Q",
            y=alt.Y("연간 비용(억):Q", axis=alt.Axis(titleColor=C3)))
        st.markdown("**도달과 예산** — 어디까지 사는가")
        st.altair_chart(alt.layer(la, lb).resolve_scale(
            y="independent").properties(height=190), width="stretch")
        st.caption(
            f"대당 표준운송원가 {sim.COST_DAY:,}원/일(수원시 2024 원가안)."
            f" 지금 설정은 연 {cur.cost_year / 1e8:.1f}억원이고, 새로 닿는"
            f" 사람 1명당 "
            + (f"{cur.cost_year / R0['gain_pop'] / 1e4:,.0f}만원이다."
               if R0["gain_pop"] > 0 else "— (아직 아무도 못 닿는다)"))

with st.expander("방법과 한계", expanded=False):
    s = B["split"]["유형"]
    x1, x2 = st.columns(2)
    x1.markdown(f"""
**대상** — 고양시 저접근 하위 25% 격자의 연결성분 가운데 1순위 군집(대표동
가좌동). 44격자 · 출발점 124 · **고령 {M['고령']:,.1f}명**. 집 앞에서 30분
안에 닿는 의료기관이 평균 **{M['A']:.2f}곳**으로 시 평균 {M['A_시평균']:.1f}곳의
{100 * M['A'] / M['A_시평균']:.2f}%다. {M['완전단절_격자']}격자는 80분을 줘도
0곳이고, 45분을 줘도 {s['미도달']:,.0f}명이 닿지 못한다. 느린 것이 아니라
길이 없다.

**도달을 어떻게 셌나** — 도입 후 한 사람의 문앞-의원 시간은
`집→승차장 보행(0.8 m/s) + 대기 + 승차장→의원 차내(30 km/h)` 이고, 예산 안에
의료기관이 하나라도 들면 '닿는다'로 센다. 대기는 그 대수의 시뮬레이션 평균이다.
**똑버스가 들어와도 기존 대중교통이 없어지지 않으므로**, 도입 후 도달은 현행과
똑버스 가운데 나은 쪽이다.

**합승으로 생기는 우회는 차내시간에 반영하지 않았다.** 같은 방향 승객끼리
묶으므로 차이가 크지 않지만, 이 값은 **낙관 쪽**이다.
""")
    x2.markdown(f"""
**수요** — 고령 의료통행만 센다. 실제 똑버스는 전 연령·전 목적이므로 소요
자원은 **보수적 하한**이다. 고령 1인 연 {sim.TRIPS_PER_YEAR}회 외래를 가정해
하루 78통행, 한 통행은 집→의원·의원→집 두 구간이다.

**주행거리** — {D['meta']['거리출처']} 기준 실제 도로 경로. 카카오가 돌려준
소요시간에서 역산한 평균속도가 31.0 km/h 로 모형 가정 30 km/h 와 맞았고,
카카오 거리와 국가표준노드링크 최단경로의 비는 표본 50쌍에서 **1.011**이었다.

**배출계수** — 환경부 2021년 승인 국가계수. 택시 {sim.EF_TAXI:.1f},
자가용 {sim.EF_CAR:.1f} g/대·km(30 km/h). 전력 0.4747 kgCO₂/kWh(소비단).
국가계수에 소형승합이 없어 쏠라티 디젤은 승용 대형 경유~화물 소형 경유로
범위를 잡았다.

**지정 승차장은 임시 가정이다** — 고령인구를 가중치로 둔 군집 중심을 도로에
붙인 것이다. 실제 선정에는 보행 접근권·정차 가능 지점·소유관계 분석이 따로
필요하다. 목록이 나오면 `sim.designated_stops()` 만 갈아끼우면 된다.
""")

st.caption(
    f"고양시 고령자 의료접근성 연구 v6 · 시나리오 {len(ROWS):,}개 "
    f"(차종 {len(VEH)} × 대수 12 × 승차장 13) × 예산 3 · 자료 생성 "
    f"{D['meta']['생성']} · 운송원가·재차인원 수원시정연구원 2024"
    f"(광교 똑버스 호출 329,501건)")
