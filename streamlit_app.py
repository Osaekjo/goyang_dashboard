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
import streamlit as st
from pyproj import Transformer

import kakao_map
import keys
import mclp
import sim


def _hot_reload() -> None:
    """고친 모듈이 실제로 쓰이게 한다.

    Streamlit Cloud 는 푸시하면 streamlit_app.py 만 다시 읽고, 이미 import 된
    모듈은 sys.modules 에 그대로 둔다. 그래서 sim.py 에 함수를 더해도 배포본은
    옛 모듈을 보고 AttributeError 를 낸다. 파일이 바뀐 모듈만 다시 읽고,
    바뀌었으면 캐시도 비운다 — 캐시에 담긴 객체는 옛 클래스로 만들어졌다.
    """
    import importlib
    changed = []
    for m in (keys, sim, mclp, kakao_map):
        s = Path(m.__file__).stat()
        sig = (s.st_size, s.st_mtime_ns)
        old = getattr(m, "_file_sig", None)
        if old is not None and old != sig:
            importlib.reload(m)
            changed.append(Path(m.__file__).name)
        setattr(m, "_file_sig", sig)
    if changed:
        st.cache_data.clear()
        st.cache_resource.clear()
        st.caption("모듈을 다시 읽었습니다 — " + ", ".join(changed))


_hot_reload()

HERE = Path(__file__).resolve().parent
TO_WGS = Transformer.from_crs(5179, 4326, always_xy=True)
C1, C2, C3, C4 = "#16707f", "#b8540f", "#6b4a9e", "#8c9499"
GOOD, BAD, NONE_ = "#16707f", "#b8540f", "#b9c3c7"

st.set_page_config(page_title="가좌·송포 똑버스 도입 검토판",
                   page_icon="🚐", layout="wide")


def _sig(p: Path) -> tuple[int, int]:
    """파일이 바뀌면 캐시가 깨지게 하는 열쇠.

    Streamlit Cloud 는 푸시하면 스크립트만 다시 읽고 @st.cache_data 는 그대로
    둔다. 자료 파일만 바뀐 경우 새 코드가 옛 자료를 보게 되어 뜻 모를 오류가
    난다. 크기와 수정시각을 열쇠에 넣어 그 일을 막는다.
    """
    s = p.stat()
    return (s.st_size, s.st_mtime_ns)


@st.cache_data(show_spinner=False)
def _load(sig: tuple) -> dict:
    d = json.loads((HERE / "data" / "scenarios.json").read_text("utf-8"))
    if not d.get("rows") or "sel" not in d["rows"][0]:
        raise RuntimeError(
            "data/scenarios.json 이 옛 판이다 (승차장 선정법 구분이 없다). "
            "build_scenarios.py 를 다시 돌리거나, 배포본이라면 앱을 "
            "Reboot 하라.")
    return d


def load() -> dict:
    return _load(_sig(HERE / "data" / "scenarios.json"))


@st.cache_data(show_spinner=False)
def _base(sig: tuple) -> dict:
    return sim.load_meta()


def base() -> dict:
    return _base(_sig(HERE / "data" / "data.json"))


@st.cache_resource(show_spinner="도로망을 읽는 중")
def road() -> sim.Road:
    return sim.Road()


@st.cache_data(show_spinner="경로를 그리는 중", max_entries=32)
def routes(sel: str, k: int) -> list[dict]:
    """승차장은 scenarios.json 이 유일한 출처다. 화면에서 다시 풀지 않는다.
    다시 풀면 시드·알고리즘이 바뀔 때 표와 지도가 조용히 어긋난다."""
    R = sim.stop_routes(road(), D["stops"][str(k)], base(), 2)
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

c0 = st.columns([1.5, 4.5])
SEL = c0[0].radio("운영 방식", ["hub", "direct"], horizontal=True,
                  format_func=lambda s: {"hub": "거점 연계",
                                         "direct": "의원 직결"}[s],
                  help="거점 연계 = 승차장에서 대화역·일산서구청까지만 태우고 "
                       "거기서 기존 대중교통으로 갈아탄다(고봉동 똑버스 방식). "
                       "의원 직결 = 집 앞 승차장에서 의원까지 바로 간다.")
HUBT = D["meta"].get("hub_target", "일산백병원")
HUB_TXT = " · ".join(f"{n} {h['onward_min']:.1f}분"
                     for n, h in D.get("hubs", {}).items())
c0[1].caption(
    (f"**거점 연계** — 승차장 → "
     + " · ".join(f"{n}({h['onward_min']:.1f}분)"
                  for n, h in D.get("hubs", {}).items())
     + f" → **{HUBT}**. 구간이 짧아 차량이 적게 든다."
     if SEL == "hub" else
     "**의원 직결** — 집 앞 승차장에서 의원까지 바로. 구간이 길어 차량이 많이 든다.")
    + f" · 승차장 = **기존 정류장 {D['meta'].get('기존정류장', 0)}곳**"
      f" + 신설 (MCLP, 기준 {D['meta'].get('S_m', 600):,.0f} m)")
c = st.columns([1.9, 1.3, 1.3, 1.3, 1.5])
vi = c[0].selectbox("차종", range(len(VEH)),
                    format_func=lambda i: VEH[i]["name"], index=1)
fleet = c[1].slider("도입 대수", 1, 12, 3 if SEL == "hub" else 10)
K = c[2].slider("신설 승차장 수", int(min(D["meta"]["k"])),
                int(max(D["meta"]["k"])), 4,
                help="기존 버스정류장은 똑버스가 그냥 쓴다. 여기서 정하는 "
                     "것은 그 위에 새로 만들 승차장 수다.")
BUD = c[3].select_slider("시간 예산", [30, 45, 60], value=30,
                         format_func=lambda v: f"{v}분")
bl = c[4].selectbox("현행(탄소 비교 기준)", list(BASELINE), index=0,
                    help="가좌·송포에는 갈 수 있는 대중교통이 없다. "
                         "이 통행은 지금 택시나 자가용으로 이뤄지고 있다.")
BL_NOTE, BL_EF, BL_OCC = BASELINE[bl]

ROWS = ROWS[ROWS.sel == SEL]
if not hasattr(sim, "reach_hub"):
    st.error(
        "서버가 옛 `sim.py` 를 들고 있습니다 — 거점 연계 함수가 없습니다.\n\n"
        "배포본이라면 **Manage app → ⋮ → Reboot app** 으로 다시 띄워 주세요.",
        icon="🔄")
    st.stop()
REACH = sim.reach_hub if SEL == "hub" else sim.reach
cur = ROWS[(ROWS.vi == vi) & (ROWS.fleet == fleet) & (ROWS.k == K)].iloc[0]
TAB = D["reach"][str(K)]
R0 = REACH(TAB, BUD, float(cur.wait))


def carbon_of(df: pd.DataFrame) -> pd.Series:
    return (df["km_rev"] * BL_EF / BL_OCC / 1000 - df["co2"]) * 365 / 1000


def reach_of(k: int, fleet_: int, budget: int) -> dict:
    w = float(ROWS[(ROWS.k == k) & (ROWS.fleet == fleet_)
                   & (ROWS.vi == 0)].iloc[0].wait)
    return REACH(D["reach"][str(k)], budget, w)


# ── 핵심 수치 ─────────────────────────────────────────────────────────
# 이 화면이 묻는 것은 하나다. 그것만 크게 띄우고 나머지는 작게 받친다.
HEAD = (f"{BUD}분 안에 {HUBT}에 닿는 고령인구" if SEL == "hub"
        else f"{BUD}분 안에 병원에 닿는 고령인구")
# 거점 연계는 "백병원에 닿는가"를 묻는다. 그 현행값은 우리 자료에 없어
# 협업자 분석(C002, 30분 백병원 도달 1.4%)을 기준선으로 빌려 쓴다.
# 지역 범위가 달라 정확히 같은 모집단은 아니다 — 화면에 그렇게 적는다.
HUB_BASE = 0.014
NOW_SHARE = HUB_BASE if SEL == "hub" else R0["now_share"]
NOW_POP = NOW_SHARE * M["고령"] if SEL == "hub" else R0["now_pop"]
NOW_LAB = ("현행 — 협업자 분석 1.4%" if SEL == "hub" else "원래 닿던 사람")
now_p, new_p = 100 * NOW_SHARE, 100 * R0["new_share"]
delta = new_p - now_p
BASE_TXT = f"{now_p:.1f}%"
w_now = now_p
w_gain = max(0.0, new_p - now_p)
w_no = max(0.0, 100 - new_p)
st.markdown(f"""
<div style="background:linear-gradient(103deg,#0f5a66 0%,#16707f 46%,
     #2a8d93 100%);border-radius:14px;padding:22px clamp(18px,3vw,34px) 20px;
     color:#fff;margin:4px 0 14px">
  <div style="font-size:.9rem;letter-spacing:.02em;opacity:.84;
       font-weight:500">{HEAD}</div>
  <div style="display:flex;align-items:baseline;gap:clamp(10px,2.2vw,26px);
       flex-wrap:wrap;margin-top:6px">
    <div style="font-family:'IBM Plex Mono',ui-monospace,monospace;
         font-size:clamp(1.5rem,3.4vw,2.4rem);font-weight:600;opacity:.62;
         line-height:1">{BASE_TXT}</div>
    <div style="font-size:clamp(1.1rem,2.4vw,1.7rem);opacity:.6;
         line-height:1">→</div>
    <div style="font-family:'IBM Plex Mono',ui-monospace,monospace;
         font-size:clamp(3rem,9vw,5.4rem);font-weight:700;line-height:.95;
         letter-spacing:-.02em">{new_p:.1f}<span
         style="font-size:.42em;font-weight:500;opacity:.82">%</span></div>
    <div style="font-family:'IBM Plex Mono',ui-monospace,monospace;
         font-size:clamp(1.1rem,2.6vw,1.9rem);font-weight:700;
         background:rgba(255,255,255,.17);border-radius:8px;
         padding:5px 13px;line-height:1.2">{delta:+.1f}%p</div>
  </div>
  <div style="display:flex;height:16px;border-radius:8px;overflow:hidden;
       margin-top:16px;background:rgba(255,255,255,.14)">
    <div style="width:{w_now:.2f}%;background:rgba(255,255,255,.52)"></div>
    <div style="width:{w_gain:.2f}%;background:#ffd08a"></div>
    <div style="width:{w_no:.2f}%;background:rgba(0,0,0,.26)"></div>
  </div>
  <div style="display:flex;gap:18px 26px;flex-wrap:wrap;margin-top:11px;
       font-size:.86rem">
    <span><b style="font-size:1.12rem">{NOW_POP:,.0f}명</b>
      <span style="opacity:.8">{NOW_LAB}</span></span>
    <span style="color:#ffd08a"><b style="font-size:1.12rem">
      +{max(0.0, R0['new_pop'] - NOW_POP):,.0f}명</b> 새로 닿게 됨</span>
    <span style="opacity:.78"><b style="font-size:1.12rem">
      {R0['still_pop']:,.0f}명</b> 그래도 못 닿음</span>
    <span style="opacity:.62;margin-left:auto">대상 고령 {M['고령']:,.0f}명 ·
      {VEH[vi]['name']} {fleet}대 · 승차장 {K}곳</span>
  </div>
</div>""", unsafe_allow_html=True)

cut = float(carbon_of(pd.DataFrame([cur])).iloc[0])
SM = D["meta"].get("S_m", 600)
k = st.columns(5)
k[0].metric("평균 대기", f"{cur.wait:,.0f}분",
            f"10번 중 1번은 {cur.wait_p90:,.0f}분 이상", delta_color="off")
k[1].metric("집→승차장 보행", f"{R0['walk_min']:.1f}분",
            f"중앙 {cur.walk_med:,.0f} m · "
            + ("실제 보행망" if SEL == "mclp" else "직선 추정"),
            delta_color="off")
k[4].metric(f"도보 {SM / 60 / 0.8:.0f}분 커버",
            f"{100 * cur.cover_share:.1f}%",
            f"승차장 {SM:,.0f} m 안 고령인구 — MCLP 가 최대화하는 값",
            delta_color="off")
k[2].metric("연간 탄소", ("± 0.0 t" if abs(cut) < .05 else
                       f"{'▲ +' if cut >= 0 else '▼ −'}{abs(cut):,.1f} t"),
            f"현행 {bl} 대비", delta_color="off" if abs(cut) < .05
            else "normal" if cut >= 0 else "inverse")
k[3].metric("연간 운영비", f"{cur.cost_year / 1e8:,.1f}억원",
            "새로 닿는 1인당 "
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

# ── 지도 ──────────────────────────────────────────────────────────────
st.markdown("### 어디가 달라지는가")

GRID = B["grid"]
GX0 = min(g[0] for g in GRID)
GY0 = min(g[1] for g in GRID)
GKEY = {(g[0], g[1]): i for i, g in enumerate(GRID)}


@st.cache_data(show_spinner=False)
def grid_cells(k_: int, sel: str = "") -> list[int]:
    """출발점마다 속한 500m 격자 번호. 격자망이 500m 로 떨어져 있다."""
    out = []
    for x, y in D["reach"][str(k_)]["cell_xy"]:
        gx = GX0 + ((x - GX0) // 500) * 500
        gy = GY0 + ((y - GY0) // 500) * 500
        out.append(GKEY.get((gx, gy), -1))
    return out


def grid_status() -> pd.DataFrame:
    """격자별 전후 도달 — 지도 색을 정한다."""
    gi = np.asarray(grid_cells(K, SEL))
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
STOPS = D["stops"][str(K)]
RT = routes(SEL, K)


def payload_kakao() -> dict:
    grids = []
    for r in GS.itertuples():
        g = GRID[int(r.g)]
        poly = []
        for x, y in [(g[0], g[1]), (g[0] + 500, g[1]),
                     (g[0] + 500, g[1] + 500), (g[0], g[1] + 500)]:
            lon, lat = TO_WGS.transform(x, y)
            poly.append([round(lat, 6), round(lon, 6)])
        grids.append({
            "poly": poly, "color": COLORS[r.stat],
            "opacity": 0.2 if r.stat == "원래 닿던 곳" else 0.5,
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
                      "r": 70 + max(1, d[7]) * 16, "title": d[5],
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


kakao_map.render(payload_kakao(), height=560)
st.caption(
    "격자 색 — **회색** 원래 닿던 곳 · **청록** 새로 닿게 됨 · "
    "**주황** 그래도 못 닿음. 큰 청록 점이 지정 승차장(크기는 맡은 고령인구),"
    " 선은 가까운 의원까지의 실제 도로 경로다. 도형에 마우스를 올리면 왼쪽"
    f" 아래에 설명이 뜬다. 주행거리 출처 {D['meta']['거리출처']}"
    + (f" (이 조합의 {100 * cur.kakao:.0f}%, 나머지는 국가표준노드링크 —"
       f" 두 값의 비는 표본 50쌍에서 1.011)" if cur.kakao > 0 else ""))

# ── 그래프 ────────────────────────────────────────────────────────────
st.markdown("### 자세히 보기")
PICK = st.radio("그래프", [
    "대수별 도달", "신설 승차장 수별 도달과 보행", "문앞-의원 시간 구성",
    "승차장별 고령인구", "격자별 도달 변화", "차종별 연간 탄소",
    "도달과 예산"], horizontal=True, label_visibility="collapsed")

H = 430          # 그래프를 크게 — 한 번에 하나만 보인다


def c_fleet():
    d = pd.DataFrame([{
        "대수": f, "도입 후": 100 * reach_of(K, f, BUD)["new_share"],
        "똑버스만": 100 * reach_of(K, f, BUD)["drt_share"],
        "평균 대기(분)": float(ROWS[(ROWS.k == K) & (ROWS.fleet == f)
                                & (ROWS.vi == 0)].iloc[0].wait),
    } for f in range(1, 13)])
    now = 100 * R0["now_share"]
    line = alt.Chart(d).transform_fold(
        ["도입 후", "똑버스만"], as_=["구분", "%"]).mark_line(
        point={"size": 90}, strokeWidth=3).encode(
        x=alt.X("대수:Q", axis=alt.Axis(tickMinStep=1, labelFontSize=13,
                                       titleFontSize=13)),
        y=alt.Y("%:Q", title=f"{BUD}분 안에 병원에 닿는 고령인구 (%)",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(labelFontSize=13, titleFontSize=13)),
        color=alt.Color("구분:N", scale=alt.Scale(
            domain=["도입 후", "똑버스만"], range=[C1, C3]),
            legend=alt.Legend(orient="top", title=None, labelFontSize=13)),
        strokeDash=alt.StrokeDash("구분:N", scale=alt.Scale(
            domain=["도입 후", "똑버스만"], range=[[1, 0], [5, 4]]),
            legend=None),
        tooltip=["대수", alt.Tooltip("%:Q", format=".1f"), "평균 대기(분)"])
    rule = alt.Chart(pd.DataFrame({"y": [now]})).mark_rule(
        color=C2, strokeWidth=2.5).encode(y="y:Q")
    txt = alt.Chart(pd.DataFrame({"y": [now], "t": [f"현행 {now:.1f}%"]})
                    ).mark_text(align="left", dx=8, dy=-9, color=C2,
                                fontWeight=700, fontSize=14).encode(
        y="y:Q", text="t:N")
    pick = alt.Chart(pd.DataFrame({"x": [fleet]})).mark_rule(
        color="#555", strokeDash=[3, 3], strokeWidth=1.6).encode(x="x:Q")
    return (line + rule + txt + pick), (
        f"주황 선이 현행 {now:.1f}% 다. **도입 후**는 기존 대중교통과 똑버스"
        f" 가운데 나은 쪽을 쓴 결과, **똑버스만**은 똑버스로만 갈 때다."
        f" 대수가 적으면 대기가 길어 {BUD}분 안에 들지 못한다 — 이 그래프가"
        f" 이 검토의 핵심이다.")


def c_stops():
    d = pd.DataFrame([{
        "신설 승차장": kk, "도달 (%)": 100 * reach_of(kk, fleet, BUD)["new_share"],
        "커버 (%)": 100 * float(ROWS[(ROWS.k == kk) & (ROWS.fleet == fleet)
                                   & (ROWS.vi == 0)].iloc[0].cover_share),
        "보행 중앙 (m)": int(ROWS[(ROWS.k == kk) & (ROWS.fleet == fleet)
                               & (ROWS.vi == 0)].iloc[0].walk_med),
    } for kk in D["meta"]["k"]])
    ax = alt.Axis(labelFontSize=13, titleFontSize=13)
    la = alt.Chart(d).mark_line(color=C1, point={"size": 90},
                                strokeWidth=3).encode(
        x=alt.X("신설 승차장:Q", title="신설 승차장 수 (기존 정류장은 늘 쓴다)",
                axis=alt.Axis(tickMinStep=2, labelFontSize=13,
                                         titleFontSize=13)),
        y=alt.Y("도달 (%):Q", title=f"{BUD}분 도달 (%)",
                axis=alt.Axis(titleColor=C1, labelFontSize=13,
                              titleFontSize=13)),
        tooltip=["신설 승차장", "도달 (%)", "커버 (%)", "보행 중앙 (m)"])
    lb = alt.Chart(d).mark_line(color=C2, point={"size": 80}, strokeWidth=2.6,
                                strokeDash=[6, 4]).encode(
        x="신설 승차장:Q",
        y=alt.Y("보행 중앙 (m):Q", axis=alt.Axis(titleColor=C2,
                                             labelFontSize=13,
                                             titleFontSize=13)))
    lc = alt.Chart(d).mark_line(color=C3, point={"size": 70},
                                strokeWidth=2.2, opacity=.85).encode(
        x="신설 승차장:Q", y=alt.Y("커버 (%):Q"))
    pk = alt.Chart(pd.DataFrame({"x": [K]})).mark_rule(
        color="#555", strokeDash=[3, 3], strokeWidth=1.6).encode(x="x:Q")
    return alt.layer(la, lc, lb, pk).resolve_scale(y="independent"), (
        f"청록 실선이 30분 도달, 보라 실선이 도보 {SM:,.0f} m 커버(MCLP 가"
        f" 최대화하는 값), 주황 점선이 보행 중앙값(오른쪽 축)이다. 승차장을"
        f" 늘리면 덜 걸어 도달이 늘지만, 차가 더 많이 돌아 탄소는 조금"
        f" 나빠지고 설치 비용도 는다.")


def c_split():
    walk, wait = R0["walk_min"], float(cur.wait)
    ride = max(0.0, BUD - walk - wait)
    d = pd.DataFrame([
        {"구분": "집→승차장 보행", "분": walk, "ord": 1},
        {"구분": "대기", "분": wait, "ord": 2},
        {"구분": "승차장→의원 차내", "분": ride, "ord": 3}])
    ch = alt.Chart(d).mark_bar(cornerRadius=4, size=120).encode(
        x=alt.X("분:Q", stack="zero", title=f"분 (시간 예산 {BUD}분)",
                axis=alt.Axis(labelFontSize=13, titleFontSize=13)),
        color=alt.Color("구분:N", sort=["집→승차장 보행", "대기",
                                      "승차장→의원 차내"],
                        scale=alt.Scale(range=[C1, C2, C3]),
                        legend=alt.Legend(orient="top", title=None,
                                          labelFontSize=13)),
        order="ord:Q",
        tooltip=["구분", alt.Tooltip("분:Q", format=".1f")])
    lab = alt.Chart(d).mark_text(color="white", fontWeight=700,
                                 fontSize=14).encode(
        x=alt.X("분:Q", stack="zero", bandPosition=0.5),
        text=alt.Text("분:Q", format=".1f"), order="ord:Q",
        detail="구분:N")
    return (ch + lab), (
        f"보행 {walk:.1f}분 + 대기 {wait:.0f}분을 빼고 나면 차에 탈 수 있는"
        f" 시간은 **{ride:.1f}분**이다. 30 km/h 로 {ride / 2:.1f} km 안의"
        f" 의원만 닿는다. 셋 가운데 **대기가 가장 크게 움직이고, 대수로"
        f" 줄일 수 있다.** 보행은 신설 승차장으로 줄인다.")


def c_stoppop():
    d = pd.DataFrame([{
        "승차장": s["name"].split()[-1], "고령인구": s["pop"],
        "보행 중앙 (m)": s["walk_med"], "보행 최대 (m)": s["walk_max"],
        "맡은 격자": s["cells"],
        "구분": "기존 정류장" if s.get("exist") else "신설"}
        for s in STOPS if s.get("is_new") or s.get("pop", 0) > 0])
    bar = alt.Chart(d).mark_bar(cornerRadius=3).encode(
        color=alt.Color("구분:N", scale=alt.Scale(
            domain=["기존 정류장", "신설"], range=[C3, C1]),
            legend=alt.Legend(orient="top", title=None, labelFontSize=12)),
        x=alt.X("승차장:N", sort=None, title="지정 승차장 (인구 순위)",
                axis=alt.Axis(labelFontSize=13, titleFontSize=13)),
        y=alt.Y("고령인구:Q", title="맡은 고령인구 (명)",
                axis=alt.Axis(labelFontSize=13, titleFontSize=13)),
        tooltip=list(d.columns))
    lab = alt.Chart(d).mark_text(dy=-9, fontSize=12, fontWeight=600,
                                 color=C1).encode(
        x=alt.X("승차장:N", sort=None), y="고령인구:Q",
        text=alt.Text("고령인구:Q", format=".0f"))
    ln = alt.Chart(d).mark_line(color=C2, point={"size": 80},
                                strokeWidth=2.6, strokeDash=[6, 4]).encode(
        x=alt.X("승차장:N", sort=None),
        y=alt.Y("보행 중앙 (m):Q", axis=alt.Axis(titleColor=C2,
                                             labelFontSize=13,
                                             titleFontSize=13)))
    far = d[d["보행 최대 (m)"] > 1000]
    note = (f"고령 {d['고령인구'].sum():.0f}명을 {len(d)}곳이 나눠 맡는다."
            f" 막대가 맡은 인구, 점선이 그 승차장까지의 보행 중앙값이다.")
    if len(far):
        note += (f" **{len(far)}곳은 가장 먼 사람이 1 km 넘게 걸어야 한다**"
                 f" (최대 {d['보행 최대 (m)'].max():,.0f} m · 0.8 m/s 로"
                 f" {d['보행 최대 (m)'].max() / 0.8 / 60:.0f}분). 실제"
                 f" 선정에서는 그 덩어리를 쪼개야 한다.")
    return alt.layer(bar, lab, ln).resolve_scale(y="independent"), note


def c_grid():
    d = GS.copy()
    d["격자"] = [f"{i + 1}" for i in range(len(d))]
    d = d.sort_values("pop", ascending=False).head(28)
    long = pd.concat([
        d.assign(구분="현행 도달", 명=d.now),
        d.assign(구분="새로 닿게 됨", 명=d.gain),
        d.assign(구분="그래도 못 닿음", 명=d["pop"] - d.after)])
    ch = alt.Chart(long).mark_bar(cornerRadius=2).encode(
        x=alt.X("격자:N", sort=list(d["격자"]), title="500m 격자 (고령인구 순)",
                axis=alt.Axis(labelFontSize=11, titleFontSize=13)),
        y=alt.Y("명:Q", stack="zero", title="고령인구 (명)",
                axis=alt.Axis(labelFontSize=13, titleFontSize=13)),
        color=alt.Color("구분:N", scale=alt.Scale(
            domain=["현행 도달", "새로 닿게 됨", "그래도 못 닿음"],
            range=[C4, GOOD, BAD]),
            legend=alt.Legend(orient="top", title=None, labelFontSize=13)),
        tooltip=["격자", "pop", "now", "after"])
    return ch, (
        f"고령인구가 많은 격자부터 {len(d)}개다. 회색이 원래 닿던 사람,"
        f" 청록이 똑버스로 새로 닿는 사람, 주황이 그래도 못 닿는 사람이다."
        f" 지도의 색과 같은 뜻이다.")


def c_carbon():
    d = ROWS[(ROWS.fleet == fleet) & (ROWS.k == K)].copy()
    d["차종"] = [VEH[i]["name"] for i in d.vi]
    d["연간 탄소(t)"] = carbon_of(d)
    bar = alt.Chart(d).mark_bar(cornerRadius=4, size=46).encode(
        y=alt.Y("차종:N", sort=None, title=None,
                axis=alt.Axis(labelLimit=220, labelFontSize=13)),
        x=alt.X("연간 탄소(t):Q", title="연간 탄소 (t) — 오른쪽이 좋다",
                axis=alt.Axis(labelFontSize=13, titleFontSize=13)),
        color=alt.condition(alt.datum["연간 탄소(t)"] >= 0,
                            alt.value(C1), alt.value(C2)),
        tooltip=["차종", "연간 탄소(t)", "breakeven_dead", "occ"])
    lab = alt.Chart(d).mark_text(align="left", dx=6, fontSize=14,
                                 fontWeight=700).encode(
        y=alt.Y("차종:N", sort=None), x="연간 탄소(t):Q",
        text=alt.Text("연간 탄소(t):Q", format="+.1f"),
        color=alt.condition(alt.datum["연간 탄소(t)"] >= 0,
                            alt.value(C1), alt.value(C2)))
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(
        color="#555", strokeWidth=1.6).encode(x="x:Q")
    return (bar + zero + lab), (
        f"{fleet}대 · 승차장 {K}곳 · 현행 {bl} 대비. 이 설정의 실제"
        f" 재차인원은 {cur.occ:.2f}명, 손익분기는 공차까지 세면"
        f" {cur.breakeven_dead:.2f}명이다. 고령 의료통행만으로는 합승이 더"
        f" 생기지 않는다 — 광교의 2.43~3.24는 전 연령·전 목적 수요에서 나온"
        f" 값이다. **탄소는 똑버스 도입의 근거가 아니다. 근거는 접근성이고,"
        f" 차종 선택으로 손해를 줄이는 것이다.**")


def c_cost():
    d = ROWS[(ROWS.vi == vi) & (ROWS.k == K)].sort_values("fleet").copy()
    d["연간 비용(억)"] = d.cost_year / 1e8
    d["도달(%)"] = [100 * reach_of(K, f, BUD)["new_share"] for f in d.fleet]
    la = alt.Chart(d).mark_line(color=C1, point={"size": 90},
                                strokeWidth=3).encode(
        x=alt.X("fleet:Q", title="도입 대수",
                axis=alt.Axis(tickMinStep=1, labelFontSize=13,
                              titleFontSize=13)),
        y=alt.Y("도달(%):Q", title=f"{BUD}분 도달 (%)",
                axis=alt.Axis(titleColor=C1, labelFontSize=13,
                              titleFontSize=13)),
        tooltip=["fleet", "도달(%)", "연간 비용(억)"])
    lb = alt.Chart(d).mark_line(color=C3, point={"size": 80}, strokeWidth=2.6,
                                strokeDash=[6, 4]).encode(
        x="fleet:Q",
        y=alt.Y("연간 비용(억):Q", axis=alt.Axis(titleColor=C3,
                                            labelFontSize=13,
                                            titleFontSize=13)))
    pk = alt.Chart(pd.DataFrame({"x": [fleet]})).mark_rule(
        color="#555", strokeDash=[3, 3], strokeWidth=1.6).encode(x="x:Q")
    per = (f"{cur.cost_year / R0['gain_pop'] / 1e4:,.0f}만원"
           if R0["gain_pop"] > 0 else "—")
    return alt.layer(la, lb, pk).resolve_scale(y="independent"), (
        f"대당 표준운송원가 {sim.COST_DAY:,}원/일(수원시 2024 원가안)."
        f" 지금 설정은 연 {cur.cost_year / 1e8:.1f}억원이고, 새로 닿는 사람"
        f" 1명당 {per}이다. 도달은 어느 지점부터 완만해지는데 비용은 그대로"
        f" 직선으로 는다 — 그 꺾이는 자리가 결정의 자리다.")


CHARTS = {"대수별 도달": c_fleet, "신설 승차장 수별 도달과 보행": c_stops,
          "문앞-의원 시간 구성": c_split, "승차장별 고령인구": c_stoppop,
          "격자별 도달 변화": c_grid, "차종별 연간 탄소": c_carbon,
          "도달과 예산": c_cost}
ch, note = CHARTS[PICK]()
st.altair_chart(ch.properties(height=H), width="stretch")
st.caption(note)

# ── 표 ────────────────────────────────────────────────────────────────
t1, t2 = st.columns(2)
with t1:
    st.markdown(f"**{BUD}분 도달, 500m 격자 {len(GS)}개**")
    agg = (GS.groupby("stat", as_index=False)
             .agg(격자=("g", "count"), 고령=("pop", "sum")))
    agg["고령"] = agg["고령"].round(0)
    st.dataframe(agg.rename(columns={"stat": "구분"}), hide_index=True,
                 width="stretch")
with t2:
    st.markdown("**지정 승차장**")
    st.dataframe(pd.DataFrame([{
        "승차장": s["name"].split()[-1], "고령인구": s["pop"],
        "보행 중앙": s["walk_med"], "보행 최대": s["walk_max"]}
        for s in STOPS]), hide_index=True, width="stretch", height=250)

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

**운영 방식** — 똑버스는 보통 **거점까지만** 태우고 거기서 기존 대중교통으로
갈아탄다(고봉동 3대, 식사·향동 각 4대가 그렇다). 구간이 짧아 차량이 적게 든다.
이 화면은 그 **거점 연계**와, 집 앞에서 의원까지 바로 가는 **의원 직결**을
나란히 둔다. 거점에서 {HUBT}까지 걸리는 시간
({HUB_TXT})은
협업자 분석(ParkJH0313/goyang-senior-mobility)에서 가져왔다. 대화역은
{HUBT}이 380 m 거리라 도보다.

두 방식은 **다른 것을 묻는다.** 거점 연계는 "{HUBT}에 닿는가", 의원 직결은
"의료기관에 한 곳이라도 닿는가"다. 한 숫자로 견주지 않는다.

**056번 마을버스는 현행에 이미 들어 있다.** 분석에 쓴 대중교통 번들에
노선번호 056(`BR_3100_241299005`)이 있고, 가좌·송포 2 km 안 정류장을 지나는
노선 153개 중 하나다.

**승차장을 어떻게 골랐나** — 기존 버스정류장은 똑버스가 **그냥 쓴다**. 노선이
없는 수요응답형이라 폴이 이미 서 있으면 추가 비용이 들지 않는다. 군집 400 m 안
실재 정류장 {D['meta'].get('기존정류장', 0)}곳이 그것이고, 이것만으로 도보
12.5분 커버가 44.8%다. 그 위에 **신설 몇 곳을 더할지**를 MCLP(최대 커버 입지,
Church & ReVelle 1974)로 고른다.

**신설 자리는 차를 세울 수 있는 곳이어야 한다.** 고속국도·도시고속도로와
설계속도 60 km/h 초과 구간은 후보에서 뺀다. 편도 1차로는 빼지 않고 **「정차
여건 확인 필요」로 표시**한다 — 마을버스는 그런 길에도 잠깐 서지만, 실제
설치에는 현장 확인이 따로 필요하다.

(아래는 MCLP 정식이다.)
승차장 k곳으로 **기준 거리 {SM:,.0f} m(0.8 m/s 로 {SM / 48:.1f}분) 안에 든
고령인구를 최대화**한다.

```
max Σ aᵢyᵢ   s.t.  Σ_{{j∈Nᵢ}} xⱼ ≥ yᵢ ,  Σxⱼ = k ,  x,y ∈ {{0,1}}
aᵢ = 수요점 i 고령인구,  Nᵢ = {{ j : 보행거리(i,j) ≤ S }}
```

후보지는 실재 버스정류장과 도로망 노드를 100 m 격자로 솎은 것이다. 보행거리는
OSM 보행망 최단경로이고, 이 지역에서 **직선의 1.34배**(상위 10%는 1.80배)다 —
논·하천·자유로가 가로막기 때문이다. 같은 값을 내는 해가 여럿일 때는 실재
정류장과 짧은 보행을 선호하도록 아주 작은 가중치를 둬서 해를 결정한다. 그래서
승차장 수를 바꿔도 지도가 임의로 뒤집히지 않는다.

**커버 밖 사람도 걸어서 탄다.** MCLP 는 어디에 놓을지만 정하고, 30분 도달은
커버 여부와 무관하게 실제 보행시간으로 따로 센다.

**승차장 선정 방식 비교** — 예전에 쓰던 고령인구 가중 k-평균과 견주면,
같은 보행망 자로 재어 신설 10곳 기준 커버 54.3% 대 68.5%, 보행 중앙 620 m 대
543 m 로 MCLP 가 낫다.

**그래도 임시 가정이다** — 실제 선정에는 정차 가능 지점, 보도·횡단 여건,
소유관계 확인이 따로 필요하다. 설계속도가 높은 도로변 후보는 정차 가능성을
별도로 확인해야 한다.
""")

st.caption(
    f"고양시 고령자 의료접근성 연구 v6 · 시나리오 {len(ROWS):,}개 "
    f"(차종 {len(VEH)} × 대수 12 × 승차장 13) × 예산 3 · 자료 생성 "
    f"{D['meta']['생성']} · 운송원가·재차인원 수원시정연구원 2024"
    f"(광교 똑버스 호출 329,501건) · 배경지도 카카오맵")
