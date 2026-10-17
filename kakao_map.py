# -*- coding: utf-8 -*-
"""카카오맵 지도 — Streamlit 커스텀 컴포넌트.

왜 따로 만드는가
  카카오는 지도를 자체 JS SDK 로만 내보내고 타일 주소를 열어 두지 않는다.
  그래서 deck.gl 바탕으로 끼울 수 없고, 지도 전체를 카카오 SDK 로 다시 그린다.

필요한 키
  **JavaScript 키**다. 길찾기에 쓰는 REST 키와 다르다.
    https://developers.kakao.com → 내 애플리케이션 → 앱 키 → JavaScript 키
  그리고 **플랫폼 → Web → 사이트 도메인**에 쓰는 주소를 등록해야 한다.
    로컬      http://localhost:8501
    배포      https://<앱이름>.streamlit.app
  등록하지 않으면 지도가 뜨지 않는다.

좌표
  카카오는 WGS84 위경도를 LatLng(위도, 경도) 순서로 받는다. 이 연구는
  EPSG:5179 를 쓰므로 넘기기 직전에만 바꾼다.
"""
from __future__ import annotations

import json

import streamlit as st
import streamlit.components.v1 as components

import keys

SDK = ("https://dapi.kakao.com/v2/maps/sdk.js"
       "?appkey={key}&autoload=false&libraries=services")


def js_key() -> str | None:
    return keys.get("KAKAO_JS_KEY", "KAKAO_JAVASCRIPT_KEY", "KAKAO_MAP_KEY")


def render(payload: dict, height: int = 520) -> None:
    """payload 를 그린다. 키가 없으면 안내만 띄우고 False 를 남긴다.

    payload 의 모양
      center   [lat, lng]
      level    카카오 확대 수준 (작을수록 확대)
      grids    [{poly:[[lat,lng]..], color, opacity, title, body}]
      stops    [{lat, lng, label, title, body, r}]
      dests    [{lat, lng, title, body, r}]
      paths    [{line:[[lat,lng]..], color, weight, opacity}]
      dongs    [[[lat,lng]..]]
    """
    key = js_key()
    if not key:
        st.warning(
            "**지도를 띄울 카카오 JavaScript 키가 이 서버에 없습니다.**\n\n"
            "키는 저장소에 올리지 않으므로(`.env` 는 `.gitignore` 대상), "
            "배포한 곳에는 따로 넣어야 합니다.\n\n"
            "**Streamlit Cloud 라면** — 앱 오른쪽 아래 ⋮ → *Settings* → "
            "*Secrets* 에 아래 한 줄을 넣고 저장하면 앱이 다시 뜹니다.\n"
            "```toml\nKAKAO_JS_KEY = \"발급받은_JavaScript_키\"\n```\n"
            "**내 컴퓨터에서 돌린다면** — `.env` 에 "
            "`KAKAO_JS_KEY=발급받은키` 를 추가하세요.\n\n"
            "그리고 카카오 개발자 콘솔의 **앱 → 플랫폼 키 → JavaScript 키 → "
            "JavaScript SDK 도메인**에 **이 사이트 주소**가 등록돼 있어야 "
            "합니다. 등록된 주소에서만 지도가 뜹니다.",
            icon="🗝️")
        return
    html = _HTML.replace("__SDK__", SDK.format(key=key))
    html = html.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    html = html.replace("__H__", str(height))
    _embed(html, height)


def _embed(html: str, height: int) -> None:
    """components.v1.html 은 폐기 예고가 붙었다. 사라지면 st.html 로 넘어간다.

    st.iframe 은 주소만 받으므로 HTML 문자열에는 쓸 수 없고, data: URI 는
    출처가 비어 Referer 가 안 나가 카카오 도메인 검사에 걸린다. 그래서
    iframe(components.html) → 본문 삽입(st.html) 순으로 간다.
    """
    try:
        components.html(html, height=height + 6, scrolling=False)
        return
    except Exception:
        pass
    try:
        st.html(html, unsafe_allow_javascript=True)
    except Exception as e:
        st.error(f"지도를 띄우지 못했습니다 ({type(e).__name__}). "
                 "배경지도에서 VWorld 를 쓰세요.", icon="🗺️")


_HTML = r"""
<div id="wrap" style="position:relative;width:100%;height:__H__px">
  <div id="map" style="width:100%;height:100%;border-radius:8px"></div>
  <div id="msg" style="position:absolute;inset:0;display:grid;
       place-items:center;background:#f3f6f7;border-radius:8px;
       font:14px/1.6 'Malgun Gothic',sans-serif;color:#53676f;
       text-align:center;padding:24px">카카오 지도를 불러오는 중…</div>
  <div id="tip" style="position:absolute;left:10px;bottom:10px;
       background:rgba(255,255,255,.94);border:1px solid #dde5e8;
       border-radius:6px;padding:8px 11px;font:12px/1.5 'Malgun Gothic',
       sans-serif;color:#16232b;max-width:62%;display:none"></div>
</div>
<script src="__SDK__"></script>
<script>
const D = __DATA__;
const msg = document.getElementById('msg');
const tip = document.getElementById('tip');

function fail(t){ msg.style.display='grid'; msg.innerHTML = t; }

// SDK 가 아예 안 뜨면(도메인 미등록·키 오류) 알려 준다
if (typeof kakao === 'undefined' || !kakao.maps) {
  fail("카카오 지도를 불러오지 못했습니다.<br><br>"
     + "JavaScript 키가 맞는지, 그리고 카카오 개발자 콘솔의<br>"
     + "<b>플랫폼 → Web → 사이트 도메인</b>에<br>"
     + "<code>" + location.origin + "</code> 가 등록돼 있는지 확인하세요.");
} else {
  kakao.maps.load(function () {
    try { draw(); } catch (e) { fail("지도를 그리다 막혔습니다: " + e.message); }
  });
}

function draw() {
  msg.style.display = 'none';
  const map = new kakao.maps.Map(document.getElementById('map'), {
    center: new kakao.maps.LatLng(D.center[0], D.center[1]),
    level: D.level || 7,
  });
  map.addControl(new kakao.maps.MapTypeControl(),
                 kakao.maps.ControlPosition.TOPRIGHT);
  map.addControl(new kakao.maps.ZoomControl(),
                 kakao.maps.ControlPosition.RIGHT);
  const LL = p => new kakao.maps.LatLng(p[0], p[1]);

  // 동 경계
  (D.dongs || []).forEach(function (ring) {
    new kakao.maps.Polyline({
      map: map, path: ring.map(LL), strokeWeight: 2,
      strokeColor: '#b8540f', strokeOpacity: 0.6, strokeStyle: 'solid',
    });
  });

  // 격자 — 도달 상태를 색으로
  (D.grids || []).forEach(function (g) {
    const poly = new kakao.maps.Polygon({
      map: map, path: g.poly.map(LL), strokeWeight: 1,
      strokeColor: '#ffffff', strokeOpacity: 0.7,
      fillColor: g.color, fillOpacity: g.opacity,
    });
    kakao.maps.event.addListener(poly, 'mouseover', function () {
      poly.setOptions({ fillOpacity: Math.min(0.95, g.opacity + 0.25) });
      show(g.title, g.body);
    });
    kakao.maps.event.addListener(poly, 'mouseout', function () {
      poly.setOptions({ fillOpacity: g.opacity });
    });
  });

  // 승차장에서 의원까지의 경로
  (D.paths || []).forEach(function (p) {
    new kakao.maps.Polyline({
      map: map, path: p.line.map(LL), strokeWeight: p.weight || 3,
      strokeColor: p.color || '#16707f',
      strokeOpacity: p.opacity == null ? 0.7 : p.opacity,
      strokeStyle: 'solid',
    });
  });

  // 의료기관
  (D.dests || []).forEach(function (d) {
    const c = new kakao.maps.Circle({
      map: map, center: LL([d.lat, d.lng]), radius: d.r || 90,
      strokeWeight: 1, strokeColor: '#6b4a9e', strokeOpacity: 0.8,
      fillColor: '#6b4a9e', fillOpacity: 0.55,
    });
    kakao.maps.event.addListener(c, 'mouseover',
      function () { show(d.title, d.body); });
  });

  // 지정 승차장 — 원 + 번호
  (D.stops || []).forEach(function (s) {
    const c = new kakao.maps.Circle({
      map: map, center: LL([s.lat, s.lng]), radius: s.r || 260,
      strokeWeight: 2, strokeColor: '#ffffff', strokeOpacity: 0.95,
      fillColor: '#16707f', fillOpacity: 0.82,
    });
    new kakao.maps.CustomOverlay({
      map: map, position: LL([s.lat, s.lng]), yAnchor: 0.5, zIndex: 5,
      content: '<div style="font:700 11px/1 \'IBM Plex Mono\',monospace;'
             + 'color:#fff;pointer-events:none">' + s.label + '</div>',
    });
    kakao.maps.event.addListener(c, 'mouseover',
      function () { show(s.title, s.body); });
  });

  function show(t, b) {
    tip.innerHTML = '<b>' + t + '</b>' + (b ? '<br>' + b : '');
    tip.style.display = 'block';
  }
  if (D.legend) show(D.legend.title, D.legend.body);
}
</script>
"""
