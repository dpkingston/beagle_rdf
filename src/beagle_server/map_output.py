# Copyright (c) 2026 Douglas P. Kingston III. MIT License - see LICENSE.
"""
Folium map generation for the TDOA aggregation server.

The map is split into two parts for dynamic updates:

1.  ``build_map()`` - returns the static Folium HTML shell: node markers,
    sync-TX markers, and the heatmap layer.  Fix markers and hyperbola arcs
    are NOT embedded here; they are loaded dynamically by the browser.

2.  ``build_fix_geojson()`` - returns a GeoJSON FeatureCollection served
    by ``GET /map/data``.  The page JS calls this endpoint on load and on
    every SSE ``new_fix`` event, then renders the features as Leaflet layers
    without a full page reload.  Age-preset buttons let the user choose the
    age window (1 m, 5 m, 15 m, 1 h, 6 h, 24 h, ALL) dynamically.
"""

from __future__ import annotations

import json
import logging
import math
import time
from typing import Any

_logger = logging.getLogger(__name__)

import folium  # type: ignore[import-untyped]
from branca.element import Element as _BrancaElement  # type: ignore[import-untyped]
from folium.plugins import HeatMap  # type: ignore[import-untyped]

from beagle_server.tdoa import _C_M_S, haversine_m

# ---------------------------------------------------------------------------
# Page chrome: favicon (bullseye SVG) + title
# ---------------------------------------------------------------------------
# Bullseye: red -> white -> red -> white, 4 concentric circles.
# Only # must be percent-encoded in SVG data URLs.
_FAVICON_HTML = (
    "<link rel='icon' type='image/svg+xml' href=\"data:image/svg+xml,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'>"
    "<circle cx='50' cy='50' r='50' fill='%23d73027'/>"
    "<circle cx='50' cy='50' r='36' fill='white'/>"
    "<circle cx='50' cy='50' r='22' fill='%23d73027'/>"
    "<circle cx='50' cy='50' r='8'  fill='white'/>"
    "</svg>\">"
)

# ---------------------------------------------------------------------------
# Control panel injected into every map page
#
# Layout (top-right corner, ~220 px wide):
#
#   +-----------------------------+
#   | Beagle  host:port     LIVE  |   <- header
#   +-----------------------------+
#   | Time        14:23           |
#   | Last fix    3 min ago       |   <- status rows
#   | Fixes       47 fixes        |   <- updated by loadFixes()
#   +-----------------------------+
#   | [1m][5m][15m][1h][6h][24h][ALL] |  <- age preset buttons
#   +-----------------------------+
#   | [ Hide Heat Map    ]        |
#   | [ Reset Fix History]        |   <- action buttons
#   | [ Reset Heat Map   ]        |
#   +-----------------------------+
#
# All dynamic values are passed via a single TDOA={...} JS object so that
# adding new controls only requires:
#   1. New key(s) in the TDOA data block built by _render_control_panel()
#   2. New rows/buttons in _PANEL_HTML
#   3. New JS in _PANEL_JS that reads from TDOA.*
# ---------------------------------------------------------------------------

# CSS lives in its own raw string - no f-string brace escaping needed.
_PANEL_CSS = """<style>
#tdoa-panel {
    position: fixed; top: 10px; right: 10px; z-index: 9999;
    background: rgba(22,25,37,0.93);
    color: #d8dbe8; font: 12px/1.7 monospace;
    border-radius: 7px; box-shadow: 0 3px 12px rgba(0,0,0,.55);
    min-width: 215px; max-width: 270px;
    pointer-events: auto;
}
#tdoa-panel .tp-hdr {
    background: rgba(25,75,170,0.92);
    padding: 5px 10px; border-radius: 7px 7px 0 0;
    display: flex; justify-content: space-between; align-items: center;
    font-weight: bold; font-size: 12px;
}
#tdoa-panel .tp-hdr-right { display: flex; align-items: center; gap: 6px; }
#tdoa-panel .tp-host { font-size: 10px; color: #9ab; font-weight: normal; }
#tdoa-live {
    font-size: 10px; font-weight: bold; padding: 1px 7px;
    border-radius: 3px; background: rgba(70,70,80,0.9);
    transition: background 0.4s;
}
#tdoa-panel .tp-body { padding: 5px 10px 3px; }
#tdoa-panel .tp-row {
    display: flex; justify-content: space-between; margin: 0;
}
#tdoa-panel .tp-key { color: #7a9bbf; }
#tdoa-panel .tp-age-row {
    display: flex; flex-wrap: wrap; gap: 3px;
    padding: 4px 10px 6px;
    border-top: 1px solid rgba(255,255,255,.08);
}
.tdoa-age-btn {
    flex: 1; min-width: 26px; padding: 3px 1px;
    background: rgba(30,50,90,0.8); color: #8ab;
    border: 1px solid rgba(80,110,160,0.3); border-radius: 3px;
    cursor: pointer; font: 10px monospace; text-align: center;
}
.tdoa-age-btn:hover { background: rgba(40,70,130,0.9); color: #cdf; }
.tdoa-age-btn.tdoa-age-active {
    background: rgba(20,100,170,0.9); color: #e8f4ff;
    border-color: rgba(80,160,230,0.6);
}
#tdoa-panel .tp-actions {
    padding: 5px 10px 8px;
    border-top: 1px solid rgba(255,255,255,.08);
    display: flex; flex-direction: column; gap: 4px;
}
.tdoa-btn {
    width: 100%; padding: 4px 0;
    background: rgba(155,35,35,.85); color: #f0e8e8;
    border: none; border-radius: 3px;
    cursor: pointer; font: 11px monospace; letter-spacing: .03em;
}
.tdoa-btn:hover { background: rgba(210,45,45,.9); }
.tdoa-btn:disabled { opacity: .45; cursor: default; }
.tdoa-btn-toggle {
    background: rgba(20,80,160,0.85); color: #e8f0ff;
}
.tdoa-btn-toggle:hover { background: rgba(30,110,210,0.9); }
#tdoa-panel .tp-window {
    padding: 5px 10px 6px;
    border-top: 1px solid rgba(255,255,255,.08);
    display: flex; flex-direction: column; gap: 3px;
}
#tdoa-panel input[type="datetime-local"] {
    flex: 1; min-width: 0;
    background: rgba(20,30,55,0.95); color: #c8d4e8;
    border: 1px solid rgba(80,110,160,0.4); border-radius: 3px;
    font: 10px monospace; padding: 2px 3px;
    color-scheme: dark;
}
#tdoa-panel .tp-window-btns { display: flex; gap: 4px; margin-top: 1px; }
.tdoa-window-btn {
    flex: 1; padding: 3px 1px;
    background: rgba(30,50,90,0.8); color: #8ab;
    border: 1px solid rgba(80,110,160,0.3); border-radius: 3px;
    cursor: pointer; font: 10px monospace; text-align: center;
}
.tdoa-window-btn:hover { background: rgba(40,70,130,0.9); color: #cdf; }
.tdoa-window-btn.tdoa-win-active {
    background: rgba(20,100,170,0.9); color: #e8f4ff;
    border-color: rgba(80,160,230,0.6);
}
/* --- Tab bar ---- */
#tdoa-panel .tp-tabs {
    display: flex;
    border-bottom: 1px solid rgba(255,255,255,.08);
}
.tdoa-tab {
    flex: 1; padding: 5px 0;
    background: transparent; color: #7a9bbf;
    border: none; border-bottom: 2px solid transparent;
    cursor: pointer; font: 11px monospace; text-align: center;
}
.tdoa-tab:hover { color: #c8d4e8; }
.tdoa-tab.tdoa-tab-active {
    color: #e8f4ff;
    border-bottom-color: rgba(80,160,230,0.8);
}
/* --- Node cards --- */
#tdoa-tab-nodes { padding: 4px 0 4px; }
#tdoa-node-bulk { display: flex; gap: 4px; padding: 4px 10px 4px; }
#tdoa-node-list { padding: 0 8px 4px; max-height: calc(100vh - 200px); overflow-y: auto; }
.tp-node-loading { color: #7a9bbf; font-size: 10px; padding: 8px 0; text-align: center; }
.tdoa-node-card {
    margin: 4px 0; padding: 5px 8px;
    background: rgba(20,30,55,0.7);
    border-radius: 4px;
    border: 1px solid rgba(80,110,160,0.2);
    font-size: 11px;
}
.tp-node-head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 2px;
}
.tp-node-id { font-weight: bold; color: #c8d4e8; }
.tp-node-dot {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block; margin-right: 4px; flex-shrink: 0;
}
.ns-online   { background: #2ecc71; }
.ns-stale    { background: #e67e22; }
.ns-offline  { background: #e74c3c; }
.ns-disabled { background: #666; }
.ns-notseen  { background: #f1c40f; }
.ns-unknown  { background: #888; }
.tp-node-meta { color: #7a9bbf; font-size: 10px; line-height: 1.5; }
.tp-node-actions { display: flex; gap: 4px; margin-top: 4px; }
.tdoa-btn-sm {
    flex: 1; padding: 2px 4px;
    border: none; border-radius: 3px;
    cursor: pointer; font: 10px monospace;
}
.tdoa-btn-sm:disabled { opacity: .45; cursor: default; }
.ton  { background: rgba(20,120,60,0.85); color: #e8f4ee; }
.ton:hover:not(:disabled) { background: rgba(30,160,80,0.9); }
.toff { background: rgba(90,50,10,0.85); color: #f0e8e0; }
.toff:hover:not(:disabled) { background: rgba(140,70,15,0.9); }
.tdel { background: rgba(110,20,20,0.85); color: #f0dede; }
.tdel:hover:not(:disabled) { background: rgba(190,35,35,0.9); }
/* --- Group cards --- */
#tdoa-tab-groups { padding: 4px 0 4px; }
#tdoa-group-list { padding: 0 8px 4px; max-height: calc(100vh - 200px); overflow-y: auto; }
.tdoa-group-card {
    margin: 4px 0; padding: 5px 8px;
    background: rgba(20,30,55,0.7);
    border-radius: 4px;
    border-left: 3px solid rgba(80,110,160,0.5);
    font-size: 11px; cursor: pointer;
}
.tdoa-group-card:hover { background: rgba(30,45,80,0.8); }
.tdoa-group-card.tg-selected { border-left-color: #2ecc71; background: rgba(30,55,70,0.8); }
.tp-grp-head {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 2px;
}
.tp-grp-id { font-weight: bold; color: #c8d4e8; }
.tp-grp-meta { color: #7a9bbf; font-size: 10px; line-height: 1.5; }
.tp-grp-detail { padding: 4px 10px 6px; border-top: 1px solid rgba(255,255,255,.08); }
.tp-grp-detail .tp-row { font-size: 11px; }
.tp-grp-members { padding: 2px 0; }
.tp-grp-member-tag {
    display: inline-block; padding: 1px 6px; margin: 1px 2px;
    background: rgba(30,60,100,0.7); color: #9ab; border-radius: 3px;
    font-size: 10px;
}
.tp-grp-assign { padding: 4px 10px 6px; }
.tp-grp-assign select {
    width: 100%; margin: 3px 0;
    background: rgba(20,30,55,0.95); color: #c8d4e8;
    border: 1px solid rgba(80,110,160,0.4); border-radius: 3px;
    font: 10px monospace; padding: 2px 3px;
}
.tp-grp-actions { display: flex; gap: 4px; margin-top: 4px; }
.tp-grp-color {
    width: 10px; height: 10px; border-radius: 50%;
    display: inline-block; margin-right: 4px; flex-shrink: 0;
}
.tp-node-grp-tag {
    display: inline-block; padding: 0 4px; margin-left: 4px;
    border-radius: 2px; font-size: 9px;
    background: rgba(40,80,130,0.6); color: #9bc;
}
/* --- Group form --- */
.tp-grp-form { padding: 6px 10px; font-size: 11px; }
.tp-grp-form label { display: block; color: #7a9bbf; font-size: 10px; margin-top: 4px; }
.tp-grp-form input, .tp-grp-form textarea {
    width: 100%; box-sizing: border-box; margin: 1px 0 2px;
    background: rgba(15,25,45,0.95); color: #c8d4e8;
    border: 1px solid rgba(80,110,160,0.4); border-radius: 3px;
    font: 11px monospace; padding: 3px 5px;
}
.tp-grp-form textarea { resize: vertical; min-height: 22px; }
.tp-grp-ch-row {
    display: flex; gap: 4px; align-items: center; margin: 2px 0;
}
.tp-grp-ch-row input { flex: 1; }
.tp-grp-ch-rm { cursor: pointer; color: #e74c3c; font-size: 14px; padding: 0 2px; }
.tp-grp-form-btns { display: flex; gap: 4px; margin-top: 6px; }
/* --- Member unassign --- */
.tp-grp-member-rm {
    cursor: pointer; color: #e74c3c; font-size: 11px;
    padding-left: 3px; opacity: 0.7;
}
.tp-grp-member-rm:hover { opacity: 1; }
/* --- Secret modal --- */
.tdoa-secret-overlay {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.7); z-index: 20000;
    display: flex; justify-content: center; align-items: center;
}
.tdoa-secret-card {
    background: #1a2540; border: 1px solid rgba(80,110,160,0.5);
    border-radius: 6px; padding: 16px 20px; max-width: 500px; width: 90%;
    color: #c8d4e8; font-size: 12px;
}
.tdoa-secret-card h3 { margin: 0 0 8px; font-size: 13px; color: #e8f0ff; }
.tdoa-secret-box {
    background: rgba(10,15,30,0.95); border: 1px solid rgba(80,110,160,0.3);
    border-radius: 3px; padding: 8px 10px; margin: 6px 0;
    font: 12px monospace; color: #2ecc71; word-break: break-all;
    user-select: all; -webkit-user-select: all;
}
.tdoa-secret-warn { color: #e67e22; font-size: 10px; margin: 6px 0; }
.tdoa-secret-btns { display: flex; gap: 6px; margin-top: 10px; }
/* --- Node inline edit & detail --- */
.tp-node-edit-label { cursor: pointer; color: #7a9bbf; margin-left: 4px; font-size: 10px; }
.tp-node-edit-label:hover { color: #aac; }
.tp-node-detail {
    padding: 4px 8px 6px; margin-top: 3px;
    border-top: 1px solid rgba(255,255,255,.08); font-size: 11px;
}
.tp-node-detail .tp-row { margin: 1px 0; }
.tp-node-config-ta {
    width: 100%; box-sizing: border-box; min-height: 60px;
    background: rgba(10,15,30,0.95); color: #c8d4e8;
    border: 1px solid rgba(80,110,160,0.3); border-radius: 3px;
    font: 10px monospace; padding: 4px 5px; resize: vertical;
    margin: 3px 0;
}
/* --- Carrier threshold display --- */
.tp-carrier-row { color: #7a9bbf; font-size: 10px; line-height: 1.5; margin-top: 1px; }
.tp-carrier-row .tp-margin { font-weight: bold; }
.tp-margin-good { color: #2ecc71; }
.tp-margin-warn { color: #f1c40f; }
.tp-margin-bad  { color: #e74c3c; }
/* --- Carrier threshold form in detail panel --- */
.tp-carrier-form { margin: 6px 0 4px; padding: 4px 0; border-top: 1px solid rgba(255,255,255,.06); }
.tp-carrier-form label { display: inline-block; color: #7a9bbf; font-size: 10px; width: 110px; }
.tp-carrier-form input[type=number] {
    width: 70px; background: rgba(10,15,30,0.95); color: #c8d4e8;
    border: 1px solid rgba(80,110,160,0.3); border-radius: 3px;
    font: 10px monospace; padding: 2px 4px;
}
.tp-carrier-form .tp-carrier-btns { margin-top: 4px; display: flex; gap: 6px; }
/* --- Node register form --- */
.tp-node-reg-form { padding: 4px 8px 6px; font-size: 11px; }
.tp-node-reg-form input {
    width: 100%; box-sizing: border-box; margin: 1px 0 3px;
    background: rgba(15,25,45,0.95); color: #c8d4e8;
    border: 1px solid rgba(80,110,160,0.4); border-radius: 3px;
    font: 11px monospace; padding: 3px 5px;
}
.tp-node-reg-form label { display: block; color: #7a9bbf; font-size: 10px; margin-top: 3px; }
/* --- Login overlay --- */
.tdoa-login-overlay {
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.75); z-index: 20000;
    display: flex; justify-content: center; align-items: center;
}
.tdoa-login-card {
    background: #1a2540; border: 1px solid rgba(80,110,160,0.5);
    border-radius: 8px; padding: 20px 24px; max-width: 320px; width: 90%;
    color: #c8d4e8; font-size: 12px;
}
.tdoa-login-card h3 { margin: 0 0 12px; font-size: 14px; color: #e8f0ff; text-align: center; }
.tdoa-login-card label { display: block; color: #7a9bbf; font-size: 10px; margin-top: 8px; }
.tdoa-login-card input[type=text],
.tdoa-login-card input[type=password] {
    width: 100%; box-sizing: border-box; margin: 2px 0 4px;
    background: rgba(10,15,30,0.95); color: #c8d4e8;
    border: 1px solid rgba(80,110,160,0.4); border-radius: 3px;
    font: 12px monospace; padding: 6px 8px;
}
.tdoa-login-error { color: #e74c3c; font-size: 10px; margin: 4px 0; display: none; }
.tdoa-login-card .tdoa-btn { width: 100%; margin-top: 10px; }
/* --- User info in header --- */
#tdoa-user-info { font-size: 9px; color: #9ab; font-weight: normal; }
#tdoa-logout-btn { font-size: 9px; padding: 1px 6px; cursor: pointer; }
/* --- Users tab --- */
.tdoa-user-card {
    padding: 5px 10px; border-bottom: 1px solid rgba(255,255,255,.06);
    font-size: 11px;
}
.tdoa-user-card:hover { background: rgba(255,255,255,.03); }
.tdoa-role-badge {
    font-size: 9px; padding: 1px 5px; border-radius: 3px;
    font-weight: bold; margin-left: 4px;
}
.tdoa-role-admin { background: rgba(52,152,219,.3); color: #5dade2; }
.tdoa-role-viewer { background: rgba(149,165,166,.25); color: #95a5a6; }
.tdoa-user-meta { color: #7a9bbf; font-size: 9px; }
.tdoa-user-actions { margin-top: 3px; display: flex; gap: 4px; flex-wrap: wrap; }
.tdoa-user-form { padding: 6px 10px; font-size: 11px; }
.tdoa-user-form input {
    width: 100%; box-sizing: border-box; margin: 1px 0 3px;
    background: rgba(15,25,45,0.95); color: #c8d4e8;
    border: 1px solid rgba(80,110,160,0.4); border-radius: 3px;
    font: 11px monospace; padding: 3px 5px;
}
.tdoa-user-form label { display: block; color: #7a9bbf; font-size: 10px; margin-top: 3px; }
.tdoa-user-form select {
    background: rgba(15,25,45,0.95); color: #c8d4e8;
    border: 1px solid rgba(80,110,160,0.4); border-radius: 3px;
    font: 11px monospace; padding: 3px 5px;
}
</style>"""

# Static HTML shell - dynamic values are filled by JS on load.
_PANEL_HTML = """
<div id="tdoa-login-overlay" class="tdoa-login-overlay" style="display:none">
  <div class="tdoa-login-card">
    <h3>Beagle Login</h3>
    <label>Username</label>
    <input type="text" id="tdoa-login-user" autocomplete="username">
    <label>Password</label>
    <input type="password" id="tdoa-login-pass" autocomplete="current-password">
    <div id="tdoa-login-2fa" style="display:none">
      <label>Authentication Code</label>
      <input type="text" id="tdoa-login-code" autocomplete="one-time-code"
             placeholder="6-digit code" maxlength="6" inputmode="numeric">
    </div>
    <div class="tdoa-login-error" id="tdoa-login-error"></div>
    <button class="tdoa-btn" id="tdoa-login-btn">Login</button>
    <div id="tdoa-google-login" style="display:none;margin-top:10px;text-align:center">
      <div style="color:#7a9bbf;font-size:10px;margin:6px 0">or</div>
      <a href="/auth/oauth/google" class="tdoa-btn" style="display:inline-block;text-decoration:none;text-align:center;width:100%;box-sizing:border-box">Sign in with Google</a>
    </div>
  </div>
</div>
<div id="tdoa-panel">
  <div class="tp-hdr">
    Beagle
    <div class="tp-hdr-right">
      <span id="tdoa-user-info"></span>
      <button class="tdoa-btn-sm toff" id="tdoa-logout-btn" style="display:none">Logout</button>
      <span class="tp-host" id="tdoa-host"></span>
      <span id="tdoa-live">...</span>
    </div>
  </div>
  <div class="tp-tabs">
    <button class="tdoa-tab tdoa-tab-active" id="tdoa-tab-btn-fixes">Fixes</button>
    <button class="tdoa-tab" id="tdoa-tab-btn-nodes">Nodes</button>
    <button class="tdoa-tab" id="tdoa-tab-btn-groups">Groups</button>
    <button class="tdoa-tab" id="tdoa-tab-btn-users" style="display:none">Users</button>
  </div>
  <div id="tdoa-tab-fixes">
    <div class="tp-body">
      <div class="tp-row"><span class="tp-key">Time</span><span id="tdoa-time">--:--</span></div>
      <div class="tp-row"><span class="tp-key">Last fix</span><span id="tdoa-last-fix">--</span></div>
      <div class="tp-row"><span class="tp-key">Fixes</span><span id="tdoa-fix-count">--</span></div>
      <div class="tp-row"><span class="tp-key">Visible</span><span id="tdoa-hide-status">All</span></div>
    </div>
    <div class="tp-age-row">
      <button class="tdoa-age-btn" data-age="60">1m</button>
      <button class="tdoa-age-btn" data-age="300">5m</button>
      <button class="tdoa-age-btn" data-age="900">15m</button>
      <button class="tdoa-age-btn" data-age="3600">1h</button>
      <button class="tdoa-age-btn" data-age="21600">6h</button>
      <button class="tdoa-age-btn" data-age="86400">24h</button>
      <button class="tdoa-age-btn" data-age="0">ALL</button>
    </div>
    <div class="tp-window">
      <div class="tp-row">
        <span class="tp-key">From</span>
        <input type="datetime-local" id="tdoa-window-from">
      </div>
      <div class="tp-row">
        <span class="tp-key">To&nbsp;&nbsp;</span>
        <input type="datetime-local" id="tdoa-window-to">
      </div>
      <div class="tp-window-btns">
        <button class="tdoa-window-btn" id="tdoa-window-set-btn">Set Window</button>
        <button class="tdoa-window-btn" id="tdoa-window-clear-btn">Clear</button>
      </div>
    </div>
    <div class="tp-actions">
      <button class="tdoa-btn tdoa-btn-toggle" id="tdoa-lop-toggle-btn">Hide LOPs</button>
      <button class="tdoa-btn tdoa-btn-toggle" id="tdoa-heatmap-toggle-btn">Hide Heat Map</button>
      <button class="tdoa-btn" id="tdoa-hide-btn">Hide Fixes</button>
      <button class="tdoa-btn tdoa-btn-toggle" id="tdoa-unhide-btn">Unhide All</button>
      <button class="tdoa-btn" id="tdoa-heatmap-reset-btn">Reset Heat Map</button>
    </div>
  </div>
  <div id="tdoa-tab-nodes" style="display:none">
    <div id="tdoa-event-auth-row" style="padding:4px 10px 5px;border-bottom:1px solid rgba(255,255,255,.08)">
      <div class="tp-row">
        <span class="tp-key">Event auth</span>
        <span id="tdoa-event-auth-val" style="font-size:10px">&#8230;</span>
      </div>
      <button class="tdoa-btn-sm" id="tdoa-event-auth-btn" style="margin-top:4px;width:100%">&#8230;</button>
    </div>
    <div id="tdoa-node-bulk">
      <button class="tdoa-btn-sm ton" id="tdoa-enable-all-btn">Enable All</button>
      <button class="tdoa-btn-sm toff" id="tdoa-disable-all-btn">Disable All</button>
      <button class="tdoa-btn-sm ton" id="tdoa-register-node-btn">+ Register</button>
    </div>
    <div id="tdoa-node-reg-area" style="display:none"></div>
    <div id="tdoa-node-list">
      <div class="tp-node-loading">Loading&#8230;</div>
    </div>
  </div>
  <div id="tdoa-tab-groups" style="display:none">
    <div style="padding:4px 10px">
      <button class="tdoa-btn-sm ton" id="tdoa-create-group-btn">+ Create Group</button>
    </div>
    <div id="tdoa-group-list">
      <div class="tp-node-loading">Loading&#8230;</div>
    </div>
    <div id="tdoa-group-detail" style="display:none"></div>
  </div>
  <div id="tdoa-tab-users" style="display:none">
    <div id="tdoa-user-bulk" style="padding:4px 10px">
      <button class="tdoa-btn-sm ton" id="tdoa-create-user-btn">+ Create User</button>
    </div>
    <div id="tdoa-user-create-area" style="display:none"></div>
    <div id="tdoa-user-list">
      <div class="tp-node-loading">Loading&#8230;</div>
    </div>
    <div id="tdoa-user-chpw-area" style="display:none;padding:4px 10px">
      <div style="border-top:1px solid rgba(255,255,255,.08);padding-top:6px;margin-top:4px">
        <span style="font-size:10px;color:#7a9bbf">Change your password</span>
        <input type="password" id="tdoa-chpw-input" placeholder="New password (min 8 chars)"
               style="width:100%;box-sizing:border-box;margin:2px 0;background:rgba(15,25,45,0.95);color:#c8d4e8;border:1px solid rgba(80,110,160,0.4);border-radius:3px;font:11px monospace;padding:3px 5px">
        <div id="tdoa-chpw-error" style="color:#e74c3c;font-size:10px;display:none"></div>
        <button class="tdoa-btn-sm ton" id="tdoa-chpw-btn">Update Password</button>
      </div>
    </div>
  </div>
</div>"""

# Pure JS - reads from the TDOA data object injected below.
# Uses only var/function (ES5-compatible) for broad browser support.
_PANEL_JS = """<script>
(function () {
'use strict';

/* - Guard: TDOA data block must have loaded before this script - */
if (typeof TDOA === 'undefined') {
    console.error('[Beagle] TDOA data block not found - panel disabled');
    return;
}

/* - Module state - */
var currentMaxAgeS = TDOA.defaultMaxAgeS;
var leafletMap;        /* set after window 'load' fires */
var _fixLayers = [];   /* Leaflet layers added by loadFixes(); cleared each call */
var _lopLayers = [];   /* 2-node LOP arcs; toggled independently */
var _lopVisible = true; /* LOP toggle state */
/* Group state - shared between Nodes and Groups tabs */
var _currentGroups = [];
var _GROUP_COLORS = [
    '#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c',
    '#1abc9c', '#f39c12', '#8e44ad', '#16a085', '#d35400'
];
function _groupColor(idx) { return _GROUP_COLORS[idx % _GROUP_COLORS.length]; }
function _groupLookup() {
    var m = {};
    for (var i = 0; i < _currentGroups.length; i++) {
        var g = _currentGroups[i];
        m[g.group_id] = { color: _groupColor(i), label: g.label || g.group_id };
    }
    return m;
}
var windowMode  = false;  /* true when a fixed [fromS, toS] window is active */
var windowFromS = 0;      /* window start: Unix seconds */
var windowToS   = 0;      /* window end:   Unix seconds */

/* - Helper: set element text, no-op if element missing - */
function setText(id, val) {
    var el = document.getElementById(id);
    if (el) el.textContent = val;
    else console.warn('[Beagle] element not found: ' + id);
}

/* - Populate static field from server data - */
setText('tdoa-host', TDOA.serverLabel);

/* - Helpers - */
function fmtAge(sec) {
    if (sec <= 0)   return 'never';
    if (sec < 90)   return Math.round(sec) + 's ago';
    if (sec < 5400) return Math.round(sec / 60) + ' min ago';
    return (Math.round(sec / 360) / 10) + 'h ago';
}

function setLive(text, bg) {
    var el = document.getElementById('tdoa-live');
    if (el) { el.textContent = text; el.style.background = bg; }
}

/* - Clock: updates every second - */
function updateClock() {
    var now = new Date();
    setText('tdoa-time',
        now.getHours().toString().padStart(2, '0') + ':' +
        now.getMinutes().toString().padStart(2, '0'));
}

/* - Last-fix age: updates every 30 s - */
function updateLastFix() {
    setText('tdoa-last-fix',
        TDOA.lastFixTs ? fmtAge(Date.now() / 1000 - TDOA.lastFixTs) : 'none');
}

/* - Hide-status label - */
function updateHideStatus() {
    var hiddenBefore = parseFloat(localStorage.getItem('tdoa_hidden_before_t') || '0') || 0;
    var el = document.getElementById('tdoa-hide-status');
    if (!el) return;
    if (hiddenBefore <= 0) {
        el.textContent = 'All';
    } else {
        var d = new Date(hiddenBefore * 1000);
        el.textContent = 'since ' +
            d.getHours().toString().padStart(2, '0') + ':' +
            d.getMinutes().toString().padStart(2, '0') + ':' +
            d.getSeconds().toString().padStart(2, '0');
    }
}

/* - Format Unix seconds as YYYY-MM-DDTHH:MM (for datetime-local inputs) - */
function toDatetimeLocal(unixSec) {
    var d = new Date(unixSec * 1000);
    var pad = function (n) { return n.toString().padStart(2, '0'); };
    return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) +
           'T' + pad(d.getHours()) + ':' + pad(d.getMinutes());
}

/* - Age colour: mirrors Python _age_color() --
   Maps age ratio 0..1 through red -> orange -> yellow -> grey. */
function ageColor(age_s, maxAgeS) {
    var t = (maxAgeS > 0) ? Math.max(0, Math.min(1, age_s / maxAgeS)) : 0;
    var stops = [
        [0.0,  [215,  48,  39]],
        [0.5,  [253, 174,  97]],
        [0.75, [255, 255, 191]],
        [1.0,  [170, 170, 170]]
    ];
    for (var i = 0; i < stops.length - 1; i++) {
        var t0 = stops[i][0],   c0 = stops[i][1];
        var t1 = stops[i+1][0], c1 = stops[i+1][1];
        if (t <= t1) {
            var frac = (t1 > t0) ? (t - t0) / (t1 - t0) : 0;
            return 'rgb(' +
                Math.round(c0[0] + frac*(c1[0]-c0[0])) + ',' +
                Math.round(c0[1] + frac*(c1[1]-c0[1])) + ',' +
                Math.round(c0[2] + frac*(c1[2]-c0[2])) + ')';
        }
    }
    return '#aaaaaa';
}

/* - loadFixes: fetch /map/data GeoJSON and render Leaflet layers --
   Called on: page load, age-preset button click, SSE new_fix event.
   Does NOT reload the page - only the fix/hyperbola layer group changes. */
function loadFixes(maxAgeS) {
    if (!leafletMap) return;
    currentMaxAgeS = maxAgeS;

    /* Highlight the matching age button (cleared when fixed window is active) */
    var btns = document.querySelectorAll('.tdoa-age-btn');
    for (var i = 0; i < btns.length; i++) {
        btns[i].classList.toggle(
            'tdoa-age-active',
            !windowMode && parseFloat(btns[i].getAttribute('data-age')) === maxAgeS
        );
    }

    /* Compute fetch parameters.
       Window mode: fixed [fromS, toS]; hiddenBefore still applies as lower bound.
       Rolling mode: age-preset window + hiddenBefore cutoff (existing logic). */
    var hiddenBefore = parseFloat(localStorage.getItem('tdoa_hidden_before_t') || '0') || 0;
    var nowSec = Date.now() / 1000;
    var fetchMaxAgeS, clientFilterToS;

    if (windowMode) {
        var effectiveFromS = Math.max(hiddenBefore, windowFromS);
        fetchMaxAgeS    = (effectiveFromS > 0) ? Math.max(nowSec - effectiveFromS, 1) : 0;
        clientFilterToS = windowToS;
    } else {
        var ageWindowStart = (maxAgeS > 0) ? (nowSec - maxAgeS) : 0;
        var lowerBound = Math.max(ageWindowStart, hiddenBefore);
        fetchMaxAgeS    = (lowerBound > 0) ? Math.max(nowSec - lowerBound, 1) : 0;
        clientFilterToS = 0;
    }

    fetch('/map/data?max_age_s=' + encodeURIComponent(fetchMaxAgeS))
        .then(function (r) {
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.json();
        })
        .then(function (geojson) {
            /* Remove previously rendered fix/hyperbola/LOP layers */
            for (var i = 0; i < _fixLayers.length; i++) {
                leafletMap.removeLayer(_fixLayers[i]);
            }
            _fixLayers = [];
            for (var i = 0; i < _lopLayers.length; i++) {
                leafletMap.removeLayer(_lopLayers[i]);
            }
            _lopLayers = [];

            var features = geojson.features || [];
            /* Client-side upper-bound filter for fixed window mode */
            if (clientFilterToS > 0) {
                var filtered = [];
                for (var fi = 0; fi < features.length; fi++) {
                    var ca = features[fi].properties.computed_at;
                    if (!ca || ca <= clientFilterToS) { filtered.push(features[fi]); }
                }
                features = filtered;
            }
            var fixCount = 0;
            var lopCount = 0;
            var newestFixTs = 0;

            for (var i = 0; i < features.length; i++) {
                var f = features[i];
                var layer;
                if (f.properties.feature_type === 'fix') {
                    fixCount++;
                    var color = ageColor(f.properties.age_s, maxAgeS);
                    var p = f.properties;
                    var t = new Date(p.onset_time_ns / 1e6);
                    var res = (p.residual_ns !== null && p.residual_ns !== undefined)
                        ? p.residual_ns.toFixed(1) + ' ns' : 'n/a';
                    var popupHtml =
                        '<b>Fix #' + p.fix_id + '</b><br>' +
                        t.toLocaleString() + '<br>' +
                        '<span style="color:#9ab;font-size:10px">' + t.toUTCString() + '</span><br>' +
                        'Lat: ' + p.latitude_deg.toFixed(5) +
                        '&nbsp;&nbsp;Lon: ' + p.longitude_deg.toFixed(5) + '<br>' +
                        'Residual: ' + res + '<br>' +
                        'Nodes (' + p.node_count + '): ' + p.nodes.join(', ') + '<br>' +
                        'Channel: ' + (p.channel_hz / 1e6).toFixed(4) + ' MHz<br>' +
                        'Type: ' + p.event_type;
                    layer = L.circleMarker(
                        [f.geometry.coordinates[1], f.geometry.coordinates[0]],
                        { radius: 6, color: color, fillColor: color,
                          fillOpacity: 0.85, weight: 1 }
                    ).bindPopup(popupHtml, { maxWidth: 320 })
                     .bindTooltip(f.properties.tooltip);
                    layer.addTo(leafletMap);
                    _fixLayers.push(layer);
                    if (f.properties.computed_at > newestFixTs) {
                        newestFixTs = f.properties.computed_at;
                    }
                } else if (f.properties.feature_type === 'hyperbola') {
                    var latlngs = f.geometry.coordinates.map(function (c) {
                        return [c[1], c[0]];
                    });
                    layer = L.polyline(latlngs,
                        { color: '#e74c3c', weight: 1.5, opacity: 0.6 }
                    ).bindTooltip(f.properties.tooltip);
                    layer.addTo(leafletMap);
                    _fixLayers.push(layer);
                } else if (f.properties.feature_type === 'lop') {
                    /* 2-node line-of-position: dashed amber */
                    lopCount++;
                    var lopLatLngs = f.geometry.coordinates.map(function (c) {
                        return [c[1], c[0]];
                    });
                    var lopAge = f.properties.age_s || 0;
                    var lopOpacity = Math.max(0.15, 0.7 - (lopAge / maxAgeS) * 0.55);
                    layer = L.polyline(lopLatLngs,
                        { color: '#e0a020', weight: 2.0, opacity: lopOpacity,
                          dashArray: '8 5' }
                    ).bindTooltip(f.properties.tooltip);
                    if (_lopVisible) { layer.addTo(leafletMap); }
                    _lopLayers.push(layer);
                } else if (f.properties.feature_type === 'node') {
                    var np = f.properties;
                    var nlat = f.geometry.coordinates[1];
                    var nlon = f.geometry.coordinates[0];
                    /* Green if heartbeat within 120s, red otherwise */
                    var hbAge = np.heartbeat_age_s;
                    var nodeOnline = (hbAge !== null && hbAge !== undefined && hbAge < 120);
                    var nColor = nodeOnline ? '#1b8a2e' : '#b03030';
                    var nFill  = nodeOnline ? '#2ecc40' : '#e74c3c';
                    var statusText = nodeOnline ? 'online' : 'offline';
                    if (hbAge === null || hbAge === undefined) statusText = 'no heartbeat';
                    /* Override fill with group color if assigned */
                    var grpInfo = _nodeGroupInfo(np.node_id);
                    var grpLine = '';
                    if (grpInfo) {
                        nColor = grpInfo.color;
                        nFill  = grpInfo.color;
                        grpLine = '<br>Group: <b>' + grpInfo.label + '</b>';
                    }
                    var nodePopup =
                        '<b>Node</b>: ' + np.node_id + '<br>' +
                        'Lat: ' + nlat.toFixed(5) + ', Lon: ' + nlon.toFixed(5) + '<br>' +
                        'Status: <b>' + statusText + '</b><br>' +
                        'Last seen: ' + fmtAge(np.age_s) + grpLine;
                    layer = L.circleMarker([nlat, nlon], {
                        radius: 8, color: nColor, fillColor: nFill,
                        fillOpacity: 0.8, weight: 2
                    }).bindPopup(nodePopup, { maxWidth: 250 })
                      .bindTooltip(np.node_id);
                    layer.addTo(leafletMap);
                    _fixLayers.push(layer);
                }
            }

            var countText = fixCount + ' fixes';
            if (lopCount > 0) countText += ', ' + lopCount + ' LOPs';
            setText('tdoa-fix-count', countText);
            if (newestFixTs) TDOA.lastFixTs = newestFixTs;
            updateLastFix();
        })
        .catch(function (e) {
            console.error('[Beagle] loadFixes error:', e);
        });
}

/* - loadHeatmap: fetch /map/heatmap and update the Leaflet.heat layer in place.
   Called on SSE new_fix so the heatmap accumulates without a page reload.
   The FeatureGroup is always present; setLatLngs([]) on an empty server
   response simply leaves the layer transparent. */
function loadHeatmap() {
    if (!TDOA.heatLayerId || !leafletMap) return;
    fetch('/map/heatmap')
        .then(function (r) {
            if (!r.ok) throw new Error('HTTP ' + r.status);
            return r.json();
        })
        .then(function (data) {
            var fg = window[TDOA.heatLayerId];
            if (!fg) return;
            var layers = fg.getLayers();
            if (layers.length > 0 && typeof layers[0].setLatLngs === 'function') {
                layers[0].setLatLngs(data.cells || []);
            }
        })
        .catch(function (e) {
            console.error('[Beagle] loadHeatmap error:', e);
        });
}

/* - Heat map toggle + age buttons + initial fix load --
   Folium emits layer scripts AFTER </body>, so Leaflet variables only
   exist after 'load' fires. */
window.addEventListener('load', function () {
    leafletMap = window[TDOA.mapId];

    /* Heat map toggle - button is always present; the layer may be empty */
    var toggleBtn = document.getElementById('tdoa-heatmap-toggle-btn');
    var heatFg = TDOA.heatLayerId ? window[TDOA.heatLayerId] : null;
    if (heatFg && leafletMap && toggleBtn) {
        toggleBtn.addEventListener('click', function () {
            if (leafletMap.hasLayer(heatFg)) {
                leafletMap.removeLayer(heatFg);
                toggleBtn.textContent = 'Show Heat Map';
            } else {
                leafletMap.addLayer(heatFg);
                toggleBtn.textContent = 'Hide Heat Map';
            }
        });
    } else {
        console.warn('[Beagle] heatmap toggle: FeatureGroup or map not ready',
            TDOA.heatLayerId, TDOA.mapId);
    }

    /* LOP toggle */
    var lopBtn = document.getElementById('tdoa-lop-toggle-btn');
    if (lopBtn) {
        lopBtn.addEventListener('click', function () {
            _lopVisible = !_lopVisible;
            lopBtn.textContent = _lopVisible ? 'Hide LOPs' : 'Show LOPs';
            for (var i = 0; i < _lopLayers.length; i++) {
                if (_lopVisible) {
                    _lopLayers[i].addTo(leafletMap);
                } else {
                    leafletMap.removeLayer(_lopLayers[i]);
                }
            }
        });
    }

    /* Age preset buttons */
    var ageBtns = document.querySelectorAll('.tdoa-age-btn');
    for (var i = 0; i < ageBtns.length; i++) {
        (function (btn) {
            btn.addEventListener('click', function () {
                windowMode = false;
                var wsb = document.getElementById('tdoa-window-set-btn');
                if (wsb) wsb.classList.remove('tdoa-win-active');
                loadFixes(parseFloat(btn.getAttribute('data-age')));
            });
        })(ageBtns[i]);
    }

    /* Hide / Unhide All buttons */
    var hideBtn = document.getElementById('tdoa-hide-btn');
    if (hideBtn) {
        hideBtn.addEventListener('click', function () {
            localStorage.setItem('tdoa_hidden_before_t', String(Date.now() / 1000));
            updateHideStatus();
            loadFixes(currentMaxAgeS);
        });
    }
    var unhideBtn = document.getElementById('tdoa-unhide-btn');
    if (unhideBtn) {
        unhideBtn.addEventListener('click', function () {
            localStorage.setItem('tdoa_hidden_before_t', '0');
            updateHideStatus();
            loadFixes(currentMaxAgeS);
        });
    }

    /* Fixed time window controls */
    var winFromInput = document.getElementById('tdoa-window-from');
    var winToInput   = document.getElementById('tdoa-window-to');
    var winSetBtn    = document.getElementById('tdoa-window-set-btn');
    var winClearBtn  = document.getElementById('tdoa-window-clear-btn');

    /* Pre-fill: To = now, From = now - defaultMaxAgeS (or 1 h) */
    if (winFromInput && winToInput) {
        var nowS0  = Date.now() / 1000;
        var defAge = (currentMaxAgeS > 0) ? currentMaxAgeS : 3600;
        winFromInput.value = toDatetimeLocal(nowS0 - defAge);
        winToInput.value   = toDatetimeLocal(nowS0);
    }

    if (winSetBtn) {
        winSetBtn.addEventListener('click', function () {
            if (!winFromInput || !winToInput) return;
            var fromS = new Date(winFromInput.value).getTime() / 1000;
            var toS   = new Date(winToInput.value).getTime()   / 1000;
            if (isNaN(fromS) || isNaN(toS)) {
                setText('tdoa-last-fix', 'invalid date');
                return;
            }
            if (fromS >= toS) {
                setText('tdoa-last-fix', 'From must be before To');
                return;
            }
            windowMode  = true;
            windowFromS = fromS;
            windowToS   = toS;
            winSetBtn.classList.add('tdoa-win-active');
            loadFixes(currentMaxAgeS);
        });
    }

    if (winClearBtn) {
        winClearBtn.addEventListener('click', function () {
            windowMode = false;
            if (winSetBtn) winSetBtn.classList.remove('tdoa-win-active');
            loadFixes(currentMaxAgeS);
        });
    }

    /* Initial fix layer load */
    loadFixes(currentMaxAgeS);

    /* Periodic refresh so node markers appear without waiting for a fix */
    setInterval(function () { loadFixes(currentMaxAgeS); }, 15000);
});

/* - Reset buttons: two-click confirmation to avoid blocked confirm() - */
function makeResetHandler(btnId, url, origLabel) {
    var btn = document.getElementById(btnId);
    if (!btn) { console.warn('[Beagle] button not found: ' + btnId); return; }
    var armed = false;
    btn.addEventListener('click', function () {
        if (!armed) {
            armed = true;
            btn.textContent = 'Sure? Click again';
            btn.style.background = 'rgba(210,100,0,0.9)';
            setTimeout(function () {
                if (armed) {
                    armed = false;
                    btn.textContent = origLabel;
                    btn.style.background = '';
                }
            }, 3000);
            return;
        }
        armed = false;
        btn.disabled = true;
        btn.textContent = 'Clearing...';
        _fetch(url, { method: 'DELETE', headers: _hdr() })
            .then(function (r) {
                if (!r.ok) throw new Error('HTTP ' + r.status);
                return r.json();
            })
            .then(function (data) {
                console.log('[Beagle] reset', url, data);
                window.location.reload();
            })
            .catch(function (e) {
                console.error('[Beagle] reset failed:', e);
                btn.disabled = false;
                btn.textContent = origLabel;
                btn.style.background = '';
                setText('tdoa-last-fix', 'reset error: ' + e.message);
            });
    });
}
makeResetHandler('tdoa-heatmap-reset-btn', '/api/v1/heatmap', 'Reset Heat Map');

/* - SSE live connection --
   On new_fix: update fix layers without a full page reload. */
function connect() {
    var src = new EventSource('/api/v1/fixes/stream');
    src.onopen = function () { setLive('LIVE', 'rgba(20,120,40,0.9)'); };
    src.addEventListener('new_fix', function () {
        setLive('NEW FIX', 'rgba(200,120,0,0.9)');
        loadFixes(currentMaxAgeS);
        loadHeatmap();
    });
    src.onerror = function () {
        src.close();
        setLive('OFFLINE', 'rgba(155,40,40,0.9)');
        setTimeout(connect, 5000);
    };
}

try {
    setInterval(updateClock, 1000);
    setInterval(updateLastFix, 30000);
    updateClock();
    updateLastFix();
    updateHideStatus();
    connect();
} catch (err) {
    console.error('[Beagle] panel startup error:', err);
}

/* ================================================================
   Nodes tab
   ================================================================ */

var _currentNodes = [];
var _nodeTabActive = false;
var _nodeEditing = false;  /* suppress auto-refresh while forms are open */

/* HTML-escape a value for safe insertion into markup */
function _esc(s) {
    return String(s === null || s === undefined ? '' : s)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/* Build fetch headers, adding Bearer token when configured */
function _hdr() {
    var h = {};
    var t = sessionStorage.getItem('tdoa_token') || TDOA.authToken;
    if (t) { h['Authorization'] = 'Bearer ' + t; }
    return h;
}
function _hdrJson() {
    var h = _hdr();
    h['Content-Type'] = 'application/json';
    return h;
}

/* Authenticated fetch wrapper - shows login overlay on 401 */
function _fetch(url, opts) {
    opts = opts || {};
    if (!opts.headers) opts.headers = _hdr();
    return fetch(url, opts).then(function (r) {
        if (r.status === 401 && TDOA.authMode === 'userdb') {
            _showLogin('Session expired - please log in again.');
            return Promise.reject(new Error('Unauthorized'));
        }
        return r;
    });
}

/* ================================================================
   Login / Logout (userdb mode)
   ================================================================ */
var _currentUser = null;

function _showLogin(msg) {
    var overlay = document.getElementById('tdoa-login-overlay');
    if (overlay) overlay.style.display = 'flex';
    var errEl = document.getElementById('tdoa-login-error');
    if (msg && errEl) { errEl.textContent = msg; errEl.style.display = 'block'; }
    /* Hide 2FA field on fresh display */
    var tfaDiv = document.getElementById('tdoa-login-2fa');
    if (tfaDiv) tfaDiv.style.display = 'none';
    var passEl = document.getElementById('tdoa-login-pass');
    if (passEl) passEl.style.display = '';
}

function _hideLogin() {
    var overlay = document.getElementById('tdoa-login-overlay');
    if (overlay) overlay.style.display = 'none';
}

function _setUserInfo(user) {
    _currentUser = user;
    var info = document.getElementById('tdoa-user-info');
    var logoutBtn = document.getElementById('tdoa-logout-btn');
    var usersTab = document.getElementById('tdoa-tab-btn-users');
    if (info && user) {
        info.textContent = user.username + ' (' + user.role + ')';
    }
    if (logoutBtn) logoutBtn.style.display = user ? '' : 'none';
    /* Show Users tab only for admin in userdb mode */
    if (usersTab) {
        usersTab.style.display = (TDOA.authMode === 'userdb' && user && user.role === 'admin') ? '' : 'none';
    }
}

/* Check for existing session on page load (userdb mode only) */
function _checkAuth() {
    if (TDOA.authMode !== 'userdb') return;

    /* Show Google login button if configured */
    if (TDOA.googleOAuthEnabled) {
        var gBtn = document.getElementById('tdoa-google-login');
        if (gBtn) gBtn.style.display = '';
    }

    /* Check for OAuth token or pending 2FA from URL params */
    var params = new URLSearchParams(window.location.search);
    var oauthToken = params.get('oauth_token');
    var pending2fa = params.get('pending_2fa');
    if (oauthToken) {
        /* Clean URL */
        history.replaceState(null, '', window.location.pathname);
        sessionStorage.setItem('tdoa_token', oauthToken);
    }
    if (pending2fa) {
        /* Clean URL and show 2FA input for OAuth login */
        history.replaceState(null, '', window.location.pathname);
        _showLogin('Enter the code from your authenticator app.');
        var tfaDiv = document.getElementById('tdoa-login-2fa');
        if (tfaDiv) tfaDiv.style.display = 'block';
        /* Hide username/password fields */
        var userEl = document.getElementById('tdoa-login-user');
        var passEl = document.getElementById('tdoa-login-pass');
        if (userEl) { userEl.style.display = 'none'; userEl.previousElementSibling.style.display = 'none'; }
        if (passEl) { passEl.style.display = 'none'; passEl.previousElementSibling.style.display = 'none'; }
        var loginBtn = document.getElementById('tdoa-login-btn');
        if (loginBtn) loginBtn.textContent = 'Verify';
        /* Store partial token for the login handler */
        sessionStorage.setItem('tdoa_pending_2fa', pending2fa);
        return;
    }

    var token = sessionStorage.getItem('tdoa_token');
    if (!token) { _showLogin(); return; }
    fetch('/auth/me', { headers: { 'Authorization': 'Bearer ' + token } })
        .then(function (r) {
            if (!r.ok) throw new Error('Invalid session');
            return r.json();
        })
        .then(function (data) {
            _setUserInfo(data);
            _hideLogin();
        })
        .catch(function () {
            sessionStorage.removeItem('tdoa_token');
            _showLogin();
        });
}

/* Login form submit handler */
(function () {
    var loginBtn = document.getElementById('tdoa-login-btn');
    if (!loginBtn) return;
    var _partialToken = null;

    loginBtn.addEventListener('click', function () {
        var errEl = document.getElementById('tdoa-login-error');
        var tfaDiv = document.getElementById('tdoa-login-2fa');
        var codeEl = document.getElementById('tdoa-login-code');

        /* Check for OAuth pending 2FA */
        if (!_partialToken) {
            var stored2fa = sessionStorage.getItem('tdoa_pending_2fa');
            if (stored2fa) { _partialToken = stored2fa; sessionStorage.removeItem('tdoa_pending_2fa'); }
        }

        /* 2FA verification step */
        if (_partialToken) {
            var code = (codeEl ? codeEl.value : '').trim();
            if (!code) { errEl.textContent = 'Enter your authentication code.'; errEl.style.display = 'block'; return; }
            fetch('/auth/2fa/verify', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ partial_token: _partialToken, code: code })
            })
            .then(function (r) {
                if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || 'Invalid code'); });
                return r.json();
            })
            .then(function (data) {
                _partialToken = null;
                sessionStorage.setItem('tdoa_token', data.token);
                _setUserInfo({ username: data.username, role: data.role });
                _hideLogin();
                if (errEl) { errEl.style.display = 'none'; }
            })
            .catch(function (e) {
                if (errEl) { errEl.textContent = e.message; errEl.style.display = 'block'; }
            });
            return;
        }

        /* Normal login step */
        var user = (document.getElementById('tdoa-login-user') || {}).value || '';
        var pass = (document.getElementById('tdoa-login-pass') || {}).value || '';
        if (!user || !pass) { errEl.textContent = 'Enter username and password.'; errEl.style.display = 'block'; return; }
        fetch('/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: user, password: pass })
        })
        .then(function (r) {
            if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || 'Login failed'); });
            return r.json();
        })
        .then(function (data) {
            if (data.requires_2fa) {
                _partialToken = data.partial_token;
                if (tfaDiv) tfaDiv.style.display = 'block';
                /* Hide password field, change button text */
                var passEl = document.getElementById('tdoa-login-pass');
                if (passEl) passEl.parentElement.querySelector('label').textContent = '';
                if (passEl) passEl.style.display = 'none';
                loginBtn.textContent = 'Verify';
                if (errEl) { errEl.textContent = 'Enter the code from your authenticator app.'; errEl.style.display = 'block'; errEl.style.color = '#7a9bbf'; }
                if (codeEl) codeEl.focus();
                return;
            }
            sessionStorage.setItem('tdoa_token', data.token);
            _setUserInfo({ username: data.username, role: data.role });
            _hideLogin();
            if (errEl) { errEl.style.display = 'none'; }
        })
        .catch(function (e) {
            if (errEl) { errEl.textContent = e.message; errEl.style.display = 'block'; errEl.style.color = '#e74c3c'; }
        });
    });

    /* Allow Enter key in login fields */
    ['tdoa-login-user', 'tdoa-login-pass', 'tdoa-login-code'].forEach(function (id) {
        var el = document.getElementById(id);
        if (el) el.addEventListener('keydown', function (e) { if (e.key === 'Enter') loginBtn.click(); });
    });
})();

/* Logout handler */
(function () {
    var btn = document.getElementById('tdoa-logout-btn');
    if (!btn) return;
    btn.addEventListener('click', function () {
        var token = sessionStorage.getItem('tdoa_token');
        if (token) {
            fetch('/auth/logout', {
                method: 'POST',
                headers: { 'Authorization': 'Bearer ' + token }
            }).catch(function () {});
        }
        sessionStorage.removeItem('tdoa_token');
        _currentUser = null;
        _setUserInfo(null);
        _showLogin();
    });
})();

/* Run auth check on load */
_checkAuth();

/* Look up group info for a node by matching against _currentNodes */
function _nodeGroupInfo(nodeId) {
    for (var i = 0; i < _currentNodes.length; i++) {
        var nd = _currentNodes[i];
        if (nd.node_id === nodeId && nd.freq_group_id) {
            var gl = _groupLookup();
            if (gl[nd.freq_group_id]) return gl[nd.freq_group_id];
        }
    }
    return null;
}

/* Derive a display-friendly node type from sdr_mode */
function _nodeType(node) {
    var m = node.sdr_mode;
    if (!m) return 'unknown';
    if (m === 'freq_hop')   return 'freq-hop';
    if (m === 'two_sdr')    return 'two-sdr';
    if (m === 'single_sdr') return 'single-sdr';
    return m;
}

/* Return {cls, text} for the status dot */
function _nodeStatus(node) {
    if (!node.registered)  return { cls: 'ns-unknown',  text: 'unregistered' };
    if (!node.enabled)     return { cls: 'ns-disabled', text: 'disabled' };
    /* Prefer heartbeat_age_s for status; fall back to last_seen_at */
    var hbAge = node.heartbeat_age_s;
    if (hbAge !== null && hbAge !== undefined) {
        if (hbAge < 120)  return { cls: 'ns-online',  text: 'online' };
        return { cls: 'ns-offline', text: 'offline (' + fmtAge(hbAge) + ')' };
    }
    var age = (node.last_seen_at !== null && node.last_seen_at !== undefined)
        ? (Date.now() / 1000 - node.last_seen_at) : null;
    if (age === null)   return { cls: 'ns-notseen', text: 'no heartbeat' };
    if (age < 60)       return { cls: 'ns-online',  text: 'online' };
    if (age < 300)      return { cls: 'ns-stale',   text: 'stale' };
    return { cls: 'ns-offline', text: 'offline' };
}

/* Render node cards into #tdoa-node-list */
function renderNodes(nodes) {
    _currentNodes = nodes;
    var list = document.getElementById('tdoa-node-list');
    if (!list) return;
    if (!nodes || nodes.length === 0) {
        list.innerHTML = '<div class="tp-node-loading">No nodes found.</div>';
        return;
    }
    var html = '';
    var grpLookup = _groupLookup();
    for (var i = 0; i < nodes.length; i++) {
        var n = nodes[i];
        var st = _nodeStatus(n);
        var label = _esc(n.label || n.node_id);
        var lat = (n.location_lat !== null && n.location_lat !== undefined)
            ? parseFloat(n.location_lat).toFixed(4) : '?';
        var lon = (n.location_lon !== null && n.location_lon !== undefined)
            ? parseFloat(n.location_lon).toFixed(4) : '?';
        var age = (n.last_seen_at !== null && n.last_seen_at !== undefined)
            ? fmtAge(Date.now() / 1000 - n.last_seen_at) : 'never';
        var type = _nodeType(n);
        var nid = _esc(n.node_id);
        html += '<div class="tdoa-node-card" data-nid="' + nid + '">';
        html += '<div class="tp-node-head">';
        html += '<span><span class="tp-node-dot ' + st.cls + '"></span>'
              + '<span class="tp-node-id">' + label + '</span>';
        if (n.registered) {
            html += '<span class="tp-node-edit-label" onclick="window._tdoaEditLabel(&apos;'
                  + nid + '&apos;,&apos;' + _esc(n.label || '') + '&apos;)">&#9998;</span>';
        }
        html += '</span>';
        html += '<span style="color:#7a9bbf;font-size:10px">' + st.text + '</span>';
        html += '</div>';
        /* Group tag */
        var grpTag = '';
        if (n.freq_group_id && grpLookup[n.freq_group_id]) {
            var gi = grpLookup[n.freq_group_id];
            grpTag = '<span class="tp-node-grp-tag" style="border-left:2px solid '
                   + gi.color + '">' + _esc(gi.label) + '</span>';
        }
        html += '<div class="tp-node-meta">';
        html += lat + ', ' + lon + ' &bull; ' + _esc(type);
        if (grpTag) html += grpTag;
        html += '<br>Last: ' + _esc(age);
        if (n.last_ip) { html += ' &bull; ' + _esc(n.last_ip); }
        html += '</div>';
        /* Carrier threshold status row */
        if (n.noise_floor_db != null && n.onset_threshold_db != null) {
            var margin = (n.onset_threshold_db - n.noise_floor_db).toFixed(1);
            var marginCls = margin >= 10 ? 'tp-margin-good' : (margin >= 5 ? 'tp-margin-warn' : 'tp-margin-bad');
            html += '<div class="tp-carrier-row">';
            html += 'Floor: ' + n.noise_floor_db.toFixed(1) + ' dB'
                  + ' &bull; Onset: ' + n.onset_threshold_db.toFixed(1)
                  + ' &bull; Offset: ' + (n.offset_threshold_db != null ? n.offset_threshold_db.toFixed(1) : '?')
                  + ' &bull; Margin: <span class="tp-margin ' + marginCls + '">' + margin + ' dB</span>';
            html += '</div>';
        }
        if (n.registered) {
            var togEnable = !n.enabled;
            var togCls   = togEnable ? 'ton' : 'toff';
            var togLabel = togEnable ? 'Enable' : 'Disable';
            html += '<div class="tp-node-actions">';
            html += '<button class="tdoa-btn-sm ' + togCls + '"'
                  + ' onclick="window._tdoaToggle(this,&apos;' + nid + '&apos;,' + togEnable + ')">'
                  + togLabel + '</button>';
            html += '<button class="tdoa-btn-sm toff" data-armed="0"'
                  + ' onclick="window._tdoaRegenSecret(this,&apos;' + nid + '&apos;)">'
                  + 'Regen Secret</button>';
            html += '<button class="tdoa-btn-sm tdel" data-armed="0"'
                  + ' onclick="window._tdoaDelete(this,&apos;' + nid + '&apos;)">'
                  + 'Delete</button>';
            html += '</div>';
            /* Expandable detail panel */
            html += '<div style="text-align:right"><span style="cursor:pointer;font-size:10px;color:#7a9bbf"'
                  + ' onclick="window._tdoaToggleDetail(&apos;' + nid + '&apos;)">&#9660; details</span></div>';
            html += '<div class="tp-node-detail" id="nd-' + nid + '" style="display:none"></div>';
        }
        html += '</div>';
    }
    list.innerHTML = html;
}

/* PATCH enable/disable a single node */
window._tdoaToggle = function (btn, nodeId, enable) {
    btn.disabled = true;
    _fetch('/api/v1/nodes/' + encodeURIComponent(nodeId), {
        method: 'PATCH',
        headers: _hdrJson(),
        body: JSON.stringify({ enabled: enable })
    })
    .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(function ()  { loadNodes(); })
    .catch(function (e) {
        console.error('[Beagle] toggle error:', e);
        btn.disabled = false;
    });
};

/* DELETE a node (two-click confirm) */
window._tdoaDelete = function (btn, nodeId) {
    if (btn.getAttribute('data-armed') !== '1') {
        btn.setAttribute('data-armed', '1');
        btn.textContent = 'Sure?';
        setTimeout(function () {
            if (btn.getAttribute('data-armed') === '1') {
                btn.setAttribute('data-armed', '0');
                btn.textContent = 'Delete';
            }
        }, 3000);
        return;
    }
    btn.setAttribute('data-armed', '0');
    btn.disabled = true;
    _fetch('/api/v1/nodes/' + encodeURIComponent(nodeId), {
        method: 'DELETE',
        headers: _hdr()
    })
    .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(function ()  { loadNodes(); })
    .catch(function (e) {
        console.error('[Beagle] delete error:', e);
        btn.disabled = false;
    });
};

/* Fetch node list and render */
function loadNodes() {
    _fetch('/map/nodes')
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function (data) { renderNodes(data.nodes || []); })
        .catch(function (e) { console.error('[Beagle] loadNodes error:', e); });
}

/* Event-auth settings row */
function loadEventAuthSetting() {
    _fetch('/api/v1/settings')
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function (data) { _renderEventAuth(data.require_event_auth); })
        .catch(function (e) { console.error('[Beagle] settings load error:', e); });
}
function _renderEventAuth(required) {
    var val = document.getElementById('tdoa-event-auth-val');
    var btn = document.getElementById('tdoa-event-auth-btn');
    if (!val || !btn) return;
    if (required) {
        val.textContent = 'required';
        val.style.color = '#e67e22';
        btn.textContent = 'Open to unauthenticated events';
        btn.className = 'tdoa-btn-sm ton';
    } else {
        val.textContent = 'open (no auth)';
        val.style.color = '#2ecc71';
        btn.textContent = 'Require authentication';
        btn.className = 'tdoa-btn-sm toff';
    }
    btn.disabled = false;
    btn._authRequired = required;
}
(function () {
    var btn = document.getElementById('tdoa-event-auth-btn');
    if (!btn) return;
    btn.addEventListener('click', function () {
        var newVal = !btn._authRequired;
        btn.disabled = true;
        _fetch('/api/v1/settings', {
            method: 'PATCH',
            headers: _hdrJson(),
            body: JSON.stringify({ require_event_auth: newVal })
        })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function (data) { _renderEventAuth(data.require_event_auth); })
        .catch(function (e) {
            console.error('[Beagle] settings patch error:', e);
            btn.disabled = false;
        });
    });
})();

/* Tab switching */
var _groupTabActive = false;
var _usersTabActive = false;
(function () {
    var fixesTabBtn  = document.getElementById('tdoa-tab-btn-fixes');
    var nodesTabBtn  = document.getElementById('tdoa-tab-btn-nodes');
    var groupsTabBtn = document.getElementById('tdoa-tab-btn-groups');
    var usersTabBtn  = document.getElementById('tdoa-tab-btn-users');
    var fixesPanel   = document.getElementById('tdoa-tab-fixes');
    var nodesPanel   = document.getElementById('tdoa-tab-nodes');
    var groupsPanel  = document.getElementById('tdoa-tab-groups');
    var usersPanel   = document.getElementById('tdoa-tab-users');
    if (!fixesTabBtn || !nodesTabBtn || !groupsTabBtn) return;

    function switchTab(active) {
        _nodeTabActive = (active === 'nodes');
        _groupTabActive = (active === 'groups');
        _usersTabActive = (active === 'users');
        fixesTabBtn.classList.toggle('tdoa-tab-active', active === 'fixes');
        nodesTabBtn.classList.toggle('tdoa-tab-active', active === 'nodes');
        groupsTabBtn.classList.toggle('tdoa-tab-active', active === 'groups');
        if (usersTabBtn) usersTabBtn.classList.toggle('tdoa-tab-active', active === 'users');
        if (fixesPanel)  fixesPanel.style.display  = (active === 'fixes')  ? '' : 'none';
        if (nodesPanel)  nodesPanel.style.display  = (active === 'nodes')  ? '' : 'none';
        if (groupsPanel) groupsPanel.style.display = (active === 'groups') ? '' : 'none';
        if (usersPanel)  usersPanel.style.display  = (active === 'users')  ? '' : 'none';
    }

    fixesTabBtn.addEventListener('click', function () { switchTab('fixes'); });
    nodesTabBtn.addEventListener('click', function () {
        switchTab('nodes');
        loadNodes();
        loadEventAuthSetting();
    });
    groupsTabBtn.addEventListener('click', function () {
        switchTab('groups');
        loadGroups();
    });
    if (usersTabBtn) usersTabBtn.addEventListener('click', function () {
        switchTab('users');
        loadUsers();
    });
})();

/* Bulk enable / disable all registered nodes */
(function () {
    function setBulk(enable) {
        var registered = _currentNodes.filter(function (n) { return n.registered; });
        var pending = registered.length;
        if (pending === 0) return;
        registered.forEach(function (n) {
            _fetch('/api/v1/nodes/' + encodeURIComponent(n.node_id), {
                method: 'PATCH',
                headers: _hdrJson(),
                body: JSON.stringify({ enabled: enable })
            })
            .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); })
            .then(function () { pending--; if (pending === 0) loadNodes(); })
            .catch(function (e) { console.error('[Beagle] bulk toggle error:', e); });
        });
    }
    var enableAllBtn  = document.getElementById('tdoa-enable-all-btn');
    var disableAllBtn = document.getElementById('tdoa-disable-all-btn');
    if (enableAllBtn)  enableAllBtn.addEventListener('click',  function () { setBulk(true);  });
    if (disableAllBtn) disableAllBtn.addEventListener('click', function () { setBulk(false); });
})();

/* Auto-refresh nodes every 10 s when the nodes tab is active */
setInterval(function () { if (_nodeTabActive && !_nodeEditing) loadNodes(); }, 10000);

/* ================================================================
   Node management: register, regen secret, label edit, detail panel
   ================================================================ */

/* --- Secret display modal --- */
function _showSecretModal(nodeId, secret) {
    var overlay = document.createElement('div');
    overlay.className = 'tdoa-secret-overlay';
    overlay.innerHTML = '<div class="tdoa-secret-card">'
        + '<h3>Node Secret for ' + _esc(nodeId) + '</h3>'
        + '<div class="tdoa-secret-box" id="tdoa-secret-text">' + _esc(secret) + '</div>'
        + '<div class="tdoa-secret-warn">This secret will not be shown again. Copy it now.</div>'
        + '<div class="tdoa-secret-btns">'
        + '<button class="tdoa-btn-sm ton" id="tdoa-secret-copy">Copy</button>'
        + '<button class="tdoa-btn-sm toff" id="tdoa-secret-close">Close</button>'
        + '</div></div>';
    document.body.appendChild(overlay);
    overlay.querySelector('#tdoa-secret-copy').addEventListener('click', function () {
        navigator.clipboard.writeText(secret).then(function () {
            overlay.querySelector('#tdoa-secret-copy').textContent = 'Copied!';
        });
    });
    overlay.querySelector('#tdoa-secret-close').addEventListener('click', function () {
        overlay.remove();
    });
}

/* --- Register node form --- */
var regBtn = document.getElementById('tdoa-register-node-btn');
if (regBtn) regBtn.addEventListener('click', function () {
    var area = document.getElementById('tdoa-node-reg-area');
    if (!area) return;
    if (area.style.display !== 'none') { area.style.display = 'none'; _nodeEditing = false; return; }
    _nodeEditing = true;
    area.style.display = '';
    area.innerHTML = '<div class="tp-node-reg-form">'
        + '<label>Node ID (required)</label>'
        + '<input id="nrf-id" placeholder="e.g. seattle-north-01">'
        + '<label>Label (optional)</label>'
        + '<input id="nrf-label" placeholder="Human-readable name">'
        + '<div id="nrf-error" style="color:#e74c3c;font-size:10px;margin-top:3px"></div>'
        + '<div style="display:flex;gap:4px;margin-top:4px">'
        + '<button class="tdoa-btn-sm ton" id="nrf-submit">Register</button>'
        + '<button class="tdoa-btn-sm toff" id="nrf-cancel">Cancel</button>'
        + '</div></div>';
    area.querySelector('#nrf-cancel').addEventListener('click', function () {
        area.style.display = 'none';
        _nodeEditing = false;
    });
    area.querySelector('#nrf-submit').addEventListener('click', function () {
        var nodeId = (document.getElementById('nrf-id').value || '').trim();
        var label  = (document.getElementById('nrf-label').value || '').trim() || null;
        var errEl  = document.getElementById('nrf-error');
        if (!nodeId) { if (errEl) errEl.textContent = 'Node ID is required.'; return; }
        _fetch('/api/v1/nodes', {
            method: 'POST',
            headers: _hdrJson(),
            body: JSON.stringify({ node_id: nodeId, label: label })
        })
        .then(function (r) {
            if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || 'HTTP ' + r.status); });
            return r.json();
        })
        .then(function (data) {
            area.style.display = 'none';
            _nodeEditing = false;
            _showSecretModal(data.node_id, data.secret);
            loadNodes();
        })
        .catch(function (e) { if (errEl) errEl.textContent = e.message || String(e); });
    });
});

/* --- Regen secret (armed confirmation) --- */
window._tdoaRegenSecret = function (btn, nodeId) {
    if (btn.getAttribute('data-armed') !== '1') {
        btn.setAttribute('data-armed', '1');
        btn.textContent = 'Sure?';
        setTimeout(function () {
            if (btn.getAttribute('data-armed') === '1') {
                btn.setAttribute('data-armed', '0');
                btn.textContent = 'Regen Secret';
            }
        }, 3000);
        return;
    }
    btn.setAttribute('data-armed', '0');
    btn.disabled = true;
    _fetch('/api/v1/nodes/' + encodeURIComponent(nodeId) + '/regen-secret', {
        method: 'POST',
        headers: _hdr()
    })
    .then(function (r) {
        if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || 'HTTP ' + r.status); });
        return r.json();
    })
    .then(function (data) {
        btn.disabled = false;
        btn.textContent = 'Regen Secret';
        _showSecretModal(data.node_id, data.secret);
    })
    .catch(function (e) {
        console.error('[Beagle] regen-secret error:', e);
        btn.disabled = false;
        btn.textContent = 'Regen Secret';
    });
};

/* --- Inline label editing --- */
window._tdoaEditLabel = function (nodeId, currentLabel) {
    _nodeEditing = true;
    var card = document.querySelector('.tdoa-node-card[data-nid="' + nodeId + '"]');
    if (!card) return;
    var head = card.querySelector('.tp-node-head');
    if (!head) return;
    var html = '<input class="tp-node-label-inp" value="' + _esc(currentLabel) + '"'
             + ' style="width:100px;font:11px monospace;background:rgba(15,25,45,0.95);'
             + 'color:#c8d4e8;border:1px solid rgba(80,110,160,0.5);border-radius:3px;padding:1px 4px">'
             + ' <button class="tdoa-btn-sm ton" style="font-size:9px;padding:1px 5px"'
             + ' onclick="window._tdoaSaveLabel(&apos;' + _esc(nodeId) + '&apos;)">Save</button>'
             + ' <button class="tdoa-btn-sm toff" style="font-size:9px;padding:1px 5px"'
             + ' onclick="window._tdoaCancelEditLabel()">Cancel</button>';
    head.innerHTML = html;
    head.querySelector('.tp-node-label-inp').focus();
};

window._tdoaSaveLabel = function (nodeId) {
    var inp = document.querySelector('.tp-node-label-inp');
    if (!inp) return;
    var newLabel = inp.value.trim() || null;
    _fetch('/api/v1/nodes/' + encodeURIComponent(nodeId), {
        method: 'PATCH',
        headers: _hdrJson(),
        body: JSON.stringify({ label: newLabel })
    })
    .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(function () { _nodeEditing = false; loadNodes(); })
    .catch(function (e) { console.error('[Beagle] label save error:', e); });
};

window._tdoaCancelEditLabel = function () {
    _nodeEditing = false;
    loadNodes();
};

/* --- Expandable node detail panel --- */
window._tdoaToggleDetail = function (nodeId) {
    var det = document.getElementById('nd-' + nodeId);
    if (!det) return;
    if (det.style.display === 'none') {
        _nodeEditing = true;
        det.style.display = '';
        /* Fetch full node details */
        _fetch('/api/v1/nodes/' + encodeURIComponent(nodeId), { headers: _hdr() })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function (node) {
            var html = '';
            html += '<div class="tp-row"><span class="tp-key">Config ver</span><span>'
                  + (node.config_version != null ? node.config_version : '?') + '</span></div>';
            html += '<div class="tp-row"><span class="tp-key">Registered</span><span>'
                  + (node.registered_at ? new Date(node.registered_at * 1000).toISOString().slice(0,19) + 'Z' : '?')
                  + '</span></div>';
            html += '<div class="tp-row"><span class="tp-key">Group</span><span>'
                  + _esc(node.freq_group_id || 'none') + '</span></div>';
            html += '<div style="margin-top:4px"><span class="tp-key">Config JSON:</span></div>';
            var cfgText = '';
            if (node.config_json) {
                try { cfgText = JSON.stringify(JSON.parse(node.config_json), null, 2); }
                catch (e) { cfgText = node.config_json; }
            }
            /* --- Carrier threshold editing --- */
            var cfgObj = {};
            if (node.config_json) {
                try { cfgObj = JSON.parse(node.config_json); } catch (e) { cfgObj = {}; }
            }
            var carrier = cfgObj.carrier || {};
            /* Find live values from the nodes list */
            var liveNode = null;
            if (_currentNodes) {
                for (var k = 0; k < _currentNodes.length; k++) {
                    if (_currentNodes[k].node_id === nodeId) { liveNode = _currentNodes[k]; break; }
                }
            }
            var liveFloor = liveNode && liveNode.noise_floor_db != null ? liveNode.noise_floor_db : null;
            var liveOnset = liveNode && liveNode.onset_threshold_db != null ? liveNode.onset_threshold_db : null;
            var liveOffset = liveNode && liveNode.offset_threshold_db != null ? liveNode.offset_threshold_db : null;
            var curOnset = carrier.onset_db != null ? carrier.onset_db : (liveOnset != null ? liveOnset : -30);
            var curOffset = carrier.offset_db != null ? carrier.offset_db : (liveOffset != null ? liveOffset : -40);
            var curHold = carrier.min_hold_windows != null ? carrier.min_hold_windows : 1;
            var curRelease = carrier.min_release_windows != null ? carrier.min_release_windows : 1;

            html += '<div class="tp-carrier-form">';
            html += '<div style="color:#c8d4e8;font-size:11px;margin-bottom:4px">Carrier Thresholds';
            if (liveFloor != null) {
                var m = (curOnset - liveFloor).toFixed(1);
                var mc = m >= 10 ? 'tp-margin-good' : (m >= 5 ? 'tp-margin-warn' : 'tp-margin-bad');
                html += ' <span style="font-size:10px;color:#7a9bbf">(noise floor: '
                      + liveFloor.toFixed(1) + ' dB, margin: <span class="' + mc + '">' + m + ' dB</span>)</span>';
            }
            html += '</div>';
            html += '<div><label>Onset (dBFS)</label><input type="number" step="1" id="ct-onset-' + _esc(nodeId) + '" value="' + curOnset + '"></div>';
            html += '<div><label>Offset (dBFS)</label><input type="number" step="1" id="ct-offset-' + _esc(nodeId) + '" value="' + curOffset + '"></div>';
            html += '<div><label>Hold windows</label><input type="number" step="1" min="1" id="ct-hold-' + _esc(nodeId) + '" value="' + curHold + '"></div>';
            html += '<div><label>Release windows</label><input type="number" step="1" min="1" id="ct-release-' + _esc(nodeId) + '" value="' + curRelease + '"></div>';
            html += '<div class="tp-carrier-btns">';
            html += '<button class="tdoa-btn-sm ton" onclick="window._tdoaSaveCarrier(&apos;'
                  + _esc(nodeId) + '&apos;)">Save Thresholds</button>';
            if (liveFloor != null) {
                html += '<button class="tdoa-btn-sm" style="background:rgba(46,204,113,0.15);color:#2ecc71;border-color:rgba(46,204,113,0.3)"'
                      + ' onclick="window._tdoaAutoCalibrate(&apos;' + _esc(nodeId) + '&apos;,' + liveFloor + ')">'
                      + 'Auto-Calibrate</button>';
            }
            html += '</div></div>';

            html += '<div style="margin-top:6px"><span class="tp-key">Config JSON:</span></div>';
            html += '<textarea class="tp-node-config-ta" id="ncfg-' + _esc(nodeId) + '">'
                  + _esc(cfgText) + '</textarea>';
            html += '<button class="tdoa-btn-sm ton" onclick="window._tdoaSaveConfig(&apos;'
                  + _esc(nodeId) + '&apos;)">Save Config</button>';
            det.innerHTML = html;
        })
        .catch(function (e) {
            det.innerHTML = '<span style="color:#e74c3c;font-size:10px">' + _esc(e.message) + '</span>';
        });
    } else {
        det.style.display = 'none';
        _nodeEditing = false;
    }
};

window._tdoaSaveConfig = function (nodeId) {
    var ta = document.getElementById('ncfg-' + nodeId);
    if (!ta) return;
    var val = ta.value.trim();
    var configJson = null;
    if (val) {
        try { configJson = JSON.parse(val); }
        catch (e) { alert('Invalid JSON: ' + e.message); return; }
    }
    _fetch('/api/v1/nodes/' + encodeURIComponent(nodeId), {
        method: 'PATCH',
        headers: _hdrJson(),
        body: JSON.stringify({ config_json: configJson })
    })
    .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(function () { _nodeEditing = false; loadNodes(); })
    .catch(function (e) { console.error('[Beagle] config save error:', e); });
};

/* Save carrier thresholds by merging into config_json */
window._tdoaSaveCarrier = function (nodeId) {
    var onset  = parseFloat(document.getElementById('ct-onset-'  + nodeId).value);
    var offset = parseFloat(document.getElementById('ct-offset-' + nodeId).value);
    var hold   = parseInt(document.getElementById('ct-hold-'    + nodeId).value, 10);
    var release= parseInt(document.getElementById('ct-release-' + nodeId).value, 10);
    if (isNaN(onset) || isNaN(offset) || isNaN(hold) || isNaN(release)) {
        alert('All fields must be numbers'); return;
    }
    if (offset >= onset) {
        alert('Offset must be less than onset'); return;
    }
    if (hold < 1 || release < 1) {
        alert('Hold and release must be >= 1'); return;
    }
    /* Read current config_json, merge carrier block, PATCH */
    _fetch('/api/v1/nodes/' + encodeURIComponent(nodeId), { headers: _hdr() })
    .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(function (node) {
        var cfg = {};
        if (node.config_json) {
            try { cfg = JSON.parse(node.config_json); } catch (e) { cfg = {}; }
        }
        cfg.carrier = cfg.carrier || {};
        cfg.carrier.onset_db = onset;
        cfg.carrier.offset_db = offset;
        cfg.carrier.min_hold_windows = hold;
        cfg.carrier.min_release_windows = release;
        return _fetch('/api/v1/nodes/' + encodeURIComponent(nodeId), {
            method: 'PATCH',
            headers: _hdrJson(),
            body: JSON.stringify({ config_json: cfg })
        });
    })
    .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(function () { _nodeEditing = false; loadNodes(); })
    .catch(function (e) { alert('Save failed: ' + e.message); });
};

/* Auto-calibrate: set thresholds relative to current noise floor */
window._tdoaAutoCalibrate = function (nodeId, noiseFloor) {
    var onset  = Math.round(noiseFloor + 12);
    var offset = Math.round(noiseFloor + 6);
    document.getElementById('ct-onset-'  + nodeId).value = onset;
    document.getElementById('ct-offset-' + nodeId).value = offset;
    /* Flash the margin text to indicate values changed */
    var btn = event.target;
    btn.textContent = 'Set to ' + onset + ' / ' + offset + ' - click Save';
    setTimeout(function () { btn.textContent = 'Auto-Calibrate'; }, 3000);
};

/* ================================================================
   Groups tab
   ================================================================ */

var _selectedGroupId = null;
var _groupEditing = false;  /* suppress auto-refresh while forms are open */

function loadGroups() {
    fetch('/map/groups')
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function (data) { renderGroups(data.groups || []); })
        .catch(function (e) { console.error('[Beagle] loadGroups error:', e); });
}

function renderGroups(groups) {
    _currentGroups = groups;
    var list = document.getElementById('tdoa-group-list');
    var detail = document.getElementById('tdoa-group-detail');
    if (!list) return;
    if (!groups || groups.length === 0) {
        list.innerHTML = '<div class="tp-node-loading">No frequency groups.</div>';
        if (detail) detail.style.display = 'none';
        return;
    }
    var html = '';
    for (var i = 0; i < groups.length; i++) {
        var g = groups[i];
        var color = _groupColor(i);
        var sel = (g.group_id === _selectedGroupId) ? ' tg-selected' : '';
        var gid = _esc(g.group_id);
        html += '<div class="tdoa-group-card' + sel + '" data-gid="' + gid + '"'
              + ' onclick="window._tdoaSelectGroup(&apos;' + gid + '&apos;)">';
        html += '<div class="tp-grp-head">';
        html += '<span><span class="tp-grp-color" style="background:' + color + '"></span>'
              + '<span class="tp-grp-id">' + _esc(g.label || g.group_id) + '</span></span>';
        html += '<span style="color:#7a9bbf;font-size:10px">'
              + g.member_count + ' node' + (g.member_count !== 1 ? 's' : '') + '</span>';
        html += '</div>';
        html += '<div class="tp-grp-meta">';
        html += _esc(g.sync_station_id) + ' &bull; '
              + (g.sync_freq_hz / 1e6).toFixed(3) + ' MHz &bull; '
              + (g.target_channels ? g.target_channels.length : 0) + ' ch';
        html += '</div>';
        html += '</div>';
    }
    list.innerHTML = html;

    /* If a group was selected, re-render its detail */
    if (_selectedGroupId) {
        var found = false;
        for (var j = 0; j < groups.length; j++) {
            if (groups[j].group_id === _selectedGroupId) {
                _renderGroupDetail(groups[j]); found = true; break;
            }
        }
        if (!found && detail) detail.style.display = 'none';
    }
}

window._tdoaSelectGroup = function (groupId) {
    if (_selectedGroupId === groupId) {
        _selectedGroupId = null;
        var detail = document.getElementById('tdoa-group-detail');
        if (detail) detail.style.display = 'none';
        /* Deselect card */
        var cards = document.querySelectorAll('.tdoa-group-card');
        for (var i = 0; i < cards.length; i++) cards[i].classList.remove('tg-selected');
        return;
    }
    _selectedGroupId = groupId;
    /* Find group data */
    for (var j = 0; j < _currentGroups.length; j++) {
        if (_currentGroups[j].group_id === groupId) {
            _renderGroupDetail(_currentGroups[j]);
            break;
        }
    }
    /* Highlight card */
    var cards = document.querySelectorAll('.tdoa-group-card');
    for (var k = 0; k < cards.length; k++) {
        cards[k].classList.toggle('tg-selected', cards[k].getAttribute('data-gid') === groupId);
    }
};

function _renderGroupDetail(grp) {
    var detail = document.getElementById('tdoa-group-detail');
    if (!detail) return;
    detail.style.display = '';
    var html = '<div class="tp-grp-detail">';
    html += '<div class="tp-row"><span class="tp-key">ID</span><span>' + _esc(grp.group_id) + '</span></div>';
    html += '<div class="tp-row"><span class="tp-key">Sync</span><span>'
          + _esc(grp.sync_station_id) + ' ' + (grp.sync_freq_hz / 1e6).toFixed(3) + ' MHz</span></div>';
    html += '<div class="tp-row"><span class="tp-key">Location</span><span>'
          + (grp.sync_station_lat != null ? parseFloat(grp.sync_station_lat).toFixed(4) : '?') + ', '
          + (grp.sync_station_lon != null ? parseFloat(grp.sync_station_lon).toFixed(4) : '?') + '</span></div>';
    /* Target channels */
    if (grp.target_channels && grp.target_channels.length > 0) {
        html += '<div style="margin-top:3px"><span class="tp-key">Channels:</span></div>';
        for (var i = 0; i < grp.target_channels.length; i++) {
            var ch = grp.target_channels[i];
            html += '<div style="padding-left:8px;font-size:10px;color:#9ab">'
                  + (ch.frequency_hz / 1e6).toFixed(3) + ' MHz'
                  + (ch.label ? ' (' + _esc(ch.label) + ')' : '')
                  + '</div>';
        }
    }
    /* Members */
    html += '<div style="margin-top:4px"><span class="tp-key">Members:</span></div>';
    html += '<div class="tp-grp-members">';
    if (grp.member_node_ids && grp.member_node_ids.length > 0) {
        for (var m = 0; m < grp.member_node_ids.length; m++) {
            html += '<span class="tp-grp-member-tag">' + _esc(grp.member_node_ids[m])
                  + '<span class="tp-grp-member-rm" onclick="window._tdoaUnassignNode(&apos;'
                  + _esc(grp.member_node_ids[m]) + '&apos;)">&times;</span></span>';
        }
    } else {
        html += '<span style="font-size:10px;color:#666">No members</span>';
    }
    html += '</div>';
    /* Assign node dropdown */
    html += '<div class="tp-grp-assign">';
    html += '<select id="tdoa-grp-assign-select">';
    html += '<option value="">-- assign node --</option>';
    /* Populate from current nodes (registered, not already in this group) */
    for (var n = 0; n < _currentNodes.length; n++) {
        var nd = _currentNodes[n];
        if (!nd.registered) continue;
        if (nd.freq_group_id === grp.group_id) continue;
        html += '<option value="' + _esc(nd.node_id) + '">' + _esc(nd.label || nd.node_id) + '</option>';
    }
    html += '</select>';
    html += '<div class="tp-grp-actions">';
    html += '<button class="tdoa-btn-sm ton" onclick="window._tdoaAssignNode()">Assign</button>';
    html += '<button class="tdoa-btn-sm toff" onclick="window._tdoaEditGroup()">Edit</button>';
    html += '<button class="tdoa-btn-sm tdel" data-armed="0"'
          + ' onclick="window._tdoaDeleteGroup(this,&apos;' + _esc(grp.group_id) + '&apos;)">Delete Group</button>';
    html += '</div>';
    html += '</div>';
    html += '</div>';
    detail.innerHTML = html;
}

window._tdoaAssignNode = function () {
    var sel = document.getElementById('tdoa-grp-assign-select');
    if (!sel || !sel.value || !_selectedGroupId) return;
    var nodeId = sel.value;
    _fetch('/api/v1/nodes/' + encodeURIComponent(nodeId), {
        method: 'PATCH',
        headers: _hdrJson(),
        body: JSON.stringify({ freq_group_id: _selectedGroupId })
    })
    .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(function () { loadGroups(); loadNodes(); })
    .catch(function (e) { console.error('[Beagle] assign node error:', e); });
};

window._tdoaDeleteGroup = function (btn, groupId) {
    if (btn.getAttribute('data-armed') !== '1') {
        btn.setAttribute('data-armed', '1');
        btn.textContent = 'Sure?';
        setTimeout(function () {
            if (btn.getAttribute('data-armed') === '1') {
                btn.setAttribute('data-armed', '0');
                btn.textContent = 'Delete Group';
            }
        }, 3000);
        return;
    }
    btn.setAttribute('data-armed', '0');
    btn.disabled = true;
    _fetch('/api/v1/groups/' + encodeURIComponent(groupId), {
        method: 'DELETE',
        headers: _hdr()
    })
    .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(function () { _selectedGroupId = null; loadGroups(); loadNodes(); })
    .catch(function (e) {
        console.error('[Beagle] delete group error:', e);
        btn.disabled = false;
    });
};

/* Unassign a node from its group */
window._tdoaUnassignNode = function (nodeId) {
    _fetch('/api/v1/nodes/' + encodeURIComponent(nodeId), {
        method: 'PATCH',
        headers: _hdrJson(),
        body: JSON.stringify({ freq_group_id: null })
    })
    .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(function () { loadGroups(); loadNodes(); })
    .catch(function (e) { console.error('[Beagle] unassign error:', e); });
};

/* Edit an existing group */
window._tdoaEditGroup = function () {
    if (!_selectedGroupId) return;
    for (var i = 0; i < _currentGroups.length; i++) {
        if (_currentGroups[i].group_id === _selectedGroupId) {
            _renderGroupForm(_currentGroups[i]);
            return;
        }
    }
};

/* Render create/edit group form into detail panel */
function _renderGroupForm(grp) {
    _groupEditing = true;
    var detail = document.getElementById('tdoa-group-detail');
    if (!detail) return;
    detail.style.display = '';
    var isNew = !grp;
    var g = grp || {};
    var html = '<div class="tp-grp-form">';
    html += '<label>Group ID' + (isNew ? '' : ' (read-only)') + '</label>';
    html += '<input id="gf-id" value="' + _esc(g.group_id || '') + '"'
          + (isNew ? '' : ' disabled') + ' placeholder="e.g. seattle-fm">';
    html += '<label>Label</label>';
    html += '<input id="gf-label" value="' + _esc(g.label || '') + '" placeholder="Display name">';
    html += '<label>Description</label>';
    html += '<input id="gf-desc" value="' + _esc(g.description || '') + '">';
    html += '<label>Sync Station ID</label>';
    html += '<input id="gf-ssid" value="' + _esc(g.sync_station_id || '') + '">';
    html += '<label>Sync Freq (MHz)</label>';
    html += '<input id="gf-freq" type="number" step="any" value="'
          + (g.sync_freq_hz ? (g.sync_freq_hz / 1e6).toFixed(6) : '') + '">';
    html += '<label>Sync Station Lat</label>';
    html += '<input id="gf-lat" type="number" step="any" value="'
          + (g.sync_station_lat != null ? g.sync_station_lat : '') + '">';
    html += '<label>Sync Station Lon</label>';
    html += '<input id="gf-lon" type="number" step="any" value="'
          + (g.sync_station_lon != null ? g.sync_station_lon : '') + '">';
    html += '<label>Target Channels</label>';
    html += '<div id="gf-channels">';
    var chs = g.target_channels || [];
    for (var i = 0; i < chs.length; i++) {
        html += _groupChannelRow(chs[i].frequency_hz / 1e6, chs[i].label || '');
    }
    if (chs.length === 0) html += _groupChannelRow('', '');
    html += '</div>';
    html += '<button class="tdoa-btn-sm ton" onclick="window._tdoaAddChannelRow()" '
          + 'style="margin-top:2px">+ Channel</button>';
    html += '<div id="gf-error" style="color:#e74c3c;font-size:10px;margin-top:4px"></div>';
    html += '<div class="tp-grp-form-btns">';
    html += '<button class="tdoa-btn-sm ton" onclick="window._tdoaSaveGroup(' + isNew + ')">Save</button>';
    html += '<button class="tdoa-btn-sm toff" onclick="window._tdoaCancelGroupForm()">Cancel</button>';
    html += '</div>';
    html += '</div>';
    detail.innerHTML = html;
}

function _groupChannelRow(freqMhz, label) {
    return '<div class="tp-grp-ch-row">'
         + '<input type="number" step="any" placeholder="MHz" value="' + (freqMhz || '') + '" class="gf-ch-freq">'
         + '<input placeholder="label (opt)" value="' + _esc(label) + '" class="gf-ch-label" style="max-width:80px">'
         + '<span class="tp-grp-ch-rm" onclick="this.parentNode.remove()">&times;</span>'
         + '</div>';
}

window._tdoaAddChannelRow = function () {
    var box = document.getElementById('gf-channels');
    if (!box) return;
    var tmp = document.createElement('div');
    tmp.innerHTML = _groupChannelRow('', '');
    box.appendChild(tmp.firstChild);
};

window._tdoaCancelGroupForm = function () {
    _groupEditing = false;
    var detail = document.getElementById('tdoa-group-detail');
    if (detail) detail.style.display = 'none';
    _selectedGroupId = null;
};

window._tdoaSaveGroup = function (isNew) {
    var errEl = document.getElementById('gf-error');
    function err(msg) { if (errEl) errEl.textContent = msg; }
    var groupId = (document.getElementById('gf-id').value || '').trim();
    var label   = (document.getElementById('gf-label').value || '').trim();
    if (!groupId) return err('Group ID is required.');
    if (!label) return err('Label is required.');
    var freqMhz = parseFloat(document.getElementById('gf-freq').value);
    if (isNaN(freqMhz)) return err('Sync frequency is required.');
    var ssid = (document.getElementById('gf-ssid').value || '').trim();
    if (!ssid) return err('Sync station ID is required.');
    var lat = parseFloat(document.getElementById('gf-lat').value);
    var lon = parseFloat(document.getElementById('gf-lon').value);
    if (isNaN(lat) || isNaN(lon)) return err('Sync station lat/lon required.');
    /* Collect channels */
    var chFreqs = document.querySelectorAll('.gf-ch-freq');
    var chLabels = document.querySelectorAll('.gf-ch-label');
    var channels = [];
    for (var i = 0; i < chFreqs.length; i++) {
        var f = parseFloat(chFreqs[i].value);
        if (!isNaN(f) && f > 0) {
            var ch = { frequency_hz: f * 1e6 };
            var cl = (chLabels[i] ? chLabels[i].value : '').trim();
            if (cl) ch.label = cl;
            channels.push(ch);
        }
    }
    if (channels.length === 0) return err('At least one target channel is required.');
    var body = {
        label: label,
        sync_freq_hz: freqMhz * 1e6,
        sync_station_id: ssid,
        sync_station_lat: lat,
        sync_station_lon: lon,
        target_channels: channels
    };
    var desc = (document.getElementById('gf-desc').value || '').trim();
    if (desc) body.description = desc;
    if (isNew) body.group_id = groupId;
    var url = isNew ? '/api/v1/groups' : '/api/v1/groups/' + encodeURIComponent(groupId);
    var method = isNew ? 'POST' : 'PATCH';
    _fetch(url, { method: method, headers: _hdrJson(), body: JSON.stringify(body) })
    .then(function (r) {
        if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || 'HTTP ' + r.status); });
        return r.json();
    })
    .then(function () {
        _groupEditing = false;
        _selectedGroupId = groupId;
        loadGroups(); loadNodes();
    })
    .catch(function (e) { err(e.message || String(e)); });
};

/* Create Group button handler */
var cgBtn = document.getElementById('tdoa-create-group-btn');
if (cgBtn) cgBtn.addEventListener('click', function () { _renderGroupForm(null); });

/* Also load groups on startup so _groupLookup() is available for node cards */
loadGroups();

/* Auto-refresh groups every 10 s when the groups tab is active */
setInterval(function () { if (_groupTabActive && !_groupEditing) loadGroups(); }, 10000);

/* ================================================================
   Users tab (admin-only, userdb mode)
   ================================================================ */

var _usersEditing = false;

function loadUsers() {
    _fetch('/auth/users', { headers: _hdr() })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function (users) { renderUsers(users); })
        .catch(function (e) { console.error('[Beagle] loadUsers error:', e); });
}

function renderUsers(users) {
    var list = document.getElementById('tdoa-user-list');
    if (!list) return;
    if (!users || users.length === 0) {
        list.innerHTML = '<div class="tp-node-loading">No users found.</div>';
        return;
    }
    var html = '';
    for (var i = 0; i < users.length; i++) {
        var u = users[i];
        var roleCls = (u.role === 'admin') ? 'tdoa-role-admin' : 'tdoa-role-viewer';
        var lastLogin = u.last_login_at ? new Date(u.last_login_at * 1000).toLocaleString() : 'never';
        var created = u.created_at ? new Date(u.created_at * 1000).toLocaleDateString() : '';
        var isSelf = (_currentUser && _currentUser.username === u.username);
        var tfaBadge = u.totp_enabled ? ' <span style="color:#2ecc71;font-size:9px">2FA</span>' : '';
        html += '<div class="tdoa-user-card" data-uid="' + _esc(u.user_id) + '">'
            + '<div class="tp-row">'
            + '<span><b>' + _esc(u.username) + '</b>'
            + '<span class="tdoa-role-badge ' + roleCls + '">' + _esc(u.role) + '</span>'
            + tfaBadge
            + (isSelf ? ' <span style="color:#7a9bbf;font-size:9px">(you)</span>' : '')
            + '</span></div>'
            + '<div class="tdoa-user-meta">Created: ' + _esc(created) + ' &middot; Last login: ' + _esc(lastLogin) + '</div>'
            + '<div class="tdoa-user-actions">';
        if (!isSelf) {
            var otherRole = (u.role === 'admin') ? 'viewer' : 'admin';
            html += '<button class="tdoa-btn-sm toff" onclick="_tdoaChangeRole(&apos;' + _esc(u.user_id) + '&apos;,&apos;' + otherRole + '&apos;)">Make ' + otherRole + '</button>';
        }
        html += '<button class="tdoa-btn-sm toff" onclick="_tdoaResetPw(this,&apos;' + _esc(u.user_id) + '&apos;)">Reset Password</button>';
        if (!isSelf) {
            html += '<button class="tdoa-btn-sm tdel" onclick="_tdoaDeleteUser(this,&apos;' + _esc(u.user_id) + '&apos;)" data-armed="0">Delete</button>';
        }
        if (u.totp_enabled && !isSelf) {
            html += '<button class="tdoa-btn-sm toff" onclick="_tdoaDisable2fa(this,&apos;' + _esc(u.user_id) + '&apos;)" data-armed="0">Disable 2FA</button>';
        }
        html += '</div></div>';
    }
    list.innerHTML = html;
    /* Show change-own-password area if userdb */
    var chpwArea = document.getElementById('tdoa-user-chpw-area');
    if (chpwArea && TDOA.authMode === 'userdb') chpwArea.style.display = '';
}

/* Change user role */
window._tdoaChangeRole = function (userId, newRole) {
    _fetch('/auth/users/' + encodeURIComponent(userId), {
        method: 'PATCH',
        headers: _hdrJson(),
        body: JSON.stringify({ role: newRole })
    })
    .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(function () { loadUsers(); })
    .catch(function (e) { console.error('[Beagle] role change error:', e); });
};

/* Reset password for a user */
window._tdoaResetPw = function (btn, userId) {
    var card = btn.closest('.tdoa-user-card');
    if (!card) return;
    /* Toggle inline password input */
    var existing = card.querySelector('.tdoa-pw-reset-row');
    if (existing) { existing.remove(); return; }
    var row = document.createElement('div');
    row.className = 'tdoa-pw-reset-row';
    row.style.cssText = 'margin-top:3px;display:flex;gap:4px;align-items:center';
    row.innerHTML = '<input type="password" placeholder="New password (min 8)" style="flex:1;background:rgba(15,25,45,0.95);color:#c8d4e8;border:1px solid rgba(80,110,160,0.4);border-radius:3px;font:10px monospace;padding:3px 5px">'
        + '<button class="tdoa-btn-sm ton">Set</button>'
        + '<button class="tdoa-btn-sm toff">X</button>';
    card.appendChild(row);
    var inp = row.querySelector('input');
    row.querySelector('.ton').addEventListener('click', function () {
        var pw = inp.value;
        if (pw.length < 8) { alert('Password must be at least 8 characters.'); return; }
        _fetch('/auth/users/' + encodeURIComponent(userId), {
            method: 'PATCH',
            headers: _hdrJson(),
            body: JSON.stringify({ password: pw })
        })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function () { row.remove(); })
        .catch(function (e) { alert('Error: ' + e.message); });
    });
    row.querySelector('.toff').addEventListener('click', function () { row.remove(); });
};

/* Delete user (armed confirmation) */
window._tdoaDeleteUser = function (btn, userId) {
    if (btn.getAttribute('data-armed') !== '1') {
        btn.setAttribute('data-armed', '1');
        btn.textContent = 'Sure?';
        setTimeout(function () {
            if (btn.getAttribute('data-armed') === '1') {
                btn.setAttribute('data-armed', '0');
                btn.textContent = 'Delete';
            }
        }, 3000);
        return;
    }
    btn.setAttribute('data-armed', '0');
    btn.disabled = true;
    _fetch('/auth/users/' + encodeURIComponent(userId), {
        method: 'DELETE',
        headers: _hdr()
    })
    .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(function () { loadUsers(); })
    .catch(function (e) { console.error('[Beagle] delete user error:', e); btn.disabled = false; });
};

/* Disable 2FA for a user (admin recovery, armed confirmation) */
window._tdoaDisable2fa = function (btn, userId) {
    if (btn.getAttribute('data-armed') !== '1') {
        btn.setAttribute('data-armed', '1');
        btn.textContent = 'Sure?';
        setTimeout(function () {
            if (btn.getAttribute('data-armed') === '1') {
                btn.setAttribute('data-armed', '0');
                btn.textContent = 'Disable 2FA';
            }
        }, 3000);
        return;
    }
    btn.setAttribute('data-armed', '0');
    btn.disabled = true;
    _fetch('/auth/2fa/disable', {
        method: 'POST',
        headers: _hdrJson(),
        body: JSON.stringify({ user_id: userId })
    })
    .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
    .then(function () { loadUsers(); })
    .catch(function (e) { console.error('[Beagle] disable 2fa error:', e); btn.disabled = false; });
};

/* Create user form */
(function () {
    var createBtn = document.getElementById('tdoa-create-user-btn');
    var area = document.getElementById('tdoa-user-create-area');
    if (!createBtn || !area) return;
    createBtn.addEventListener('click', function () {
        if (area.style.display !== 'none') { area.style.display = 'none'; _usersEditing = false; return; }
        _usersEditing = true;
        area.style.display = '';
        area.innerHTML = '<div class="tdoa-user-form">'
            + '<label>Username (required)</label>'
            + '<input id="uf-user" placeholder="e.g. alice">'
            + '<label>Password (min 8 chars)</label>'
            + '<input type="password" id="uf-pass" placeholder="Password">'
            + '<label>Role</label>'
            + '<select id="uf-role"><option value="viewer">viewer</option><option value="admin">admin</option></select>'
            + '<div id="uf-error" style="color:#e74c3c;font-size:10px;margin-top:3px"></div>'
            + '<div style="display:flex;gap:4px;margin-top:4px">'
            + '<button class="tdoa-btn-sm ton" id="uf-submit">Create</button>'
            + '<button class="tdoa-btn-sm toff" id="uf-cancel">Cancel</button>'
            + '</div></div>';
        document.getElementById('uf-cancel').addEventListener('click', function () {
            area.style.display = 'none'; _usersEditing = false;
        });
        document.getElementById('uf-submit').addEventListener('click', function () {
            var username = (document.getElementById('uf-user').value || '').trim();
            var password = document.getElementById('uf-pass').value || '';
            var role = document.getElementById('uf-role').value;
            var errEl = document.getElementById('uf-error');
            if (!username) { errEl.textContent = 'Username is required.'; return; }
            if (password.length < 8) { errEl.textContent = 'Password must be at least 8 characters.'; return; }
            _fetch('/auth/register', {
                method: 'POST',
                headers: _hdrJson(),
                body: JSON.stringify({ username: username, password: password, role: role })
            })
            .then(function (r) {
                if (!r.ok) return r.json().then(function (d) { throw new Error(d.detail || 'HTTP ' + r.status); });
                return r.json();
            })
            .then(function () {
                area.style.display = 'none'; _usersEditing = false;
                loadUsers();
            })
            .catch(function (e) { errEl.textContent = e.message; });
        });
    });
})();

/* Change own password */
(function () {
    var btn = document.getElementById('tdoa-chpw-btn');
    if (!btn) return;
    btn.addEventListener('click', function () {
        var inp = document.getElementById('tdoa-chpw-input');
        var errEl = document.getElementById('tdoa-chpw-error');
        var pw = inp ? inp.value : '';
        if (pw.length < 8) {
            if (errEl) { errEl.textContent = 'Password must be at least 8 characters.'; errEl.style.display = 'block'; }
            return;
        }
        if (!_currentUser) return;
        _fetch('/auth/users/' + encodeURIComponent(_currentUser.user_id || ''), {
            method: 'PATCH',
            headers: _hdrJson(),
            body: JSON.stringify({ password: pw })
        })
        .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
        .then(function () {
            /* Sessions revoked - must re-login */
            sessionStorage.removeItem('tdoa_token');
            _currentUser = null;
            _setUserInfo(null);
            _showLogin('Password changed. Please log in again.');
        })
        .catch(function (e) {
            if (errEl) { errEl.textContent = 'Error: ' + e.message; errEl.style.display = 'block'; }
        });
    });
})();

/* Auto-refresh users every 10 s when the users tab is active */
setInterval(function () { if (_usersTabActive && !_usersEditing) loadUsers(); }, 10000);

})();
</script>"""


def _render_control_panel(
    server_label: str,
    last_fix_ts: float,
    default_max_age_s: float,
    map_id: str = "",
    heat_layer_id: str = "",
    auth_token: str = "",
    auth_mode: str = "token",
    google_oauth_enabled: bool = False,
) -> str:
    """
    Return the HTML/CSS/JS control panel to inject before </body>.

    To add a new control in a future feature:
      - Add key(s) to the TDOA dict below
      - Add a row or button to _PANEL_HTML
      - Add handler JS to _PANEL_JS that reads TDOA.<key>
    """
    tdoa_data = json.dumps({
        "serverLabel": server_label,
        "lastFixTs": last_fix_ts,
        "defaultMaxAgeS": default_max_age_s,
        "mapId": map_id,
        "heatLayerId": heat_layer_id,
        "authToken": auth_token,
        "authMode": auth_mode,
        "googleOAuthEnabled": google_oauth_enabled,
    })
    # Escape </ to prevent accidental </script> tag termination inside the data block.
    tdoa_data = tdoa_data.replace("</", r"<\/")
    data_block = f"<script>var TDOA = {tdoa_data};</script>"
    return _PANEL_CSS + _PANEL_HTML + data_block + _PANEL_JS


# ---------------------------------------------------------------------------
# Color helpers
# ---------------------------------------------------------------------------

def _age_color(age_s: float, max_age_s: float) -> str:
    """
    Map age (0 = newest, max_age_s = oldest) to a CSS hex colour.
    Newest -> red (#d73027), oldest -> grey (#aaaaaa).
    """
    if max_age_s <= 0:
        t = 0.0
    else:
        t = max(0.0, min(1.0, age_s / max_age_s))

    # red -> orange -> yellow -> grey
    # Defined as three RGB stops
    stops = [
        (0.0,  (215, 48,  39)),   # red
        (0.5,  (253, 174,  97)),  # orange
        (0.75, (255, 255, 191)),  # yellow
        (1.0,  (170, 170, 170)),  # grey
    ]
    # Linear interpolation between stops
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        if t <= t1:
            frac = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            r = int(c0[0] + frac * (c1[0] - c0[0]))
            g = int(c0[1] + frac * (c1[1] - c0[1]))
            b = int(c0[2] + frac * (c1[2] - c0[2]))
            return f"#{r:02x}{g:02x}{b:02x}"
    return "#aaaaaa"


# ---------------------------------------------------------------------------
# Hyperbola arc generation
# ---------------------------------------------------------------------------

def _hyperbola_points(
    lat_a: float, lon_a: float,
    lat_b: float, lon_b: float,
    tdoa_s: float,
    n_points: int = 500,
    radius_km: float = 150.0,
) -> list[tuple[float, float]]:
    """
    Return n_points ordered points on the hyperbola branch defined by:
        dist(P, A) - dist(P, B) = tdoa_s * c

    Uses the analytic parametric form in a local flat-earth coordinate frame
    with foci at A and B, then converts back to (lat, lon).

    Canonical hyperbola (A at (-c_h, 0), B at (+c_h, 0)):
        x = sgn * a * cosh(t),  y = b_h * sinh(t)
    where
        a   = |tdoa_s * c| / 2      (semi-transverse axis)
        c_h = baseline / 2           (focal half-distance)
        b_h = sqrt(c_h^2 - a^2)     (semi-conjugate axis)
        sgn = sign(tdoa_s)           (selects the branch: +1 -> P closer to B)

    The parameter t runs from -t_max to +t_max, clipped so all points lie
    within radius_km of the baseline midpoint.
    """
    target_diff_m = tdoa_s * _C_M_S

    mid_lat = (lat_a + lat_b) / 2.0
    mid_lon = (lon_a + lon_b) / 2.0
    cos_mid = math.cos(math.radians(mid_lat))
    m_per_deg_lat = 111_195.0
    m_per_deg_lon = 111_195.0 * cos_mid if cos_mid > 0 else 111_195.0

    def to_local(lat: float, lon: float) -> tuple[float, float]:
        return (lon - mid_lon) * m_per_deg_lon, (lat - mid_lat) * m_per_deg_lat

    def to_latlon(x_east: float, y_north: float) -> tuple[float, float]:
        return mid_lat + y_north / m_per_deg_lat, mid_lon + x_east / m_per_deg_lon

    ax, ay = to_local(lat_a, lon_a)
    bx, by = to_local(lat_b, lon_b)

    baseline_m = math.hypot(bx - ax, by - ay)
    c_h = baseline_m / 2.0
    a = abs(target_diff_m) / 2.0

    # Degenerate: |TDOA| >= baseline/c is physically impossible for a real
    # transmitter but can occur with noisy solver output.
    if a >= c_h or c_h < 1.0:
        if c_h < 1.0:
            _logger.debug(
                "LOP not rendered: baseline too short (%.1f m) - "
                "nodes are co-located or have identical positions",
                baseline_m,
            )
        else:
            _logger.info(
                "LOP not rendered: |TDOA| (%.1f m) >= baseline (%.1f m) - "
                "degenerate hyperbola",
                abs(target_diff_m), baseline_m,
            )
        return []

    b_h = math.sqrt(c_h ** 2 - a ** 2)

    # Angle of the AB baseline measured from east (x-axis in local frame)
    theta = math.atan2(by - ay, bx - ax)

    # Branch selection: positive tdoa_s means d_a > d_b (P closer to B)
    sgn = 1.0 if target_diff_m >= 0.0 else -1.0

    # Clip t so the arc stays within radius_km of the midpoint.
    # Distance from origin along the curve: sqrt(a^2 + c_h^2 * sinh^2(t))
    # Solve for t_max: sinh(t_max) = sqrt(max(radius_m^2 - a^2, 0)) / c_h
    radius_m = radius_km * 1000.0
    sinh_arg_sq = (radius_m ** 2 - a ** 2) / c_h ** 2
    t_max = math.asinh(math.sqrt(max(sinh_arg_sq, 0.0)))
    if t_max < 1e-9:
        t_max = 0.1  # ensure at least a small arc

    pts: list[tuple[float, float]] = []
    for k in range(n_points):
        t = -t_max + 2.0 * t_max * k / max(n_points - 1, 1)
        x_can = sgn * a * math.cosh(t)
        y_can = b_h * math.sinh(t)
        # Rotate from canonical (AB along x-axis) to local east/north frame
        x_east  = x_can * math.cos(theta) - y_can * math.sin(theta)
        y_north = x_can * math.sin(theta) + y_can * math.cos(theta)
        pts.append(to_latlon(x_east, y_north))

    return pts


# ---------------------------------------------------------------------------
# GeoJSON builder for dynamic fix layer
# ---------------------------------------------------------------------------

def build_fix_geojson(
    fixes: list[dict[str, Any]],
    recent_events: list[dict[str, Any]],
    max_age_s: float,
    hyperbola_points: int = 500,
    now: float | None = None,
    heartbeats: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Build a GeoJSON FeatureCollection of fix markers and hyperbola arcs.

    Served by ``GET /map/data?max_age_s=N``.  The browser JS renders these
    as Leaflet layers without a full page reload.

    Fix features are ordered oldest-first so that Leaflet renders the most
    recent fix on top (later additions sit above earlier ones).

    Hyperbola arcs are generated for:
    - The single most recent 3+-node fix (full hyperbola set)
    - All 2-node fixes (LOP arcs, styled differently by the JS)

    Parameters
    ----------
    fixes : list of fix dicts from db.fetch_fixes() - newest first
    recent_events : list of event dicts (for node lat/lon lookup)
    max_age_s : age-out window (0 = all fixes)
    hyperbola_points : points per hyperbola arc
    now : override for current time (tests)
    heartbeats : node_id -> heartbeat dict (from app.state.heartbeats)
    """
    if now is None:
        now = time.time()

    # Collect the most recent node position for each node_id
    node_pos: dict[str, tuple[float, float]] = {}
    for ev in recent_events:
        nid = ev["node_id"]
        if nid not in node_pos:
            node_pos[nid] = (ev["node_lat"], ev["node_lon"])

    features: list[dict[str, Any]] = []
    newest_full_fix: dict[str, Any] | None = None  # newest 3+-node fix
    lop_fixes: list[dict[str, Any]] = []            # all 2-node LOPs
    last_fix_ts: float = 0.0

    # Process oldest-first so Leaflet renders newest on top
    for fix in reversed(fixes):
        age_s = now - fix["computed_at"]
        if max_age_s > 0 and age_s > max_age_s:
            continue

        is_lop = fix["node_count"] == 2
        if is_lop:
            lop_fixes.append(fix)

        lat, lon = fix["latitude_deg"], fix["longitude_deg"]
        nodes = fix.get("nodes", [])
        # 2-node LOPs: don't emit a point marker (the position is arbitrary);
        # only the hyperbola arc is meaningful.
        if not is_lop:
            features.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "feature_type": "fix",
                    "fix_id": fix["id"],
                    "age_s": age_s,
                    "computed_at": fix["computed_at"],
                    "onset_time_ns": fix["onset_time_ns"],
                    "latitude_deg": lat,
                    "longitude_deg": lon,
                    "residual_ns": fix.get("residual_ns"),
                    "node_count": fix["node_count"],
                    "nodes": nodes,
                    "channel_hz": fix["channel_hz"],
                    "event_type": fix["event_type"],
                    "tooltip": f"Fix {fix['id']} - {age_s:.0f}s ago",
                },
            })

        if not is_lop and (newest_full_fix is None or fix["computed_at"] > newest_full_fix["computed_at"]):
            newest_full_fix = fix
        if fix["computed_at"] > last_fix_ts:
            last_fix_ts = fix["computed_at"]

    # Hyperbola arcs for the most recent 3+-node fix
    if newest_full_fix is not None:
        _collect_hyperbola_features(features, newest_full_fix, node_pos, hyperbola_points)

    # Hyperbola arcs for ALL 2-node LOPs (styled differently by the JS)
    for lop_fix in lop_fixes:
        _collect_hyperbola_features(
            features, lop_fix, node_pos, hyperbola_points, feature_type="lop",
        )

    # Node features - most recent position per node_id from events and heartbeats.
    # Included here so the JS receives fresh node state on every loadFixes()
    # call and on every SSE new_fix event, without requiring a page reload.
    seen_nodes: dict[str, dict[str, Any]] = {}
    for ev in recent_events:
        nid = ev["node_id"]
        if nid not in seen_nodes or ev["received_at"] > seen_nodes[nid]["received_at"]:
            seen_nodes[nid] = ev

    hb_map = heartbeats or {}
    for nid, ev in seen_nodes.items():
        lat, lon = ev["node_lat"], ev["node_lon"]
        age_s = now - ev["received_at"]
        hb = hb_map.get(nid)
        hb_age = now - hb["received_at"] if hb else None
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "feature_type": "node",
                "node_id": nid,
                "latitude_deg": lat,
                "longitude_deg": lon,
                "age_s": age_s,
                "received_at": ev["received_at"],
                "heartbeat_age_s": hb_age,
                "tooltip": nid,
            },
        })

    # Nodes known only from heartbeats (no events yet)
    for nid, hb in hb_map.items():
        if nid in seen_nodes:
            continue
        lat = hb.get("latitude_deg")
        lon = hb.get("longitude_deg")
        if lat is None or lon is None:
            continue
        hb_age = now - hb["received_at"]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "feature_type": "node",
                "node_id": nid,
                "latitude_deg": lat,
                "longitude_deg": lon,
                "age_s": hb_age,
                "received_at": hb["received_at"],
                "heartbeat_age_s": hb_age,
                "tooltip": nid,
            },
        })

    fix_count = sum(
        1 for f in features if f["properties"].get("feature_type") == "fix"
    )
    return {
        "type": "FeatureCollection",
        "properties": {
            "lastFixTs": last_fix_ts,
            "fixCount": fix_count,
            "maxAgeS": max_age_s,
        },
        "features": features,
    }


def _collect_hyperbola_features(
    features: list[dict[str, Any]],
    fix: dict[str, Any],
    node_pos: dict[str, tuple[float, float]],
    hyperbola_points: int,
    feature_type: str = "hyperbola",
) -> None:
    """Append GeoJSON LineString features for hyperbola arcs to features.

    feature_type controls the GeoJSON property:
      "hyperbola" - arcs for a 3+-node full fix (solid red)
      "lop"       - 2-node line-of-position (dashed amber)
    """
    nodes: list[str] = fix.get("nodes", [])
    fix_lat = fix["latitude_deg"]
    fix_lon = fix["longitude_deg"]

    node_list = [n for n in nodes if n in node_pos]
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            n_a, n_b = node_list[i], node_list[j]
            lat_a, lon_a = node_pos[n_a]
            lat_b, lon_b = node_pos[n_b]

            d_a = haversine_m(fix_lat, fix_lon, lat_a, lon_a)
            d_b = haversine_m(fix_lat, fix_lon, lat_b, lon_b)
            tdoa_s = (d_a - d_b) / _C_M_S

            pts = _hyperbola_points(
                lat_a, lon_a, lat_b, lon_b, tdoa_s, n_points=hyperbola_points
            )
            if pts:
                label = "LOP" if feature_type == "lop" else "TDOA"
                tooltip = f"{label} {n_a}<->{n_b}: {tdoa_s * 1e6:.2f} usec"
                age_s = time.time() - fix["computed_at"]
                # GeoJSON uses [lon, lat] coordinate order
                coordinates = [[p[1], p[0]] for p in pts]
                features.append({
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coordinates},
                    "properties": {
                        "feature_type": feature_type,
                        "fix_id": fix["id"],
                        "pair": f"{n_a}<->{n_b}",
                        "tdoa_us": tdoa_s * 1e6,
                        "age_s": age_s,
                        "computed_at": fix["computed_at"],
                        "tooltip": tooltip,
                    },
                })


# ---------------------------------------------------------------------------
# Main map builder (static shell - fix layer loaded dynamically)
# ---------------------------------------------------------------------------

def build_map(
    fixes: list[dict[str, Any]],
    recent_events: list[dict[str, Any]],
    max_age_s: float = 3600.0,
    center_lat: float = 47.6,
    center_lon: float = -122.3,
    server_label: str = "",
    heatmap_cells: list[list[float]] | None = None,
    auth_token: str = "",
    auth_mode: str = "token",
    google_oauth_enabled: bool = False,
) -> str:
    """
    Build a Folium map and return the HTML as a string.

    Renders the static shell: node markers, sync-TX markers, and the heatmap
    layer.  Fix markers and hyperbola arcs are NOT embedded here; the browser
    JS fetches them from ``GET /map/data`` on load and on each SSE event.

    Parameters
    ----------
    fixes : list of fix dicts from db.fetch_fixes() - used only for map centre
    recent_events : list of event dicts (for sync tx markers; node markers are dynamic)
    max_age_s : default age-out window passed to the page JS as ``defaultMaxAgeS``
    center_lat, center_lon : fallback map centre if no fixes available
    server_label : host:port string shown in the control panel header
    heatmap_cells : [[lat, lon, weight], ...] from db.fetch_heatmap_cells()
    """
    # Determine map centre from the most recent fix, falling back to config
    if fixes:
        map_lat = fixes[0]["latitude_deg"]
        map_lon = fixes[0]["longitude_deg"]
    else:
        map_lat, map_lon = center_lat, center_lon

    m = folium.Map(location=[map_lat, map_lon], zoom_start=11, tiles="OpenStreetMap")

    # Node markers are rendered dynamically by the browser JS via the
    # /map/data GeoJSON response (build_fix_geojson adds feature_type="node"
    # entries), so they update live as new events arrive without page reloads.

    # -----------------------------------------------------------------------
    # Sync transmitter markers - unique sync_tx_id
    # -----------------------------------------------------------------------
    seen_sync: dict[str, dict[str, Any]] = {}
    for ev in recent_events:
        sid = ev.get("sync_tx_id", "")
        if sid and sid not in seen_sync:
            seen_sync[sid] = ev

    for sid, ev in seen_sync.items():
        lat, lon = ev["sync_tx_lat"], ev["sync_tx_lon"]
        popup_html = (
            f"<b>Sync TX</b>: {sid}<br>"
            f"Lat: {lat:.5f}, Lon: {lon:.5f}"
        )
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=200),
            tooltip=sid,
            icon=folium.Icon(color="gray", icon="signal", prefix="fa"),
        ).add_to(m)

    # -----------------------------------------------------------------------
    # Heat map layer (togglable via Leaflet LayerControl)
    # Always created so the toggle button is always visible.
    # An empty cell list renders as a transparent layer; the JS loadHeatmap()
    # function updates the data live when SSE new_fix events arrive.
    # -----------------------------------------------------------------------
    heat_fg = folium.FeatureGroup(name="Heat Map", show=True)
    heat_layer_id = heat_fg.get_name()
    HeatMap(
        heatmap_cells or [],
        radius=20,
        blur=15,
        min_opacity=0.3,
    ).add_to(heat_fg)
    heat_fg.add_to(m)
    folium.LayerControl(position="bottomright", collapsed=False).add_to(m)

    last_fix_ts = fixes[0]["computed_at"] if fixes else 0.0
    panel = _render_control_panel(
        server_label=server_label,
        last_fix_ts=last_fix_ts,
        default_max_age_s=max_age_s,
        map_id=m.get_name(),
        heat_layer_id=heat_layer_id,
        auth_token=auth_token,
        auth_mode=auth_mode,
        google_oauth_enabled=google_oauth_enabled,
    )
    m.get_root().header.add_child(_BrancaElement("<title>Beagle</title>\n" + _FAVICON_HTML))
    m.get_root().html.add_child(_BrancaElement(panel))
    return m.get_root().render()
