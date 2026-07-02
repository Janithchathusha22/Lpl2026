"""LPL 2026 premium Streamlit prediction dashboard."""

import sys, os, json, time, base64, html
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

matplotlib.rcParams['axes.facecolor']  = '#0f111a'
matplotlib.rcParams['figure.facecolor']= '#0f111a'
matplotlib.rcParams['text.color']      = '#e2e8f0'
matplotlib.rcParams['axes.labelcolor'] = '#94a3b8'
matplotlib.rcParams['xtick.color']     = '#94a3b8'
matplotlib.rcParams['ytick.color']     = '#94a3b8'

API = os.environ.get("LPL_API_URL", "http://127.0.0.1:8000").rstrip("/")

# Keep navigation inside the public Streamlit URL. The FastAPI service is an
# internal prediction engine on Community Cloud and is not exposed publicly.
HTML_DASHBOARD_URL = "?page=command"
HTML_FIELD_URL = "?page=field_lab"
HTML_FORECAST_URL = "?page=forecast"
HTML_TACTICS_URL = "?page=tactics"

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LPL 2026 Prediction Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="collapsed",
)

ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "assets")


def asset_data_uri(filename):
    path = os.path.join(ASSET_DIR, filename)
    try:
        with open(path, "rb") as image_file:
            payload = base64.b64encode(image_file.read()).decode("ascii")
        return f"data:image/png;base64,{payload}"
    except OSError:
        return ""


HERO_ART = asset_data_uri("lpl-command-center.png")
PLAYER_ART = asset_data_uri("player-analysis-lab.png")
LIVE_ART = asset_data_uri("live-simulation-lab.png")
TOURNAMENT_ART = asset_data_uri("tournament-command-center.png")

# ── Premium Dark CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}
.stApp { background: #080b14 !important; }
header[data-testid="stHeader"] { height:0; background:transparent; }
[data-testid="stToolbar"], [data-testid="stDecoration"], #MainMenu { display:none !important; }

.hero {
    background: linear-gradient(135deg,#0d1b4b 0%,#1a3a8f 40%,#0a2260 100%);
    border: 1px solid rgba(99,179,237,.25);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 70% 50%, rgba(99,179,237,.12) 0%, transparent 70%);
}
.hero h1 { font-size: 2.6rem; font-weight: 800; color: #fff; margin: 0; line-height: 1.2; }
.hero p  { color: #90cdf4; font-size: 1.05rem; margin-top: .5rem; }

.card {
    background: rgba(22,33,62,.6);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}
.card h3 { margin: 0 0 .8rem; color: #63b3ed; font-weight: 700; font-size: 1rem; }

.metric-big { font-size: 2.4rem; font-weight: 800; color: #fff; line-height: 1; }
.metric-sub { font-size: .8rem; color: #718096; text-transform: uppercase; letter-spacing: .08em; }

.win-bar-wrap { height: 14px; border-radius: 7px; background: rgba(255,255,255,.07); overflow: hidden; margin: .5rem 0; }
.win-bar { height: 100%; border-radius: 7px; transition: width .4s ease; }

.pill {
    display: inline-block;
    padding: .2rem .8rem;
    border-radius: 999px;
    font-size: .78rem;
    font-weight: 600;
    margin: .15rem .1rem;
}
.pill-blue  { background: rgba(59,130,246,.25); color: #93c5fd; border: 1px solid rgba(59,130,246,.4); }
.pill-green { background: rgba(16,185,129,.2); color: #6ee7b7; border: 1px solid rgba(16,185,129,.35); }
.pill-red   { background: rgba(239,68,68,.2);  color: #fca5a5; border: 1px solid rgba(239,68,68,.35); }
.pill-gold  { background: rgba(245,158,11,.2); color: #fcd34d; border: 1px solid rgba(245,158,11,.35); }
.pill-purple{ background: rgba(139,92,246,.2); color: #c4b5fd; border: 1px solid rgba(139,92,246,.35); }

.strength-box { background: rgba(16,185,129,.1); border: 1px solid rgba(16,185,129,.3); border-radius: 10px; padding: 1rem; }
.weakness-box { background: rgba(239,68,68,.1);  border: 1px solid rgba(239,68,68,.3);  border-radius: 10px; padding: 1rem; }
.strength-box h4, .weakness-box h4 { margin: 0 0 .5rem; font-size: .9rem; }

.xai-bar-pos { background: linear-gradient(90deg,#10b981,#059669); border-radius: 4px; height: 10px; }
.xai-bar-neg { background: linear-gradient(90deg,#ef4444,#b91c1c); border-radius: 4px; height: 10px; }

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(22,33,62,.7);
    border-radius: 12px;
    padding: .4rem;
    gap: .2rem;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #94a3b8;
    font-weight: 600;
    padding: .5rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    background: rgba(59,130,246,.3) !important;
    color: #93c5fd !important;
}

/* Creative command-center refresh */
.stApp {
    background:
      radial-gradient(circle at 12% 6%, rgba(71,209,140,.13), transparent 30%),
      radial-gradient(circle at 86% 12%, rgba(101,215,255,.11), transparent 32%),
      linear-gradient(135deg,#05070d 0%,#101117 48%,#07110d 100%) !important;
}
.hero {
    background:
      linear-gradient(135deg,rgba(11,16,27,.96),rgba(7,15,15,.94)),
      radial-gradient(circle at 68% 44%, rgba(245,166,35,.2), transparent 42%);
    border: 1px solid rgba(245,166,35,.42);
    border-radius: 8px;
    padding: 2rem 2.4rem;
    box-shadow: 0 24px 70px rgba(0,0,0,.28);
}
.hero h1 { color: #ffd37a; }
.hero p { color: #8c99b5; }
.card {
    background: linear-gradient(145deg,rgba(255,255,255,.055),rgba(255,255,255,.025));
    border: 1px solid rgba(255,255,255,.085);
    border-radius: 8px;
    box-shadow: 0 18px 42px rgba(0,0,0,.18), inset 0 1px 0 rgba(255,255,255,.05);
}
.card h3 { color: #ffd37a; }
.stTabs [data-baseweb="tab-list"] {
    background: rgba(7,11,18,.86);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 8px;
}
.stTabs [data-baseweb="tab"] { border-radius: 6px; color: #8c99b5; }
.stTabs [aria-selected="true"] {
    background: rgba(245,166,35,.12) !important;
    color: #ffd37a !important;
}
.launch-grid {
    display: grid;
    grid-template-columns: minmax(260px, .9fr) minmax(280px, 1.1fr);
    gap: 1rem;
    align-items: stretch;
    margin-top: .8rem;
}
.launch-panel {
    background: linear-gradient(145deg,rgba(255,255,255,.06),rgba(255,255,255,.025));
    border: 1px solid rgba(255,255,255,.09);
    border-radius: 8px;
    padding: 1.15rem;
    box-shadow: 0 18px 42px rgba(0,0,0,.18);
}
.launch-panel h3 {
    margin: 0 0 .55rem;
    color: #ffd37a;
    font-size: .78rem;
    letter-spacing: .08em;
    text-transform: uppercase;
}
.launch-title { font-size: 1.28rem; font-weight: 800; color: #fff; line-height: 1.25; }
.launch-copy { color: #8c99b5; font-size: .86rem; line-height: 1.62; margin-top: .6rem; }
.launch-metrics {
    display: grid;
    grid-template-columns: repeat(2,minmax(0,1fr));
    gap: .55rem;
    margin-top: .9rem;
}
.launch-metric {
    background: rgba(0,0,0,.24);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 8px;
    padding: .72rem;
}
.launch-metric b { display:block; color:#fff; font-size:1.08rem; }
.launch-metric span { display:block; color:#8c99b5; font-size:.68rem; margin-top:.1rem; }
.launch-action {
    display:flex;
    align-items:center;
    justify-content:center;
    min-height:58px;
    margin-top: 1rem;
    border-radius: 8px;
    border: 1px solid rgba(245,166,35,.6);
    background: linear-gradient(135deg,#f5a623,#ffd37a);
    color: #07100b !important;
    text-decoration: none !important;
    font-size: 1rem;
    font-weight: 900;
    box-shadow: 0 16px 34px rgba(245,166,35,.16);
}
.launch-action.secondary {
    border-color: rgba(255,255,255,.22);
    background: #121720;
    color: #f0f4ff !important;
}
.field-mini {
    position: relative;
    min-height: 300px;
    overflow: hidden;
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,.1);
    background: radial-gradient(ellipse at 50% 55%,rgba(48,128,64,.52),rgba(8,20,15,.52) 42%,rgba(5,8,15,.96) 74%);
}
.field-mini::before {
    content:"";
    position:absolute;
    inset:16%;
    border:2px solid rgba(245,166,35,.68);
    border-radius:50%;
    transform:perspective(520px) rotateX(58deg);
}
.field-mini::after {
    content:"";
    position:absolute;
    left:50%;
    top:50%;
    width:72px;
    height:210px;
    background:linear-gradient(#c79a4f,#a87335);
    border-radius:3px;
    transform:translate(-50%,-44%) perspective(520px) rotateX(58deg);
    box-shadow:0 0 0 1px rgba(255,255,255,.16) inset;
}
.field-dot { position:absolute; width:14px; height:14px; border-radius:50%; box-shadow:0 0 22px currentColor; z-index:2; }
.field-dot.d1 { left:47%; top:28%; color:#fff; background:#fff; }
.field-dot.d2 { left:24%; top:46%; color:#ffe055; background:#ffe055; }
.field-dot.d3 { left:72%; top:36%; color:#55e07a; background:#55e07a; }
.field-dot.d4 { left:34%; top:68%; color:#ff6868; background:#ff6868; }
@media (max-width: 900px) {
    .launch-grid { grid-template-columns: 1fr; }
}

/* Streamlit home command center */
.block-container { padding-top: 1.15rem; max-width: 1820px; }
.command-hero {
    position: relative;
    min-height: 620px;
    margin: 0 0 1.15rem;
    overflow: hidden;
    border: 1px solid rgba(84,171,255,.36);
    border-radius: 22px;
    background: #030718;
    isolation: isolate;
    box-shadow: 0 32px 90px rgba(0,0,0,.5), 0 0 70px rgba(73,92,255,.08);
}
.command-hero > img {
    position: absolute;
    inset: 0;
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center;
    z-index: -3;
    filter: saturate(1.08) contrast(1.04);
}
.command-hero::before {
    content: "";
    position: absolute;
    inset: 0;
    z-index: -2;
    background:
      linear-gradient(90deg,rgba(2,6,20,.98) 0%,rgba(2,7,22,.9) 34%,rgba(3,7,19,.2) 72%,rgba(2,5,16,.44) 100%),
      linear-gradient(0deg,rgba(2,5,16,.98),transparent 52%);
}
.command-hero::after {
    content: "";
    position: absolute;
    inset: 0;
    z-index: -1;
    background: radial-gradient(circle at 72% 49%,transparent 0 21%,rgba(3,7,20,.12) 44%,rgba(3,7,20,.45) 82%);
}
.command-copy { max-width: 710px; padding: 66px 54px 205px; }
.command-live {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 7px 12px;
    border: 1px solid rgba(83,212,255,.38);
    border-radius: 99px;
    background: rgba(3,19,43,.68);
    color: #92e8ff;
    font-size: .67rem;
    font-weight: 900;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    backdrop-filter: blur(12px);
}
.command-live i {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #45f59a;
    box-shadow: 0 0 15px #45f59a;
}
.command-copy h1 {
    margin: 24px 0 16px;
    color: #fff;
    font-size: clamp(3rem,5.5vw,5.4rem);
    font-weight: 900;
    letter-spacing: -.06em;
    line-height: .92;
    text-shadow: 0 16px 42px rgba(0,0,0,.65);
}
.command-copy h1 span {
    display: block;
    background: linear-gradient(105deg,#54dcff 5%,#9882ff 50%,#fa58c5 88%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.command-copy p {
    max-width: 590px;
    margin: 0;
    color: #bdc9df;
    font-size: .98rem;
    line-height: 1.75;
}
.command-actions { display:flex; flex-wrap:wrap; gap:10px; margin-top:26px; }
.command-actions a {
    display:inline-flex;
    min-height:46px;
    align-items:center;
    justify-content:center;
    padding:0 18px;
    border:1px solid rgba(255,255,255,.16);
    border-radius:10px;
    background:rgba(5,12,30,.72);
    color:#eef7ff !important;
    text-decoration:none !important;
    font-size:.8rem;
    font-weight:900;
    backdrop-filter:blur(12px);
}
.command-actions a:first-child {
    border:0;
    color:#020817 !important;
    background:linear-gradient(115deg,#4bdcff,#8b7dff 55%,#f65cc3);
    box-shadow:0 14px 36px rgba(72,149,255,.28);
}
.command-signal {
    position:absolute;
    top:24px;
    right:24px;
    width:310px;
    padding:15px 16px;
    border:1px solid rgba(91,207,255,.26);
    border-radius:14px;
    background:linear-gradient(145deg,rgba(3,12,33,.82),rgba(10,8,29,.6));
    backdrop-filter:blur(16px);
}
.command-signal small {
    color:#84e1ff;
    font-size:.58rem;
    font-weight:900;
    letter-spacing:1.4px;
    text-transform:uppercase;
}
.command-signal strong { display:block; margin-top:5px; color:#fff; font-size:1.06rem; }
.command-signal span { color:#ffcf7d; font-size:.67rem; }
.command-stats {
    position:absolute;
    left:18px;
    right:18px;
    bottom:18px;
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:8px;
    padding:9px;
    border:1px solid rgba(91,165,255,.2);
    border-radius:16px;
    background:rgba(2,6,19,.78);
    backdrop-filter:blur(18px);
}
.command-stat { padding:11px 16px; border-right:1px solid rgba(255,255,255,.08); }
.command-stat:last-child { border-right:0; }
.command-stat b { display:block; color:#fff; font-size:1.18rem; }
.command-stat span { display:block; margin-top:3px; color:#8190ae; font-size:.57rem; font-weight:800; letter-spacing:.8px; text-transform:uppercase; }

/* Player intelligence experience */
.player-visual {
    position:relative;
    min-height:370px;
    margin:0 0 1.2rem;
    overflow:hidden;
    border:1px solid rgba(67,157,255,.32);
    border-radius:18px;
    background:#02091a;
    box-shadow:0 25px 65px rgba(0,0,0,.4);
}
.player-visual img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; object-position:center; }
.player-visual::after { content:""; position:absolute; inset:0; background:linear-gradient(90deg,rgba(2,7,22,.97),rgba(2,7,22,.7) 38%,rgba(2,7,22,.1) 75%),linear-gradient(0deg,rgba(2,7,22,.82),transparent 50%); }
.player-visual-copy { position:relative; z-index:1; width:44%; padding:48px 42px; }
.player-visual-copy small { color:#65dcff; font-size:.64rem; font-weight:900; letter-spacing:1.6px; text-transform:uppercase; }
.player-visual-copy h2 { margin:12px 0 11px; color:#fff; font-size:2.3rem; line-height:1.02; letter-spacing:-.04em; }
.player-visual-copy p { color:#a8b6d0; font-size:.84rem; line-height:1.68; }
.player-visual-pills { display:flex; flex-wrap:wrap; gap:7px; margin-top:18px; }
.player-visual-pills span { padding:6px 10px; border:1px solid rgba(102,210,255,.22); border-radius:99px; background:rgba(5,20,45,.62); color:#c8eaff; font-size:.62rem; font-weight:800; }
.intel-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-top:.2rem; }
.intel-panel {
    position:relative;
    overflow:hidden;
    padding:18px;
    border:1px solid rgba(255,255,255,.08);
    border-radius:14px;
    background:rgba(5,10,22,.62);
}
.intel-panel::before { content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--accent); }
.intel-panel.strength { --accent:#42e39b; background:linear-gradient(135deg,rgba(29,174,104,.12),rgba(4,9,21,.7)); }
.intel-panel.weakness { --accent:#ff6689; background:linear-gradient(135deg,rgba(232,52,94,.12),rgba(4,9,21,.7)); }
.intel-kicker { color:var(--accent); font-size:.61rem; font-weight:900; letter-spacing:1.2px; text-transform:uppercase; }
.intel-panel h3 { margin:5px 0 13px; color:#fff; font-size:1.08rem; }
.intel-row { display:grid; grid-template-columns:1fr auto; gap:12px; align-items:center; margin-top:10px; }
.intel-row span { color:#c8d2e5; font-size:.74rem; font-weight:700; }
.intel-row b { color:var(--accent); font-size:.72rem; }
.intel-track { grid-column:1/-1; height:5px; overflow:hidden; border-radius:9px; background:rgba(255,255,255,.07); margin-top:-7px; }
.intel-fill { height:100%; border-radius:9px; background:linear-gradient(90deg,var(--accent),rgba(255,255,255,.78)); }
.decision-brief {
    display:grid;
    grid-template-columns:auto 1fr auto;
    gap:15px;
    align-items:center;
    margin:12px 0 1rem;
    padding:17px 19px;
    border:1px solid rgba(96,207,255,.24);
    border-radius:14px;
    background:linear-gradient(115deg,rgba(25,124,207,.12),rgba(134,75,208,.1),rgba(4,9,20,.7));
}
.decision-icon { width:46px; height:46px; display:grid; place-items:center; border-radius:13px; background:linear-gradient(145deg,#35d9ff,#9a78ff); color:#030817; font-size:1.3rem; }
.decision-brief small { color:#6ddfff; font-size:.59rem; font-weight:900; letter-spacing:1.2px; text-transform:uppercase; }
.decision-brief h3 { margin:3px 0; color:#fff; font-size:1rem; }
.decision-brief p { margin:0; color:#9eabc3; font-size:.72rem; }
.decision-score { text-align:right; }
.decision-score b { display:block; color:#fff; font-size:1.55rem; }
.decision-score span { color:#8190aa; font-size:.56rem; text-transform:uppercase; }
.st-key-main_navigation {
    position: sticky;
    top: 0;
    z-index: 999;
    margin: -.3rem 0 1rem;
    padding: .45rem;
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 14px;
    background: rgba(4,8,17,.9);
    box-shadow: 0 14px 38px rgba(0,0,0,.28);
    backdrop-filter: blur(18px);
}
.st-key-main_navigation [data-testid="stButtonGroup"] { width:100%; }
.page-heading { margin:.8rem 0 1.1rem; }
.page-heading small { color:#65dcff; font-size:.63rem; font-weight:900; letter-spacing:1.5px; text-transform:uppercase; }
.page-heading h1 { margin:.25rem 0 .35rem; color:#fff; font-size:2.15rem; letter-spacing:-.035em; }
.page-heading p { margin:0; color:#8f9db7; font-size:.84rem; }
.player-command-card {
    display:grid;
    grid-template-columns:minmax(360px,.82fr) minmax(430px,1.18fr);
    min-height:390px;
    margin:.8rem 0 1.2rem;
    overflow:hidden;
    border:1px solid rgba(71,166,255,.3);
    border-radius:18px;
    background:#03091a;
    box-shadow:0 26px 68px rgba(0,0,0,.38);
}
.player-profile-copy { position:relative; z-index:2; padding:36px; background:linear-gradient(135deg,rgba(4,12,31,.98),rgba(4,10,24,.88)); }
.player-profile-copy small { color:#64dcff; font-size:.61rem; font-weight:900; letter-spacing:1.4px; text-transform:uppercase; }
.player-profile-copy h2 { margin:9px 0 2px; color:#fff; font-size:2.25rem; line-height:1; letter-spacing:-.04em; }
.player-team-line { color:#7ba8ff; font-size:.82rem; font-weight:800; margin-bottom:13px; }
.player-tag-row { display:flex; flex-wrap:wrap; gap:6px; }
.player-tag-row span { padding:5px 9px; border-radius:99px; border:1px solid rgba(255,255,255,.1); background:rgba(255,255,255,.05); color:#c8d4e9; font-size:.6rem; font-weight:800; }
.profile-signal-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:22px; }
.profile-signal { padding:12px; border:1px solid rgba(255,255,255,.07); border-radius:11px; background:rgba(0,0,0,.22); }
.profile-signal b { display:block; color:#fff; font-size:.92rem; }
.profile-signal span { display:block; margin-top:3px; color:#7787a6; font-size:.55rem; font-weight:800; letter-spacing:.7px; text-transform:uppercase; }
.profile-decision { margin-top:9px; padding:11px 12px; border:1px solid rgba(67,223,158,.2); border-radius:10px; background:rgba(27,161,102,.08); color:#a9b9d0; font-size:.68rem; line-height:1.5; }
.profile-decision b { color:#49e3a0; }
.player-map { position:relative; min-height:390px; overflow:hidden; background:#02091a; }
.player-map img { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; object-position:72% center; filter:saturate(1.08) contrast(1.06); }
.player-map::after { content:""; position:absolute; inset:0; background:linear-gradient(90deg,#040a19 0%,rgba(4,10,25,.34) 26%,rgba(2,7,20,.06) 80%),linear-gradient(0deg,rgba(2,7,20,.55),transparent 42%); }
.player-map-label { position:absolute; z-index:2; right:17px; bottom:16px; padding:8px 10px; border:1px solid rgba(89,211,255,.3); border-radius:8px; background:rgba(2,10,27,.78); color:#91e9ff; font-size:.58rem; font-weight:900; letter-spacing:1px; text-transform:uppercase; backdrop-filter:blur(12px); }
.st-key-site_nav {
    position: sticky;
    top: .35rem;
    z-index: 9999;
    margin: 0 0 1.25rem;
}
.st-key-site_nav [data-testid="stMarkdownContainer"] { width:100%; }
.website-nav {
    display:flex;
    align-items:center;
    gap:18px;
    min-height:64px;
    padding:8px 10px 8px 18px;
    border:1px solid rgba(255,255,255,.1);
    border-radius:16px;
    background:rgba(4,8,18,.9);
    box-shadow:0 18px 48px rgba(0,0,0,.35);
    backdrop-filter:blur(22px);
}
.website-brand {
    display:flex;
    align-items:center;
    gap:10px;
    min-width:190px;
    color:#fff !important;
    text-decoration:none !important;
}
.website-brand i {
    width:36px;
    height:36px;
    display:grid;
    place-items:center;
    border-radius:11px;
    background:linear-gradient(135deg,#4ee0ff,#8a7dff 56%,#f45ac3);
    color:#030817;
    font-style:normal;
    box-shadow:0 0 26px rgba(88,142,255,.3);
}
.website-brand b { display:block; font-size:.8rem; line-height:1.05; }
.website-brand span { display:block; margin-top:3px; color:#71819e; font-size:.51rem; font-weight:800; letter-spacing:1px; text-transform:uppercase; }
.website-links { display:flex; align-items:center; justify-content:flex-end; gap:3px; flex:1; overflow-x:auto; scrollbar-width:none; }
.website-links::-webkit-scrollbar { display:none; }
.website-links a {
    flex:0 0 auto;
    padding:10px 12px;
    border:1px solid transparent;
    border-radius:9px;
    color:#9ba8be !important;
    text-decoration:none !important;
    font-size:.67rem;
    font-weight:800;
    white-space:nowrap;
    transition:.18s ease;
}
.website-links a:hover {
    color:#fff !important;
    border-color:rgba(91,210,255,.25);
    background:rgba(83,174,255,.08);
    transform:translateY(-1px);
}
.website-links a.active {
    color:#fff !important;
    border-color:rgba(88,213,255,.32);
    background:linear-gradient(135deg,rgba(75,190,255,.14),rgba(148,103,255,.12));
    box-shadow:inset 0 -2px 0 #62ddff;
}
.website-links a.primary {
    color:#06101b !important;
    background:linear-gradient(115deg,#54dcff,#9b7eff);
}
.section-anchor { height:1px; scroll-margin-top:92px; }
.web-section-head {
    display:grid;
    grid-template-columns:minmax(0,1fr) auto;
    gap:20px;
    align-items:end;
    margin:1.8rem 0 1.25rem;
    padding-top:1.2rem;
    border-top:1px solid rgba(255,255,255,.08);
}
.web-section-head small {
    color:#66dcff;
    font-size:.61rem;
    font-weight:900;
    letter-spacing:1.5px;
    text-transform:uppercase;
}
.web-section-head h2 {
    margin:5px 0 5px;
    color:#fff;
    font-size:2.1rem;
    line-height:1.05;
    letter-spacing:-.04em;
}
.web-section-head p { max-width:720px; margin:0; color:#8897b1; font-size:.8rem; line-height:1.6; }
.web-section-index { color:rgba(111,221,255,.28); font-size:3.4rem; font-weight:900; letter-spacing:-.08em; }
.site-footer {
    margin:5rem 0 1rem;
    padding:24px;
    border-top:1px solid rgba(255,255,255,.08);
    color:#65738d;
    text-align:center;
    font-size:.68rem;
}
.home-modules {
    display:grid;
    grid-template-columns:repeat(3,minmax(0,1fr));
    gap:12px;
    margin:1rem 0 2rem;
}
.home-module {
    position:relative;
    min-height:165px;
    padding:20px;
    overflow:hidden;
    border:1px solid rgba(255,255,255,.08);
    border-radius:14px;
    background:linear-gradient(145deg,rgba(255,255,255,.045),rgba(255,255,255,.015));
    color:inherit !important;
    text-decoration:none !important;
    transition:.2s ease;
}
.home-module::after {
    content:"";
    position:absolute;
    right:-35px;
    bottom:-55px;
    width:130px;
    height:130px;
    border-radius:50%;
    background:radial-gradient(circle,rgba(96,208,255,.15),transparent 68%);
}
.home-module:hover {
    transform:translateY(-4px);
    border-color:rgba(93,210,255,.35);
    box-shadow:0 22px 48px rgba(0,0,0,.3);
}
.home-module i { font-style:normal; font-size:1.45rem; }
.home-module small { display:block; margin-top:16px; color:#64dcff; font-size:.56rem; font-weight:900; letter-spacing:1.2px; text-transform:uppercase; }
.home-module h3 { margin:5px 0 6px; color:#fff; font-size:1rem; }
.home-module p { margin:0; color:#8190a9; font-size:.68rem; line-height:1.5; }
.home-module b { position:absolute; right:17px; top:17px; color:#61718e; font-size:.72rem; }
.product-art-hero {
    position:relative;
    min-height:470px;
    margin:1.1rem 0 1.35rem;
    overflow:hidden;
    border:1px solid rgba(78,181,255,.32);
    border-radius:19px;
    background:#020817;
    box-shadow:0 28px 72px rgba(0,0,0,.42);
}
.product-art-hero > img {
    position:absolute;
    inset:0;
    width:100%;
    height:100%;
    object-fit:cover;
    object-position:center;
    filter:saturate(1.05) contrast(1.04);
}
.product-art-hero::after {
    content:"";
    position:absolute;
    inset:0;
    background:linear-gradient(90deg,rgba(2,7,21,.98) 0%,rgba(2,8,23,.88) 34%,rgba(2,8,21,.18) 72%,rgba(2,6,17,.38)),linear-gradient(0deg,rgba(2,6,17,.94),transparent 52%);
}
.product-hero-copy {
    position:relative;
    z-index:2;
    max-width:620px;
    padding:48px 42px 155px;
}
.product-hero-copy small {
    color:#65deff;
    font-size:.62rem;
    font-weight:900;
    letter-spacing:1.5px;
    text-transform:uppercase;
}
.product-hero-copy h1 {
    margin:11px 0 12px;
    color:#fff;
    font-size:clamp(2.2rem,4vw,3.65rem);
    line-height:.98;
    letter-spacing:-.05em;
}
.product-hero-copy p { max-width:540px; margin:0; color:#a7b6cf; font-size:.82rem; line-height:1.7; }
.product-chip-row { display:flex; flex-wrap:wrap; gap:7px; margin-top:17px; }
.product-chip-row span { padding:6px 9px; border:1px solid rgba(103,218,255,.22); border-radius:99px; background:rgba(4,18,42,.58); color:#c6eaff; font-size:.6rem; font-weight:800; backdrop-filter:blur(10px); }
.product-hero-stats {
    position:absolute;
    z-index:3;
    left:17px;
    right:17px;
    bottom:17px;
    display:grid;
    grid-template-columns:repeat(4,1fr);
    gap:7px;
    padding:8px;
    border:1px solid rgba(98,188,255,.2);
    border-radius:14px;
    background:rgba(2,7,20,.78);
    backdrop-filter:blur(17px);
}
.product-hero-stat { padding:10px 13px; border-right:1px solid rgba(255,255,255,.08); }
.product-hero-stat:last-child { border-right:0; }
.product-hero-stat b { display:block; color:#fff; font-size:1rem; }
.product-hero-stat span { display:block; margin-top:3px; color:#7384a1; font-size:.52rem; font-weight:900; letter-spacing:.7px; text-transform:uppercase; }
.tournament-art-hero::after { background:linear-gradient(90deg,rgba(2,8,20,.98),rgba(2,9,22,.84) 38%,rgba(2,8,20,.16) 76%),linear-gradient(0deg,rgba(2,7,19,.94),transparent 52%); }
.live-art-hero::after { background:linear-gradient(90deg,rgba(2,8,20,.97),rgba(2,9,23,.82) 36%,rgba(2,8,20,.12) 76%),linear-gradient(0deg,rgba(2,7,19,.92),transparent 48%); }
@media (max-width: 900px) {
    .command-hero { min-height:680px; }
    .command-copy { padding:52px 30px 210px; }
    .command-signal { display:none; }
    .command-copy h1 { font-size:3.5rem; }
    .player-visual-copy { width:68%; padding:38px 28px; }
    .player-command-card { grid-template-columns:1fr; }
    .player-map { min-height:310px; }
    .website-brand { min-width:auto; }
    .website-brand div { display:none; }
    .home-modules { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .product-art-hero { min-height:540px; }
    .product-hero-copy { padding:38px 28px 190px; }
}
@media (max-width: 640px) {
    .command-hero { min-height:710px; border-radius:16px; }
    .command-copy { padding:38px 20px 230px; }
    .command-copy h1 { font-size:2.8rem; }
    .command-actions a { width:100%; }
    .command-stats { grid-template-columns:1fr 1fr; }
    .command-stat { border-right:0; padding:8px 10px; }
    .player-visual { min-height:430px; }
    .player-visual-copy { width:100%; padding:34px 22px; }
    .player-visual-copy h2 { font-size:1.85rem; }
    .intel-grid { grid-template-columns:1fr; }
    .decision-brief { grid-template-columns:auto 1fr; }
    .decision-score { display:none; }
    .website-nav { padding-left:10px; }
    .website-links a { padding:9px 10px; }
    .web-section-head { grid-template-columns:1fr; margin-top:3.8rem; }
    .web-section-index { display:none; }
    .home-modules { grid-template-columns:1fr; }
    .product-art-hero { min-height:610px; }
    .product-hero-copy { padding:32px 21px 250px; }
    .product-hero-stats { grid-template-columns:1fr 1fr; }
    .product-hero-stat { border-right:0; }
}
</style>
""", unsafe_allow_html=True)

# ── Backend helpers ───────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def api_get(path):
    try:
        r = requests.get(f"{API}{path}", timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

def api_post(path, payload):
    try:
        r = requests.post(f"{API}{path}", json=payload, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return None

# ── Load Static Data ─────────────────────────────────────────────────────────
TEAMS = ["Colombo Kaps","Dambulla Sixers","Galle Gallants","Jaffna Kings","Kandy Royals"]
VENUES = ["R. Premadasa Stadium","Pallekele Stadium","Rangiri Dambulla","SSC Ground"]
VENUE_SHORT = {
    "R. Premadasa Stadium": "Premadasa",
    "R.Premadasa Stadium": "Premadasa",
    "Pallekele Stadium": "Pallekele",
    "Pallekele International Stadium": "Pallekele",
    "Rangiri Dambulla": "Dambulla",
    "Rangiri Dambulla International": "Dambulla",
    "SSC Ground": "SSC",
}

TEAM_COLORS = {
    "Colombo Kaps":    "#3b82f6",
    "Dambulla Sixers": "#f59e0b",
    "Galle Gallants":  "#10b981",
    "Jaffna Kings":    "#8b5cf6",
    "Kandy Royals":    "#ef4444",
}

ROLE_ICONS = {
    "WK-Batsman": "🧤", "Batsman": "🏏", "Bowler": "🎳", "All-Rounder": "⚡",
    "Bat/Spin": "🏏", "WK": "🧤",
}

backend_ok = api_get("/") is not None
if not backend_ok:
    st.markdown(
        "<div style='margin-bottom:1rem;padding:11px 16px;border-radius:12px;"
        "border:1px solid rgba(101,215,255,.28);background:rgba(101,215,255,.07);"
        "color:#bfe7ff;font-size:.85rem;font-weight:600'>"
        "📡 Connecting to the live prediction engine… this page will fill in automatically."
        "</div>",
        unsafe_allow_html=True,
    )

squads_data  = api_get("/squads") or []
teams_from_api = api_get("/teams") or TEAMS
venues_payload = api_get("/venues") or {}
VENUES = venues_payload.get("venues") or VENUES
strengths_data = api_get("/team-strengths") or []
h2h_data     = api_get("/h2h-matchups") or []

squad_df = pd.DataFrame(squads_data) if squads_data else pd.DataFrame()

VALID_VIEWS = {
    "home", "match", "player", "live", "squad", "tournament", "field",
    "command", "forecast", "tactics", "field_lab",
}
current_view = st.query_params.get("page", "home")
if current_view not in VALID_VIEWS:
    current_view = "home"


def nav_class(view, primary=False):
    classes = []
    view_groups = {
        "tournament": {"tournament", "command", "forecast", "tactics"},
        "field": {"field", "field_lab"},
    }
    if current_view == view or current_view in view_groups.get(view, set()):
        classes.append("active")
    if primary:
        classes.append("primary")
    return " ".join(classes)


with st.container(key="site_nav"):
    st.markdown(f"""
    <nav class="website-nav" aria-label="Main navigation">
      <a class="website-brand" href="?page=home">
        <i>🏏</i><div><b>LPL Intelligence</b><span>2026 Decision OS</span></div>
      </a>
      <div class="website-links">
        <a class="{nav_class('home')}" href="?page=home">Home</a>
        <a class="{nav_class('match')}" href="?page=match">Match Predictor</a>
        <a class="{nav_class('player')}" href="?page=player">Player Analysis</a>
        <a class="{nav_class('live')}" href="?page=live">Live Simulator</a>
        <a class="{nav_class('squad')}" href="?page=squad">Squad Intel</a>
        <a class="{nav_class('tournament')}" href="?page=tournament">Tournament</a>
        <a class="{nav_class('field', True)}" href="?page=field">3D Field Lab</a>
      </div>
    </nav>
    """, unsafe_allow_html=True)

# ─── HERO ─────────────────────────────────────────────────────────────────────
_dash_home = api_get("/api/dashboard") or {}
_bracket = _dash_home.get("bracket") or {}
_stand = _dash_home.get("standings") or []


def _cup_for(team):
    for s in _stand:
        if s.get("team") == team:
            v = s.get("cup")
            return f"{v}%" if v is not None else "—"
    return "—"


champion_home = _bracket.get("champion") or (_stand[0].get("team") if _stand else "TBD")
runner_home = _bracket.get("runner_up") or (_stand[1].get("team") if len(_stand) > 1 else "TBD")
champ_color = TEAM_COLORS.get(champion_home, "#f5a623")
runner_color = TEAM_COLORS.get(runner_home, "#65d7ff")
champ_cup = _cup_for(champion_home)
runner_cup = _cup_for(runner_home)
champ_tagline = next(
    (s.get("strength_1") for s in strengths_data
     if s.get("team") == champion_home and s.get("strength_1")),
    "Strongest overall squad profile across batting depth and bowling control.",
)
n_teams = len(teams_from_api) if isinstance(teams_from_api, list) and teams_from_api else 5
n_venues = len(VENUES) if VENUES else 4

if current_view == "home":
    st.markdown(f"""
<section class="command-hero" id="home">
  <img src="{HERO_ART}" alt="LPL analysis command center">
  <div class="command-copy">
    <div class="command-live"><i></i>LPL 2026 decision engine online</div>
    <h1>Strategic <span>Decision Making</span></h1>
    <p>Transform player intelligence, venue conditions and tournament simulations into the clearest match-day call. See the opportunity, understand the risk, then act with confidence.</p>
    <div class="command-actions">
      <a href="{HTML_TACTICS_URL}" target="_blank" rel="noopener">Open Strategy Room&nbsp; →</a>
      <a href="{HTML_FORECAST_URL}" target="_blank" rel="noopener">View Tournament Forecast</a>
    </div>
  </div>
  <div class="command-signal">
    <small>AI tournament signal</small>
    <strong>👑 {html.escape(str(champion_home))}</strong>
    <span>{html.escape(str(champ_cup))} title probability · {html.escape(str(champ_tagline))}</span>
  </div>
  <div class="command-stats">
    <div class="command-stat"><b>{html.escape(str(champion_home))}</b><span>Projected champion</span></div>
    <div class="command-stat"><b>{html.escape(str(runner_home))}</b><span>Closest challenger</span></div>
    <div class="command-stat"><b>{n_teams} teams · {n_venues} venues</b><span>Tournament intelligence</span></div>
    <div class="command-stat"><b>7 AI models</b><span>Monte-Carlo simulated</span></div>
  </div>
</section>
""", unsafe_allow_html=True)
    st.markdown("""
    <div class="home-modules">
      <a class="home-module" href="?page=match"><i>🔮</i><b>01</b><small>Pre-match</small><h3>Match Predictor</h3><p>Compare teams, conditions and explainable win signals.</p></a>
      <a class="home-module" href="?page=player"><i>👤</i><b>02</b><small>Scouting</small><h3>Player Analysis</h3><p>Strengths, weaknesses, venue fit and PP-NN projections.</p></a>
      <a class="home-module" href="?page=live"><i>📡</i><b>03</b><small>In-play</small><h3>Live Simulator</h3><p>Track win probability, pressure and chase requirements.</p></a>
      <a class="home-module" href="?page=squad"><i>🛡️</i><b>04</b><small>Team build</small><h3>Squad Intel</h3><p>Read team balance, depth and structural vulnerabilities.</p></a>
      <a class="home-module" href="?page=tournament"><i>🏆</i><b>05</b><small>Competition</small><h3>Tournament View</h3><p>Standings, bracket logic, awards and simulations.</p></a>
      <a class="home-module" href="?page=field"><i>🎯</i><b>06</b><small>Tactical lab</small><h3>3D Field Lab</h3><p>Field placements, shot paths and catch simulations.</p></a>
    </div>
    """, unsafe_allow_html=True)


def _rating(value):
    try:
        number = float(value)
        return 0.0 if np.isnan(number) else number
    except (TypeError, ValueError):
        return 0.0


def _player_intelligence(row, role, venue):
    batting = [
        ("Powerplay attack", _rating(row.get("batting_power_play_rating"))),
        ("Middle-over batting", _rating(row.get("batting_middle_over_rating"))),
        ("Death-over finishing", _rating(row.get("batting_death_over_rating"))),
    ]
    bowling = [
        ("New-ball impact", _rating(row.get("bowling_power_play_rating"))),
        ("Middle-over control", _rating(row.get("bowling_middle_over_rating"))),
        ("Death-over execution", _rating(row.get("bowling_death_over_rating"))),
    ]
    support = [
        ("Pressure handling", _rating(row.get("pressure_handling"))),
        ("Fielding value", _rating(row.get("fielding_rating"))),
        ("Consistency", _rating(row.get("consistency_score"))),
    ]
    role_lower = str(role).lower()
    if "all" in role_lower or "bat/spin" in role_lower:
        relevant = batting + bowling + support
        identity = "multi-phase all-round option"
    elif "bowl" in role_lower:
        relevant = bowling + support
        identity = "strike-bowling option"
    else:
        relevant = batting + support
        identity = "specialist batting option"

    relevant = [(label, score) for label, score in relevant if score > 0]
    if len(relevant) < 4:
        relevant = [(label, score) for label, score in batting + bowling + support if score > 0]
    ranked = sorted(relevant, key=lambda item: item[1], reverse=True)
    strengths = ranked[:3]
    weaknesses = sorted(relevant, key=lambda item: item[1])[:2]
    overall = round(sum(score for _, score in relevant) / len(relevant), 1) if relevant else 0.0
    top_skill = strengths[0][0].lower() if strengths else "primary role"
    low_skill = weaknesses[0][0].lower() if weaknesses else "lowest-rated phase"
    player_name = html.escape(str(row.get("player_name", "This player")))
    best_venue = str(row.get("best_venue", "")).strip()
    venue_note = f" Best venue profile: {html.escape(best_venue)}." if best_venue and best_venue.lower() != "nan" else ""
    decision = (
        f"Deploy {player_name} as a {identity} when {top_skill} matters most at "
        f"{html.escape(str(venue))}. Protect the {low_skill} exposure with role support.{venue_note}"
    )
    return strengths, weaknesses, overall, decision


def _intel_rows(items):
    if not items:
        return '<div class="intel-row"><span>Insufficient rating data</span><b>—</b></div>'
    rows = []
    for label, score in items:
        width = max(4, min(100, score * 10))
        rows.append(
            f'<div class="intel-row"><span>{html.escape(label)}</span><b>{score:.1f}/10</b>'
            f'<div class="intel-track"><div class="intel-fill" style="width:{width:.0f}%"></div></div></div>'
        )
    return "".join(rows)


# ════════════════════════════════════════════════════════════════════════
# TAB 1 — MATCH PREDICTOR (PyTorch MT-DNN + XAI)
# ════════════════════════════════════════════════════════════════════════
if current_view == "match":
    st.markdown("""
<div id="match-predictor" class="section-anchor"></div>
<div class="web-section-head">
  <div><small>Prediction workspace</small><h2>Match Outcome Intelligence</h2>
  <p>Compare teams, venue conditions and model explanations before making the match call.</p></div>
  <div class="web-section-index">01</div>
</div>
""", unsafe_allow_html=True)
if current_view == "match":
    st.caption("Powered by PyTorch Multi-Task Deep Neural Network (60 epochs) + Gradient×Input XAI")

    col_l, col_r, col_v = st.columns([2, 2, 2])
    with col_l:
        team1 = st.selectbox("🏠 Team 1", TEAMS, key="t1_sel")
        t1_home = st.toggle("Team 1 playing at Home?", value=True)
    with col_r:
        remaining = [t for t in TEAMS if t != team1]
        team2 = st.selectbox("✈️ Team 2", remaining, key="t2_sel")
    with col_v:
        venue = st.selectbox("🏟️ Venue", VENUES, key="venue_sel")
        dew   = st.toggle("🌧️ Dew Factor Active?", value=False)

    run_pred = st.button("⚡ Run Deep Learning Prediction", use_container_width=True, type="primary")

    if run_pred:
        with st.spinner("Running PyTorch MT-DNN inference …"):
            result = api_post("/predict/winner", {
                "team1": team1, "team2": team2,
                "venue": venue, "team1_home": t1_home, "dew": dew
            })

        if result is None:
            st.info("📡 Still connecting to the prediction engine — please try again in a moment.")
        else:
            nn   = result.get("pytorch_nn", {})
            lr   = result.get("logistic_regression", {})
            xai  = result.get("xai_attributions", {})

            wp1_nn = nn.get("win_prob_team1") or 50.0
            wp2_nn = 100 - wp1_nn
            wp1_lr = lr.get("win_prob_team1") or wp1_nn

            c1, c2 = st.columns(2)
            t1c = TEAM_COLORS.get(team1, "#3b82f6")
            t2c = TEAM_COLORS.get(team2, "#f59e0b")

            with c1:
                st.markdown(f"""
                <div class="card">
                  <h3 style="color:{t1c};">🏆 {team1}</h3>
                  <div class="metric-big" style="color:{t1c};">{wp1_nn:.1f}%</div>
                  <div class="metric-sub">Neural Network Win Probability</div>
                  <div class="win-bar-wrap" style="margin-top:.8rem;">
                    <div class="win-bar" style="width:{wp1_nn:.1f}%;background:{t1c};"></div>
                  </div>
                  <hr style="border:none;border-top:1px solid rgba(255,255,255,.07);margin:.9rem 0;">
                  <div style="display:flex;gap:2rem;">
                    <div>
                      <div style="font-size:1.4rem;font-weight:700;">{int(nn.get('score_team1') or 0)}</div>
                      <div class="metric-sub">Projected Runs</div>
                    </div>
                    <div>
                      <div style="font-size:1.4rem;font-weight:700;">{nn.get('wickets_team1',0):.1f}</div>
                      <div class="metric-sub">Proj. Wickets</div>
                    </div>
                    <div>
                      <div style="font-size:1.4rem;font-weight:700;">{wp1_lr:.1f}%</div>
                      <div class="metric-sub">LR Baseline</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

            with c2:
                st.markdown(f"""
                <div class="card">
                  <h3 style="color:{t2c};">🛡️ {team2}</h3>
                  <div class="metric-big" style="color:{t2c};">{wp2_nn:.1f}%</div>
                  <div class="metric-sub">Neural Network Win Probability</div>
                  <div class="win-bar-wrap" style="margin-top:.8rem;">
                    <div class="win-bar" style="width:{wp2_nn:.1f}%;background:{t2c};"></div>
                  </div>
                  <hr style="border:none;border-top:1px solid rgba(255,255,255,.07);margin:.9rem 0;">
                  <div style="display:flex;gap:2rem;">
                    <div>
                      <div style="font-size:1.4rem;font-weight:700;">{int(nn.get('score_team2') or 0)}</div>
                      <div class="metric-sub">Projected Runs</div>
                    </div>
                    <div>
                      <div style="font-size:1.4rem;font-weight:700;">{nn.get('wickets_team2',0):.1f}</div>
                      <div class="metric-sub">Proj. Wickets</div>
                    </div>
                    <div>
                      <div style="font-size:1.4rem;font-weight:700;">{100-wp1_lr:.1f}%</div>
                      <div class="metric-sub">LR Baseline</div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

            # XAI Attributions Chart
            if xai:
                st.markdown("#### 🧠 Explainable AI — Feature Contributions to Win Probability")
                st.caption("Gradient × Input attribution: positive = increases Team 1 win probability, negative = decreases it")

                sorted_xai = sorted(xai.items(), key=lambda x: x[1], reverse=True)
                labels = [k for k, _ in sorted_xai]
                vals   = [v for _, v in sorted_xai]
                colors = [(0.24, 0.74, 0.51, 0.85) if v >= 0 else (0.94, 0.27, 0.27, 0.85) for v in vals]

                fig, ax = plt.subplots(figsize=(10, max(4, len(labels)*0.45)), facecolor="#0f111a")
                ax.set_facecolor("#0f111a")
                bars = ax.barh(labels, vals, color=colors, edgecolor="none", height=0.6)
                ax.axvline(0, color=(1,1,1,0.25), linewidth=0.8)
                ax.set_xlabel("Attribution Value", color="#94a3b8", fontsize=9)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.xaxis.grid(True, color=(1,1,1,0.05), linewidth=0.5)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    # H2H Match Schedule
    if h2h_data:
        st.markdown("#### 📋 Predicted Group Stage Results")
        h2h_df = pd.DataFrame(h2h_data)
        st.dataframe(
            h2h_df[["team_home","team_away","venue","home_win_probability",
                     "away_win_probability","prediction_winner","margin_estimate"]].rename(columns={
                "team_home":"Home","team_away":"Away","venue":"Venue",
                "home_win_probability":"Home Win %","away_win_probability":"Away Win %",
                "prediction_winner":"Predicted Winner","margin_estimate":"Margin"
            }),
            use_container_width=True, hide_index=True
        )

# ════════════════════════════════════════════════════════════════════════
# TAB 2 — PLAYER ANALYZER (PP-NN + Venue Pitch + Strengths/Weaknesses)
# ════════════════════════════════════════════════════════════════════════
if current_view == "player":
    st.markdown("""
<div id="player-analysis" class="section-anchor"></div>
""", unsafe_allow_html=True)
if current_view == "player":
    st.markdown("""
    <div class="web-section-head">
      <div><small>AI scouting workspace</small><h2>Player Analysis</h2>
      <p>Select a squad, player and venue to build a complete performance and tactical profile.</p></div>
      <div class="web-section-index">02</div>
    </div>
    """, unsafe_allow_html=True)

    pa_col1, pa_col2, pa_col3 = st.columns([2, 3, 2])
    with pa_col1:
        sel_team = st.selectbox("Select Team", TEAMS, key="pa_team")
    with pa_col3:
        sel_venue_pa = st.selectbox("Venue for Prediction", VENUES, key="pa_venue")

    players_raw = api_get(f"/players/{requests.utils.quote(sel_team)}") or []
    if not players_raw:
        st.warning(f"No player data for {sel_team}")
    else:
        players_df = pd.DataFrame(players_raw)
        player_names = players_df["player_name"].tolist() if "player_name" in players_df.columns else []

        with pa_col2:
            sel_player = st.selectbox("Select Player", player_names, key="pa_player")

        if sel_player and "player_name" in players_df.columns:
            p_row = players_df[players_df["player_name"] == sel_player].iloc[0]
            p_id  = p_row.get("player_id", "")
            role  = str(p_row.get("role", "Batsman"))
            role_icon = next((v for k,v in ROLE_ICONS.items() if k in role), "🏏")
            tc = TEAM_COLORS.get(sel_team, "#3b82f6")
            cat = p_row.get("category","Classic")
            cat_pill_cls = {"Star":"pill-gold","Icon":"pill-gold","Platinum":"pill-purple",
                            "Gold":"pill-gold","Classic":"pill-blue"}.get(str(cat),"pill-blue")
            player_strengths, player_weaknesses, intel_score, strategic_call = _player_intelligence(
                p_row, role, sel_venue_pa
            )
            pp_result = api_post(
                "/predict/player-performance",
                {"player_id": p_id, "venue": sel_venue_pa},
            ) if p_id else None
            top_skill, top_score = player_strengths[0] if player_strengths else ("Profile building", 0.0)
            pressure_score = _rating(p_row.get("pressure_handling"))
            best_venue = p_row.get("best_venue", "N/A")

            # Integrated player command profile: live identity + visual analysis map
            st.markdown(f"""
            <section class="player-command-card">
              <div class="player-profile-copy">
                <small>Live player intelligence · {html.escape(str(cat))}</small>
                <h2>{role_icon} {html.escape(str(sel_player))}</h2>
                <div class="player-team-line">{html.escape(str(sel_team))}</div>
                <div class="player-tag-row">
                  <span>{html.escape(str(role))}</span>
                  <span>{html.escape(str(p_row.get('nationality','SL')))}</span>
                  <span>{html.escape(str(sel_venue_pa))}</span>
                </div>
                <div class="profile-signal-grid">
                  <div class="profile-signal"><b>{html.escape(top_skill)}</b><span>Strongest signal · {top_score:.1f}/10</span></div>
                  <div class="profile-signal"><b>{html.escape(str(best_venue))}</b><span>Best venue profile</span></div>
                  <div class="profile-signal"><b>{pressure_score:.1f}/10</b><span>Pressure handling</span></div>
                  <div class="profile-signal"><b>{pp_result.get('expected_runs','—') if pp_result else '—'}</b><span>Expected runs</span></div>
                </div>
                <div class="profile-decision"><b>Strategic read:</b> {strategic_call}</div>
              </div>
              <div class="player-map">
                <img src="{PLAYER_ART}" alt="Player analysis charts and tactical heat maps">
                <div class="player-map-label">Visual scouting map · live model data below</div>
              </div>
            </section>
            """, unsafe_allow_html=True)

            col_sk, col_v, col_pp = st.columns(3)

            # ── Skill Radar ───────────────────────────────────────────
            with col_sk:
                st.markdown('<div class="card"><h3>📊 Phase Skill Ratings</h3>', unsafe_allow_html=True)
                skills = {
                    "Bat PP": float(p_row.get("batting_power_play_rating",0)),
                    "Bat Mid": float(p_row.get("batting_middle_over_rating",0)),
                    "Bat Death": float(p_row.get("batting_death_over_rating",0)),
                    "Bowl PP": float(p_row.get("bowling_power_play_rating",0)),
                    "Bowl Mid": float(p_row.get("bowling_middle_over_rating",0)),
                    "Bowl Death": float(p_row.get("bowling_death_over_rating",0)),
                    "Fielding": float(p_row.get("fielding_rating",0)),
                    "Pressure": float(p_row.get("pressure_handling",0)),
                }
                fig, ax = plt.subplots(figsize=(4, 3), facecolor="#0f111a")
                ax.set_facecolor("#0f111a")
                ax.barh(list(skills.keys()), list(skills.values()),
                        color=tc + "cc", edgecolor="none", height=0.65)
                ax.set_xlim(0, 10)
                ax.xaxis.grid(True, color=(1,1,1,0.05), linewidth=0.5)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(labelsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── Venue Suitability ─────────────────────────────────────
            with col_v:
                st.markdown('<div class="card"><h3>🏟️ Venue Suitability</h3>', unsafe_allow_html=True)
                venue_scores = {
                    "Premadasa": float(p_row.get("premadasa_rating", 7)),
                    "Pallekele": float(p_row.get("pallekele_rating", 7)),
                    "Dambulla":  float(p_row.get("dambulla_rating", 7)),
                    "SSC":       float(p_row.get("premadasa_rating", 7)),
                }
                sel_short = VENUE_SHORT.get(sel_venue_pa, "Premadasa")
                v_colors = [(0.24, 0.74, 0.51, 0.9) if k == sel_short else (0.37, 0.51, 0.73, 0.5)
                            for k in venue_scores.keys()]
                fig, ax = plt.subplots(figsize=(4, 2.5), facecolor="#0f111a")
                ax.set_facecolor("#0f111a")
                ax.bar(venue_scores.keys(), venue_scores.values(), color=v_colors, edgecolor="none", width=0.6)
                ax.set_ylim(0, 10)
                ax.yaxis.grid(True, color=(1,1,1,0.05))
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(labelsize=8)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                best = p_row.get("best_venue","N/A")
                worst= p_row.get("worst_venue","N/A")
                st.markdown(f"<span class='pill pill-green'>Best: {best}</span> <span class='pill pill-red'>Worst: {worst}</span>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # ── PP-NN Prediction ──────────────────────────────────────
            with col_pp:
                st.markdown('<div class="card"><h3>🤖 PP-NN Prediction</h3>', unsafe_allow_html=True)
                if p_id:
                    if pp_result:
                        st.markdown(f"""
                        <div style="margin-bottom:.5rem;">
                          <div class="metric-sub">Expected Runs</div>
                          <div class="metric-big" style="font-size:1.8rem;color:#60a5fa;">{pp_result.get('expected_runs','-')}</div>
                        </div>
                        <div style="margin-bottom:.5rem;">
                          <div class="metric-sub">Expected Strike Rate</div>
                          <div class="metric-big" style="font-size:1.8rem;color:#34d399;">{pp_result.get('expected_sr','-')}</div>
                        </div>
                        <div style="margin-bottom:.5rem;">
                          <div class="metric-sub">Expected Wickets</div>
                          <div class="metric-big" style="font-size:1.8rem;color:#f59e0b;">{pp_result.get('expected_wickets','-')}</div>
                        </div>
                        <div>
                          <div class="metric-sub">Expected Economy</div>
                          <div class="metric-big" style="font-size:1.8rem;color:#a78bfa;">{pp_result.get('expected_economy','-')}</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.info("PP-NN model not available.")
                st.markdown('</div>', unsafe_allow_html=True)

            # Creative scouting intelligence: strengths, weaknesses and deployment call
            st.markdown(f"""
            <div class="intel-grid">
              <div class="intel-panel strength">
                <div class="intel-kicker">▲ Competitive edge</div>
                <h3>Core strengths</h3>
                {_intel_rows(player_strengths)}
              </div>
              <div class="intel-panel weakness">
                <div class="intel-kicker">▼ Opposition target</div>
                <h3>Tactical weaknesses</h3>
                {_intel_rows(player_weaknesses)}
              </div>
            </div>
            <div class="decision-brief">
              <div class="decision-icon">◎</div>
              <div>
                <small>Strategic decision</small>
                <h3>Recommended deployment for {html.escape(str(sel_player))}</h3>
                <p>{strategic_call}</p>
              </div>
              <div class="decision-score"><b>{intel_score:.1f}</b><span>Intelligence score / 10</span></div>
            </div>
            """, unsafe_allow_html=True)

            # H2H Records
            if p_id:
                h2h_p = api_get(f"/player/{p_id}/h2h")
                if h2h_p:
                    as_bat  = h2h_p.get("as_batsman", [])
                    as_bowl = h2h_p.get("as_bowler", [])
                    if as_bat:
                        st.markdown("#### ⚔️ Batting H2H Records")
                        bat_h = pd.DataFrame(as_bat)
                        if not bat_h.empty:
                            st.dataframe(bat_h[["bowler","bowling_type","balls_faced","runs_scored",
                                                "dismissals","strike_rate_in_matchup",
                                                "matchup_advantage_score","advantage_favors"]].rename(columns={
                                "bowler":"Bowler","bowling_type":"Type","balls_faced":"Balls",
                                "runs_scored":"Runs","dismissals":"Dismissals",
                                "strike_rate_in_matchup":"SR","matchup_advantage_score":"Adv Score",
                                "advantage_favors":"Favors"
                            }), use_container_width=True, hide_index=True)
                    if as_bowl:
                        st.markdown("#### 🎳 Bowling H2H Records")
                        bowl_h = pd.DataFrame(as_bowl)
                        if not bowl_h.empty:
                            st.dataframe(bowl_h[["batsman","batsman_team","balls_faced","runs_scored",
                                                 "dismissals","matchup_advantage_score","advantage_favors"]].rename(columns={
                                "batsman":"Batsman","batsman_team":"Team","balls_faced":"Balls",
                                "runs_scored":"Runs","dismissals":"Dismissals",
                                "matchup_advantage_score":"Adv Score","advantage_favors":"Favors"
                            }), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE MATCH SIMULATOR
# ════════════════════════════════════════════════════════════════════════
if current_view == "live":
    st.markdown(f"""
<div id="live-simulator" class="section-anchor"></div>
<section class="product-art-hero live-art-hero">
  <img src="{LIVE_ART}" alt="Live cricket simulation and tactical field laboratory">
  <div class="product-hero-copy">
    <small>Live tactical engine · real-time state model</small>
    <h1>Read the match<br>before it turns.</h1>
    <p>Move the score, overs and wickets to simulate the chase. The engine recalculates win probability, pressure and the required scoring path in real time.</p>
    <div class="product-chip-row"><span>Win probability</span><span>Pressure index</span><span>RRR vs CRR</span><span>20-over sweep</span></div>
  </div>
  <div class="product-hero-stats">
    <div class="product-hero-stat"><b>120 balls</b><span>Full innings model</span></div>
    <div class="product-hero-stat"><b>0–100</b><span>Pressure scale</span></div>
    <div class="product-hero-stat"><b>Live state</b><span>Instant recalculation</span></div>
    <div class="product-hero-stat"><b>XAI ready</b><span>Decision context</span></div>
  </div>
</section>
""", unsafe_allow_html=True)
if current_view == "live":
    st.caption("Logistic sigmoid model calibrated with RRR, CRR, wickets-in-hand. Pressure Index (0-100).")

    lc1, lc2 = st.columns(2)
    with lc1:
        target      = st.slider("🎯 Target Score", 100, 250, 170, 5, key="l_tgt")
        runs_scored = st.slider("🏏 Runs Scored", 0, 250, 90, 1, key="l_runs")
    with lc2:
        overs_done  = st.slider("⏱️ Overs Completed", 0.0, 20.0, 12.0, 0.1, key="l_ov")
        wickets_lost= st.slider("💀 Wickets Lost", 0, 10, 3, 1, key="l_wk")

    whole_ov   = int(overs_done)
    part_balls = min(int(round((overs_done - whole_ov)*10)), 6)
    balls_done = whole_ov * 6 + part_balls

    live_result = api_post("/predict/live", {
        "target": target, "runs_scored": runs_scored,
        "balls_bowled": balls_done, "wickets_lost": wickets_lost
    })

    if live_result:
        wp_chase  = live_result.get("win_probability_chasing", 50.0)
        wp_defend = 100 - wp_chase
        pi        = live_result.get("pressure_index", 50.0)
        p_level   = live_result.get("pressure_level", "Moderate")

        p_color = {"Low":"#10b981","Moderate":"#f59e0b","High":"#ef4444","Extreme":"#7f1d1d"}.get(p_level,"#f59e0b")

        lres1, lres2, lres3 = st.columns(3)
        with lres1:
            st.markdown(f"""
            <div class="card">
              <h3>🏃 Chasing Win Prob</h3>
              <div class="metric-big" style="color:{'#10b981' if wp_chase>50 else '#ef4444'};">{wp_chase:.1f}%</div>
              <div class="win-bar-wrap"><div class="win-bar" style="width:{wp_chase:.1f}%;background:{'#10b981' if wp_chase>50 else '#ef4444'};"></div></div>
            </div>""", unsafe_allow_html=True)
        with lres2:
            st.markdown(f"""
            <div class="card">
              <h3>🛡️ Defending Win Prob</h3>
              <div class="metric-big" style="color:{'#10b981' if wp_defend>50 else '#ef4444'};">{wp_defend:.1f}%</div>
              <div class="win-bar-wrap"><div class="win-bar" style="width:{wp_defend:.1f}%;background:{'#10b981' if wp_defend>50 else '#ef4444'};"></div></div>
            </div>""", unsafe_allow_html=True)
        with lres3:
            st.markdown(f"""
            <div class="card">
              <h3>⚡ Pressure Index</h3>
              <div class="metric-big" style="color:{p_color};">{pi:.1f}</div>
              <div class="metric-sub">Level: <b style="color:{p_color};">{p_level}</b></div>
              <div class="win-bar-wrap"><div class="win-bar" style="width:{pi:.1f}%;background:{p_color};"></div></div>
            </div>""", unsafe_allow_html=True)

        # Situation summary
        runs_req  = target - runs_scored
        balls_rem = 120 - balls_done
        rrr = (runs_req / balls_rem * 6) if balls_rem > 0 else 99
        crr = (runs_scored / balls_done * 6) if balls_done > 0 else 0

        st.markdown(f"""
        <div class="card" style="margin-top:1rem;">
          <h3>📈 Match Situation Summary</h3>
          <div style="display:flex;gap:3rem;flex-wrap:wrap;">
            <div><div class="metric-sub">Runs Required</div><div style="font-size:1.4rem;font-weight:700;">{runs_req}</div></div>
            <div><div class="metric-sub">Balls Remaining</div><div style="font-size:1.4rem;font-weight:700;">{balls_rem}</div></div>
            <div><div class="metric-sub">Required Run Rate</div><div style="font-size:1.4rem;font-weight:700;">{rrr:.2f}</div></div>
            <div><div class="metric-sub">Current Run Rate</div><div style="font-size:1.4rem;font-weight:700;">{crr:.2f}</div></div>
            <div><div class="metric-sub">Wickets in Hand</div><div style="font-size:1.4rem;font-weight:700;">{10-wickets_lost}</div></div>
          </div>
        </div>""", unsafe_allow_html=True)

        # WP sweep curve (varying overs_remaining)
        st.markdown("#### 📉 Win Probability Sweep (varying overs bowled, current state)")
        sweep_overs = list(range(1, 21))
        sweep_wp    = []
        for ov in sweep_overs:
            r = api_post("/predict/live", {
                "target": target, "runs_scored": runs_scored,
                "balls_bowled": ov * 6, "wickets_lost": wickets_lost
            })
            sweep_wp.append(r["win_probability_chasing"] if r else 50)

        fig, ax = plt.subplots(figsize=(9, 3.5), facecolor="#0f111a")
        ax.set_facecolor("#0f111a")
        ax.fill_between(sweep_overs, sweep_wp, alpha=0.25, color="#3b82f6")
        ax.plot(sweep_overs, sweep_wp, color="#60a5fa", linewidth=2.5)
        ax.axhline(50, color=(1,1,1,0.25), linewidth=0.8, linestyle="--")
        ax.axvline(whole_ov, color="#f59e0b", linewidth=1.2, linestyle=":")
        ax.set_xlabel("Overs Bowled", fontsize=9)
        ax.set_ylabel("Chase Win %", fontsize=9)
        ax.set_xlim(1, 20)
        ax.set_ylim(0, 100)
        ax.yaxis.grid(True, color=(1,1,1,0.05))
        for spine in ax.spines.values():
            spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ════════════════════════════════════════════════════════════════════════
# TAB 4 — SQUAD INTEL (Team strengths, weaknesses, ratings comparison)
# ════════════════════════════════════════════════════════════════════════
if current_view == "squad":
    st.markdown("""
<div id="squad-intel" class="section-anchor"></div>
<div class="web-section-head">
  <div><small>Team architecture</small><h2>Squad Intelligence</h2>
  <p>Compare batting, bowling, experience, strengths and structural weaknesses across every squad.</p></div>
  <div class="web-section-index">04</div>
</div>
""", unsafe_allow_html=True)
if current_view == "squad":
    if not squad_df.empty:
        # Team comparison radar / bar
        st.markdown("#### 📊 All-Team Rating Comparison")
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), facecolor="#0f111a")
        metrics   = ["batting_rating", "bowling_rating", "experience_rating"]
        m_labels  = ["Batting", "Bowling", "Experience"]
        for i, (metric, mlabel) in enumerate(zip(metrics, m_labels)):
            ax = axes[i]
            ax.set_facecolor("#0f111a")
            colors = [TEAM_COLORS.get(t, "#3b82f6") for t in squad_df["team"]]
            vals   = squad_df[metric].astype(float).values if metric in squad_df.columns else [0]*len(squad_df)
            bars   = ax.bar(squad_df["team"], vals, color=colors, edgecolor="none", width=0.65)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                        f"{val:.1f}", ha="center", va="bottom", color="#e2e8f0", fontsize=8, fontweight="bold")
            ax.set_title(mlabel, color="#94a3b8", fontsize=10)
            ax.set_ylim(0, 10.5)
            ax.yaxis.grid(True, color=(1,1,1,0.05))
            ax.set_xticklabels(squad_df["team"], rotation=30, ha="right", fontsize=7)
            for spine in ax.spines.values():
                spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("#### 📋 Squad Ratings Table")
        st.dataframe(squad_df.style.background_gradient(cmap="Blues",
            subset=[c for c in ["batting_rating","bowling_rating","experience_rating","allrounder_depth"] if c in squad_df.columns]),
            use_container_width=True, hide_index=True)
    else:
        st.warning("Squad data not available from backend.")

    # Team Strengths & Weaknesses Cards
    if strengths_data:
        st.markdown("#### 🧠 Team Strengths & Weaknesses Analysis")
        for row in strengths_data:
            team_name = row.get("team","")
            tc = TEAM_COLORS.get(team_name,"#3b82f6")
            strengths = [row.get(f"strength_{i}","") for i in range(1,6) if row.get(f"strength_{i}","")]
            weaknesses= [row.get(f"weakness_{i}","") for i in range(1,6) if row.get(f"weakness_{i}","")]
            bat_strat = row.get("best_strategy_batting","")
            bowl_strat= row.get("best_strategy_bowling","")

            with st.expander(f"  {team_name}", expanded=False):
                sc, wc = st.columns(2)
                with sc:
                    st.markdown(f'<div class="strength-box"><h4 style="color:#10b981;">✅ Strengths</h4>{"".join(f"<div style=margin-bottom:.3rem;>• {s}</div>" for s in strengths)}</div>', unsafe_allow_html=True)
                with wc:
                    st.markdown(f'<div class="weakness-box"><h4 style="color:#ef4444;">⚠️ Weaknesses</h4>{"".join(f"<div style=margin-bottom:.3rem;>• {w}</div>" for w in weaknesses)}</div>', unsafe_allow_html=True)
                if bat_strat or bowl_strat:
                    st.markdown(f"""
                    <div class="card" style="margin-top:.6rem;">
                      <h3>📋 Strategy</h3>
                      <div><b style="color:#60a5fa;">Batting:</b> {bat_strat}</div>
                      <div style="margin-top:.3rem;"><b style="color:#34d399;">Bowling:</b> {bowl_strat}</div>
                    </div>""", unsafe_allow_html=True)
    else:
        st.info("Team strengths/weaknesses data not available from backend.")

    # Tournament features table
    feat_data = api_get("/tournament-features")
    if feat_data:
        st.markdown("#### 🏆 Tournament Prediction Feature Weights")
        feat_df = pd.DataFrame(feat_data)
        if "feature_name" in feat_df.columns:
            disp_cols = [c for c in ["feature_name","description","weight_in_model",
                                     "colombo_kaps_score","galle_gallants_score","jaffna_kings_score",
                                     "dambulla_sixers_score","kandy_royals_score"] if c in feat_df.columns]
            st.dataframe(feat_df[disp_cols].rename(columns={
                "feature_name":"Feature","description":"Description","weight_in_model":"Weight",
                "colombo_kaps_score":"Colombo","galle_gallants_score":"Galle",
                "jaffna_kings_score":"Jaffna","dambulla_sixers_score":"Dambulla","kandy_royals_score":"Kandy"
            }), use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════
# TAB 5 — TOURNAMENT COMMAND CENTER LAUNCH
# ════════════════════════════════════════════════════════════════════════
if current_view == "tournament":
    st.markdown('<div id="tournament" class="section-anchor"></div>', unsafe_allow_html=True)
if current_view == "tournament":
    dash_payload = api_get("/api/dashboard") or {}
    standings = dash_payload.get("standings") or []
    venue_profiles = dash_payload.get("venue_profiles") or []
    awards_cards = (dash_payload.get("awards") or {}).get("cards") or []
    champion = (dash_payload.get("bracket") or {}).get("champion") or (standings[0].get("team") if standings else "TBD")
    top_cup = standings[0].get("cup", "-") if standings else "-"
    st.markdown(f"""
    <section class="product-art-hero tournament-art-hero">
      <img src="{TOURNAMENT_ART}" alt="LPL tournament squad architecture command center">
      <div class="product-hero-copy">
        <small>Tournament intelligence · squad architecture</small>
        <h1>Build the route<br>to the trophy.</h1>
        <p>Connect squad depth, player roles, venue profiles and bracket probabilities in one tournament command layer.</p>
        <div class="product-chip-row"><span>Bracket model</span><span>Squad structure</span><span>Award projections</span><span>Venue intelligence</span></div>
      </div>
      <div class="product-hero-stats">
        <div class="product-hero-stat"><b>{html.escape(str(champion))}</b><span>Projected champion</span></div>
        <div class="product-hero-stat"><b>{top_cup}</b><span>Top cup probability</span></div>
        <div class="product-hero-stat"><b>{len(standings) or "—"} teams</b><span>Ranked by model</span></div>
        <div class="product-hero-stat"><b>{len(awards_cards) or "—"} awards</b><span>Individual projections</span></div>
      </div>
    </section>
    <div class="launch-grid">
      <div class="launch-panel">
        <h3>Live Model View</h3>
        <div class="launch-title">Tournament predictions in a proper full-screen command center.</div>
        <div class="launch-copy">
          This section now opens the actual full dashboard page instead of rendering a cramped iframe inside Streamlit.
          The full view keeps the bracket, simulator, awards, cap race, and 3D field lab readable.
        </div>
        <div class="launch-metrics">
          <div class="launch-metric"><b>{len(standings) or "-"}</b><span>Teams ranked</span></div>
          <div class="launch-metric"><b>{len(venue_profiles) or "-"}</b><span>Venue profiles</span></div>
          <div class="launch-metric"><b>{top_cup}</b><span>Top cup probability</span></div>
          <div class="launch-metric"><b>{len(awards_cards) or "-"}</b><span>Award models</span></div>
        </div>
        <a class="launch-action" href="{HTML_DASHBOARD_URL}" target="_blank" rel="noopener">Open Tournament Command Center</a>
      </div>
      <div class="launch-panel">
        <h3>Current Signal</h3>
        <div class="launch-title">Projected leader: {champion}</div>
        <div class="launch-copy">
          The full dashboard is designed as a match-control room: first glance shows the tournament story,
          then each tab opens the evidence behind standings, simulations, players, caps, and field plans.
        </div>
        <a class="launch-action secondary" href="{HTML_FORECAST_URL}" target="_blank" rel="noopener">Open AI Tournament Forecast</a>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not backend_ok:
        st.caption("📡 Live model view connects automatically once the prediction engine is ready.")

# ════════════════════════════════════════════════════════════════════════
# TAB 6 — 3D FIELD + CATCH SIMULATION LAUNCH
# ════════════════════════════════════════════════════════════════════════
if current_view == "field":
    st.markdown("""
<div id="field-lab" class="section-anchor"></div>
<div class="web-section-head">
  <div><small>Tactical visualization</small><h2>3D Field & Catch Lab</h2>
  <p>Explore venue-aware field placements, shot paths, delivery simulations and catch difficulty.</p></div>
  <div class="web-section-index">06</div>
</div>
""", unsafe_allow_html=True)
if current_view == "field":
    field_payload = api_get("/api/fielding") or {}
    field_teams = field_payload.get("teams") or []
    field_venues = field_payload.get("venues") or []
    first_team = field_teams[0].get("name") if field_teams else "Team model"
    first_player = ""
    if field_teams and field_teams[0].get("batsmen"):
        first_player = field_teams[0]["batsmen"][0].get("name", "")
    st.markdown(f"""
    <div class="launch-grid">
      <div class="launch-panel">
        <h3>3D Tactical Lab</h3>
        <div class="launch-title">Field placements, venue surface, and wicket-catch replay in one view.</div>
        <div class="launch-copy">
          Select team, venue, and key batter. Then use <b>Predictive Delivery Simulation</b> to choose a fielder,
          choose a shot type, and press <b>START DELIVERY</b>. The ball travels from bowler to batter,
          shows bat impact, then follows a gravity-based predicted shot path toward the fielder.
        </div>
        <div class="launch-metrics">
          <div class="launch-metric"><b>{len(field_teams) or "-"}</b><span>Teams</span></div>
          <div class="launch-metric"><b>{len(field_venues) or "-"}</b><span>Grounds</span></div>
          <div class="launch-metric"><b>{first_team}</b><span>Default team</span></div>
          <div class="launch-metric"><b>{first_player or "-"}</b><span>Opening batter model</span></div>
        </div>
        <a class="launch-action" href="{HTML_FIELD_URL}" target="_blank" rel="noopener">Open Full 3D Field View</a>
      </div>
      <div class="field-mini">
        <span class="field-dot d1"></span>
        <span class="field-dot d2"></span>
        <span class="field-dot d3"></span>
        <span class="field-dot d4"></span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="card" style="margin-top:1rem;">
      <h3>Model Accuracy Boundary</h3>
      <div style="color:#8c99b5;line-height:1.6;font-size:.88rem;">
        True real-world catch accuracy needs match video tracking, release speed, bat-contact speed, spin, wind,
        and fielder movement data. This app uses a practical simulation layer: backend field positions plus
        projectile motion, flight time, apex height, catch distance, shot type, and dew difficulty.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not backend_ok:
        st.caption("📡 The 3D field view connects automatically once the prediction engine is ready.")


# ════════════════════════════════════════════════════════════════════════
# CLOUD-SAFE FULL VIEWS
# FastAPI's localhost HTML routes are not publicly exposed by Streamlit
# Community Cloud, so the launch buttons open native Streamlit views.
# ════════════════════════════════════════════════════════════════════════
def _subpage_header(eyebrow, title, copy, back_page):
    st.markdown(
        f"""
        <div style="margin:1rem 0 1.5rem">
          <a href="?page={back_page}" style="color:#ffd37a;text-decoration:none;font-weight:800">← Back</a>
          <div class="web-section-head" style="margin-top:1.3rem">
            <div><small>{eyebrow}</small><h2>{title}</h2><p>{copy}</p></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _field_figure(positions, title):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor="#080b14")
    ax.set_facecolor("#0b2b1c")
    boundary = plt.Circle((0, 0), 1.02, fill=False, color="#f5a623", linewidth=2)
    inner = plt.Circle((0, 0), 0.66, fill=False, color="#ffffff", alpha=.18, linewidth=1)
    ax.add_patch(boundary)
    ax.add_patch(inner)
    ax.plot([0, 0], [-0.34, .34], color="#d8b36b", linewidth=12, alpha=.85)
    ax.scatter([0], [0], s=90, color="#ffd37a", edgecolor="#080b14", zorder=5)
    type_colors = {"wk": "#65d7ff", "close": "#ef4444", "ring": "#f5a623", "deep": "#8b5cf6"}
    for pos in positions or []:
        x = float(pos.get("x", 0))
        y = float(pos.get("z", 0))
        kind = pos.get("type", pos.get("t", "ring"))
        name = pos.get("name", pos.get("n", "Fielder"))
        ax.scatter([x], [y], s=105, color=type_colors.get(kind, "#f5a623"),
                   edgecolor="#ffffff", linewidth=.7, zorder=6)
        ax.annotate(name, (x, y), xytext=(0, 8), textcoords="offset points",
                    ha="center", color="#f4f7fb", fontsize=8, fontweight="bold")
    ax.set_title(title, color="#ffd37a", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout()
    return fig


if current_view == "command":
    _subpage_header(
        "Tournament operations",
        "Tournament Command Center",
        "Standings, qualification picture, awards and venue intelligence in one cloud-ready view.",
        "tournament",
    )
    command_data = api_get("/api/dashboard") or {}
    command_standings = command_data.get("standings") or []
    bracket = command_data.get("bracket") or {}
    awards = (command_data.get("awards") or {}).get("labels") or []
    venue_profiles = command_data.get("venue_profiles") or []

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projected champion", bracket.get("champion", "—"))
    c2.metric("Runner-up", bracket.get("runner_up", "—"))
    c3.metric("Qualified teams", len(bracket.get("qualified") or []))
    c4.metric("Model status", "Online" if (command_data.get("model_status") or {}).get("match_model_loaded") else "Fallback")

    st.markdown("### Projected standings")
    if command_standings:
        standings_df = pd.DataFrame(command_standings)
        standings_cols = [c for c in ["pos", "team", "p", "w", "l", "pts", "nrr", "cup", "qualifies"] if c in standings_df.columns]
        st.dataframe(
            standings_df[standings_cols].rename(columns={
                "pos": "Rank", "team": "Team", "p": "P", "w": "W", "l": "L",
                "pts": "Points", "nrr": "NRR", "cup": "Cup %", "qualifies": "Qualifies",
            }),
            width="stretch",
            hide_index=True,
        )

    left, right = st.columns(2)
    with left:
        st.markdown("### Tournament awards")
        if awards:
            award_df = pd.DataFrame(awards)
            award_cols = [c for c in ["label_type", "label_value", "confidence_pct", "basis"] if c in award_df.columns]
            st.dataframe(
                award_df[award_cols].rename(columns={
                    "label_type": "Award", "label_value": "Prediction",
                    "confidence_pct": "Confidence %", "basis": "Model basis",
                }),
                width="stretch",
                hide_index=True,
            )
    with right:
        st.markdown("### Venue intelligence")
        if venue_profiles:
            venue_df = pd.DataFrame(venue_profiles)
            venue_cols = [c for c in [
                "label", "pitch_type", "safe_score_target", "spin_advantage_pct",
                "pace_advantage_pct", "dew_factor",
            ] if c in venue_df.columns]
            st.dataframe(
                venue_df[venue_cols].rename(columns={
                    "label": "Venue", "pitch_type": "Pitch", "safe_score_target": "Safe target",
                    "spin_advantage_pct": "Spin %", "pace_advantage_pct": "Pace %",
                    "dew_factor": "Dew",
                }),
                width="stretch",
                hide_index=True,
            )


if current_view == "forecast":
    _subpage_header(
        "Monte-Carlo outlook",
        "AI Tournament Forecast",
        "Match probabilities and tournament progression generated from the trained prediction stack.",
        "tournament",
    )
    with st.spinner("Running tournament simulations…"):
        forecast = api_get("/api/tournament-forecast?sims=2000") or {}

    if forecast:
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Champion", forecast.get("projected_champion", "—"))
        f2.metric("Runner-up", forecast.get("projected_runner_up", "—"))
        f3.metric("Simulations", f"{forecast.get('n_simulations', 0):,}")
        f4.metric("Scheduled matches", len(forecast.get("matches") or []))

        progression = pd.DataFrame(forecast.get("progression") or [])
        if not progression.empty:
            st.markdown("### Qualification probabilities")
            chart_df = progression.set_index("team")[
                [c for c in ["semi_pct", "final_pct", "champion_pct"] if c in progression.columns]
            ].rename(columns={"semi_pct": "Semi-final %", "final_pct": "Final %", "champion_pct": "Champion %"})
            st.bar_chart(chart_df, color=["#65d7ff", "#f5a623", "#8b5cf6"])

        projected = pd.DataFrame(forecast.get("standings_projection") or [])
        if not projected.empty:
            st.markdown("### Projected points table")
            st.dataframe(projected, width="stretch", hide_index=True)

        matches = pd.DataFrame(forecast.get("matches") or [])
        if not matches.empty:
            st.markdown("### Match-by-match forecast")
            match_cols = [c for c in [
                "date", "time", "team1", "team2", "venue", "win_prob_team1",
                "win_prob_team2", "predicted_winner",
            ] if c in matches.columns]
            st.dataframe(
                matches[match_cols].rename(columns={
                    "date": "Date", "time": "Time", "team1": "Team 1", "team2": "Team 2",
                    "venue": "Venue", "win_prob_team1": "Team 1 %", "win_prob_team2": "Team 2 %",
                    "predicted_winner": "Predicted winner",
                }),
                width="stretch",
                hide_index=True,
            )
    else:
        st.error("Tournament forecast is unavailable. Check the prediction-engine logs.")


if current_view in {"tactics", "field_lab"}:
    cloud_field_data = api_get("/api/fielding") or {}
    cloud_field_teams = cloud_field_data.get("teams") or []
    cloud_field_venues = cloud_field_data.get("venues") or []


if current_view == "tactics":
    _subpage_header(
        "AI match planning",
        "Strategy Room",
        "Choose a batter, venue and match phase to generate a model-backed delivery and field plan.",
        "home",
    )
    player_options = []
    for team in cloud_field_teams:
        for batter in team.get("batsmen") or []:
            player_options.append((f"{batter.get('name')} · {team.get('name')}", batter.get("name")))
    venue_options = [v.get("venue_name") for v in cloud_field_venues if v.get("venue_name")]

    if player_options and venue_options:
        p1, p2, p3 = st.columns(3)
        player_label = p1.selectbox("Target batter", [p[0] for p in player_options], key="strategy_player")
        strategy_venue = p2.selectbox("Venue", venue_options, key="strategy_venue")
        strategy_phase = p3.selectbox("Match phase", ["Powerplay", "Middle Overs", "Death Overs"], key="strategy_phase")
        target_player = dict(player_options)[player_label]

        if st.button("Generate tactical plan", type="primary", width="stretch"):
            st.session_state["cloud_tactical_result"] = api_post("/api/tactical-plan", {
                "target_batsman": target_player,
                "venue": strategy_venue,
                "match_phase": strategy_phase,
            })

        tactical_result = st.session_state.get("cloud_tactical_result")
        if tactical_result:
            plan = tactical_result.get("tactical_plan") or {}
            delivery = plan.get("recommended_delivery") or {}
            outcome = plan.get("expected_outcome") or {}
            analysis = plan.get("analysis") or {}
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Bowler", plan.get("bowler", "—"))
            t2.metric("Delivery", delivery.get("type", "—"))
            t3.metric("Expected outcome", outcome.get("outcome", "—"))
            t4.metric("Probability", f"{outcome.get('probability', 0)}%")
            st.info(
                f"{delivery.get('line', '—')} · {delivery.get('length', '—')} · "
                f"{delivery.get('movement', '—')} · {delivery.get('speed', '—')}"
            )
            fig = _field_figure(
                plan.get("optimal_field_setup") or [],
                f"Optimal field · {analysis.get('pitch_type', strategy_venue)}",
            )
            st.pyplot(fig)
            plt.close(fig)


if current_view == "field_lab":
    _subpage_header(
        "Interactive placement map",
        "Full Field Lab",
        "Explore batter-specific, venue-aware field placements from the tactical dataset.",
        "field",
    )
    if cloud_field_teams and cloud_field_venues:
        team_names = [t.get("name") for t in cloud_field_teams]
        fl1, fl2 = st.columns(2)
        selected_team_name = fl1.selectbox("Team", team_names, key="field_lab_team")
        selected_team = next(t for t in cloud_field_teams if t.get("name") == selected_team_name)
        batters = selected_team.get("batsmen") or []
        selected_batter_name = fl2.selectbox(
            "Batter", [b.get("name") for b in batters], key="field_lab_batter"
        )
        selected_batter = next(b for b in batters if b.get("name") == selected_batter_name)

        available_plans = selected_batter.get("plans") or {}
        plan_venues = list(available_plans) or [v.get("venue_name") for v in cloud_field_venues]
        selected_plan_venue = st.selectbox("Venue", plan_venues, key="field_lab_venue")
        field_plan = available_plans.get(selected_plan_venue) or {}

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Model score", selected_batter.get("modelScore", "—"))
        m2.metric("Pitch", field_plan.get("pitchType", "—"))
        m3.metric("Timing", field_plan.get("matchTiming", "—"))
        m4.metric("Safe target", (field_plan.get("summary") or "—").split("|")[0].strip())
        st.info(field_plan.get("fieldType") or selected_batter.get("fieldType") or "Field plan ready.")
        fig = _field_figure(
            field_plan.get("pos") or [],
            f"{selected_batter_name} · {selected_plan_venue}",
        )
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.error("Fielding data is unavailable. Check the prediction-engine logs.")


st.markdown("""
<footer class="site-footer">
  LPL 2026 Intelligence Platform · Data, insight and strategic decision support<br>
  Model projections are designed for analysis and learning.
</footer>
""", unsafe_allow_html=True)
