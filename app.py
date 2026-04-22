"""
Smart ATS Resume Analyzer — Premium SaaS UI Edition  (v2 — Enhanced)
=====================================================================
Original features kept intact. Five new features added:
  1. JD File Upload  (PDF / TXT fallback to text_area)
  2. Keyword Highlighting  (new "🔍 Keyword Highlight" tab)
  3. Section-wise Scoring  (inside Overview tab)
  4. ATS Resume Checker     (new "🛡️ ATS Checker" tab)
  5. Resume Improvement Generator (new "✨ Improve Resume" tab)

NEW CODE is clearly delimited with  ─── NEW CODE START / NEW CODE END ───
"""

import io
import re
import warnings
from collections import Counter

import streamlit as st
import streamlit.components.v1 as components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors
import numpy as np

import time
import pandas as pd
import folium
from folium.plugins import HeatMap

import pdfplumber
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Smart ATS Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ══════════════════════════════════════════════════════════════════
#  GLOBAL CSS — Premium SaaS dark theme
# ══════════════════════════════════════════════════════════════════
PREMIUM_CSS = """
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Design tokens ── */
:root {
  --bg-base:      #050810;
  --bg-surface:   #0c1120;
  --bg-card:      rgba(255,255,255,0.04);
  --border:       rgba(255,255,255,0.08);
  --border-glow:  rgba(99,179,237,0.35);
  --accent-cyan:  #63b3ed;
  --accent-green: #68d391;
  --accent-purple:#b794f4;
  --accent-amber: #f6ad55;
  --danger:       #fc8181;
  --text-primary: #f0f4ff;
  --text-muted:   #718096;
  --text-dim:     #4a5568;
  --radius-lg:    16px;
  --radius-md:    10px;
  --radius-sm:    6px;
  --shadow-glass: 0 8px 32px rgba(0,0,0,0.4);
  --shadow-glow:  0 0 20px rgba(99,179,237,0.15);
}

/* ── Base reset — transparent so particles shine through ── */
html, body {
  background: #050810 !important;
  overflow-x: hidden;
}
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background: transparent !important;
  position: relative;
  z-index: 1;
}
/* Deep radial gradient overlay so particles don't clash with content */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 50% at 20% 20%, rgba(99,179,237,0.04) 0%, transparent 60%),
    radial-gradient(ellipse 60% 60% at 80% 80%, rgba(167,139,250,0.04) 0%, transparent 60%),
    radial-gradient(ellipse 100% 80% at 50% 50%, rgba(5,8,16,0.65) 0%, rgba(5,8,16,0.85) 100%);
  z-index: -1;
  pointer-events: none;
}
[data-testid="stHeader"]         { background: transparent !important; }
[data-testid="stSidebar"]        { background: rgba(12,17,32,0.9) !important; backdrop-filter: blur(20px); }
[data-testid="stDecoration"]     { display: none !important; }
section[data-testid="stMain"] > div { padding-top: 0 !important; }

/* ── Fade-in animation ── */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(18px); }
  to   { opacity: 1; transform: translateY(0); }
}
@keyframes glowPulse {
  0%, 100% { box-shadow: 0 0 20px rgba(99,179,237,0.15), 0 0 40px rgba(99,179,237,0.05); }
  50%       { box-shadow: 0 0 30px rgba(99,179,237,0.25), 0 0 60px rgba(99,179,237,0.1); }
}
@keyframes neonFlicker {
  0%, 100% { opacity: 1; }
  92%       { opacity: 1; }
  93%       { opacity: 0.85; }
  94%       { opacity: 1; }
}

/* Apply fade-in to main blocks */
[data-testid="stVerticalBlock"] > div {
  animation: fadeInUp 0.45s ease both;
}

/* ── Neon glow on interactive metric values ── */
[data-testid="stMetricValue"] {
  text-shadow: 0 0 18px rgba(99,179,237,0.55) !important;
  animation: glowPulse 3s ease-in-out infinite;
}

/* ── Typography ── */
*, h1, h2, h3, h4, p, span, div, label {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  color: var(--text-primary) !important;
}
code, pre, .mono { font-family: 'JetBrains Mono', monospace !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar            { width: 6px; }
::-webkit-scrollbar-track      { background: transparent; }
::-webkit-scrollbar-thumb      { background: rgba(99,179,237,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover{ background: rgba(99,179,237,0.4); }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
  background: rgba(12,17,32,0.7) !important;
  backdrop-filter: blur(12px) !important;
  border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
  padding: 6px 6px 0 !important;
  gap: 4px;
  border-bottom: 1px solid rgba(99,179,237,0.12) !important;
}
[data-testid="stTabs"] button {
  font-weight: 600 !important;
  font-size: .82rem !important;
  letter-spacing: .04em !important;
  color: var(--text-muted) !important;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
  padding: .5rem .9rem !important;
  transition: all .2s ease !important;
  border: none !important;
}
[data-testid="stTabs"] button:hover {
  color: var(--text-primary) !important;
  background: rgba(99,179,237,0.06) !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--accent-cyan) !important;
  background: rgba(99,179,237,0.1) !important;
  border-bottom: 2px solid var(--accent-cyan) !important;
  text-shadow: 0 0 16px rgba(99,179,237,0.5) !important;
  box-shadow: 0 0 12px rgba(99,179,237,0.08) inset !important;
}

/* ── Analyze button ── */
 /* ✅ SAFE BUTTON (FINAL FIXED VERSION) */
div[data-testid="stButton"] > button {
  background: linear-gradient(135deg, #1a56db, #7c3aed);
  color: #ffffff;
  border: none;
  border-radius: 10px;
  font-weight: 700;
  font-size: 1rem;
  letter-spacing: 0.05em;
  padding: 0.6rem 1.5rem;
  transition: all 0.2s ease;
  box-shadow: 0 4px 15px rgba(124,58,237,0.35);
}

/* Hover */
div[data-testid="stButton"] > button:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(124,58,237,0.5);
}

/* Click */
div[data-testid="stButton"] > button:active {
  transform: scale(0.98);
}

/* ── Download button ── */
.stDownloadButton > button {
  background: linear-gradient(135deg, #065f46 0%, #047857 100%) !important;
  color: var(--accent-green) !important;
  border: 1px solid rgba(104,211,145,.3) !important;
  border-radius: var(--radius-md) !important;
  font-weight: 700 !important;
  font-size: .95rem !important;
  padding: .65rem 1.5rem !important;
  transition: all .2s ease !important;
  box-shadow: 0 4px 15px rgba(6,95,70,.3) !important;
}
.stDownloadButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 20px rgba(6,95,70,.45) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: rgba(99,179,237,0.03) !important;
  border: 1.5px dashed rgba(99,179,237,0.25) !important;
  border-radius: var(--radius-lg) !important;
  transition: all .25s ease !important;
  backdrop-filter: blur(8px) !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: rgba(99,179,237,0.6) !important;
  box-shadow: 0 0 20px rgba(99,179,237,0.1), inset 0 0 20px rgba(99,179,237,0.03) !important;
}

/* ── Textarea ── */
textarea {
  background: rgba(12,17,32,0.7) !important;
  border: 1.5px solid rgba(255,255,255,0.08) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-primary) !important;
  font-size: .88rem !important;
  transition: all .2s ease !important;
  backdrop-filter: blur(8px) !important;
}
textarea:focus {
  border-color: rgba(99,179,237,0.5) !important;
  box-shadow: 0 0 0 3px rgba(99,179,237,.08), 0 0 20px rgba(99,179,237,.1) !important;
}

/* ── Native metric widget ── */
[data-testid="stMetric"] {
  background: rgba(12,17,32,0.6) !important;
  border: 1px solid rgba(99,179,237,0.12) !important;
  border-top: 1px solid rgba(255,255,255,0.08) !important;
  border-radius: var(--radius-lg) !important;
  padding: 1.1rem 1.3rem !important;
  backdrop-filter: blur(16px) !important;
  -webkit-backdrop-filter: blur(16px) !important;
  transition: all .25s ease !important;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.05) !important;
}
[data-testid="stMetric"]:hover {
  border-color: rgba(99,179,237,0.3) !important;
  box-shadow: 0 8px 32px rgba(0,0,0,0.4), 0 0 20px rgba(99,179,237,0.08) !important;
  transform: translateY(-1px) scale(1.01) !important;
}
[data-testid="stMetricValue"] {
  font-size: 1.75rem !important;
  font-weight: 800 !important;
  color: var(--accent-cyan) !important;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size:.8rem !important; }
[data-testid="stMetricDelta"]  { font-size: .75rem !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
  background: rgba(12,17,32,0.55) !important;
  border: 1px solid rgba(255,255,255,0.07) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden;
  backdrop-filter: blur(12px) !important;
  transition: border-color .2s ease !important;
}
[data-testid="stExpander"]:hover {
  border-color: rgba(99,179,237,0.18) !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpanderHeader"] {
  background: transparent !important;
  padding: .5rem .8rem !important;
  cursor: pointer;
}
[data-testid="stExpander"] summary > div > svg,
[data-testid="stExpander"] details > summary span.material-icons,
[data-testid="stExpander"] details > summary > div > span:first-child {
  font-size: 0 !important;
  line-height: 0 !important;
  overflow: hidden !important;
}
[data-testid="stExpander"] summary svg {
  font-size: initial !important;
  width: 18px !important;
  height: 18px !important;
  color: var(--text-muted) !important;
  fill: var(--text-muted) !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] .streamlit-expanderHeader p {
  font-weight: 600 !important;
  font-size: .88rem !important;
  color: var(--text-primary) !important;
  margin: 0 !important;
}
[data-testid="stExpander"] summary span {
    font-size: 0 !important;
    line-height: 0 !important;
}
[data-testid="stExpander"] summary svg {
    font-size: initial !important;
    width: 16px !important;
    height: 16px !important;
    display: inline-block !important;
}
[data-testid="stExpander"] summary {
    display: flex !important;
    align-items: center !important;
    gap: 6px !important;
}

/* ── Alert / info boxes ── */
[data-testid="stAlert"] {
  background: rgba(99,179,237,.06) !important;
  border: 1px solid rgba(99,179,237,.18) !important;
  border-radius: var(--radius-md) !important;
  backdrop-filter: blur(8px) !important;
}

/* ── Dataframe ── */
[data-testid="stDataFrame"] {
  border: 1px solid rgba(99,179,237,0.12) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden;
  backdrop-filter: blur(8px) !important;
}

/* ── Caption ── */
[data-testid="stCaptionContainer"] p {
  color: var(--text-dim) !important;
  font-size: .78rem !important;
}

/* ── Divider ── */
hr {
  border: none !important;
  height: 1px !important;
  background: linear-gradient(90deg,transparent,rgba(99,179,237,.2),transparent) !important;
  margin: 1.4rem 0 !important;
}

/* ── Images (charts) ── */
[data-testid="stImage"] img {
  border-radius: var(--radius-md) !important;
  box-shadow: 0 4px 20px rgba(0,0,0,0.4) !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--accent-cyan) !important; }

/* ─── MAP INSIGHTS STYLES ─── */
.map-glass-card {
  background: rgba(12,17,32,0.75);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  border: 1px solid rgba(99,179,237,0.14);
  border-top: 1px solid rgba(255,255,255,0.09);
  border-radius: 20px;
  padding: 1.6rem 1.8rem 1.4rem;
  margin-bottom: 1.2rem;
  box-shadow: 0 12px 40px rgba(0,0,0,0.5),
              0 0 60px rgba(99,179,237,0.06),
              inset 0 1px 0 rgba(255,255,255,0.05);
}
.map-title {
  font-size: 1.1rem !important;
  font-weight: 800 !important;
  color: #63b3ed !important;
  letter-spacing: .03em;
  margin: 0 0 .25rem !important;
  text-shadow: 0 0 22px rgba(99,179,237,0.4);
}
.map-subtitle {
  font-size: .82rem !important;
  color: #718096 !important;
  margin: 0 0 1.2rem !important;
}
.loc-badge {
  display:inline-flex; align-items:center; gap:.4rem;
  background: rgba(99,179,237,0.09);
  border: 1px solid rgba(99,179,237,0.22);
  border-radius: 999px;
  padding: 4px 14px;
  font-size: .79rem;
  font-weight: 600;
  color: #63b3ed !important;
  margin: 3px 3px;
  transition: all .2s ease;
}
.loc-freq-bar {
  background: linear-gradient(90deg, rgba(99,179,237,0.18), rgba(183,148,244,0.18));
  border: 1px solid rgba(99,179,237,0.12);
  border-radius: 10px;
  padding: .7rem 1rem;
  margin-bottom: .5rem;
  display: flex;
  align-items: center;
  gap: .8rem;
}
/* ─── NEW CODE START ─── keyword highlight box ─── */
.kw-highlight-box {
  background: rgba(12,17,32,0.6);
  border: 1px solid rgba(99,179,237,0.12);
  border-radius: 14px;
  padding: 1.2rem 1.4rem;
  line-height: 2.1;
  font-size: .9rem;
  color: #e2e8f0 !important;
  max-height: 420px;
  overflow-y: auto;
  backdrop-filter: blur(12px);
}
/* ─── NEW CODE END ─── */
</style>
"""
st.markdown(PREMIUM_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PARTICLES BACKGROUND  — Full-viewport animated neon particles
# ══════════════════════════════════════════════════════════════════

def advanced_particles_background():
    """
    Inject a full-screen animated particles.js background that covers the
    entire viewport.  Uses an invisible iframe (height=0) whose JS reaches
    into the parent document to mount the canvas — this bypasses the iframe
    boundary so the effect fills the whole page, not just a widget box.
    """
    _particles_html = """
<!DOCTYPE html>
<html>
<head><style>body{margin:0;padding:0;background:transparent;}</style></head>
<body>
<script>
(function () {
  "use strict";
  var parent = window.parent || window;
  var doc    = parent.document;

  /* ── Only inject once ── */
  if (doc.getElementById("ats-particles-host")) return;

  /* ── Create fixed canvas host ── */
  var host = doc.createElement("div");
  host.id  = "ats-particles-host";
  host.style.cssText = [
    "position:fixed",
    "top:0","left:0",
    "width:100vw","height:100vh",
    "z-index:0",
    "pointer-events:none",
    "overflow:hidden",
  ].join(";");
  doc.body.insertBefore(host, doc.body.firstChild);

  /* ── Ambient gradient overlay sitting on top of particles ── */
  var overlay = doc.createElement("div");
  overlay.style.cssText = [
    "position:fixed","inset:0",
    "background:radial-gradient(ellipse 100% 60% at 50% 0%,rgba(5,8,16,0) 0%,rgba(5,8,16,0.55) 100%)",
    "z-index:1","pointer-events:none",
  ].join(";");
  doc.body.insertBefore(overlay, host.nextSibling);

  /* ── Ensure app content renders above particles ── */
  var appStyle = doc.createElement("style");
  appStyle.textContent = [
    "#root,",
    "[data-testid='stApp'],",
    "[data-testid='stAppViewContainer'],",
    ".main { position:relative !important; z-index:2 !important; }",
    /* Subtle neon connection-line glow via SVG filter */
    "#ats-particles-host canvas { filter: brightness(1.05); }",
  ].join("\n");
  doc.head.appendChild(appStyle);

  /* ── Load particles.js from CDN, then init ── */
  var script  = doc.createElement("script");
  script.src  = "https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js";
  script.onload = function () {
    parent.particlesJS("ats-particles-host", {
      "particles": {
        "number":  { "value": 90, "density": { "enable": true, "value_area": 900 } },
        "color":   { "value": ["#60a5fa", "#6ee7b7", "#a78bfa"] },
        "shape":   { "type": "circle" },
        "opacity": {
          "value": 0.45, "random": true,
          "anim": { "enable": true, "speed": 0.6, "opacity_min": 0.1, "sync": false }
        },
        "size": {
          "value": 2.8, "random": true,
          "anim": { "enable": true, "speed": 1.5, "size_min": 0.5, "sync": false }
        },
        "line_linked": {
          "enable": true, "distance": 130,
          "color": "#60a5fa", "opacity": 0.13, "width": 1
        },
        "move": {
          "enable": true, "speed": 0.9, "direction": "none",
          "random": true, "straight": false, "out_mode": "out",
          "bounce": false,
          "attract": { "enable": false }
        }
      },
      "interactivity": {
        "detect_on": "window",
        "events": {
          "onhover": { "enable": true, "mode": "repulse" },
          "onclick": { "enable": false },
          "resize":  true
        },
        "modes": {
          "repulse": { "distance": 100, "duration": 0.4 },
          "grab":    { "distance": 140, "line_linked": { "opacity": 0.3 } }
        }
      },
      "retina_detect": true
    });
  };
  doc.head.appendChild(script);
})();
</script>
</body>
</html>
"""
    components.html(_particles_html, height=0, scrolling=False)

def render_hero():
    """Full-width hero banner with animated neon gradient mesh background."""
    st.markdown("""
    <div style="
        background: linear-gradient(135deg,rgba(5,8,16,0.95) 0%,rgba(13,27,62,0.85) 40%,rgba(26,10,46,0.85) 70%,rgba(5,8,16,0.95) 100%);
        border: 1px solid rgba(99,179,237,0.15);
        border-top: 1px solid rgba(255,255,255,0.1);
        border-radius: 24px;
        padding: 3.5rem 2rem 3rem;
        text-align: center;
        position: relative;
        overflow: hidden;
        margin-bottom: 1.8rem;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 20px 60px rgba(0,0,0,0.5), 0 0 80px rgba(99,179,237,0.05), inset 0 1px 0 rgba(255,255,255,0.06);
    ">
      <!-- Corner glow blobs -->
      <div style="
        position:absolute; top:-80px; left:-80px;
        width:280px; height:280px; border-radius:50%;
        background:radial-gradient(circle,rgba(124,58,237,.22),transparent 70%);
        pointer-events:none; filter:blur(2px);
      "></div>
      <div style="
        position:absolute; bottom:-60px; right:-60px;
        width:260px; height:260px; border-radius:50%;
        background:radial-gradient(circle,rgba(99,179,237,.18),transparent 70%);
        pointer-events:none; filter:blur(2px);
      "></div>
      <div style="
        position:absolute; top:30%; left:50%; transform:translateX(-50%);
        width:400px; height:1px; border-radius:50%;
        background:linear-gradient(90deg,transparent,rgba(99,179,237,.15),transparent);
        pointer-events:none;
      "></div>
      <!-- Badge -->
      <div style="
        display:inline-block;
        background:linear-gradient(135deg,rgba(99,179,237,.12),rgba(167,139,250,.12));
        border:1px solid rgba(99,179,237,.3);
        border-radius:50px;
        padding:5px 18px;
        font-size:.76rem;
        font-weight:700;
        letter-spacing:.1em;
        color:#63b3ed !important;
        margin-bottom:1.2rem;
        text-transform:uppercase;
        box-shadow: 0 0 20px rgba(99,179,237,.12);
      ">✦ AI-Powered Resume Intelligence</div>
      <!-- Title -->
      <h1 style="
        font-size:clamp(2rem,5vw,3.2rem) !important;
        font-weight:800 !important;
        background: linear-gradient(90deg,#63b3ed 0%,#b794f4 45%,#68d391 90%);
        -webkit-background-clip:text;
        -webkit-text-fill-color:transparent;
        line-height:1.15 !important;
        margin:0 0 1rem !important;
        filter: drop-shadow(0 0 30px rgba(99,179,237,.25));
      ">🎯 Smart ATS Resume Analyzer</h1>
      <!-- Subtitle -->
      <p style="
        color:#8899b5 !important;
        font-size:1.05rem;
        max-width:580px;
        margin:0 auto;
        line-height:1.7;
      ">
        Upload your resume · Paste or upload a job description · Get a precision fit score,
        skill gap analysis, ATS check &amp; a downloadable recruiter report.
      </p>
      <!-- Stat chips row -->
      <div style="display:flex;justify-content:center;gap:1.2rem;margin-top:1.8rem;flex-wrap:wrap;">
        <span style="background:rgba(104,211,145,.1);border:1px solid rgba(104,211,145,.25);border-radius:8px;padding:5px 14px;font-size:.78rem;font-weight:600;color:#68d391 !important;">⚡ TF-IDF Similarity</span>
        <span style="background:rgba(99,179,237,.1);border:1px solid rgba(99,179,237,.25);border-radius:8px;padding:5px 14px;font-size:.78rem;font-weight:600;color:#63b3ed !important;">🔍 Skill Gap Analysis</span>
        <span style="background:rgba(167,139,250,.1);border:1px solid rgba(167,139,250,.25);border-radius:8px;padding:5px 14px;font-size:.78rem;font-weight:600;color:#b794f4 !important;">🛡️ ATS Checker</span>
        <span style="background:rgba(246,173,85,.1);border:1px solid rgba(246,173,85,.25);border-radius:8px;padding:5px 14px;font-size:.78rem;font-weight:600;color:#f6ad55 !important;">📄 PDF Report</span>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_glass_section(title: str, icon: str, content_html: str, accent="#63b3ed"):
    safe = content_html.strip()
    html = (
        '<div style="background:rgba(12,17,32,0.65);backdrop-filter:blur(20px);'
        '-webkit-backdrop-filter:blur(20px);'
        'border:1px solid rgba(255,255,255,0.07);'
        'border-top:1px solid rgba(255,255,255,0.1);'
        'border-radius:18px;'
        'padding:1.5rem 1.6rem;margin-bottom:1rem;'
        f'box-shadow:0 8px 32px rgba(0,0,0,0.4),0 0 30px {accent}11,'
        'inset 0 1px 0 rgba(255,255,255,0.05);'
        'transition:box-shadow .3s ease,border-color .3s ease;">'
        '<div style="display:flex;align-items:center;gap:.6rem;margin-bottom:1rem;'
        'padding-bottom:.75rem;border-bottom:1px solid rgba(255,255,255,0.06);">'
        f'<span style="font-size:1.2rem;filter:drop-shadow(0 0 8px {accent}88);">{icon}</span>'
        f'<span style="font-weight:700;font-size:1rem;color:{accent} !important;'
        f'letter-spacing:.02em;text-shadow:0 0 20px {accent}66;">{title}</span>'
        '</div>'
        f'{safe}'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_big_score(fit_score: float):
    if fit_score >= 80:
        color    = "#68d391"
        bg_glow  = "rgba(104,211,145,.12)"
        label    = "🟢 Strong Match"
        border_c = "rgba(104,211,145,.3)"
    elif fit_score >= 50:
        color    = "#f6ad55"
        bg_glow  = "rgba(246,173,85,.1)"
        label    = "🟡 Moderate Match"
        border_c = "rgba(246,173,85,.3)"
    else:
        color    = "#fc8181"
        bg_glow  = "rgba(252,129,129,.1)"
        label    = "🔴 Needs Improvement"
        border_c = "rgba(252,129,129,.3)"

    st.markdown(f"""
    <div style="
        background: linear-gradient(145deg, {bg_glow}, rgba(12,17,32,0.8));
        border: 1px solid {border_c};
        border-top: 1px solid rgba(255,255,255,0.1);
        border-radius: 24px;
        padding: 2.5rem 1rem 2rem;
        text-align: center;
        margin-bottom: 1.2rem;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        box-shadow: 0 0 60px {bg_glow}, 0 20px 40px rgba(0,0,0,0.4),
                    inset 0 1px 0 rgba(255,255,255,0.06);
        position: relative; overflow: hidden;
    ">
      <!-- Radial glow behind number -->
      <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-60%);
        width:200px;height:200px;border-radius:50%;
        background:radial-gradient(circle,{bg_glow},transparent 70%);
        pointer-events:none;filter:blur(8px);"></div>
      <div style="color:#8899b5 !important; font-size:.72rem; font-weight:700;
                  letter-spacing:.14em; text-transform:uppercase; margin-bottom:.5rem;">
        FINAL FIT SCORE
      </div>
      <div style="
        font-size:clamp(3.2rem,9vw,5.5rem) !important;
        font-weight:800 !important;
        color:{color} !important;
        line-height:1;
        font-family:'JetBrains Mono',monospace !important;
        text-shadow: 0 0 30px {color}88, 0 0 60px {color}44;
        position:relative; z-index:1;
      ">{fit_score:.1f}<span style="font-size:2.2rem;color:{color} !important;">%</span></div>
      <div style="
        display:inline-block;
        background:rgba(255,255,255,0.06);
        border:1px solid {border_c};
        border-radius:50px;
        padding:5px 20px;
        margin-top:1rem;
        font-size:.85rem;
        font-weight:700;
        color:{color} !important;
        box-shadow: 0 0 16px {bg_glow};
        letter-spacing:.04em;
      ">{label}</div>
      <div style="color:#4a5568 !important; font-size:.72rem; margin-top:.9rem; letter-spacing:.02em;">
        Fit Score = 0.6 × TF-IDF Similarity + 0.4 × Skill Match Score
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_metric_card(icon: str, label: str, value: str, sub: str, accent: str):
    st.markdown(f"""
    <div style="
        background: rgba(12,17,32,0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.07);
        border-top: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.2rem 1.3rem;
        height: 100%;
        transition: border-color .25s ease, box-shadow .25s ease, transform .2s ease;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.04);
        position: relative; overflow: hidden;
    ">
      <div style="position:absolute;top:0;right:0;width:80px;height:80px;border-radius:50%;
        background:radial-gradient(circle,{accent}18,transparent 70%);
        pointer-events:none;transform:translate(20px,-20px);"></div>
      <div style="font-size:1.5rem; margin-bottom:.5rem; filter:drop-shadow(0 0 8px {accent}88);">{icon}</div>
      <div style="
        font-size:2.1rem !important; font-weight:800 !important;
        color:{accent} !important; line-height:1; margin-bottom:.3rem;
        font-family:'JetBrains Mono',monospace !important;
        text-shadow: 0 0 20px {accent}66;
      ">{value}</div>
      <div style="font-weight:700; font-size:.85rem; color:#e2e8f0 !important;
                  margin-bottom:.25rem;">{label}</div>
      <div style="font-size:.74rem; color:#718096 !important; letter-spacing:.03em;">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def render_gradient_progress(label: str, value: float,
                              color_start="#63b3ed", color_end="#b794f4"):
    pct = min(max(value, 0), 100)
    st.markdown(f"""
    <div style="margin-bottom:1rem;">
      <div style="
        display:flex; justify-content:space-between;
        margin-bottom:5px;
      ">
        <span style="font-size:.82rem; font-weight:600; color:#a0aec0 !important;">
          {label}
        </span>
        <span style="
          font-size:.82rem; font-weight:700;
          color:{color_start} !important;
          font-family:'JetBrains Mono',monospace !important;
        ">{pct:.1f}%</span>
      </div>
      <div style="
        background: rgba(255,255,255,0.05);
        border-radius:999px;
        height:10px;
        overflow:hidden;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);
      ">
        <div style="
          width:{pct}%;
          height:100%;
          background:linear-gradient(90deg,{color_start},{color_end});
          border-radius:999px;
          transition:width .8s cubic-bezier(.4,0,.2,1);
          box-shadow: 0 0 12px {color_start}77, 0 0 24px {color_start}33;
          position:relative;
        "></div>
      </div>
    </div>
    """, unsafe_allow_html=True)


def render_skill_pills(skills: list, variant: str = "matched"):
    styles = {
        "matched": ("rgba(6,95,70,0.55)",  "#68d391", "rgba(104,211,145,.35)", "rgba(104,211,145,.15)"),
        "missing": ("rgba(120,10,10,0.45)", "#fc8181", "rgba(252,129,129,.35)", "rgba(252,129,129,.12)"),
        "extra":   ("rgba(20,30,80,0.55)",  "#63b3ed", "rgba(99,179,237,.35)",  "rgba(99,179,237,.12)"),
    }
    bg, text, border, glow = styles.get(variant, styles["matched"])
    if not skills:
        return ""
    pills = "".join(
        f"""<span style="
            display:inline-block;
            background:{bg};
            color:{text} !important;
            border:1px solid {border};
            border-radius:999px;
            padding:4px 14px;
            margin:3px 3px;
            font-size:.79rem;
            font-weight:600;
            letter-spacing:.02em;
            white-space:nowrap;
            transition:transform .15s ease, box-shadow .15s ease;
            box-shadow: 0 0 8px {glow};
        ">{s}</span>"""
        for s in sorted(skills)
    )
    return f"<div style='line-height:2.2; padding:4px 0;'>{pills}</div>"


def render_suggestion_card(text: str, priority: str = "tip"):
    palettes = {
        "high":   ("#fc8181", "rgba(252,129,129,.08)", "rgba(252,129,129,.25)", "⚠️"),
        "medium": ("#f6ad55", "rgba(246,173,85,.08)",  "rgba(246,173,85,.25)",  "⚡"),
        "good":   ("#68d391", "rgba(104,211,145,.08)", "rgba(104,211,145,.25)", "✅"),
        "tip":    ("#63b3ed", "rgba(99,179,237,.08)",  "rgba(99,179,237,.25)",  "💡"),
    }
    tc, bg, br, icon = palettes.get(priority, palettes["tip"])
    clean = re.sub(r"^[^\w]+\s*", "", text)
    return f"""
    <div style="
        background:{bg};
        border:1px solid {br};
        border-left:4px solid {tc};
        border-radius:0 12px 12px 0;
        padding:.8rem 1rem;
        margin-bottom:.65rem;
        display:flex;
        align-items:flex-start;
        gap:.65rem;
    ">
      <span style="font-size:1.1rem; flex-shrink:0; padding-top:1px;">{icon}</span>
      <span style="font-size:.88rem; line-height:1.55; color:#e2e8f0 !important;">
        {clean}
      </span>
    </div>
    """


def render_footer():
    st.markdown("""
    <div style="
        text-align:center;
        padding: 2.5rem 0 1.5rem;
        border-top: 1px solid rgba(99,179,237,0.1);
        margin-top: 2.5rem;
        position: relative;
    ">
      <div style="
        position:absolute;top:0;left:50%;transform:translateX(-50%);
        width:120px;height:1px;
        background:linear-gradient(90deg,transparent,rgba(99,179,237,.4),transparent);
      "></div>
      <p style="color:#4a5568 !important; font-size:.82rem; margin:0; letter-spacing:.04em;">
        Built with ❤️ using &nbsp;
        <span style="color:#63b3ed !important; font-weight:700;">Streamlit</span>
        &nbsp;·&nbsp;
        <span style="color:#b794f4 !important; font-weight:700;">scikit-learn</span>
        &nbsp;·&nbsp;
        <span style="color:#68d391 !important; font-weight:700;">ReportLab</span>
        &nbsp;·&nbsp;
        <span style="color:#f6ad55 !important; font-weight:700;">Matplotlib</span>
        &nbsp;·&nbsp;
        <span style="color:#fc8181 !important; font-weight:700;">Folium</span>
        &nbsp;·&nbsp;
        <span style="color:#b794f4 !important; font-weight:700;">Geopy</span>
      </p>
      <p style="color:#2d3748 !important; font-size:.73rem; margin:.5rem 0 0; letter-spacing:.05em;">
        Smart ATS Resume Analyzer &nbsp;·&nbsp; For demo &amp; educational use
      </p>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  PREDEFINED SKILLS LIST  (unchanged)
# ══════════════════════════════════════════════════════════════════
PREDEFINED_SKILLS = {
    "python","java","javascript","typescript","c++","c#","go","rust",
    "kotlin","swift","r","scala","ruby","php","dart","matlab",
    "html","css","react","angular","vue","next.js","nuxt","svelte",
    "bootstrap","tailwind","jquery","graphql","rest","api",
    "django","flask","fastapi","spring","node.js","express","laravel",
    "rails","asp.net",
    "machine learning","deep learning","nlp","computer vision","tensorflow",
    "pytorch","keras","scikit-learn","pandas","numpy","scipy",
    "data analysis","data science","statistics","regression","classification",
    "clustering","neural network",
    "sql","mysql","postgresql","mongodb","redis","sqlite","oracle",
    "cassandra","dynamodb","elasticsearch","firebase",
    "aws","gcp","azure","docker","kubernetes","jenkins","terraform",
    "ansible","ci/cd","git","github","gitlab","linux","bash",
    "spark","hadoop","airflow","kafka","etl","dbt","snowflake",
    "bigquery","databricks","data pipeline",
    "tableau","power bi","looker","excel","powerpoint","a/b testing",
    "agile","scrum","jira","confluence","product management",
    "communication","leadership","teamwork","problem solving",
    "cybersecurity","penetration testing","cryptography","owasp",
}


# ══════════════════════════════════════════════════════════════════
#  BACKEND LOGIC  (unchanged — bug fixes only)
# ══════════════════════════════════════════════════════════════════

def extract_text_from_pdf(uploaded_file) -> str:
    text = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            pt = page.extract_text()
            if pt:
                text.append(pt)
    return "\n".join(text)


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s\.\+\#]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_tfidf_similarity(resume_text: str, jd_text: str) -> float:
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    try:
        mat   = vectorizer.fit_transform([resume_text, jd_text])
        score = cosine_similarity(mat[0:1], mat[1:2])[0][0]
        return round(float(score) * 100, 2)
    except Exception:
        return 0.0


def extract_dynamic_jd_keywords(jd_text: str, top_n: int = 40) -> set:
    sentences = re.split(r"[.\n!?;]", jd_text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if len(sentences) < 2:
        words = re.findall(r"\b[a-z][a-z\+\#\.]{2,}\b", jd_text.lower())
        freq  = Counter(words)
        stops = {"the","and","for","with","that","this","are","have","will","you",
                 "from","your","our","they","not","but","all","can","its","was",
                 "been","more","also","any","has","who","how","what"}
        return {w for w, _ in freq.most_common(top_n) if w not in stops}
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1,2),
                              max_features=top_n, sublinear_tf=True)
        vec.fit_transform(sentences)
        return set(vec.get_feature_names_out())
    except Exception:
        return set()


def extract_skills(text: str, skill_pool: set) -> set:
    tl = text.lower()
    found = set()
    for skill in skill_pool:
        if re.search(r"\b" + re.escape(skill) + r"\b", tl):
            found.add(skill)
    return found


def compute_skill_score(matched: set, required: set) -> float:
    if not required:
        return 0.0
    return round(len(matched) / len(required) * 100, 2)


def compute_fit_score(tfidf_sim: float, skill_score: float) -> float:
    return round(0.6 * tfidf_sim + 0.4 * skill_score, 2)


def compute_keyword_frequency(text: str, skill_pool: set) -> dict:
    tl = text.lower()
    freq = {}
    for skill in skill_pool:
        c = len(re.findall(r"\b" + re.escape(skill) + r"\b", tl))
        if c > 0:
            freq[skill] = c
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))


def generate_suggestions(matched: set, missing: set,
                          tfidf_score: float, skill_score: float) -> list:
    tips = []
    if skill_score < 50:
        tips.append(("high",   "Skill coverage is below 50%. Add more relevant technical skills to your resume."))
    elif skill_score < 75:
        tips.append(("medium", "Skill coverage is moderate. Consider adding the missing skills if you have experience with them."))
    else:
        tips.append(("good",   "Excellent skill coverage! Your resume matches this role very well."))

    if tfidf_score < 30:
        tips.append(("high",   "Use more keywords from the job description in your resume to improve contextual alignment."))
    elif tfidf_score < 55:
        tips.append(("medium", "Moderate keyword alignment. Mirror the JD language more closely in your experience section."))
    else:
        tips.append(("good",   "Strong keyword alignment with the job description."))

    top_missing = sorted(missing)[:8]
    if top_missing:
        tips.append(("high", f"Consider adding these high-priority skills: {', '.join(top_missing)}."))

    if len(matched) > 0:
        tips.append(("tip", f"You matched {len(matched)} skills. Highlight them prominently in your summary and skills section."))

    tips.append(("tip", "Quantify your achievements (e.g., 'Improved model accuracy by 20%') to strengthen impact."))
    tips.append(("tip", "Tailor your professional summary to reflect the exact job title and key responsibilities in the JD."))
    return tips


# ══════════════════════════════════════════════════════════════════
#  MATPLOTLIB CHARTS  (dark-themed, bug fixes applied)
# ══════════════════════════════════════════════════════════════════

def _dark(fig, ax):
    fig.patch.set_facecolor("#0c1120")
    ax.set_facecolor("#0c1120")
    for spine in ax.spines.values():
        spine.set_edgecolor("#1e293b")
    ax.tick_params(colors="#718096", labelsize=9)
    ax.xaxis.label.set_color("#718096")
    ax.yaxis.label.set_color("#718096")
    ax.title.set_color("#e2e8f0")
    return fig, ax


def _fig_to_buf(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf


def plot_matched_vs_missing(matched_count: int, missing_count: int) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    fig, ax = _dark(fig, ax)
    cats    = ["Matched", "Missing"]
    vals    = [matched_count, missing_count]
    bcolors = ["#68d391", "#fc8181"]
    bars = ax.bar(cats, vals, color=bcolors, width=0.45,
                  edgecolor="#0c1120", linewidth=1.5, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.18,
                str(v), ha="center", va="bottom",
                color="#f0f4ff", fontsize=13, fontweight="bold")
    ax.set_title("Matched vs Missing Skills", fontsize=12,
                 fontweight="bold", pad=12, color="#e2e8f0")
    ax.set_ylabel("Count", fontsize=10)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.grid(axis="y", color="#1e293b", linestyle="--", linewidth=0.7, zorder=0)
    ax.set_ylim(0, max(vals) * 1.35 if max(vals) > 0 else 5)
    plt.tight_layout()
    return _fig_to_buf(fig)


def plot_skill_frequency(freq_dict: dict, top_n: int = 10):
    items = list(freq_dict.items())[:top_n]
    if not items:
        return None
    skills, counts = zip(*items)
    skills = list(skills)[::-1]
    counts = list(counts)[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(skills) * 0.6)))
    fig, ax = _dark(fig, ax)
    cmap    = matplotlib.colormaps["Blues"]
    palette = cmap(np.linspace(0.4, 0.9, len(skills)))
    bars = ax.barh(skills, counts, color=palette,
                   edgecolor="#0c1120", linewidth=0.6,
                   height=0.62, zorder=3)
    for bar, v in zip(bars, counts):
        ax.text(v + 0.05, bar.get_y() + bar.get_height()/2,
                str(v), va="center", ha="left",
                color="#f0f4ff", fontsize=9, fontweight="bold")
    ax.set_title(f"Top {len(skills)} Keyword Frequencies in Resume",
                 fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Occurrences", fontsize=10)
    ax.grid(axis="x", color="#1e293b", linestyle="--", linewidth=0.7, zorder=0)
    ax.set_xlim(0, max(counts) * 1.3)
    plt.tight_layout()
    return _fig_to_buf(fig)


def plot_score_breakdown(fit: float, tfidf: float, skill: float) -> io.BytesIO:
    fig, ax = plt.subplots(figsize=(7, 2.8))
    fig, ax = _dark(fig, ax)
    labels  = ["Fit Score", "TF-IDF Similarity", "Skill Match"]
    vals    = [fit, tfidf, skill]
    bcolors = ["#68d391", "#63b3ed", "#b794f4"]
    bars = ax.barh(labels, vals, color=bcolors,
                   edgecolor="#0c1120", linewidth=0.7,
                   height=0.5, zorder=3)
    for bar, v in zip(bars, vals):
        ax.text(min(v + 1.2, 97), bar.get_y() + bar.get_height()/2,
                f"{v:.1f}%", va="center", ha="left",
                color="#f0f4ff", fontsize=10, fontweight="bold")
    ax.set_xlim(0, 112)
    ax.set_title("Score Breakdown", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Score (%)", fontsize=10)
    ax.axvline(x=100, color="#334155", linestyle="--", linewidth=0.8)
    ax.grid(axis="x", color="#1e293b", linestyle="--", linewidth=0.6, zorder=0)
    plt.tight_layout()
    return _fig_to_buf(fig)


# ─── NEW CODE START ─────────────────────────────────────────────
# Feature 3 helper: section-wise scoring chart
def plot_section_scores(section_scores: dict) -> io.BytesIO:
    """Horizontal bar chart for section-wise resume scores."""
    sections = list(section_scores.keys())
    scores   = [section_scores[s] for s in sections]
    palette  = ["#63b3ed", "#b794f4", "#68d391", "#f6ad55"]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    fig, ax = _dark(fig, ax)

    bars = ax.barh(sections, scores,
                   color=palette[:len(sections)],
                   edgecolor="#0c1120", linewidth=0.6,
                   height=0.52, zorder=3)
    for bar, v in zip(bars, scores):
        ax.text(min(v + 1.2, 96), bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}%", va="center", ha="left",
                color="#f0f4ff", fontsize=10, fontweight="bold")

    ax.set_xlim(0, 112)
    ax.set_title("Section-wise Resume Score vs JD", fontsize=12,
                 fontweight="bold", pad=12)
    ax.set_xlabel("Match Score (%)", fontsize=10)
    ax.axvline(x=100, color="#334155", linestyle="--", linewidth=0.8)
    ax.grid(axis="x", color="#1e293b", linestyle="--", linewidth=0.6, zorder=0)
    plt.tight_layout()
    return _fig_to_buf(fig)
# ─── NEW CODE END ───────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════
#  PDF REPORT  (unchanged, bug-fixed TableStyle)
# ══════════════════════════════════════════════════════════════════

def generate_pdf_report(fit_score, tfidf_score, skill_score,
                         matched, missing, suggestions) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=2*cm, leftMargin=2*cm,
                            topMargin=2*cm,  bottomMargin=2*cm)
    sty = getSampleStyleSheet()

    T = lambda name, **kw: ParagraphStyle(name, parent=sty["Normal"], **kw)
    title_s   = T("TT", fontSize=22, textColor=colors.HexColor("#065f46"),
                  spaceAfter=6, fontName="Helvetica-Bold")
    heading_s = T("TH", fontSize=13, textColor=colors.HexColor("#1d4ed8"),
                  spaceBefore=14, spaceAfter=4, fontName="Helvetica-Bold")
    body_s    = T("TB", fontSize=10, leading=15,
                  textColor=colors.HexColor("#1e293b"))
    sub_s     = T("TS", fontSize=9,  textColor=colors.HexColor("#475569"))
    green_s   = T("TG", fontSize=10, textColor=colors.HexColor("#047857"),
                  fontName="Helvetica-Bold")
    red_s     = T("TR", fontSize=10, textColor=colors.HexColor("#dc2626"),
                  fontName="Helvetica-Bold")

    def label(s):
        return ("Excellent" if s>=75 else "Good" if s>=55
                else "Moderate" if s>=35 else "Weak")

    elems = []
    elems += [
        Paragraph("🎯  Smart ATS Resume Analyzer", title_s),
        Paragraph("Automated Candidate Fit Report", sub_s),
        HRFlowable(width="100%", thickness=1.5,
                   color=colors.HexColor("#065f46"), spaceAfter=10),
        Paragraph("Score Summary", heading_s),
    ]

    score_data = [
        ["Metric", "Score", "Rating"],
        ["Final Fit Score",   f"{fit_score:.1f}%",   label(fit_score)],
        ["TF-IDF Similarity", f"{tfidf_score:.1f}%", label(tfidf_score)],
        ["Skill Match Score", f"{skill_score:.1f}%", label(skill_score)],
    ]
    tbl = Table(score_data, colWidths=[8*cm, 4*cm, 4.5*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,0),  colors.HexColor("#065f46")),
        ("TEXTCOLOR",     (0,0),(-1,0),  colors.white),
        ("FONTNAME",      (0,0),(-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",      (0,0),(-1,0),  11),
        ("TOPPADDING",    (0,0),(-1,0),  8),
        ("BOTTOMPADDING", (0,0),(-1,0),  8),
        ("BACKGROUND",    (0,1),(-1,1),  colors.HexColor("#d1fae5")),
        ("BACKGROUND",    (0,2),(-1,2),  colors.HexColor("#f0f9ff")),
        ("BACKGROUND",    (0,3),(-1,3),  colors.HexColor("#fef3c7")),
        ("ALIGN",         (1,0),(-1,-1), "CENTER"),
        ("FONTNAME",      (0,1),(-1,-1), "Helvetica"),
        ("FONTSIZE",      (0,1),(-1,-1), 10),
        ("GRID",          (0,0),(-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING",    (0,1),(-1,-1), 7),
        ("BOTTOMPADDING", (0,1),(-1,-1), 7),
    ]))
    elems += [tbl, Spacer(1,12)]

    elems += [
        Paragraph("Skill Analysis", heading_s),
        Paragraph(
            f"<b>Total Required:</b> {len(matched)+len(missing)}  |  "
            f"<b>Matched:</b> {len(matched)}  |  <b>Missing:</b> {len(missing)}",
            body_s),
        Spacer(1,8),
        Paragraph("Matched Skills", green_s),
        Paragraph(", ".join(sorted(matched)) if matched
                  else "No matched skills found.", body_s),
        Spacer(1,8),
        Paragraph("Missing Skills", red_s),
        Paragraph(", ".join(sorted(missing)) if missing
                  else "No missing skills — great match!", body_s),
        Spacer(1,12),
        HRFlowable(width="100%", thickness=0.8,
                   color=colors.HexColor("#cbd5e1"), spaceAfter=6),
        Paragraph("Improvement Suggestions", heading_s),
    ]
    for i, (_, tip) in enumerate(suggestions, 1):
        clean = re.sub(r"[^\x00-\x7F]+", "", tip).strip()
        elems += [Paragraph(f"{i}.  {clean}", body_s), Spacer(1,4)]

    elems += [
        Spacer(1,20),
        HRFlowable(width="100%", thickness=0.8,
                   color=colors.HexColor("#e2e8f0"), spaceAfter=4),
        Paragraph("Generated by Smart ATS Resume Analyzer  •  For internal use only.",
                  ParagraphStyle("Ftr", parent=sub_s, alignment=1)),
    ]
    doc.build(elems)
    buf.seek(0)
    return buf.read()


# ─── NEW CODE START ─────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════
#  NEW FEATURE HELPERS
# ══════════════════════════════════════════════════════════════════

# ── Feature 1: JD text extraction from TXT file ──────────────────
def extract_text_from_txt(uploaded_file) -> str:
    """Read plain-text JD file with UTF-8 fallback."""
    try:
        return uploaded_file.read().decode("utf-8")
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return uploaded_file.read().decode("latin-1", errors="replace")


# ── Feature 2: Keyword Highlighting ──────────────────────────────
def build_highlighted_jd(jd_text: str, matched: set, missing: set) -> str:
    """
    Return HTML string of the JD with:
      - matched skills highlighted green
      - missing skills highlighted red
    Uses word-boundary regex, case-insensitive, longest match first.
    """
    # Sort by length desc so multi-word phrases are replaced before substrings
    all_skills = sorted(matched | missing, key=lambda x: -len(x))

    # Escape HTML entities first
    safe_text = (jd_text
                 .replace("&", "&amp;")
                 .replace("<", "&lt;")
                 .replace(">", "&gt;"))

    # Build a single-pass replacement using a combined regex
    if not all_skills:
        return safe_text

    pattern = re.compile(
        r"\b(" + "|".join(re.escape(s) for s in all_skills) + r")\b",
        re.IGNORECASE
    )

    def replacer(m):
        word = m.group(0)
        skill_lower = word.lower()
        # Check against original matched / missing sets (lowercase)
        matched_lower = {s.lower() for s in matched}
        missing_lower = {s.lower() for s in missing}
        if skill_lower in matched_lower:
            return (
                f'<mark style="background:rgba(104,211,145,0.25);'
                f'color:#68d391 !important;border-radius:3px;'
                f'padding:1px 4px;font-weight:600;">{word}</mark>'
            )
        elif skill_lower in missing_lower:
            return (
                f'<mark style="background:rgba(252,129,129,0.22);'
                f'color:#fc8181 !important;border-radius:3px;'
                f'padding:1px 4px;font-weight:600;">{word}</mark>'
            )
        return word

    highlighted = pattern.sub(replacer, safe_text)
    # Preserve line breaks
    highlighted = highlighted.replace("\n", "<br>")
    return highlighted


# ── Feature 3: Section-wise Scoring ──────────────────────────────
# Heuristic section markers (case-insensitive)
SECTION_PATTERNS = {
    "Skills":     r"(skill|technical|technologies|tools|competenc)",
    "Projects":   r"(project|portfolio|work sample|case study)",
    "Experience": r"(experience|employment|work history|career|professional)",
    "Education":  r"(education|academic|degree|university|college|certification)",
}

def extract_resume_sections(resume_text: str) -> dict:
    """
    Split resume text into sections using header heuristics.
    Returns dict { section_name: section_text }.
    Any text not assigned to a section goes to a catch-all.
    """
    lines = resume_text.split("\n")
    sections = {k: [] for k in SECTION_PATTERNS}
    sections["Other"] = []
    current = "Other"

    for line in lines:
        matched_section = None
        for sec, pat in SECTION_PATTERNS.items():
            if re.search(pat, line, re.IGNORECASE) and len(line.strip()) < 60:
                matched_section = sec
                break
        if matched_section:
            current = matched_section
        else:
            sections[current].append(line)

    return {k: " ".join(v) for k, v in sections.items() if v}


def compute_section_scores(resume_text: str, jd_required: set) -> dict:
    """
    For each resume section, compute how many JD-required skills appear in it.
    Returns { section: score_pct }.
    """
    if not jd_required:
        return {}
    raw_sections = extract_resume_sections(resume_text)
    scores = {}
    for sec, text in raw_sections.items():
        if sec == "Other":
            continue
        found = extract_skills(clean_text(text), jd_required)
        scores[sec] = round(len(found) / len(jd_required) * 100, 1)
    return scores


# ── Feature 4: ATS Checker ────────────────────────────────────────
def run_ats_checks(resume_text: str) -> list:
    """
    Rule-based ATS compatibility checks.
    Returns list of (status, message) where status in {'pass','warn','fail'}.
    """
    checks = []
    word_count = len(resume_text.split())

    # Length check
    if word_count < 150:
        checks.append(("fail",  f"Resume is too short ({word_count} words). ATS may rank it lower. Aim for 300–700 words."))
    elif word_count > 1200:
        checks.append(("warn",  f"Resume is quite long ({word_count} words). Consider trimming to under 1000 words for better ATS parsing."))
    else:
        checks.append(("pass",  f"Resume length is good ({word_count} words)."))

    # Quantified achievements
    quant_pattern = r"\b\d+[\s]*(%|percent|x|times|million|k\b|thousand|years|months|members|users|projects|clients)"
    quant_matches = re.findall(quant_pattern, resume_text, re.IGNORECASE)
    if len(quant_matches) < 2:
        checks.append(("warn",  "Few or no quantified achievements found. Add numbers/percentages to demonstrate impact (e.g., 'Improved performance by 30%')."))
    else:
        checks.append(("pass",  f"Found {len(quant_matches)} quantified achievement(s) — great for ATS and recruiters!"))

    # Contact info
    has_email = bool(re.search(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", resume_text))
    has_phone = bool(re.search(r"(\+?\d[\d\s\-]{7,}\d)", resume_text))
    if not has_email:
        checks.append(("fail", "No email address detected. Ensure your contact email is present."))
    else:
        checks.append(("pass", "Email address detected."))
    if not has_phone:
        checks.append(("warn", "No phone number detected. Consider including one for recruiter contact."))
    else:
        checks.append(("pass", "Phone number detected."))

    # Tables / graphics heuristic (pdfplumber strips them; if text is sparse, flag)
    lines = [l.strip() for l in resume_text.split("\n") if l.strip()]
    avg_line_len = sum(len(l) for l in lines) / max(len(lines), 1)
    if avg_line_len < 18:
        checks.append(("warn",  "Unusually short lines detected — resume may contain tables or columns that ATS parsers struggle with. Consider a single-column layout."))
    else:
        checks.append(("pass",  "No obvious table/column layout issues detected."))

    # Section headers presence
    has_exp  = bool(re.search(r"\b(experience|employment|work)\b", resume_text, re.I))
    has_edu  = bool(re.search(r"\b(education|degree|university|college)\b", resume_text, re.I))
    has_skl  = bool(re.search(r"\b(skills|technologies|tools)\b", resume_text, re.I))
    missing_sections = [s for s, f in [("Experience", has_exp), ("Education", has_edu), ("Skills", has_skl)] if not f]
    if missing_sections:
        checks.append(("warn",  f"Missing standard section header(s): {', '.join(missing_sections)}. ATS systems rely on these to categorize content."))
    else:
        checks.append(("pass",  "All standard section headers (Experience, Education, Skills) detected."))

    return checks


# ── Feature 5: Resume Improvement Generator ───────────────────────
def generate_improved_summary(resume_text: str, jd_input: str,
                               matched_skills: set, missing_skills: set,
                               fit_score: float) -> str:
    """
    Rule-based professional summary generator.
    Extracts candidate name heuristic, years of experience, top skills,
    and frames them against the JD context.
    """
    # Heuristic: first non-empty line is often the name
    lines = [l.strip() for l in resume_text.split("\n") if l.strip()]
    candidate_name = lines[0] if lines and len(lines[0].split()) <= 4 else "The candidate"

    # Extract years of experience (e.g. "3 years", "5+ years")
    yoe_match = re.search(r"(\d+)\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience)?", resume_text, re.I)
    years_exp  = yoe_match.group(1) if yoe_match else None

    # Top matched skills (up to 6)
    top_skills = sorted(matched_skills)[:6]
    skills_str = ", ".join(top_skills) if top_skills else "a broad set of technical skills"

    # Detect likely job title from JD first line
    jd_lines   = [l.strip() for l in jd_input.split("\n") if l.strip()]
    job_title  = jd_lines[0][:60] if jd_lines else "this role"

    # Compose summary
    exp_phrase = f"{years_exp}+ years of experience" if years_exp else "proven experience"
    score_note = (
        "a strong match" if fit_score >= 70
        else "a solid foundation" if fit_score >= 45
        else "relevant potential"
    )

    missing_top = sorted(missing_skills)[:3]
    growth_note = (
        f" Currently expanding expertise in {', '.join(missing_top)}."
        if missing_top else ""
    )

    summary = (
        f"Results-driven professional with {exp_phrase} in {skills_str}. "
        f"Demonstrated ability to deliver impactful solutions aligned with "
        f"the requirements of {job_title}. "
        f"Brings {score_note} for this position with a track record of "
        f"technical excellence, cross-functional collaboration, and measurable outcomes."
        f"{growth_note} "
        f"Committed to leveraging expertise to drive organisational growth and innovation."
    )
    return summary.strip()

# ─── NEW CODE END ───────────────────────────────────────────────



# ══════════════════════════════════════════════════════════════════
#  ─── NEW CODE START ─── Feature 6: Interactive Job Location Map ──
# ══════════════════════════════════════════════════════════════════

# ── Known major cities / countries for fast regex pre-scan ────────
_KNOWN_CITIES = [
    "New York","Los Angeles","San Francisco","Chicago","Boston","Seattle",
    "Austin","Dallas","Houston","Denver","Atlanta","Miami","Washington DC",
    "Washington D.C.","Portland","Phoenix","Minneapolis","Detroit","San Diego",
    "London","Manchester","Birmingham","Edinburgh","Bristol","Leeds","Glasgow",
    "Paris","Lyon","Marseille","Berlin","Munich","Hamburg","Frankfurt",
    "Amsterdam","Rotterdam","Brussels","Zurich","Geneva","Vienna","Stockholm",
    "Copenhagen","Oslo","Helsinki","Madrid","Barcelona","Lisbon","Rome","Milan",
    "Dublin","Warsaw","Prague","Budapest","Kyiv","Bucharest","Sofia",
    "Bangalore","Bengaluru","Mumbai","Delhi","New Delhi","Hyderabad","Chennai",
    "Pune","Kolkata","Ahmedabad","Jaipur","Surat","Lucknow","Noida","Gurgaon",
    "Gurugram","Kochi","Chandigarh","Bhubaneswar","Indore",
    "Singapore","Tokyo","Osaka","Seoul","Beijing","Shanghai","Shenzhen",
    "Hong Kong","Taipei","Bangkok","Jakarta","Kuala Lumpur","Manila",
    "Sydney","Melbourne","Brisbane","Perth","Auckland",
    "Toronto","Vancouver","Montreal","Calgary","Ottawa",
    "São Paulo","Mexico City","Buenos Aires","Bogotá","Lima","Santiago",
    "Dubai","Abu Dhabi","Riyadh","Doha","Cairo","Lagos","Nairobi",
    "Cape Town","Johannesburg",
]

_COUNTRY_PATTERN = (
    r"\b(India|USA|United States|UK|United Kingdom|Germany|France|Canada|"
    r"Australia|Singapore|Japan|China|Netherlands|Ireland|Poland|Brazil|"
    r"Mexico|Spain|Italy|Sweden|Denmark|Norway|Finland|Switzerland|"
    r"Belgium|Austria|Portugal|South Korea|UAE|Saudi Arabia|New Zealand|"
    r"South Africa|Nigeria|Kenya|Egypt)\b"
)



# ── Location normalisation map — variants → canonical name ────────
_LOCATION_ALIASES: dict[str, str] = {
    # Delhi variants
    "delhi":                    "New Delhi, India",
    "new delhi":                "New Delhi, India",
    "new delhi, india":         "New Delhi, India",
    # NCR satellites
    "noida":                    "Noida, India",
    "gurgaon":                  "Gurugram, India",
    "gurugram":                 "Gurugram, India",
    # Bangalore
    "bangalore":                "Bengaluru, India",
    "bengaluru":                "Bengaluru, India",
    # Mumbai
    "mumbai":                   "Mumbai, India",
    "bombay":                   "Mumbai, India",
    # US cities
    "washington dc":            "Washington, D.C., USA",
    "washington d.c.":          "Washington, D.C., USA",
    "new york":                 "New York, USA",
    "new york city":            "New York, USA",
    "nyc":                      "New York, USA",
    "san francisco":            "San Francisco, USA",
    "sf":                       "San Francisco, USA",
    "los angeles":              "Los Angeles, USA",
    "la":                       "Los Angeles, USA",
    # UK
    "london":                   "London, UK",
    # China
    "beijing":                  "Beijing, China",
    "shanghai":                 "Shanghai, China",
    "shenzhen":                 "Shenzhen, China",
    # General country aliases
    "usa":                      "United States",
    "united states":            "United States",
    "us":                       "United States",
    "uk":                       "United Kingdom",
    "united kingdom":           "United Kingdom",
    "uae":                      "United Arab Emirates",
}


def _normalise_location(raw: str) -> str:
    """Return the canonical form of a location string."""
    key = raw.strip().lower()
    return _LOCATION_ALIASES.get(key, raw.strip())


def extract_locations(jd_text: str) -> list[str]:
    """
    Extract all location mentions from job description text.
    Returns a deduplicated, normalised list preserving order of first appearance.
    Strategy:
      1. Scan for known city names (case-insensitive, word-boundary).
      2. Scan for known country names.
      3. Scan for "City, State/Country" style patterns.
      4. Normalise → deduplicate while preserving insertion order.
    """
    found: list[str] = []
    seen: set[str] = set()

    def _add(raw_loc: str):
        normalised = _normalise_location(raw_loc)
        key = normalised.lower()
        if key and key not in seen:
            seen.add(key)
            found.append(normalised)

    # 1. Known cities (longest first to avoid partial matches)
    for city in sorted(_KNOWN_CITIES, key=len, reverse=True):
        if re.search(r"\b" + re.escape(city) + r"\b", jd_text, re.IGNORECASE):
            _add(city)

    # 2. Known countries
    for match in re.finditer(_COUNTRY_PATTERN, jd_text, re.IGNORECASE):
        _add(match.group(0).title())

    # 3. "City, State" / "City, Country" patterns  e.g. "Austin, TX" "Pune, India"
    inline_pattern = re.compile(
        r"\b([A-Z][a-zA-Z .]{2,25}),\s*([A-Z][a-zA-Z .]{1,25})\b"
    )
    for m in inline_pattern.finditer(jd_text):
        combo = f"{m.group(1).strip()}, {m.group(2).strip()}"
        _add(combo)

    # 4. Remote / hybrid hints → add "Remote" as a virtual location
    if re.search(r"\b(remote|work from home|wfh|fully remote|hybrid)\b",
                 jd_text, re.IGNORECASE):
        _add("Remote")

    return found


@st.cache_data(show_spinner=False)
def geocode_locations(locations: tuple) -> pd.DataFrame:
    """
    Convert a tuple of location strings to a DataFrame with lat/lon/frequency.
    Uses Nominatim (no API key). Cached so repeated calls for same JD are free.
    Normalises and deduplicates before geocoding.
    """
    # Normalise all locations first, then count frequencies
    normalised = [_normalise_location(loc) for loc in locations
                  if loc.lower() != "remote"]
    freq_counter: Counter = Counter(normalised)
    unique_locs = list(freq_counter.keys())

    geolocator = Nominatim(user_agent="smart_ats_analyzer_v3", timeout=10)
    rows = []
    geocoded_keys: set[str] = set()

    for loc in unique_locs:
        key = loc.lower()
        if key in geocoded_keys:
            continue
        try:
            geo = geolocator.geocode(loc, language="en", exactly_one=True)
            time.sleep(0.15)   # Nominatim rate limit: max 1 req/s
            if geo and geo.latitude is not None:
                rows.append({
                    "location_name": loc,
                    "lat":           float(geo.latitude),
                    "lon":           float(geo.longitude),
                    "frequency":     freq_counter[loc],
                })
                geocoded_keys.add(key)
        except (GeocoderTimedOut, GeocoderUnavailable):
            continue
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(columns=["location_name", "lat", "lon", "frequency"])

    df = pd.DataFrame(rows).drop_duplicates(subset=["location_name"])
    return df


def render_job_map(df: pd.DataFrame, top_n: int = 50) -> None:
    """
    Render an interactive Folium map (OpenStreetMap, no API key) with:
      • HeatMap layer  — location density glow
      • CircleMarker   — per-location dot with popup + tooltip
    Auto-zooms to fit all visible markers.
    """
    if df.empty:
        st.warning(
            "⚠️ No geocodable locations were found in this job description. "
            "Try a JD that mentions specific cities or countries."
        )
        return

    # Apply top-N filter
    df_plot = df.nlargest(top_n, "frequency").copy() if len(df) > top_n else df.copy()

    # ── Map centre ────────────────────────────────────────────────
    centre_lat = float(df_plot["lat"].mean())
    centre_lon = float(df_plot["lon"].mean())

    # ── Create Folium map (OpenStreetMap — no API key) ────────────
    m = folium.Map(
        location=[centre_lat, centre_lon],
        zoom_start=3,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # ── HeatMap layer for density ─────────────────────────────────
    max_freq = max(df_plot["frequency"].max(), 1)
    heat_data = [
        [row["lat"], row["lon"], row["frequency"] / max_freq]
        for _, row in df_plot.iterrows()
    ]
    HeatMap(
        heat_data,
        radius=30,
        blur=25,
        min_opacity=0.25,
        gradient={
            "0.2": "#1a3a6b",
            "0.4": "#2563eb",
            "0.6": "#60a5fa",
            "0.8": "#fbbf24",
            "1.0": "#ef4444",
        },
    ).add_to(m)

    # ── CircleMarkers with popup + tooltip ────────────────────────
    for _, row in df_plot.iterrows():
        freq      = int(row["frequency"])
        norm_r    = freq / max_freq            # 0–1
        radius    = int(7 + norm_r * 14)       # 7–21 px
        opacity   = round(0.60 + norm_r * 0.35, 2)

        popup_html = f"""
        <div style="
            font-family: 'Segoe UI', Arial, sans-serif;
            min-width: 160px;
            background: #0c1120;
            color: #e2e8f0;
            border-radius: 8px;
            padding: 10px 14px;
        ">
          <b style="color:#63b3ed; font-size:14px;">📍 {row['location_name']}</b>
          <hr style="border:none;border-top:1px solid #2d3748;margin:6px 0;">
          <span style="color:#b794f4; font-size:13px;">
            Mentions: <b>{freq}</b>
          </span><br>
          <span style="color:#718096; font-size:11px;">
            {row['lat']:.3f}°, {row['lon']:.3f}°
          </span>
        </div>
        """

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=radius,
            color="#63b3ed",
            fill=True,
            fill_color="#3b82f6",
            fill_opacity=opacity,
            weight=2,
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=folium.Tooltip(
                f"<b>{row['location_name']}</b> — {freq} mention{'s' if freq!=1 else ''}",
                style=(
                    "background-color:#0c1120;"
                    "color:#e2e8f0;"
                    "border:1px solid #63b3ed;"
                    "border-radius:6px;"
                    "font-size:12px;"
                    "padding:4px 8px;"
                ),
            ),
        ).add_to(m)

    # ── Auto-zoom to fit all markers ──────────────────────────────
    if len(df_plot) > 1:
        sw = [df_plot["lat"].min(), df_plot["lon"].min()]
        ne = [df_plot["lat"].max(), df_plot["lon"].max()]
        m.fit_bounds([sw, ne], padding=(30, 30))

    # ── Render inside Streamlit via HTML component ────────────────
    map_html = m._repr_html_()
    # Wrap in a styled container matching the dark theme
    styled_html = f"""
    <div style="
        border-radius: 16px;
        overflow: hidden;
        border: 1px solid rgba(99,179,237,0.18);
        box-shadow: 0 8px 32px rgba(0,0,0,0.5), 0 0 40px rgba(99,179,237,0.06);
    ">
        {map_html}
    </div>
    """
    components.html(styled_html, height=480, scrolling=False)


def render_location_stats(df: pd.DataFrame, raw_locations: list,
                           remote_found: bool) -> None:
    """Render location frequency cards and top-location callout."""
    if df.empty:
        return

    total_unique = len(df)
    top_row      = df.loc[df["frequency"].idxmax()]
    top_name     = top_row["location_name"]
    top_count    = int(top_row["frequency"])

    # ── Stat row ──────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3, gap="small")
    with c1:
        render_metric_card("📍", "Locations Detected",
                           str(len(raw_locations)),
                           "total location mentions", "#63b3ed")
    with c2:
        render_metric_card("🌏", "Unique Locations",
                           str(total_unique),
                           "successfully geocoded", "#b794f4")
    with c3:
        label = "🏠 Remote Available" if remote_found else "🏢 On-site Only"
        render_metric_card("📡", "Work Mode",
                           "Yes" if remote_found else "No",
                           label, "#68d391" if remote_found else "#f6ad55")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Top location callout ───────────────────────────────────────
    st.markdown(
        f"""
        <div style="
            background:linear-gradient(135deg,rgba(99,179,237,0.08),rgba(183,148,244,0.06));
            border:1px solid rgba(99,179,237,0.2);
            border-radius:14px; padding:1rem 1.4rem;
            display:flex; align-items:center; gap:1rem;
            margin-bottom:1rem;
        ">
          <span style="font-size:2rem;">🏆</span>
          <div>
            <div style="font-size:.75rem;color:#718096 !important;
                        font-weight:700;letter-spacing:.1em;
                        text-transform:uppercase;margin-bottom:3px;">
              Most Frequent Job Location
            </div>
            <div style="font-size:1.25rem !important; font-weight:800 !important;
                        color:#63b3ed !important;
                        text-shadow:0 0 18px rgba(99,179,237,0.45);">
              {top_name}
            </div>
            <div style="font-size:.8rem;color:#a0aec0 !important;margin-top:2px;">
              Mentioned&nbsp;<b style="color:#b794f4 !important;">{top_count}&nbsp;time{"s" if top_count!=1 else ""}</b>
              &nbsp;in the job description
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Location frequency breakdown ──────────────────────────────
    st.markdown(
        "<p style='font-size:.82rem;font-weight:700;color:#a0aec0 !important;"
        "letter-spacing:.08em;text-transform:uppercase;margin-bottom:.6rem;'>"
        "Location Frequency Breakdown</p>",
        unsafe_allow_html=True,
    )
    df_sorted = df.sort_values("frequency", ascending=False).head(10)
    max_f     = df_sorted["frequency"].max()
    for _, row in df_sorted.iterrows():
        bar_pct = int(row["frequency"] / max_f * 100)
        st.markdown(
            f"""
            <div class="loc-freq-bar">
              <span style="min-width:140px;font-size:.83rem;font-weight:600;
                           color:#e2e8f0 !important;">
                📍 {row['location_name']}
              </span>
              <div style="flex:1;background:rgba(255,255,255,0.05);
                          border-radius:999px;height:8px;overflow:hidden;">
                <div style="width:{bar_pct}%;height:100%;
                  background:linear-gradient(90deg,#63b3ed,#b794f4);
                  border-radius:999px;
                  box-shadow:0 0 10px rgba(99,179,237,0.4);"></div>
              </div>
              <span style="min-width:28px;text-align:right;
                           font-size:.82rem;font-weight:700;
                           color:#b794f4 !important;
                           font-family:'JetBrains Mono',monospace !important;">
                ×{int(row['frequency'])}
              </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── All detected location badges (use geocoded, deduplicated names) ──
    badge_locs = list(df.sort_values("frequency", ascending=False)["location_name"])
    if badge_locs:
        badges = "".join(
            f"<span class='loc-badge'>📍 {loc}</span>"
            for loc in badge_locs
        )
        remote_badge = (
            "<span style='display:inline-flex;align-items:center;gap:.4rem;"
            "background:rgba(104,211,145,0.1);border:1px solid rgba(104,211,145,0.3);"
            "border-radius:999px;padding:4px 14px;font-size:.79rem;font-weight:600;"
            "color:#68d391 !important;margin:3px;'>🏠 Remote Available</span>"
            if remote_found else ""
        )
        st.markdown(
            f"<div style='margin-top:.8rem;line-height:2.2;'>{badges}{remote_badge}</div>",
            unsafe_allow_html=True,
        )

# ─── NEW CODE END ─── Feature 6 ───────────────────────────────────


# ══════════════════════════════════════════════════════════════════
#  MAIN APP
# ══════════════════════════════════════════════════════════════════

def main():

    # ── 0. PARTICLES BACKGROUND (full viewport) ───────────────────
    advanced_particles_background()

    # ── 1. HERO SECTION ──────────────────────────────────────────
    render_hero()

    # ── 2. INPUT PANEL ────────────────────────────────────────────
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown(
            "<p style='font-weight:700;font-size:.95rem;"
            "color:#63b3ed !important;margin-bottom:.5rem;'>"
            "📄 Upload Resume (PDF)</p>",
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "resume_upload", type=["pdf"], label_visibility="collapsed"
        )

    with col_r:
        st.markdown(
            "<p style='font-weight:700;font-size:.95rem;"
            "color:#63b3ed !important;margin-bottom:.5rem;'>"
            "📋 Job Description</p>",
            unsafe_allow_html=True,
        )

        # ─── NEW CODE START ─── Feature 1: JD File Upload ───────
        jd_file = st.file_uploader(
            "Upload JD (PDF or TXT) — or paste below",
            type=["pdf", "txt"],
            label_visibility="visible",
            key="jd_file_uploader",
        )
        jd_input = ""
        if jd_file is not None:
            if jd_file.type == "application/pdf":
                jd_input = extract_text_from_pdf(jd_file)
            else:
                jd_input = extract_text_from_txt(jd_file)
            if jd_input.strip():
                st.success("✅ JD extracted from uploaded file.")
            else:
                st.warning("Could not extract text from uploaded file. Please paste the JD below.")

        if not jd_input.strip():
            jd_input = st.text_area(
                "Paste Job Description",
                height=200,
                placeholder="Paste the complete job description here…",
                label_visibility="collapsed",
            )
        # ─── NEW CODE END ───────────────────────────────────────

    st.markdown("<br>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([2, 1, 2])
    with btn_col:
        analyze_btn = st.button("🔍  Analyze Resume", use_container_width=True)

    st.divider()

    # ── 3. ANALYSIS ───────────────────────────────────────────────
    if analyze_btn:
        if not uploaded_file:
            st.error("⚠️  Please upload a PDF resume before analyzing.")
            return
        if not jd_input.strip():
            st.error("⚠️  Please enter or upload a job description before analyzing.")
            return

        with st.spinner("⚙️  Extracting text and computing scores…"):
            resume_text = extract_text_from_pdf(uploaded_file)
            if not resume_text.strip():
                st.error("❌  Could not extract text. Ensure this is a text-based (not scanned) PDF.")
                return

            resume_clean = clean_text(resume_text)
            jd_clean     = clean_text(jd_input)

            tfidf_score         = compute_tfidf_similarity(resume_clean, jd_clean)
            dynamic_kw          = extract_dynamic_jd_keywords(jd_clean)
            skill_pool          = PREDEFINED_SKILLS | dynamic_kw
            jd_required         = extract_skills(jd_clean, skill_pool)
            resume_matched      = extract_skills(resume_clean, jd_required)
            missing_skills      = jd_required - resume_matched
            skill_score         = compute_skill_score(resume_matched, jd_required)
            fit_score           = compute_fit_score(tfidf_score, skill_score)
            freq_dict           = compute_keyword_frequency(resume_clean, skill_pool)
            suggestions         = generate_suggestions(
                                      resume_matched, missing_skills,
                                      tfidf_score, skill_score)
            pdf_bytes           = generate_pdf_report(
                                      fit_score, tfidf_score, skill_score,
                                      resume_matched, missing_skills, suggestions)

            # ─── NEW CODE START ─── pre-compute new feature data ───
            section_scores  = compute_section_scores(resume_text, jd_required)
            ats_checks      = run_ats_checks(resume_text)
            # ─── NEW CODE END ────────────────────────────────────

        # ── BIG SCORE DISPLAY ─────────────────────────────────────
        render_big_score(fit_score)

        # ── DASHBOARD METRIC CARDS ────────────────────────────────
        m1, m2, m3, m4 = st.columns(4, gap="small")
        with m1:
            render_metric_card("📐", "TF-IDF Similarity",
                               f"{tfidf_score:.1f}%",
                               "Contextual keyword match", "#63b3ed")
        with m2:
            render_metric_card("🧩", "Skill Match Score",
                               f"{skill_score:.1f}%",
                               "Based on JD + predefined skills", "#b794f4")
        with m3:
            render_metric_card("✅", "Matched Skills",
                               str(len(resume_matched)),
                               f"out of {len(jd_required)} required", "#68d391")
        with m4:
            render_metric_card("❌", "Missing Skills",
                               str(len(missing_skills)),
                               "Skills to add to resume", "#fc8181")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── GRADIENT PROGRESS BARS ────────────────────────────────
        pb_l, pb_r = st.columns([1, 1], gap="large")
        with pb_l:
            render_gradient_progress("🎯 Final Fit Score",
                                     fit_score, "#68d391", "#63b3ed")
            render_gradient_progress("📐 TF-IDF Similarity",
                                     tfidf_score, "#63b3ed", "#b794f4")
        with pb_r:
            render_gradient_progress("🧩 Skill Match Score",
                                     skill_score, "#b794f4", "#f6ad55")
            pct_pool = min(len(jd_required) / max(len(skill_pool), 1) * 100, 100)
            render_gradient_progress("📋 JD Coverage in Skill Pool",
                                     pct_pool, "#f6ad55", "#fc8181")

        st.divider()

        # ── TABS ──────────────────────────────────────────────────
        # ─── NEW CODE START ─── extended tab list ────────────────
        tab_ov, tab_sk, tab_freq, tab_sug, tab_kw, tab_ats, tab_improve, tab_map, tab_exp = st.tabs([
            "📊 Overview",
            "🧠 Skill Analysis",
            "📈 Keyword Frequency",
            "💡 Suggestions",
            "🔍 Keyword Highlight",
            "🛡️ ATS Checker",
            "✨ Improve Resume",
            "🌍 Map Insights",
            "📄 Export Report",
        ])
        # ─── NEW CODE END ────────────────────────────────────────

        # ── TAB: OVERVIEW ─────────────────────────────────────────
        with tab_ov:
            c1, c2 = st.columns([1, 1], gap="large")
            with c1:
                st.markdown(
                    "<p style='font-weight:700;color:#a0aec0 !important;"
                    "font-size:.88rem;margin-bottom:.5rem;'>SKILL COVERAGE</p>",
                    unsafe_allow_html=True)
                st.image(plot_matched_vs_missing(
                             len(resume_matched), len(missing_skills)),
                         use_container_width=True)
            with c2:
                st.markdown(
                    "<p style='font-weight:700;color:#a0aec0 !important;"
                    "font-size:.88rem;margin-bottom:.5rem;'>SCORE BREAKDOWN</p>",
                    unsafe_allow_html=True)
                st.image(plot_score_breakdown(fit_score, tfidf_score, skill_score),
                         use_container_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.info(
                f"**Dynamic JD Analysis:** Extracted **{len(dynamic_kw)}** terms "
                f"from the JD + **{len(PREDEFINED_SKILLS)}** predefined skills "
                f"→ **{len(skill_pool)}** total in skill pool. "
                f"**{len(jd_required)}** matched the job description."
            )

            # ─── NEW CODE START ─── Feature 3: Section-wise Scoring ───
            if section_scores:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    "<p style='font-weight:700;color:#a0aec0 !important;"
                    "font-size:.88rem;margin-bottom:.5rem;'>SECTION-WISE RESUME SCORE</p>",
                    unsafe_allow_html=True)
                sec_l, sec_r = st.columns([3, 2], gap="large")
                with sec_l:
                    st.image(plot_section_scores(section_scores),
                             use_container_width=True)
                with sec_r:
                    for sec, score in section_scores.items():
                        color = "#68d391" if score >= 50 else "#f6ad55" if score >= 25 else "#fc8181"
                        render_gradient_progress(f"📌 {sec}", score, color, "#63b3ed")
            # ─── NEW CODE END ─────────────────────────────────────

        # ── TAB: SKILL ANALYSIS ───────────────────────────────────
        with tab_sk:
            sk_l, sk_r = st.columns([1, 1], gap="large")
            with sk_l:
                pills_html = render_skill_pills(list(resume_matched), "matched")
                render_glass_section(
                    f"Matched Skills  ({len(resume_matched)})", "✅",
                    pills_html or "<p style='color:#718096'>None found.</p>",
                    accent="#68d391"
                )
            with sk_r:
                pills_html = render_skill_pills(list(missing_skills), "missing")
                render_glass_section(
                    f"Missing Skills  ({len(missing_skills)})", "❌",
                    pills_html or "<p style='color:#68d391'>🎉 No missing skills — full coverage!</p>",
                    accent="#fc8181"
                )

            with st.expander("🔍 View all JD-detected required skills"):
                all_req = sorted(jd_required)
                st.markdown(
                    render_skill_pills(all_req, "extra") if all_req
                    else "None detected.",
                    unsafe_allow_html=True
                )

        # ── TAB: KEYWORD FREQUENCY ────────────────────────────────
        with tab_freq:
            if freq_dict:
                freq_buf = plot_skill_frequency(freq_dict, top_n=10)
                if freq_buf:
                    _, fc, _ = st.columns([0.3, 3, 0.3])
                    with fc:
                        st.image(freq_buf, use_container_width=True)

                st.markdown("<br>", unsafe_allow_html=True)
                table_data = [{"Skill": sk, "Frequency": cnt}
                              for sk, cnt in list(freq_dict.items())[:20]]
                st.dataframe(table_data, use_container_width=True, hide_index=True)
            else:
                st.warning("No matching skill keywords found in the resume.")

        # ── TAB: SUGGESTIONS ──────────────────────────────────────
        with tab_sug:
            st.markdown(
                "<p style='color:#718096 !important; font-size:.88rem;"
                "margin-bottom:1rem;'>"
                "Personalised recommendations to improve your resume's ATS score:</p>",
                unsafe_allow_html=True,
            )
            cards_html = ""
            for priority, text in suggestions:
                cards_html += render_suggestion_card(text, priority)
            st.markdown(cards_html, unsafe_allow_html=True)

        # ─── NEW CODE START ─── Feature 2: Keyword Highlight Tab ─
        with tab_kw:
            st.markdown(
                "<p style='color:#a0aec0 !important;font-size:.88rem;"
                "margin-bottom:.8rem;'>"
                "Job description text with skill keywords highlighted. "
                "<span style='color:#68d391 !important;font-weight:700;'>Green</span>"
                " = matched in your resume &nbsp;|&nbsp; "
                "<span style='color:#fc8181 !important;font-weight:700;'>Red</span>"
                " = missing from your resume.</p>",
                unsafe_allow_html=True,
            )

            legend_html = (
                '<div style="display:flex;gap:1.5rem;margin-bottom:1rem;flex-wrap:wrap;">'
                '<span style="display:flex;align-items:center;gap:.4rem;font-size:.82rem;">'
                '<mark style="background:rgba(104,211,145,0.25);color:#68d391 !important;'
                'border-radius:3px;padding:1px 8px;font-weight:600;">sample</mark>'
                '<span style="color:#718096 !important;">Matched skill</span></span>'
                '<span style="display:flex;align-items:center;gap:.4rem;font-size:.82rem;">'
                '<mark style="background:rgba(252,129,129,0.22);color:#fc8181 !important;'
                'border-radius:3px;padding:1px 8px;font-weight:600;">sample</mark>'
                '<span style="color:#718096 !important;">Missing skill</span></span>'
                '</div>'
            )
            st.markdown(legend_html, unsafe_allow_html=True)

            highlighted_html = build_highlighted_jd(jd_input, resume_matched, missing_skills)
            st.markdown(
                f'<div class="kw-highlight-box">{highlighted_html}</div>',
                unsafe_allow_html=True,
            )
        # ─── NEW CODE END ─────────────────────────────────────────

        # ─── NEW CODE START ─── Feature 4: ATS Checker Tab ───────
        with tab_ats:
            st.markdown(
                "<p style='color:#a0aec0 !important;font-size:.88rem;"
                "margin-bottom:1rem;'>"
                "Rule-based ATS compatibility check for your resume:</p>",
                unsafe_allow_html=True,
            )

            pass_count = sum(1 for s, _ in ats_checks if s == "pass")
            warn_count = sum(1 for s, _ in ats_checks if s == "warn")
            fail_count = sum(1 for s, _ in ats_checks if s == "fail")

            ats_l, ats_r, ats_rr = st.columns([2, 1, 1], gap="small")
            with ats_l:
                total = len(ats_checks)
                ats_pct = round(pass_count / total * 100) if total else 0
                render_gradient_progress(
                    f"🛡️ ATS Compatibility Score  ({pass_count}/{total} checks passed)",
                    ats_pct, "#68d391", "#63b3ed"
                )
            with ats_r:
                render_metric_card("✅", "Passed", str(pass_count), "checks", "#68d391")
            with ats_rr:
                issues = warn_count + fail_count
                render_metric_card("⚠️", "Issues", str(issues), "to review", "#f6ad55")

            st.markdown("<br>", unsafe_allow_html=True)
            for status, msg in ats_checks:
                if status == "pass":
                    st.success(f"✅  {msg}")
                elif status == "warn":
                    st.warning(f"⚡  {msg}")
                else:
                    st.error(f"❌  {msg}")
        # ─── NEW CODE END ─────────────────────────────────────────

        # ─── NEW CODE START ─── Feature 5: Improve Resume Tab ────
        with tab_improve:
            st.markdown(
                "<p style='color:#a0aec0 !important;font-size:.88rem;"
                "margin-bottom:1rem;'>"
                "Generate a tailored professional summary and improvement tips "
                "based on your resume and the target job description.</p>",
                unsafe_allow_html=True,
            )

            if st.button("✨  Generate Improved Summary", key="gen_summary_btn",
                         use_container_width=False):
                improved = generate_improved_summary(
                    resume_text, jd_input,
                    resume_matched, missing_skills, fit_score
                )
                render_glass_section(
                    "Suggested Professional Summary", "✨",
                    f"<p style='line-height:1.8;font-size:.92rem;"
                    f"color:#e2e8f0 !important;margin:0;'>{improved}</p>",
                    accent="#b794f4"
                )
                st.markdown("<br>", unsafe_allow_html=True)

            # Static rule-based improvement tips
            tips_html = ""

            # Tip: add missing skills
            if missing_skills:
                top_m = sorted(missing_skills)[:5]
                tips_html += render_suggestion_card(
                    f"Add these missing skills to your Skills section if applicable: "
                    f"{', '.join(top_m)}.", "high"
                )

            # Tip: quantification
            quant = re.findall(r"\b\d+[\s]*(%|percent|x\b|times|million|k\b)", resume_text, re.I)
            if len(quant) < 3:
                tips_html += render_suggestion_card(
                    "Strengthen impact by quantifying achievements. "
                    "For example: 'Reduced load time by 40%' or 'Led a team of 8 engineers'.",
                    "medium"
                )
            else:
                tips_html += render_suggestion_card(
                    f"Good use of quantified achievements ({len(quant)} found). "
                    "Continue adding metrics wherever possible.", "good"
                )

            # Tip: action verbs
            action_verbs = ["led","built","designed","developed","improved",
                            "optimised","delivered","architected","launched",
                            "reduced","increased","managed","created","automated"]
            found_verbs  = [v for v in action_verbs if re.search(r"\b" + v + r"\b", resume_text, re.I)]
            if len(found_verbs) < 4:
                tips_html += render_suggestion_card(
                    "Use strong action verbs to open bullet points: Led, Built, Designed, "
                    "Optimised, Delivered, Launched, Reduced, Increased.", "medium"
                )
            else:
                tips_html += render_suggestion_card(
                    f"Strong action verbs detected ({', '.join(found_verbs[:5])}…). "
                    "Well done!", "good"
                )

            # Tip: tailor summary
            tips_html += render_suggestion_card(
                "Tailor your Professional Summary to mirror the exact job title and "
                "top 2–3 responsibilities from the job description.", "tip"
            )

            # Tip: keywords in context
            if tfidf_score < 50:
                tips_html += render_suggestion_card(
                    "Your TF-IDF alignment is below 50%. Use JD keywords naturally in "
                    "your experience bullet points, not just in the Skills section.", "medium"
                )

            st.markdown(tips_html, unsafe_allow_html=True)
        # ─── NEW CODE END ─────────────────────────────────────────

        # ─── NEW CODE START ─── Feature 6: Map Insights Tab ─────
        with tab_map:
            # ── Header ──────────────────────────────────────────
            st.markdown(
                """
                <div class="map-glass-card">
                  <p class="map-title">🌍 Job Location Insights</p>
                  <p class="map-subtitle">
                    Geographic distribution of job opportunities detected in this job description
                  </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # ── Extract locations ────────────────────────────────
            with st.spinner("🔍 Extracting locations from job description…"):
                raw_locs = extract_locations(jd_input)

            remote_found = any(
                loc.lower() == "remote" for loc in raw_locs
            )
            map_locs = [loc for loc in raw_locs if loc.lower() != "remote"]

            if not map_locs:
                st.info(
                    "📭 No mappable locations were detected in this job description. "
                    "Try pasting a JD that mentions specific cities or countries."
                )
                if remote_found:
                    st.success("🏠 The job description mentions **remote** work.")
            else:
                # ── Geocode ───────────────────────────────────────
                with st.spinner(
                    f"🌐 Geocoding {len(set(map_locs))} unique location(s)… "
                    "(cached after first run)"
                ):
                    geo_df = geocode_locations(tuple(map_locs))

                # ── Stats panel ───────────────────────────────────
                render_location_stats(geo_df, map_locs, remote_found)

                st.markdown("<br>", unsafe_allow_html=True)

                # ── Top-N filter ──────────────────────────────────
                map_col, ctrl_col = st.columns([4, 1], gap="large")
                with ctrl_col:
                    top_n = st.slider(
                        "Show Top Locations",
                        min_value=1,
                        max_value=max(len(geo_df), 1),
                        value=min(len(geo_df), 50),
                        step=1,
                        key="map_top_n",
                        help="Filter the map to show only the top N most-mentioned locations",
                    )
                    st.markdown(
                        "<p style='font-size:.75rem;color:#718096 !important;"
                        "margin-top:.3rem;'>Use the slider to focus on the "
                        "most prominent job markets.</p>",
                        unsafe_allow_html=True,
                    )

                with map_col:
                    st.markdown(
                        "<p style='font-weight:700;color:#a0aec0 !important;"
                        "font-size:.82rem;letter-spacing:.08em;text-transform:uppercase;"
                        "margin-bottom:.5rem;'>🗺️ Interactive Map</p>",
                        unsafe_allow_html=True,
                    )
                    render_job_map(geo_df, top_n=top_n)
                    st.caption(
                        "🔵 Blue dots = job locations  ·  "
                        "🔥 Heatmap = location density  ·  "
                        "Hover over a dot for details"
                    )

                # ── Raw table expander ────────────────────────────
                with st.expander("📋 View raw geocoded location data"):
                    display_df = geo_df[
                        ["location_name","lat","lon","frequency"]
                    ].sort_values("frequency", ascending=False).reset_index(drop=True)
                    display_df.columns = ["Location", "Latitude", "Longitude", "Mentions"]
                    st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ─── NEW CODE END ─────────────────────────────────────────

        # ── TAB: EXPORT REPORT ────────────────────────────────────
        with tab_exp:
            _export_desc = (
                "<p style='color:#a0aec0 !important;font-size:.88rem;"
                "line-height:1.6;margin:0;'>"
                "Your report includes &nbsp;&middot;&nbsp; Final Fit Score "
                "&nbsp;&middot;&nbsp; Score Breakdown &nbsp;&middot;&nbsp; "
                "Matched &amp; Missing Skills &nbsp;&middot;&nbsp; "
                "Improvement Suggestions.<br>"
                "Clean, recruiter-ready formatting via ReportLab.</p>"
            )
            render_glass_section("Download PDF Report", "📄", _export_desc, accent="#68d391")
            st.download_button(
                label="⬇️  Download Analysis Report (PDF)",
                data=pdf_bytes,
                file_name="ATS_Resume_Analysis_Report.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="pdf_download_btn",
            )

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("📋 Report contents preview"):
                preview_data = {
                    "Section": [
                        "Fit Score", "TF-IDF Similarity", "Skill Match",
                        "Matched Skills", "Missing Skills", "Suggestions"
                    ],
                    "Content": [
                        f"{fit_score:.1f}%",
                        f"{tfidf_score:.1f}%",
                        f"{skill_score:.1f}%",
                        f"{len(resume_matched)} skills listed",
                        f"{len(missing_skills)} skills listed",
                        f"{len(suggestions)} actionable tips",
                    ]
                }
                st.table(preview_data)

    # ── FOOTER ────────────────────────────────────────────────────
    render_footer()


if __name__ == "__main__":
    main()
