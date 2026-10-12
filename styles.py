"""Application CSS.  The lesson card uses normal document flow and no fixed height."""


APP_CSS = r"""
<style>
:root {
  --bg: #080d16;
  --card: #101824;
  --panel: #151f2e;
  --panel-2: #111b29;
  --border: #263448;
  --text: #eef4ff;
  --muted: #9eabc0;
  --blue: #5fa8ff;
  --blue-bg: rgba(58, 139, 253, .12);
  --green: #55d6a5;
  --green-bg: rgba(51, 196, 141, .11);
  --orange: #ffae62;
  --orange-bg: rgba(255, 158, 68, .11);
}

.stApp { background: radial-gradient(circle at 50% -10%, #122039 0, var(--bg) 42%); color: var(--text); }
header[data-testid="stHeader"] { background: transparent; }
.stMainBlockContainer { max-width: 980px; padding-top: 2.2rem; padding-bottom: 4rem; }

[class*="st-key-lesson_card"] {
  background: rgba(16, 24, 36, .97);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: clamp(1.15rem, 3vw, 2.2rem);
  box-shadow: 0 24px 80px rgba(0,0,0,.30);
}

h1 { color: var(--text) !important; font-size: clamp(1.75rem, 4vw, 2.55rem) !important; line-height: 1.12 !important; letter-spacing: -.035em; }
h2, h3 { color: var(--text) !important; }
p, li, label { color: #dce5f4; line-height: 1.65; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

.chapter-kicker { color: var(--blue); font-size: .74rem; font-weight: 800; letter-spacing: .13em; text-transform: uppercase; margin-bottom: .3rem; }
.chapter-name { color: var(--muted); font-size: .9rem; margin-bottom: .8rem; }
.cat-stage { display:flex; align-items:center; gap:1rem; background:var(--panel-2); border:1px solid var(--border); border-radius:18px; padding:1rem 1.15rem; margin:.55rem 0 1rem; }
.cat-emoji { font-size:2.8rem; line-height:1; }
.cat-question { color:var(--text); font-weight:750; font-size:1.08rem; }
.cat-sub { color:var(--muted); font-size:.86rem; margin-top:.15rem; }

.phase-card, .expert-card, .number-card, .flow-card, .comparison-card {
  border:1px solid var(--border); border-radius:16px; padding:1rem 1.1rem; background:var(--panel); margin:.35rem 0 .65rem;
}
.phase-card.blue, .flow-card.blue { border-left:4px solid var(--blue); background:var(--blue-bg); }
.phase-card.green, .flow-card.green { border-left:4px solid var(--green); background:var(--green-bg); }
.phase-card.orange { border-left:4px solid var(--orange); background:var(--orange-bg); }
.card-label { color:var(--muted); font-size:.72rem; letter-spacing:.09em; font-weight:800; text-transform:uppercase; }
.card-title { color:var(--text); font-weight:800; font-size:1.06rem; margin:.22rem 0 .35rem; }
.big-number { color:var(--text); font-weight:850; font-size:clamp(1.55rem,4vw,2.15rem); letter-spacing:-.03em; }
.small-copy { color:#c7d2e3; font-size:.89rem; line-height:1.55; }
.blue-text { color:var(--blue); }
.green-text { color:var(--green); }
.orange-text { color:var(--orange); }

.meter-row { display:grid; grid-template-columns:minmax(100px, 155px) 1fr 58px; align-items:center; gap:.7rem; margin:.6rem 0; }
.meter-label { color:#dce5f4; font-size:.86rem; }
.meter-track { height:10px; background:#263246; border-radius:99px; overflow:hidden; }
.meter-fill { height:100%; border-radius:99px; min-width:2px; }
.meter-fill.blue { background:var(--blue); }
.meter-fill.green { background:var(--green); }
.meter-fill.orange { background:var(--orange); }
.meter-value { color:var(--text); text-align:right; font-size:.84rem; font-variant-numeric:tabular-nums; }

.equation-box { background:#0b121d; border:1px solid #2b3b50; border-radius:14px; padding:.9rem 1rem; margin:.8rem 0 1rem; overflow-x:auto; }
.equation-label { color:var(--muted); font-size:.72rem; font-weight:800; letter-spacing:.08em; text-transform:uppercase; margin-bottom:.35rem; }
.equation { color:#f4f7ff; font-family:"SFMono-Regular",Consolas,monospace; font-size:clamp(.88rem,2vw,1.03rem); white-space:nowrap; }

.takeaway { border:1px solid rgba(85,214,165,.35); background:var(--green-bg); border-radius:15px; padding:1rem 1.1rem; margin:1.2rem 0 .7rem; }
.takeaway-title { color:var(--green); font-size:.75rem; font-weight:850; letter-spacing:.1em; text-transform:uppercase; margin-bottom:.3rem; }
.takeaway-copy { color:#e4f7ef; font-weight:650; line-height:1.55; }
.warning-box { border:1px solid rgba(255,174,98,.35); background:var(--orange-bg); border-radius:15px; padding:1rem 1.1rem; margin:.8rem 0; color:#ffe3c8; }
.source-note { color:#7f8ca1; font-size:.77rem; line-height:1.45; margin-top:.8rem; }
.flow-line { color:#cbd7e8; line-height:1.9; font-weight:600; }
.arrow { color:var(--blue); padding:0 .2rem; }
.divider { height:1px; background:var(--border); margin:1.3rem 0 1rem; }

div[data-testid="stProgress"] > div > div > div { background-color:var(--blue); }
div[data-testid="stSlider"] [role="slider"] { background-color:var(--blue); }
div[data-testid="stRadio"] { background:#0d1521; border:1px solid var(--border); border-radius:14px; padding:.3rem .85rem; }
div[data-testid="stMetric"] { background:var(--panel); border:1px solid var(--border); border-radius:14px; padding:.75rem .9rem; }
div[data-testid="stExpander"] { border-color:var(--border); border-radius:14px; background:#0d1521; }
div[data-testid="stAlert"] { border-radius:14px; }
.stButton > button { width:100%; border-radius:12px; min-height:2.75rem; font-weight:750; border-color:#34445b; }
[class*="st-key-nav_reset"] button { color:#ffd6b1; border-color:#ba6d2e; background:rgba(186,109,46,.14); }
[class*="st-key-nav_next"] button { background:#2878db; border-color:#2878db; color:white; }
[class*="st-key-special_button"] button { border-color:#55d6a5; color:#bdf5df; }

@media (max-width: 640px) {
  .stMainBlockContainer { padding-left:.7rem; padding-right:.7rem; padding-top:.7rem; }
  [class*="st-key-lesson_card"] { border-radius:18px; }
  .meter-row { grid-template-columns: 92px 1fr 48px; gap:.45rem; }
}
</style>
"""
