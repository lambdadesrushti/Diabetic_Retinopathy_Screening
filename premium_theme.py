"""
premium_theme.py  -  drop-in premium look for the
"Diabetic Retinopathy Screening System" Streamlit app.

Put this file in  DR_Project/app/  (next to app_premium.py), then add to the
TOP of app_premium.py, right after `import streamlit as st`:

    import premium_theme
    premium_theme.apply()

What it does (no model / metric code is touched):
  * dark medical-tech theme, animated retina-scan + ECG-pulse background
  * glassmorphism cards, glowing buttons, styled tabs / uploader / expanders
  * header titled "Diabetic Retinopathy Screening System"
  * removes the "Model Performance" tab
  * removes every metric box (st.metric) in all tabs
"""
import streamlit as st
import streamlit.components.v1 as components

TITLE = "Diabetic Retinopathy Screening System"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=IBM+Plex+Sans:wght@400;500&display=swap');

:root{
  --bg0:#040b16; --bg1:#07182b; --cyan:#22d3ee; --teal:#2dd4bf; --red:#fb7185;
  --ink:#e6f1fb; --muted:#8aa4bd; --glass:rgba(12,32,54,.55); --line:rgba(34,211,238,.22);
}

html, body, [class*="css"], .stApp { font-family:'IBM Plex Sans',sans-serif; color:var(--ink); }
h1,h2,h3,h4 { font-family:'Sora',sans-serif !important; letter-spacing:.2px; color:var(--ink) !important; }

/* ---------- background ---------- */
.stApp{
  background:
    radial-gradient(1200px 600px at 85% -10%, rgba(34,211,238,.16), transparent 60%),
    radial-gradient(900px 600px at -10% 110%, rgba(45,212,191,.12), transparent 60%),
    linear-gradient(160deg,var(--bg0),var(--bg1) 60%, #050e1c);
  background-attachment: fixed;
}
/* fine medical grid */
.stApp::before{
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:
    linear-gradient(rgba(34,211,238,.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(34,211,238,.06) 1px, transparent 1px);
  background-size:44px 44px;
  mask-image: radial-gradient(ellipse at center, #000 30%, transparent 85%);
  -webkit-mask-image: radial-gradient(ellipse at center, #000 30%, transparent 85%);
}
/* rotating retina scanner ring */
.stApp::after{
  content:""; position:fixed; z-index:0; pointer-events:none;
  width:760px; height:760px; right:-260px; top:50%; margin-top:-380px; border-radius:50%;
  background:
    radial-gradient(circle, transparent 38%, rgba(34,211,238,.10) 39%, transparent 40%),
    radial-gradient(circle, transparent 55%, rgba(45,212,191,.10) 56%, transparent 57%),
    radial-gradient(circle, transparent 70%, rgba(34,211,238,.08) 71%, transparent 72%),
    conic-gradient(from 0deg, rgba(34,211,238,.28), transparent 18%, transparent 100%);
  animation: sweep 9s linear infinite;
  opacity:.55;
}
@keyframes sweep { to { transform:rotate(360deg);} }

/* ECG pulse line running along the bottom */
#ecg-bg{ position:fixed; left:0; right:0; bottom:0; height:120px; z-index:0; pointer-events:none; opacity:.55; overflow:hidden;}
#ecg-bg svg{ width:200%; height:100%; animation: ecg 7s linear infinite; }
@keyframes ecg { from{transform:translateX(0);} to{transform:translateX(-50%);} }
#ecg-bg path{ fill:none; stroke:var(--cyan); stroke-width:2; filter:drop-shadow(0 0 6px var(--cyan)); }

@media (prefers-reduced-motion: reduce){
  .stApp::after, #ecg-bg svg { animation:none; }
}

/* keep content above background */
.main, [data-testid="stAppViewContainer"] > .main, .block-container{ position:relative; z-index:1; }
.block-container{ padding-top:2rem; max-width:1200px; }
header[data-testid="stHeader"]{ background:transparent; }

/* ---------- hero header ---------- */
.dr-hero{
  display:flex; align-items:center; gap:22px; padding:26px 30px; margin-bottom:22px;
  border:1px solid var(--line); border-radius:20px;
  background:linear-gradient(120deg, rgba(14,42,70,.75), rgba(8,24,44,.55));
  backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  box-shadow:0 20px 60px rgba(0,0,0,.45), inset 0 0 0 1px rgba(255,255,255,.03);
}
.dr-eye{
  flex:0 0 74px; height:74px; border-radius:50%;
  background:radial-gradient(circle at 50% 50%, #04101e 0 16%, #0b3a52 17% 30%, #0e5a6e 31% 46%, #0a2d46 47% 100%);
  border:2px solid var(--cyan); box-shadow:0 0 28px rgba(34,211,238,.55), inset 0 0 18px rgba(0,0,0,.7);
  position:relative;
}
.dr-eye::after{ content:""; position:absolute; left:50%; top:50%; width:12px; height:12px; margin:-6px 0 0 -6px;
  border-radius:50%; background:var(--cyan); box-shadow:0 0 14px var(--cyan); animation:pulse 2.4s ease-in-out infinite;}
@keyframes pulse{ 50%{ transform:scale(1.7); opacity:.55;} }
.dr-hero h1{ margin:0; font-size:2.1rem; font-weight:700; line-height:1.15;
  background:linear-gradient(90deg,#fff,#a5f3fc 55%, #5eead4); -webkit-background-clip:text; background-clip:text; color:transparent !important; }
.dr-hero p{ margin:6px 0 0; color:var(--muted); font-size:.98rem; }

/* ---------- glass cards ---------- */
[data-testid="stVerticalBlockBorderWrapper"], [data-testid="stExpander"], div[data-testid="stFileUploader"] section,
[data-testid="stAlert"], [data-testid="stDataFrame"]{
  background:var(--glass) !important; border:1px solid var(--line) !important; border-radius:16px !important;
  backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
  box-shadow:0 10px 36px rgba(0,0,0,.35);
}
div[data-testid="stFileUploader"] section{ border:1.5px dashed rgba(34,211,238,.5) !important; }
div[data-testid="stFileUploader"] section:hover{ box-shadow:0 0 0 3px rgba(34,211,238,.15), 0 10px 36px rgba(0,0,0,.35); }

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"]{ gap:6px; background:rgba(8,24,44,.55); padding:6px; border-radius:14px; border:1px solid var(--line); }
.stTabs [data-baseweb="tab"]{ border-radius:10px; padding:10px 18px; color:var(--muted); font-family:'Sora',sans-serif; }
.stTabs [aria-selected="true"]{ background:linear-gradient(90deg, rgba(34,211,238,.25), rgba(45,212,191,.2)) !important; color:#fff !important; }
.stTabs [data-baseweb="tab-highlight"]{ background:var(--cyan) !important; }
.stTabs [data-baseweb="tab-border"]{ display:none; }

/* ---------- buttons ---------- */
.stButton>button, .stDownloadButton>button{
  border:none; color:#04101e; font-weight:600; font-family:'Sora',sans-serif; border-radius:12px; padding:.6rem 1.3rem;
  background:linear-gradient(90deg,var(--cyan),var(--teal));
  box-shadow:0 8px 26px rgba(34,211,238,.35); transition:transform .15s, box-shadow .15s;
}
.stButton>button:hover, .stDownloadButton>button:hover{ transform:translateY(-1px); box-shadow:0 12px 34px rgba(34,211,238,.55); color:#04101e; }
.stButton>button:focus-visible{ outline:2px solid #fff; outline-offset:2px; }

/* images get a scanner frame */
[data-testid="stImage"] img{ border-radius:14px; border:1px solid var(--line); box-shadow:0 8px 30px rgba(0,0,0,.5); }

/* progress, sliders */
.stProgress > div > div > div{ background:linear-gradient(90deg,var(--cyan),var(--teal)); }
[data-testid="stSidebar"]{ background:rgba(5,14,28,.92); border-right:1px solid var(--line); }

/* ---------- REMOVE the 4 metric boxes everywhere ---------- */
[data-testid="stMetric"], [data-testid="metric-container"]{ display:none !important; }
/* collapse the empty column row that held them */
[data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]){ display:none !important; }

/* ---------- force ONE look in laptop light AND dark mode ---------- */
:root, html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"]{ color-scheme: dark !important; }
.stApp{ color:var(--ink); }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label{ color:var(--ink) !important; }
[data-baseweb="select"] > div, [data-baseweb="input"], [data-baseweb="base-input"], textarea, input{
  background:rgba(10,28,48,.92) !important; color:var(--ink) !important; border-color:var(--line) !important; }
[data-baseweb="select"] *{ color:var(--ink) !important; }
[data-baseweb="select"] svg{ fill:var(--ink) !important; }
[data-baseweb="popover"] ul, [data-baseweb="menu"], [data-baseweb="popover"] > div{ background:#0a1c30 !important; color:var(--ink) !important; }
[data-baseweb="popover"] li{ color:var(--ink) !important; }
[data-baseweb="popover"] li:hover{ background:rgba(34,211,238,.18) !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary, [data-testid="stSidebar"] [data-testid="stExpander"] details{ background:transparent !important; color:var(--ink) !important; }
.stTabs [data-baseweb="tab"]{ color:#a9c1d8 !important; }
.stTabs [aria-selected="true"]{ color:#fff !important; }
.stMarkdown, [data-testid="stCaptionContainer"]{ color:var(--ink); }

/* ---------- REMOVE the 5th (last) tab = Model Performance ---------- */
[data-baseweb="tab-list"] button[role="tab"]:nth-of-type(5):last-of-type{ display:none !important; }
[data-baseweb="tab-panel"]:nth-of-type(5):last-of-type{ display:none !important; }
</style>

<div id="ecg-bg">
  <svg viewBox="0 0 2000 120" preserveAspectRatio="none">
    <path d="M0 70 H180 l14 -8 l14 8 H300 l12 10 l16 -62 l18 90 l12 -38 H460 l16 -12 l16 12 H700
             M1000 70 H1180 l14 -8 l14 8 H1300 l12 10 l16 -62 l18 90 l12 -38 H1460 l16 -12 l16 12 H1700 H2000"/>
  </svg>
</div>
"""

HERO = f"""
<div class="dr-hero">
  <div class="dr-eye"></div>
  <div>
    <h1>{TITLE}</h1>
    <p>AI-assisted retinal fundus analysis: severity grading, lesion segmentation and explainable heat-maps.</p>
  </div>
</div>
"""

# Hides the "Model Performance" tab (Streamlit renders tabs as buttons[role=tab]).
JS = r"""
<script>
const doc = window.parent.document;
const TITLE = "Diabetic Retinopathy Screening System";
function isWhite(c){
  const m = c.match(/[0-9.]+/g); if(!m) return false;
  const [r,g,b] = [+m[0], +m[1], +m[2]]; const a = m.length>3 ? +m[3] : 1;
  return a > 0.5 && r > 225 && g > 225 && b > 225;
}
function clean(){
  // 1) remove Model Performance tab
  doc.querySelectorAll('button[role="tab"]').forEach(b=>{
    if ((b.innerText||'').toLowerCase().includes('model performance')) b.style.display='none';
  });
  // 2) remove metric boxes
  doc.querySelectorAll('[data-testid="stMetric"]').forEach(m=>{ m.style.display='none'; });
  // 3) white cards -> dark glass so pale text stays readable
  doc.querySelectorAll('[data-testid="stMain"] div, [data-testid="stAppViewContainer"] section div').forEach(el=>{
    if (el.dataset.drFixed) return;
    if (el.closest('[data-baseweb="select"], [data-baseweb="popover"], [data-testid="stFileUploader"], [data-testid="stImage"], button')) return;
    const r = el.getBoundingClientRect();
    if (r.width < 120 || r.height < 30) return;
    if (isWhite(getComputedStyle(el).backgroundColor)){
      el.dataset.drFixed = '1';
      el.style.setProperty('background','linear-gradient(135deg, rgba(14,42,70,.92), rgba(8,24,44,.88))','important');
      el.style.setProperty('border','1px solid rgba(34,211,238,.28)','important');
      el.style.setProperty('color','#e6f1fb','important');
      el.querySelectorAll('p,span,div,li,h1,h2,h3,h4').forEach(ch=>{
        const col = getComputedStyle(ch).color.match(/[0-9.]+/g);
        if (col && +col[0] > 200 && +col[1] > 200 && +col[2] > 200) return;      // already light
        if (col && +col[0] < 130 && +col[2] < 130 && ch.children.length) return;
      });
    }
  });
  // 4) app title
  const w = doc.createTreeWalker(doc.body, NodeFilter.SHOW_TEXT);
  let n; while ((n = w.nextNode())){
    if (n.nodeValue && n.nodeValue.includes('Diabetic Retinopathy + Metabolic Risk Assessment'))
      n.nodeValue = n.nodeValue.replace('Diabetic Retinopathy + Metabolic Risk Assessment', TITLE);
  }
}
let t = null;
function schedule(){ if (t) return; t = setTimeout(()=>{ t = null; try{ clean(); }catch(e){} }, 120); }
schedule();
new MutationObserver(schedule).observe(doc.body,{childList:true,subtree:true});
</script>
"""


def apply(show_hero: bool = False):
    """Call once, near the top of the app script."""
    try:
        st.set_page_config(page_title=TITLE, page_icon="👁️", layout="wide")
    except Exception:
        pass  # set_page_config was already called by the app - fine
    st.set_page_config = lambda *a, **k: None   # ignore the app's own later call (it would raise an error)
    st.markdown(CSS, unsafe_allow_html=True)
    if show_hero:
        st.markdown(HERO, unsafe_allow_html=True)
    components.html(JS, height=0)
