"""
โปรเจคการจำแนกผู้ป่วยโรคหัวใจด้วย K-Nearest Neighbors (KNN)
Heart disease classification with KNN — Streamlit app
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,precision_score, recall_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,train_test_split)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Heart KNN Classifier",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RED = "#E63946"
GREEN = "#2A9D8F"
NAVY = "#1D3557"
COLORS = {"ไม่เป็นโรคหัวใจ": GREEN, "เป็นโรคหัวใจ": RED}

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stButton button, .stTabs button {{
    font-family: 'Prompt', sans-serif;
}}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; }}
.hero {{
    background: linear-gradient(120deg, {NAVY} 0%, #457B9D 55%, {RED} 100%);
    border-radius: 22px; padding: 2.2rem 2.4rem; color: #fff;
    box-shadow: 0 12px 32px rgba(29,53,87,.25); margin-bottom: 1.4rem;
}}
.hero h1 {{ color: #fff; font-size: 2.1rem; font-weight: 700; margin: 0 0 .35rem 0; }}
.hero p  {{ color: rgba(255,255,255,.9); font-size: 1.05rem; margin: 0; }}
.hero .pill {{
    display: inline-block; background: rgba(255,255,255,.18); color: #fff;
    border: 1px solid rgba(255,255,255,.35); border-radius: 999px;
    padding: .2rem .8rem; font-size: .82rem; margin: .9rem .4rem 0 0;
}}
.kpi {{
    background: #fff; border-radius: 16px; padding: 1rem 1.2rem;
    border-left: 6px solid var(--accent); color: {NAVY};
    box-shadow: 0 4px 14px rgba(0,0,0,.07); height: 100%;
}}
.kpi .label {{ font-size: .85rem; color: #6c757d; }}
.kpi .value {{ font-size: 1.7rem; font-weight: 700; color: {NAVY}; line-height: 1.3; }}
.kpi .sub   {{ font-size: .78rem; color: #8a94a0; }}
.result {{
    border-radius: 20px; padding: 1.6rem 1.8rem; color: #fff;
    box-shadow: 0 10px 28px rgba(0,0,0,.18);
}}
.result h2 {{ color: #fff; margin: 0 0 .3rem 0; font-size: 1.7rem; }}
.result p  {{ color: rgba(255,255,255,.92); margin: 0; }}
.result.bad  {{ background: linear-gradient(135deg, #C1121F, {RED}); }}
.result.good {{ background: linear-gradient(135deg, #1B7A6E, {GREEN}); }}
.section-title {{ font-weight: 600; font-size: 1.15rem; margin: .4rem 0 .6rem 0; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .4rem; }}
.stTabs [data-baseweb="tab"] {{
    border-radius: 12px 12px 0 0; padding: .55rem 1.1rem; font-weight: 600;
}}
div.stButton > button[kind="primary"], div.stFormSubmitButton > button {{
    background: linear-gradient(90deg, {RED}, #F4743B); border: none; color: #fff;
    font-weight: 600; border-radius: 12px; padding: .6rem 1rem;
}}
.footer {{ text-align: center; color: #8a94a0; font-size: .82rem; margin-top: 2rem; }}
</style>
""",
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Data dictionary
# The numeric codes match the Kaggle "Heart Failure Prediction" dataset,
# label-encoded alphabetically starting from 1.
# --------------------------------------------------------------------------
TARGET = "HeartDisease"
FEATURES = ["Age", "Sex", "ChestPainType", "RestingBP", "Cholesterol",
            "FastingBS", "RestingECG", "MaxHR", "ExerciseAngina", "Oldpeak",
            "ST_Slope"]

CATEGORY_LABELS = {
    "Sex": {1: "ชาย (Male)", 0: "หญิง (Female)"},
    "ChestPainType": {
        1: "ASY – ไม่มีอาการเจ็บหน้าอก",
        2: "ATA – เจ็บหน้าอกแบบไม่ปกติ",
        3: "NAP – เจ็บที่ไม่ใช่จากหัวใจ",
        4: "TA – เจ็บหน้าอกแบบหัวใจขาดเลือด",
    },
    "FastingBS": {0: "ไม่เกิน 120 mg/dl", 1: "มากกว่า 120 mg/dl"},
    "RestingECG": {1: "LVH – หัวใจห้องล่างซ้ายโต",
                   2: "Normal – ปกติ",
                   3: "ST – ST-T ผิดปกติ"},
    "ExerciseAngina": {0: "ไม่มี", 1: "มี"},
    "ST_Slope": {1: "Down – ลาดลง", 2: "Flat – แบนราบ", 3: "Up – ลาดขึ้น"},
}

THAI_NAMES = {
    "Age": "อายุ (ปี)",
    "Sex": "เพศ",
    "ChestPainType": "ลักษณะการเจ็บหน้าอก",
    "RestingBP": "ความดันขณะพัก (mm Hg)",
    "Cholesterol": "คอเลสเตอรอล (mg/dl)",
    "FastingBS": "น้ำตาลในเลือดขณะอดอาหาร",
    "RestingECG": "คลื่นไฟฟ้าหัวใจขณะพัก",
    "MaxHR": "อัตราการเต้นหัวใจสูงสุด",
    "ExerciseAngina": "เจ็บหน้าอกขณะออกกำลังกาย",
    "Oldpeak": "Oldpeak (ST depression)",
    "ST_Slope": "ความชันของช่วง ST",
}

CLASS_NAMES = {0: "ไม่เป็นโรคหัวใจ", 1: "เป็นโรคหัวใจ"}


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("./data/Heart3.csv")


def build_model(k: int, weights: str, metric: str, scale: bool) -> Pipeline:
    scaler = StandardScaler() if scale else FunctionTransformer()
    return Pipeline([
        ("scaler", scaler),
        ("knn", KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric)),
    ])


@st.cache_resource
def train(k: int, weights: str, metric: str, scale: bool):
    """Evaluate on a held-out split, then refit on all data for predictions."""
    dt = load_data()
    X, y = dt[FEATURES], dt[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    eval_model = build_model(k, weights, metric, scale).fit(X_tr, y_tr)
    y_pred = eval_model.predict(X_te)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(build_model(k, weights, metric, scale), X, y, cv=cv)

    metrics = {
        "accuracy": accuracy_score(y_te, y_pred),
        "precision": precision_score(y_te, y_pred),
        "recall": recall_score(y_te, y_pred),
        "f1": f1_score(y_te, y_pred),
        "cv_mean": cv_scores.mean(),
        "cv_std": cv_scores.std(),
        "cm": confusion_matrix(y_te, y_pred),
    }
    final_model = build_model(k, weights, metric, scale).fit(X, y)
    return final_model, metrics


@st.cache_data
def k_curve(weights: str, metric: str, scale: bool) -> pd.DataFrame:
    dt = load_data()
    X, y = dt[FEATURES], dt[TARGET]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for k in range(1, 32, 2):
        s = cross_val_score(build_model(k, weights, metric, scale), X, y, cv=cv)
        rows.append({"k": k, "accuracy": s.mean()})
    return pd.DataFrame(rows)


def kpi(label: str, value: str, sub: str = "", accent: str = NAVY) -> str:
    return (f'<div class="kpi" style="--accent:{accent}"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>')


def readable(df: pd.DataFrame) -> pd.DataFrame:
    """Swap numeric codes for readable labels (for display only)."""
    out = df.copy()
    for col, mapping in CATEGORY_LABELS.items():
        if col in out:
            out[col] = out[col].map(mapping).fillna(out[col])
    if TARGET in out:
        out[TARGET] = out[TARGET].map(CLASS_NAMES)
    return out


dt = load_data()

# --------------------------------------------------------------------------
# Sidebar — model settings
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ ตั้งค่าโมเดล KNN")
    k = st.slider("จำนวนเพื่อนบ้าน (k)", 1, 31, 9, step=2,
                  help="ใช้เลขคี่เพื่อไม่ให้คะแนนโหวตเสมอกัน")
    weights = st.radio("การให้น้ำหนักเพื่อนบ้าน", ["uniform", "distance"],
                       format_func=lambda w: "เท่ากันทุกตัว" if w == "uniform"
                       else "ตามระยะทาง (ใกล้มีน้ำหนักมาก)")
    metric = st.selectbox("วิธีวัดระยะทาง", ["euclidean", "manhattan"],
                          format_func=lambda m: {"euclidean": "Euclidean",
                                                 "manhattan": "Manhattan"}[m])
    scale = st.toggle("ปรับมาตรฐานข้อมูล (StandardScaler)", value=True,
                      help="KNN ใช้ระยะทาง ถ้าไม่ปรับสเกล คอลัมน์ที่ตัวเลขใหญ่ "
                           "เช่น Cholesterol จะมีอิทธิพลมากเกินไป")

    model, m = train(k, weights, metric, scale)

    st.markdown("---")
    st.markdown("### 📈 ความแม่นยำของโมเดล")
    st.metric("Test accuracy", f"{m['accuracy']:.1%}")
    st.metric("5-fold CV", f"{m['cv_mean']:.1%}", f"± {m['cv_std']:.1%}",
              delta_color="off")
    st.caption("ทดสอบกับข้อมูล 20% ที่โมเดลไม่เคยเห็น "
               "และใช้ข้อมูลทั้งหมดเมื่อทำนายผลจริง")
    st.image("./img/chitsanupong.jpg", width="stretch")

# --------------------------------------------------------------------------
# Hero
# --------------------------------------------------------------------------
st.markdown(
    f"""
<div class="hero">
  <h1>❤️ ระบบจำแนกผู้ป่วยโรคหัวใจด้วย KNN</h1>
  <p>Heart Disease Classification with K-Nearest Neighbors · ป้อนข้อมูลผู้ป่วยใหม่เพื่อทำนายความเสี่ยงได้ทันที</p>
  <span class="pill">📋 {len(dt):,} ตัวอย่าง</span>
  <span class="pill">🧬 {len(FEATURES)} คุณลักษณะ</span>
  <span class="pill">🎯 k = {k}</span>
  <span class="pill">✅ ความแม่นยำ {m['cv_mean']:.1%}</span>
</div>
""",
    unsafe_allow_html=True,
)

tab_predict, tab_data, tab_model, tab_about = st.tabs(
    ["🩺 ทำนายผล", "📊 สำรวจข้อมูล", "🎯 ประสิทธิภาพโมเดล", "📘 เกี่ยวกับ KNN"])

# --------------------------------------------------------------------------
# Tab 1 — Predict
# --------------------------------------------------------------------------
with tab_predict:
    mode = st.segmented_control("รูปแบบการทำนาย", ["👤 ทีละราย", "📁 หลายรายจากไฟล์ CSV"],
                                default="👤 ทีละราย", label_visibility="collapsed")

    if mode == "📁 หลายรายจากไฟล์ CSV":
        st.markdown('<div class="section-title">📁 ทำนายหลายรายพร้อมกัน</div>',
                    unsafe_allow_html=True)
        st.write("อัปโหลดไฟล์ CSV ที่มีคอลัมน์ต่อไปนี้ (ใช้รหัสตัวเลขแบบเดียวกับชุดข้อมูล):")
        st.code(", ".join(FEATURES), language=None)
        template = dt[FEATURES].head(5).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ ดาวน์โหลดไฟล์ตัวอย่าง", template,
                           "heart_template.csv", "text/csv")
        up = st.file_uploader("เลือกไฟล์ CSV", type="csv")
        if up is not None:
            new = pd.read_csv(up)
            missing = [c for c in FEATURES if c not in new.columns]
            if missing:
                st.error(f"ไฟล์ขาดคอลัมน์: {', '.join(missing)}")
            else:
                proba = model.predict_proba(new[FEATURES])[:, 1]
                res = new.copy()
                res["โอกาสเป็นโรค"] = proba
                res["ผลทำนาย"] = np.where(proba >= 0.5, CLASS_NAMES[1], CLASS_NAMES[0])
                c1, c2, c3 = st.columns(3)
                c1.markdown(kpi("จำนวนทั้งหมด", f"{len(res):,}", "ราย"), unsafe_allow_html=True)
                c2.markdown(kpi("เป็นโรคหัวใจ", f"{(proba >= .5).sum():,}", "ราย", RED),
                            unsafe_allow_html=True)
                c3.markdown(kpi("ไม่เป็นโรคหัวใจ", f"{(proba < .5).sum():,}", "ราย", GREEN),
                            unsafe_allow_html=True)
                st.write("")
                st.dataframe(
                    res, hide_index=True, width="stretch",
                    column_config={"โอกาสเป็นโรค": st.column_config.ProgressColumn(
                        "โอกาสเป็นโรค", format="percent", min_value=0, max_value=1)})
                st.download_button("⬇️ ดาวน์โหลดผลการทำนาย",
                                   res.to_csv(index=False).encode("utf-8-sig"),
                                   "heart_predictions.csv", "text/csv")
    else:
        left, right = st.columns([1.15, 1], gap="large")

        with left:
            with st.form("predict_form", border=True):
                st.markdown('<div class="section-title">👤 ข้อมูลผู้ป่วยใหม่</div>',
                            unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    age = st.number_input(THAI_NAMES["Age"], 18, 100, 54)
                    sex = st.selectbox(THAI_NAMES["Sex"], [1, 0],
                                       format_func=CATEGORY_LABELS["Sex"].get)
                    cp = st.selectbox(THAI_NAMES["ChestPainType"], [1, 2, 3, 4],
                                      format_func=CATEGORY_LABELS["ChestPainType"].get)
                    bp = st.number_input(THAI_NAMES["RestingBP"], 0, 250, 130,
                                         help="ค่าเฉลี่ยในชุดข้อมูลประมาณ 132")
                    chol = st.number_input(THAI_NAMES["Cholesterol"], 0, 650, 223,
                                           help="ใส่ 0 หากไม่ได้ตรวจ (ชุดข้อมูลใช้ 0 แทนค่าที่ไม่มี)")
                    fbs = st.selectbox(THAI_NAMES["FastingBS"], [0, 1],
                                       format_func=CATEGORY_LABELS["FastingBS"].get)
                with c2:
                    ecg = st.selectbox(THAI_NAMES["RestingECG"], [2, 1, 3],
                                       format_func=CATEGORY_LABELS["RestingECG"].get)
                    maxhr = st.slider(THAI_NAMES["MaxHR"], 60, 210, 138)
                    angina = st.radio(THAI_NAMES["ExerciseAngina"], [0, 1], horizontal=True,
                                      format_func=CATEGORY_LABELS["ExerciseAngina"].get)
                    oldpeak = st.slider(THAI_NAMES["Oldpeak"], -3.0, 7.0, 0.6, step=0.1)
                    slope = st.selectbox(THAI_NAMES["ST_Slope"], [3, 2, 1],
                                         format_func=CATEGORY_LABELS["ST_Slope"].get)
                submitted = st.form_submit_button("🔍 ทำนายผล", width="stretch")

        with right:
            if not submitted:
                st.info("👈 กรอกข้อมูลผู้ป่วยทางซ้าย แล้วกดปุ่ม **ทำนายผล**")
                st.image("./img/heartB.jpg", width="stretch")
            else:
                x_new = pd.DataFrame([[age, sex, cp, bp, chol, fbs, ecg, maxhr,
                                       angina, oldpeak, slope]], columns=FEATURES)
                prob = model.predict_proba(x_new)[0, 1]
                pred = int(prob >= 0.5)

                if pred == 1:
                    st.markdown(
                        f'<div class="result bad"><h2>⚠️ {CLASS_NAMES[1]}</h2>'
                        f'<p>เพื่อนบ้านใกล้ที่สุด {k} รายส่วนใหญ่เป็นผู้ป่วยโรคหัวใจ '
                        f'ควรปรึกษาแพทย์เพื่อตรวจเพิ่มเติม</p></div>',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="result good"><h2>💚 {CLASS_NAMES[0]}</h2>'
                        f'<p>เพื่อนบ้านใกล้ที่สุด {k} รายส่วนใหญ่ไม่เป็นโรคหัวใจ '
                        f'ดูแลสุขภาพอย่างสม่ำเสมอต่อไป</p></div>',
                        unsafe_allow_html=True)

                gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 40}},
                    title={"text": "โอกาสเป็นโรคหัวใจ"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": RED if pred else GREEN, "thickness": 0.3},
                        "steps": [{"range": [0, 50], "color": "rgba(42,157,143,.18)"},
                                  {"range": [50, 100], "color": "rgba(230,57,70,.18)"}],
                        "threshold": {"line": {"color": NAVY, "width": 3}, "value": 50},
                    }))
                gauge.update_layout(height=260, margin=dict(t=60, b=10, l=30, r=30),
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    font=dict(family="Prompt"))
                st.plotly_chart(gauge, width="stretch")
                st.image("./img/heartA.jpg" if pred else "./img/heartB.jpg",
                         width="stretch")

                # Remember prediction history for this session
                rec = readable(x_new).iloc[0].to_dict()
                rec["ผลทำนาย"] = CLASS_NAMES[pred]
                rec["โอกาสเป็นโรค"] = prob
                st.session_state.setdefault("history", []).append(rec)

        if submitted:
            st.markdown("---")
            st.markdown(f'<div class="section-title">🧭 {k} เพื่อนบ้านที่ใกล้ที่สุดในชุดข้อมูล</div>',
                        unsafe_allow_html=True)
            st.caption("KNN ตัดสินผลจากการโหวตของผู้ป่วยในชุดข้อมูลที่มีลักษณะใกล้เคียงที่สุด")
            dist, idx = model.named_steps["knn"].kneighbors(
                model.named_steps["scaler"].transform(x_new))
            nb = dt.iloc[idx[0]].copy()
            nb.insert(0, "ระยะทาง", dist[0].round(3))
            nb_view = readable(nb).rename(columns={TARGET: "ผลจริง"})

            def color_label(v):
                if v == CLASS_NAMES[1]:
                    return f"background-color: rgba(230,57,70,.18); color: {RED}; font-weight:600"
                if v == CLASS_NAMES[0]:
                    return f"background-color: rgba(42,157,143,.18); color: {GREEN}; font-weight:600"
                return ""

            st.dataframe(nb_view.style.map(color_label, subset=["ผลจริง"]),
                         hide_index=True, width="stretch")

        history = st.session_state.get("history", [])
        if history:
            with st.expander(f"🕘 ประวัติการทำนายในรอบนี้ ({len(history)} รายการ)"):
                hist_df = pd.DataFrame(history)
                st.dataframe(
                    hist_df, hide_index=True, width="stretch",
                    column_config={"โอกาสเป็นโรค": st.column_config.ProgressColumn(
                        "โอกาสเป็นโรค", format="percent", min_value=0, max_value=1)})
                c1, c2 = st.columns(2)
                c1.download_button("⬇️ ดาวน์โหลดประวัติ",
                                   hist_df.to_csv(index=False).encode("utf-8-sig"),
                                   "prediction_history.csv", "text/csv",
                                   width="stretch")
                if c2.button("🗑️ ล้างประวัติ", width="stretch"):
                    st.session_state["history"] = []
                    st.rerun()

# --------------------------------------------------------------------------
# Tab 2 — Explore data
# --------------------------------------------------------------------------
with tab_data:
    n_pos = int(dt[TARGET].sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("จำนวนตัวอย่าง", f"{len(dt):,}", "ราย"), unsafe_allow_html=True)
    c2.markdown(kpi("เป็นโรคหัวใจ", f"{n_pos:,}", f"{n_pos/len(dt):.1%} ของทั้งหมด", RED),
                unsafe_allow_html=True)
    c3.markdown(kpi("ไม่เป็นโรคหัวใจ", f"{len(dt)-n_pos:,}",
                    f"{1-n_pos/len(dt):.1%} ของทั้งหมด", GREEN), unsafe_allow_html=True)
    c4.markdown(kpi("อายุเฉลี่ย", f"{dt['Age'].mean():.1f}",
                    f"ช่วง {dt['Age'].min()}–{dt['Age'].max()} ปี", "#F4A261"),
                unsafe_allow_html=True)
    st.write("")

    view = readable(dt)
    g1, g2 = st.columns(2, gap="large")
    with g1:
        with st.container(border=True):
            st.markdown('<div class="section-title">สัดส่วนผู้ป่วย</div>', unsafe_allow_html=True)
            pie = px.pie(view, names=TARGET, hole=.55, color=TARGET,
                         color_discrete_map=COLORS)
            pie.update_traces(textinfo="percent+label", textfont_size=13)
            pie.update_layout(showlegend=False, height=330,
                              margin=dict(t=10, b=10, l=10, r=10),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(pie, width="stretch")
    with g2:
        with st.container(border=True):
            st.markdown('<div class="section-title">ความสัมพันธ์กับการเป็นโรคหัวใจ</div>',
                        unsafe_allow_html=True)
            corr = (dt.corr()[TARGET].drop(TARGET).sort_values()
                    .rename(index=THAI_NAMES).reset_index())
            corr.columns = ["feature", "corr"]
            bar = px.bar(corr, x="corr", y="feature", orientation="h",
                         color="corr", color_continuous_scale=[GREEN, "#EEEEEE", RED],
                         range_color=[-.6, .6])
            bar.update_layout(height=330, coloraxis_showscale=False,
                              xaxis_title="ค่าสหสัมพันธ์", yaxis_title=None,
                              margin=dict(t=10, b=10, l=10, r=10),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(bar, width="stretch")

    with st.container(border=True):
        st.markdown('<div class="section-title">เปรียบเทียบการกระจายตัวของข้อมูลแต่ละคุณลักษณะ</div>',
                    unsafe_allow_html=True)
        feat = st.selectbox("เลือกคุณลักษณะ", FEATURES, index=7,
                            format_func=lambda f: f"{THAI_NAMES[f]} ({f})")
        if feat in CATEGORY_LABELS:
            cnt = view.groupby([feat, TARGET]).size().reset_index(name="จำนวน")
            fig = px.bar(cnt, x=feat, y="จำนวน", color=TARGET, barmode="group",
                         color_discrete_map=COLORS, text="จำนวน")
        else:
            fig = px.histogram(view, x=feat, color=TARGET, barmode="overlay",
                               opacity=.7, nbins=35, color_discrete_map=COLORS,
                               marginal="box")
        fig.update_layout(height=400, legend_title=None, xaxis_title=THAI_NAMES[feat],
                          margin=dict(t=10, b=10, l=10, r=10),
                          paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(orientation="h", y=1.08, x=0))
        st.plotly_chart(fig, width="stretch")

    with st.container(border=True):
        st.markdown('<div class="section-title">ความสัมพันธ์ระหว่างสองคุณลักษณะ</div>',
                    unsafe_allow_html=True)
        numeric = ["Age", "RestingBP", "Cholesterol", "MaxHR", "Oldpeak"]
        s1, s2 = st.columns(2)
        fx = s1.selectbox("แกน X", numeric, index=0, format_func=THAI_NAMES.get)
        fy = s2.selectbox("แกน Y", numeric, index=3, format_func=THAI_NAMES.get)
        sc = px.scatter(view, x=fx, y=fy, color=TARGET, color_discrete_map=COLORS,
                        opacity=.65, hover_data=["Sex", "ChestPainType"])
        sc.update_layout(height=420, legend_title=None,
                         xaxis_title=THAI_NAMES[fx], yaxis_title=THAI_NAMES[fy],
                         margin=dict(t=10, b=10, l=10, r=10),
                         paper_bgcolor="rgba(0,0,0,0)",
                         legend=dict(orientation="h", y=1.08, x=0))
        st.plotly_chart(sc, width="stretch")

    with st.expander("🔎 ดูข้อมูลทั้งหมด และคำอธิบายข้อมูล"):
        show_labels = st.toggle("แสดงเป็นข้อความแทนรหัสตัวเลข", value=True)
        st.dataframe(view if show_labels else dt, width="stretch", height=360)
        st.markdown("**ข้อสังเกตจากการวิเคราะห์ข้อมูล**")
        n_zero_chol = int((dt["Cholesterol"] == 0).sum())
        st.markdown(
            f"- ไม่มีค่าว่างและไม่มีแถวซ้ำ\n"
            f"- Cholesterol มีค่า 0 จำนวน {n_zero_chol} ราย ซึ่งหมายถึงไม่ได้ตรวจ "
            f"กลุ่มนี้เป็นโรคหัวใจถึง {dt.loc[dt.Cholesterol == 0, TARGET].mean():.0%} "
            f"จึงเก็บค่าไว้ตามเดิมเพราะช่วยให้โมเดลแม่นขึ้น\n"
            f"- คุณลักษณะที่สัมพันธ์กับโรคหัวใจมากที่สุดคือ ความชันของช่วง ST, "
            f"การเจ็บหน้าอกขณะออกกำลังกาย และ Oldpeak\n"
            f"- ข้อมูลแต่ละคอลัมน์มีหน่วยต่างกันมาก จึงต้องปรับมาตรฐานก่อนใช้ KNN")

# --------------------------------------------------------------------------
# Tab 3 — Model performance
# --------------------------------------------------------------------------
with tab_model:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(kpi("Accuracy", f"{m['accuracy']:.1%}", "ทำนายถูกทั้งหมด", NAVY),
                unsafe_allow_html=True)
    c2.markdown(kpi("Precision", f"{m['precision']:.1%}", "ทายว่าเป็นโรค แล้วเป็นจริง", "#457B9D"),
                unsafe_allow_html=True)
    c3.markdown(kpi("Recall", f"{m['recall']:.1%}", "ผู้ป่วยจริงที่ตรวจพบ", RED),
                unsafe_allow_html=True)
    c4.markdown(kpi("F1-score", f"{m['f1']:.1%}", "ค่าเฉลี่ยของ Precision และ Recall", GREEN),
                unsafe_allow_html=True)
    st.write("")

    p1, p2 = st.columns(2, gap="large")
    with p1:
        with st.container(border=True):
            st.markdown('<div class="section-title">Confusion Matrix (ชุดทดสอบ 20%)</div>',
                        unsafe_allow_html=True)
            labels = [CLASS_NAMES[0], CLASS_NAMES[1]]
            cm = px.imshow(m["cm"], x=labels, y=labels, text_auto=True,
                           color_continuous_scale=["#F1FAEE", "#457B9D", NAVY])
            cm.update_layout(height=360, coloraxis_showscale=False,
                             xaxis_title="ผลทำนาย", yaxis_title="ผลจริง",
                             margin=dict(t=10, b=10, l=10, r=10),
                             paper_bgcolor="rgba(0,0,0,0)")
            cm.update_traces(textfont_size=22)
            st.plotly_chart(cm, width="stretch")
    with p2:
        with st.container(border=True):
            st.markdown('<div class="section-title">ความแม่นยำเมื่อเปลี่ยนค่า k</div>',
                        unsafe_allow_html=True)
            curve = k_curve(weights, metric, scale)
            best = curve.loc[curve["accuracy"].idxmax()]
            line = px.line(curve, x="k", y="accuracy", markers=True)
            line.update_traces(line_color="#457B9D", line_width=3)
            line.add_scatter(x=[k], y=curve.loc[curve.k == k, "accuracy"], mode="markers",
                             marker=dict(size=16, color=RED, symbol="star"),
                             name="k ที่เลือก")
            line.update_layout(height=360, showlegend=False, yaxis_tickformat=".0%",
                               yaxis_title="5-fold CV accuracy",
                               margin=dict(t=10, b=10, l=10, r=10),
                               paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(line, width="stretch")
            st.caption(f"⭐ คือ k ที่เลือกอยู่ · k ที่ดีที่สุดคือ {int(best.k)} "
                       f"(ความแม่นยำ {best.accuracy:.1%})")

    with st.container(border=True):
        st.markdown('<div class="section-title">ผลของการปรับมาตรฐานข้อมูล</div>',
                    unsafe_allow_html=True)
        raw_acc = k_curve(weights, metric, False).set_index("k").loc[k, "accuracy"]
        std_acc = k_curve(weights, metric, True).set_index("k").loc[k, "accuracy"]
        comp = pd.DataFrame({"วิธี": ["ไม่ปรับสเกล", "StandardScaler"],
                             "accuracy": [raw_acc, std_acc]})
        cbar = px.bar(comp, x="accuracy", y="วิธี", orientation="h", text_auto=".1%",
                      color="วิธี", color_discrete_sequence=["#ADB5BD", "#457B9D"])
        cbar.update_layout(height=200, showlegend=False, xaxis_tickformat=".0%",
                           xaxis_range=[0, 1], yaxis_title=None,
                           margin=dict(t=10, b=10, l=10, r=10),
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(cbar, width="stretch")
        st.caption(f"เมื่อ k = {k} การปรับมาตรฐานข้อมูลเพิ่มความแม่นยำ "
                   f"{(std_acc - raw_acc) * 100:+.1f} จุด")

# --------------------------------------------------------------------------
# Tab 4 — About KNN
# --------------------------------------------------------------------------
with tab_about:
    a1, a2 = st.columns([1.3, 1], gap="large")
    with a1:
        st.markdown("""
### K-Nearest Neighbors คืออะไร?
KNN เป็นอัลกอริทึมการเรียนรู้ของเครื่องที่เรียบง่ายที่สุดแบบหนึ่ง
โมเดลจะ **จดจำข้อมูลทั้งหมด** ไว้ เมื่อมีข้อมูลใหม่เข้ามา จะทำงาน 3 ขั้นตอน

1. **วัดระยะทาง** ระหว่างข้อมูลใหม่กับทุกตัวอย่างในชุดข้อมูล
2. **เลือก k ตัวอย่างที่ใกล้ที่สุด** (เพื่อนบ้าน)
3. **โหวต** คลาสที่เพื่อนบ้านส่วนใหญ่เป็น คือผลการทำนาย

### ทำไมต้องปรับมาตรฐานข้อมูล?
คอเลสเตอรอลมีค่าหลักร้อย ส่วนเพศมีค่าแค่ 0 หรือ 1
ถ้าไม่ปรับสเกล ระยะทางจะถูกกำหนดโดยคอลัมน์ที่ตัวเลขใหญ่เกือบทั้งหมด
StandardScaler ปรับทุกคอลัมน์ให้มีค่าเฉลี่ย 0 และส่วนเบี่ยงเบนมาตรฐาน 1

### การเลือกค่า k
- **k น้อย** ไวต่อข้อมูลรบกวน (overfitting)
- **k มาก** เส้นแบ่งเรียบเกินไป (underfitting)
- ใช้ **เลขคี่** เพื่อไม่ให้โหวตเสมอกัน
""")
    with a2:
        with st.container(border=True):
            st.markdown('<div class="section-title">📋 คำอธิบายคุณลักษณะ</div>',
                        unsafe_allow_html=True)
            dict_rows = []
            for f in FEATURES:
                if f in CATEGORY_LABELS:
                    desc = " · ".join(f"{c} = {lbl}" for c, lbl in CATEGORY_LABELS[f].items())
                else:
                    desc = f"ตัวเลข {dt[f].min()} – {dt[f].max()}"
                dict_rows.append({"คอลัมน์": f, "ความหมาย": THAI_NAMES[f], "ค่าที่เป็นไปได้": desc})
            st.dataframe(pd.DataFrame(dict_rows), hide_index=True, width="stretch",
                         height=430)
        st.warning("⚕️ แอปนี้จัดทำเพื่อการศึกษาเท่านั้น ไม่สามารถใช้แทนการวินิจฉัยของแพทย์ได้")

st.markdown('<div class="footer">โปรเจค KNN_2_IT2026 · สร้างด้วย Streamlit และ scikit-learn</div>',
            unsafe_allow_html=True)
