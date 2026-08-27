import os
import sys
import pickle
import re
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))
from src.comparator import RiskComparator
from config import (
    COMPANIES, RISK_CATEGORIES, EMBEDDINGS_DIR,
    AVAILABLE_YEARS, DEFAULT_YEAR,
    get_embeddings_dir, get_risk_profiles_dir,
)

# ============================================================
# Page Configuration & Styling
# ============================================================
st.set_page_config(
    page_title="Financial Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS — Premium Design System
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
    /* === Global === */
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    .stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 50%, #16213e 100%); }

    /* === Sidebar === */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stTextInput label { color: #a0a0c0 !important; font-weight: 500; }

    /* === Headers === */
    .main-header {
        font-size: 2.6rem; font-weight: 800; letter-spacing: -0.5px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0; padding-top: 0.5rem;
    }
    .sub-header { font-size: 1.05rem; color: #8888aa; margin-bottom: 1.8rem; }

    /* === Year Badge === */
    .year-badge {
        display: inline-block; padding: 4px 14px; border-radius: 20px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: #fff !important; font-size: 0.82em; font-weight: 700;
        margin-left: 10px; letter-spacing: 0.5px;
        box-shadow: 0 2px 12px rgba(102,126,234,0.3);
    }

    /* === Risk Severity Pills === */
    .risk-critical {
        color: #ff0040 !important; font-weight: 800;
        text-shadow: 0 0 12px rgba(255,0,64,0.5);
        animation: pulse-critical 2s ease-in-out infinite;
    }
    @keyframes pulse-critical {
        0%, 100% { opacity: 1; } 50% { opacity: 0.7; }
    }
    .risk-high {
        color: #ff4757 !important; font-weight: 700;
        text-shadow: 0 0 8px rgba(255,71,87,0.3);
    }
    .risk-medium { color: #ffa502 !important; font-weight: 700; }
    .risk-low { color: #2ed573 !important; font-weight: 700; }
    .risk-negligible { color: #636e72 !important; font-weight: 500; font-style: italic; }

    /* === Evidence Box === */
    .evidence-box {
        background: rgba(255,255,255,0.04);
        border-left: 3px solid #667eea;
        padding: 14px 16px; margin-bottom: 10px; border-radius: 6px;
        font-family: 'JetBrains Mono', 'Courier New', monospace;
        font-size: 0.85em; line-height: 1.6; white-space: pre-wrap;
        color: #c8c8d8 !important;
        transition: border-color 0.2s;
    }
    .evidence-box:hover { border-color: #764ba2; }

    /* === Tabs === */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px; background: rgba(255,255,255,0.03);
        border-radius: 12px; padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px; padding: 8px 16px; font-weight: 500;
        color: #8888aa !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: #fff !important; font-weight: 600;
        box-shadow: 0 2px 12px rgba(102,126,234,0.25);
    }

    /* === Metric Cards === */
    .metric-card {
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 20px; text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, box-shadow 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102,126,234,0.15);
    }

    /* === Expanders === */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03) !important;
        border-radius: 8px !important; font-weight: 500;
    }

    /* === Plotly charts dark bg === */
    .js-plotly-plot .plotly .main-svg { background: transparent !important; }

    /* === Scrollbar === */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #667eea40; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Sidebar: Year Selector & Navigation
# ============================================================
st.sidebar.title("Navigation")
st.sidebar.markdown("Explore automated risk profiles extracted from Form 10-K filings using Multi-Document RAG.")
st.sidebar.markdown("---")

# Year selector — show ALL years so user can pick any and fetch data
years_with_data = RiskComparator.get_available_years()

# Initialize session state for year if not present
if "selected_year_ss" not in st.session_state:
    st.session_state.selected_year_ss = DEFAULT_YEAR

selected_year = st.sidebar.selectbox(
    "📅 Analiz Yılı (Fiscal Year)",
    options=AVAILABLE_YEARS,
    index=AVAILABLE_YEARS.index(st.session_state.selected_year_ss),
    key="main_year_selector",
    help="Hangi yılın 10-K raporlarını incelemek istiyorsunuz?",
    format_func=lambda y: f"FY{y} ✅" if y in years_with_data else f"FY{y}"
)

# Update session state whenever the selectbox changes
st.session_state.selected_year_ss = selected_year

has_data = selected_year in years_with_data

st.sidebar.markdown("---")

# ============================================================
# Data Loading (year-specific)
# ============================================================
@st.cache_data
def load_data(year):
    comp = RiskComparator(year=year)
    return comp

@st.cache_data
def load_chunk_metadata(year):
    """Load metadata to map Chunk IDs back to full text."""
    # Try year-specific first
    year_emb_dir = get_embeddings_dir(year)
    meta_path = os.path.join(year_emb_dir, "chunk_metadata.pkl")
    
    # Fallback to flat dir
    if not os.path.exists(meta_path):
        meta_path = os.path.join(EMBEDDINGS_DIR, "chunk_metadata.pkl")
    
    if os.path.exists(meta_path):
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
            return {meta["chunk_id"]: meta["text"] for meta in metadata.values()}
    return {}

comparator = load_data(selected_year)
chunk_lookup = load_chunk_metadata(selected_year)
available_companies = comparator.get_available_companies()

if not available_companies:
    has_data = False

# Sidebar: Available companies (only if data exists)
if has_data:
    st.sidebar.markdown(f"**Mevcut Şirketler (FY{selected_year}):**")
    for ticker in available_companies:
        name = COMPANIES.get(ticker, ticker)
        st.sidebar.markdown(f"- **{ticker}**: {name}")
else:
    st.sidebar.warning(f"FY{selected_year} için henüz veri yok.")

st.sidebar.markdown("---")

# Live analysis section
st.sidebar.markdown("### ⚡ Canlı Hisse Analizi")
new_ticker = st.sidebar.text_input("Ticker Girin (Örn: NFLX, GOOGL)", max_chars=5)
live_year = st.sidebar.selectbox(
    "Hangi Yıl İçin?",
    options=AVAILABLE_YEARS,
    index=AVAILABLE_YEARS.index(selected_year) if selected_year in AVAILABLE_YEARS else len(AVAILABLE_YEARS) - 1,
    key="live_year"
)

if st.sidebar.button("Verileri Çek & Analiz Et"):
    if new_ticker:
        from src.live_pipeline import run_live_analysis
        status_box = st.sidebar.empty()
        with st.spinner(f"FY{live_year} için {new_ticker.upper()} analiz ediliyor..."):
            success, msg = run_live_analysis(new_ticker, year=live_year, status_placeholder=status_box)
        if success:
            st.sidebar.success(f"✅ {new_ticker.upper()} (FY{live_year}) başarıyla eklendi!")
            st.session_state.selected_year_ss = live_year  # Set the view to the year just analyzed
            st.cache_data.clear()
            st.rerun()
        else:
            st.sidebar.error(msg)
    else:
        st.sidebar.warning("Lütfen bir Ticker girin.")

# ============================================================
# Main Header + KPI Cards
# ============================================================
st.markdown('<p class="main-header">Automated Risk Profiling Dashboard</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">Powered by Llama-3.1-8B (Groq) · FAISS Vector Search · BAAI/bge-small-en-v1.5 <span class="year-badge">FY{selected_year}</span></p>', unsafe_allow_html=True)

# If no data for selected year, show empty state
if not has_data:
    st.markdown("---")
    st.markdown(f"""
    ### 📭 FY{selected_year} için henüz veri yok
    
    Bu yılın 10-K raporları henüz analiz edilmemiş. Veri çekmek için:
    
    1. **Sol paneldeki** "⚡ Canlı Hisse Analizi" bölümüne gidin
    2. Bir **Ticker** girin (Örn: AAPL, MSFT, TSLA)
    3. **"Hangi Yıl İçin?"** bölümünden **FY{selected_year}** seçili olduğundan emin olun
    4. **"Verileri Çek & Analiz Et"** butonuna tıklayın
    
    Sistem otomatik olarak SEC EDGAR'dan o yılın 10-K raporunu çekip, 
    Llama-3 ile risk analizi yapacaktır.
    """)
    
    all_available_years = RiskComparator.get_available_years()
    if len(all_available_years) >= 2:
        st.markdown("---")
        st.markdown("💡 **Not:** Mevcut verisi olan yıllar arasında karşılaştırma yapmak için yukarıdan farklı bir yıl seçebilirsiniz.")
        st.markdown(f"**Verisi olan yıllar:** {', '.join(f'FY{y}' for y in all_available_years)}")
    
    st.stop()

# KPI Cards
total_companies = len(available_companies)
total_high = sum(
    1 for p in comparator.profiles 
    for r in p.get("risk_assessments", []) 
    if r.get("is_present") and r.get("severity") == "high"
)
avg_conf = 0
count = 0
for p in comparator.profiles:
    for r in p.get("risk_assessments", []):
        if r.get("is_present"):
            avg_conf += r.get("confidence", 0)
            count += 1
avg_conf = avg_conf / count if count else 0

kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.markdown(f"""<div class="metric-card">
        <div style="font-size:2.2rem;font-weight:800;color:#667eea">{total_companies}</div>
        <div style="font-size:0.85rem;color:#8888aa;margin-top:4px">Companies Analyzed</div>
    </div>""", unsafe_allow_html=True)
with kpi2:
    st.markdown(f"""<div class="metric-card">
        <div style="font-size:2.2rem;font-weight:800;color:#ff4757">{total_high}</div>
        <div style="font-size:0.85rem;color:#8888aa;margin-top:4px">High-Severity Risks</div>
    </div>""", unsafe_allow_html=True)
with kpi3:
    st.markdown(f"""<div class="metric-card">
        <div style="font-size:2.2rem;font-weight:800;color:#2ed573">{len(RISK_CATEGORIES)}</div>
        <div style="font-size:0.85rem;color:#8888aa;margin-top:4px">Risk Categories</div>
    </div>""", unsafe_allow_html=True)
with kpi4:
    st.markdown(f"""<div class="metric-card">
        <div style="font-size:2.2rem;font-weight:800;color:#ffa502">{avg_conf:.0%}</div>
        <div style="font-size:0.85rem;color:#8888aa;margin-top:4px">Avg. Confidence</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌡️ Risk Heatmap", 
    "🏆 Top Risks by Company", 
    "🔍 Evidence Explorer", 
    "⚖️ Company Comparison",
    "📈 Year-over-Year Analysis"
])

# ============================================================
# Tab 1: Risk Heatmap
# ============================================================
with tab1:
    st.markdown(f"### Industry Risk Overview — FY{selected_year}")
    st.markdown("A macro view of risk severity across all analyzed companies and categories.")
    
    df_scores = comparator.get_risk_heatmap_data()
    df_labels = comparator.get_severity_labels_matrix()
    
    if not df_scores.empty:
        fig = go.Figure(data=go.Heatmap(
            z=df_scores.values,
            x=[c.replace(" Risk", "") for c in df_scores.columns],
            y=df_scores.index,
            text=df_labels.values,
            texttemplate="%{text}",
            textfont=dict(size=12, color="#e0e0e0"),
            colorscale=[
                [0, "#1a1a2e"],
                [0.33, "#1b4332"],
                [0.66, "#7b2d26"],
                [1.0, "#c0392b"]
            ],
            showscale=False,
            hovertemplate="<b>%{y}</b> — %{x}<br>Severity: %{text}<extra></extra>"
        ))
        
        fig.update_layout(
            height=500,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#c8c8d8"),
            xaxis=dict(tickangle=-35, side="bottom"),
            margin=dict(t=30, l=100, r=20, b=100)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data available for heatmap.")

# ============================================================
# Tab 2: Top Risks by Company
# ============================================================
with tab2:
    st.markdown(f"### Company Deep Dive — FY{selected_year}")
    
    col1, col2 = st.columns([1, 3])
    with col1:
        selected_ticker = st.selectbox("Select Company", available_companies, key="top_risks_company")
        
    profile = comparator.get_company_profile(selected_ticker)
    
    if profile:
        with col2:
            st.markdown(f"#### {profile['company_name']} ({selected_ticker})")
            st.markdown(f"**Total Risks Identified:** {profile['risks_found']} out of {profile['total_categories']} categories.")
            
        top_risks = comparator.get_top_risks_for_company(selected_ticker, top_n=8)
        
        for i, risk in enumerate(top_risks):
            sev_class = f"risk-{risk['severity'].lower()}"
            with st.expander(f"{i+1}. {risk['risk_category']} - {risk['severity'].upper()}", expanded=(i==0)):
                st.markdown(f"**Severity:** <span class='{sev_class}'>{risk['severity'].upper()}</span> | **Confidence:** {risk['confidence']:.2f}", unsafe_allow_html=True)
                st.markdown(f"**LLM Assessment:** {risk['explanation']}")
                
                if risk.get("evidence_snippets"):
                    st.markdown("**Key Evidence:**")
                    for snippet in risk["evidence_snippets"]:
                        # Some LLMs return the Chunk ID instead of the text, e.g. "[Chunk 1] (ID: AAPL_2025_item1a_0031)"
                        chunk_id_match = re.search(r"ID:\s*([A-Za-z0-9_]+)", snippet)
                        if chunk_id_match:
                            chunk_id = chunk_id_match.group(1)
                            full_text = chunk_lookup.get(chunk_id, snippet)
                            st.markdown(f"**📄 Original 10-K Chunk: {chunk_id}**")
                            st.markdown(f"<div class='evidence-box'>{full_text}</div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div class='evidence-box'>\"{snippet}\"</div>", unsafe_allow_html=True)

# ============================================================
# Tab 3: Evidence Explorer
# ============================================================
with tab3:
    st.markdown(f"### Source Traceability — FY{selected_year}")
    st.markdown("Investigate the exact 10-K evidence snippets the LLM used to make its assessment.")
    
    col1, col2 = st.columns(2)
    with col1:
        ev_company = st.selectbox("Select Company", available_companies, key="ev_company")
    with col2:
        ev_category = st.selectbox(
            "Select Risk Category", 
            [cat["name"] for cat in RISK_CATEGORIES], 
            key="ev_category"
        )
        
    profile = comparator.get_company_profile(ev_company)
    if profile:
        risk_data = next((r for r in profile["risk_assessments"] if r["risk_category"] == ev_category), None)
        
        if risk_data:
            if not risk_data["is_present"]:
                st.info(f"The LLM determined that **{ev_category}** is not significantly mentioned for {ev_company}.")
            else:
                st.markdown(f"#### Assessment: <span class='risk-{risk_data['severity'].lower()}'>{risk_data['severity'].upper()}</span> Risk", unsafe_allow_html=True)
                st.write(risk_data["explanation"])
                
                st.markdown("#### Retrieved Source Evidence (Grounding)")
                for idx, snippet in enumerate(risk_data.get("evidence_snippets", [])):
                    chunk_id_match = re.search(r"ID:\s*([A-Za-z0-9_]+)", snippet)
                    if chunk_id_match:
                        chunk_id = chunk_id_match.group(1)
                        full_text = chunk_lookup.get(chunk_id, snippet)
                        st.markdown(f"**Snippet {idx+1} (Original 10-K Context: {chunk_id})**")
                        st.markdown(f"<div class='evidence-box'>{full_text}</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**Snippet {idx+1}**")
                        st.markdown(f"<div class='evidence-box'>\"{snippet}\"</div>", unsafe_allow_html=True)
        else:
            st.warning("No data found for this combination.")

# ============================================================
# Tab 4: Company Comparison
# ============================================================
with tab4:
    st.markdown(f"### Side-by-Side Comparison — FY{selected_year}")
    
    col1, col2 = st.columns(2)
    with col1:
        comp1 = st.selectbox("Company 1", available_companies, index=0)
    with col2:
        comp2 = st.selectbox("Company 2", available_companies, index=1 if len(available_companies) > 1 else 0)
        
    if comp1 and comp2:
        df_comp = comparator.compare_two_companies(comp1, comp2)
        
        # Display nicely
        for _, row in df_comp.iterrows():
            st.markdown(f"#### {row['Risk Category']}")
            
            c1, c2 = st.columns(2)
            with c1:
                sev1 = row[f"{comp1} Severity"]
                sev_class1 = f"risk-{sev1.lower()}" if sev1 != "None" else ""
                st.markdown(f"**{comp1}**: <span class='{sev_class1}'>{sev1}</span>", unsafe_allow_html=True)
                if sev1 != "None":
                    st.write(row[f"{comp1} Explanation"])
                    
            with c2:
                sev2 = row[f"{comp2} Severity"]
                sev_class2 = f"risk-{sev2.lower()}" if sev2 != "None" else ""
                st.markdown(f"**{comp2}**: <span class='{sev_class2}'>{sev2}</span>", unsafe_allow_html=True)
                if sev2 != "None":
                    st.write(row[f"{comp2} Explanation"])
                    
            st.markdown("---")

# ============================================================
# Tab 5: Year-over-Year Analysis (NEW)
# ============================================================
with tab5:
    st.markdown("### 📈 Year-over-Year Risk Analysis")
    st.markdown("Compare how a company's risk profile evolved over different fiscal years.")
    
    all_available_years = RiskComparator.get_available_years()
    
    if len(all_available_years) < 2:
        st.info(
            f"Yıl bazlı karşılaştırma için en az **2 farklı yılın** verisine ihtiyaç var.\n\n"
            f"Şu anda sadece **FY{all_available_years[0] if all_available_years else '—'}** verisi mevcut.\n\n"
            f"Sidebar'daki **Canlı Hisse Analizi** bölümünden farklı bir yıl seçerek yeni veri çekebilirsiniz."
        )
    else:
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            yoy_ticker = st.selectbox(
                "Şirket Seçin",
                options=list(COMPANIES.keys()),
                key="yoy_ticker"
            )
        
        with col2:
            yoy_years = st.multiselect(
                "Yılları Seçin",
                options=all_available_years,
                default=all_available_years,
                key="yoy_years"
            )
        
        if len(yoy_years) >= 2 and yoy_ticker:
            # --- Section 1: Risk Severity Grouped Bar Chart ---
            st.markdown(f"#### {COMPANIES.get(yoy_ticker, yoy_ticker)} — Risk Severity Trend")
            
            df_trend = RiskComparator.get_risk_trend(yoy_ticker, yoy_years)
            
            if not df_trend.empty:
                # Build a melted DataFrame for grouped bar chart
                df_melted = df_trend.reset_index().melt(
                    id_vars="Year",
                    var_name="Risk Category",
                    value_name="Severity Score"
                )
                severity_labels = {0: "Negligible", 1: "Low", 2: "Medium", 3: "High", 4: "Critical"}
                df_melted["Severity"] = df_melted["Severity Score"].map(severity_labels)
                df_melted["Year"] = df_melted["Year"].apply(lambda y: f"FY{y}")
                # Shorten risk category names for readability
                df_melted["Category"] = df_melted["Risk Category"].str.replace(" Risk", "")

                # Grouped bar chart — each category visible side-by-side
                fig_bar = px.bar(
                    df_melted,
                    x="Category",
                    y="Severity Score",
                    color="Year",
                    barmode="group",
                    text="Severity",
                    hover_data=["Risk Category", "Severity"],
                    title="Risk Severity Comparison by Category",
                    color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"],
                )
                fig_bar.update_traces(textposition="outside", textfont=dict(size=11, color="#c8c8d8"))
                fig_bar.update_layout(
                    height=480,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#c8c8d8"),
                    yaxis=dict(range=[0, 4.8], dtick=1,
                               tickvals=[0, 1, 2, 3, 4],
                               ticktext=["Negligible", "Low", "Medium", "High", "Critical"],
                               gridcolor="rgba(255,255,255,0.05)"),
                    xaxis=dict(tickangle=-30),
                    xaxis_title="", yaxis_title="Severity",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig_bar, use_container_width=True)
                
                # --- Section 2: Radar Chart Comparison ---
                st.markdown("#### Radar Chart Comparison")
                
                categories = list(df_trend.columns)
                short_labels = [c.replace(" / ", "/").replace(" Risk", "") for c in categories]
                
                fig_radar = go.Figure()
                radar_colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA", "#FFA15A"]
                
                for i, (year_val, row) in enumerate(df_trend.iterrows()):
                    color = radar_colors[i % len(radar_colors)]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=list(row.values) + [row.values[0]],
                        theta=short_labels + [short_labels[0]],
                        fill='toself',
                        name=f"FY{year_val}",
                        opacity=0.35,
                        line=dict(color=color, width=2.5),
                        marker=dict(size=6),
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(visible=True, range=[0, 4],
                                       tickvals=[0, 1, 2, 3, 4],
                                       ticktext=["N/A", "Low", "Med", "High", "Crit"],
                                       gridcolor="rgba(255,255,255,0.08)",
                                       color="#8888aa"),
                        angularaxis=dict(gridcolor="rgba(255,255,255,0.06)", color="#c8c8d8"),
                    ),
                    height=500, showlegend=True,
                    paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#c8c8d8"),
                )
                st.plotly_chart(fig_radar, use_container_width=True)
                
                # --- Section 3: Detailed Year-over-Year Comparison ---
                st.markdown("---")
                st.markdown("#### 📝 Detaylı Yıl Karşılaştırması (Explanation & Evidence)")
                st.markdown("Her risk kategorisi için yıllar arasındaki **LLM değerlendirme** ve **kanıt farklarını** inceleyin.")
                
                sorted_years = sorted(yoy_years)
                
                # Load profiles AND chunk metadata for each year
                year_profiles = {}
                year_chunk_lookups = {}
                for y in sorted_years:
                    comp_y = RiskComparator(year=y)
                    prof = comp_y.get_company_profile(yoy_ticker)
                    if prof:
                        year_profiles[y] = {r["risk_category"]: r for r in prof["risk_assessments"]}
                    year_chunk_lookups[y] = load_chunk_metadata(y)
                
                if year_profiles:
                    for cat in RISK_CATEGORIES:
                        cat_name = cat["name"]
                        
                        explanations = []
                        severities = []
                        for y in sorted_years:
                            risk = year_profiles.get(y, {}).get(cat_name, {})
                            explanations.append(risk.get("explanation", "N/A"))
                            severities.append(risk.get("severity", "none") if risk.get("is_present") else "none")
                        
                        sev_changed = len(set(severities)) > 1
                        exp_changed = len(set(explanations)) > 1
                        
                        if sev_changed:
                            change_icon = "🔄"
                        elif exp_changed:
                            change_icon = "📝"
                        else:
                            change_icon = "➡️"
                        
                        with st.expander(
                            f"{change_icon} {cat_name} — " + " → ".join(
                                f"FY{y}: **{sev.capitalize()}**" for y, sev in zip(sorted_years, severities)
                            ),
                            expanded=sev_changed
                        ):
                            cols = st.columns(len(sorted_years))
                            for i, y in enumerate(sorted_years):
                                risk = year_profiles.get(y, {}).get(cat_name, {})
                                sev = risk.get("severity", "none") if risk.get("is_present") else "none"
                                lookup = year_chunk_lookups.get(y, {})
                                with cols[i]:
                                    sev_class = f"risk-{sev.lower()}" if sev != "none" else ""
                                    st.markdown(f"##### FY{y}")
                                    st.markdown(f"**Severity:** <span class='{sev_class}'>{sev.upper()}</span> | **Confidence:** {risk.get('confidence', 0):.2f}", unsafe_allow_html=True)
                                    st.markdown(f"**Assessment:** {risk.get('explanation', 'N/A')}")
                                    
                                    snippets = risk.get("evidence_snippets", [])
                                    if snippets:
                                        st.markdown("**Evidence:**")
                                        for s_idx, snippet in enumerate(snippets[:2]):
                                            # Resolve chunk IDs to actual text
                                            chunk_id_match = re.search(r"ID:\s*([A-Za-z0-9_]+)", snippet)
                                            if chunk_id_match:
                                                chunk_id = chunk_id_match.group(1)
                                                full_text = lookup.get(chunk_id, snippet)
                                                display_text = full_text[:300] + "..." if len(full_text) > 300 else full_text
                                            else:
                                                display_text = snippet[:300] + "..." if len(snippet) > 300 else snippet
                                            st.markdown(f"<div class='evidence-box' style='font-size:0.8em'>{display_text}</div>", unsafe_allow_html=True)
                                    else:
                                        st.caption("No evidence snippets")
            else:
                st.warning(f"{yoy_ticker} için seçilen yıllarda veri bulunamadı.")
        
        elif yoy_ticker and len(yoy_years) < 2:
            st.info("Karşılaştırma için en az 2 yıl seçin.")

