import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import html as html_lib
import plotly.graph_objects as go
import base64
import os


# -------------------- Configurazione pagina --------------------
st.set_page_config(
    page_title="Magnethon Application",
    page_icon=":trophy:",
    layout="wide"
)

# -------------------- Tema pagina --------------------
st.markdown(
    """
<style>
.stApp {
    background: #ffffff !important;
    color: #000 !important;
}

/* Titoli e paragrafi */
h1 {
    color: #22326b !important;
    text-align: center;
}
h2, h3, h4, h5, h6, p {
    color: #000 !important;
    text-align: center;
}

/* Contenitore principale */
.block-container {
    padding-top: 2.0rem;
    max-width: none;
}

/* Badge */
.badge {
    display: inline-block;
    padding: 6px 12px;
    border-radius: 999px;
    background: #111827;
    color: #fff;
    font-weight: 600;
    font-size: 0.9rem;
}

/* Testo sottile */
.subtle {
    color: #22326b;
    text-align: center;
}

/* Linea orizzontale */
hr {
    border: 0;
    border-top: 1px solid #eee;
    margin: 1.5rem 0;
}

/* Stile globale per tabelle DataFrame */
[data-testid="stDataFrame"] table {
    background-color: white !important;
    color: black !important;
    border-collapse: collapse;
    width: 100%;
}
[data-testid="stDataFrame"] th {
    background-color: #f8f9fa !important;
    color: black !important;
    font-weight: bold;
}

/* Top 3 righe evidenziate */
[data-testid="stDataFrame"] tbody tr:nth-child(1) td {
    background-color: #a7f3a0 !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(2) td {
    background-color: #7ee787 !important;
}
[data-testid="stDataFrame"] tbody tr:nth-child(3) td {
    background-color: #34d399 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------- Logo --------------------
st.image("magnethon logo 2025 final-01 - blu.png", width=350)

# -------------------- Titolo pagina --------------------
st.title(":trophy: Laboratory Competition Dashboard")

st.markdown(
    """
<p class="subtle">
Welcome to the 🧪 <b>Magnethon Award Ceremony</b>, the final event of the three-day competition.<br>
Here we celebrate excellence and innovation across participating laboratories.<br>
Through rigorous statistical evaluation and Gaussian-based ranking,<br>
we are proud to announce the official <b>winning laboratories</b> of this edition. 
</p>
""",
    unsafe_allow_html=True,
)

st.divider()

# -------------------- Funzioni utilità tabelle --------------------
def make_colored_html(df: pd.DataFrame, highlight_top=3, colors=None):
    if colors is None:
        colors = ["#a7f3a0", "#7ee787", "#34d399"]

    df_display = df.reset_index(drop=True)

    def esc(x):
        return html_lib.escape(str(x))

    cols = list(df_display.columns)

    header_html = """
<table border="0" cellpadding="6" cellspacing="0" style="border-collapse:collapse;width:100%;">
<thead>
<tr>
<th></th>
"""
    for c in cols:
        header_html += f'<th style="background:#f8f9fa;color:black;font-weight:bold;text-align:center;">{esc(c)}</th>'
    header_html += "</tr>\n</thead>\n<tbody>\n"

    rows_html = ""
    for i, row in df_display.iterrows():
        bg = colors[i] if i < highlight_top and i < len(colors) else "white"
        rows_html += f'<tr style="background-color:{bg};">\n'
        rows_html += f'<th style="text-align:center;">{i+1}</th>\n'
        for c in cols:
            rows_html += f'<td style="text-align:center;">{esc(row[c])}</td>\n'
        rows_html += "</tr>\n"

    footer = "</tbody>\n</table>"
    return header_html + rows_html + footer

# -------------------- Lettura JSON tabellare --------------------
def json_to_df(file):
    try:
        raw = json.load(file)
    except Exception as e:
        st.error(f"JSON non valido: {e}")
        return None

    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]

    if isinstance(raw, list) and (len(raw) == 0 or isinstance(raw[0], dict)):
        return pd.DataFrame(raw)

    if isinstance(raw, dict):
        try:
            return pd.DataFrame(raw)
        except Exception as e:
            st.error(f"Impossibile costruire la tabella da dict: {e}")
            return None

    st.error("Formato JSON non riconosciuto come tabella.")
    return None

# -------------------- Ripulisco campo labname --------------------
def clean_labname(df: pd.DataFrame) -> pd.DataFrame:
    # Regex cattura "Lab" seguito da eventuali spazi/-, poi cifre (es. "Lab 1234", "Lab-1234", "lab 1234")
    pattern = r'(?i)^(Lab[\s-]*\d+)'
    # estrai la parte "Lab ####" (case-insensitive)
    df['labName_short'] = df['labName'].astype(str).str.extract(pattern, expand=False)
    # se non trovi una corrispondenza, prova a prendere la prima parte prima di " - " (se presente)
    df['labName_short'] = df['labName_short'].fillna(df['labName'].astype(str).str.split(' - ').str[0])
    return df

# -------------------- Funzioni rating & top10 --------------------
def _compute_rating_from_z(series_z: pd.Series, tab: str) -> pd.Series:
    """
    rating = (3 - |z|) / 3 se |z| <= 3, altrimenti 0.
    Range: 1 (z=0) -> 0 (|z|>=3).
    Convert z-scores into a 0–1 reliability rating (3 decimals, round half-up)."""

    abs_z = series_z.abs()
    rating = np.where(abs_z > 3, 0.0, (3 - abs_z) / 3)


    # # round half-up a 3 decimali
    # rating = np.floor(rating * 1000 + 0.5) / 1000

    # Pesi diversi per tubo

    if tab == 'XA':
        rating = rating * 40
    elif tab == 'XB':
        rating = rating * 30
    elif tab == 'YA':
        rating = rating * 15
    elif tab == 'YB':
        rating = rating * 15

    return rating

def get_top_10_winners(df, z_col="zscore", rating_col=None,
                       labname_col="labName_short", labid_col="labId", avg_col="avg"):
    # Ordina per |z| più piccolo (migliore performance)
    df_sorted = df.sort_values(by=z_col, key=lambda x: x.abs()).head(10)
    cols = [labid_col, labname_col, avg_col, z_col]
    if rating_col and rating_col in df_sorted.columns:
        cols.append(rating_col)
    cols = [c for c in cols if c in df_sorted.columns]
    return df_sorted[cols] if cols else df_sorted

# -------------------- Gaussiane / Plot --------------------
# gaussian function old
# def gaussian(x, mu, sigma):
#     return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
def gaussian(x, mu, sigma):
    return (1.0 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def plot_gaussian_competition(
    df,
    z_col="zscore",
    labname_col="labName_short",
    # labid_col="labId",
    avg_col="avg",
    mu=0.0,
    sigma=0.05,
    # drop_worst=True,
    # title="Laboratory Competition Gaussian Plot",
    chart_key=None
):
    df = df.copy()    
    df[z_col] = pd.to_numeric(df[z_col], errors="coerce")

    # Curva teorica
    z = np.linspace(- 5, + 5, 1000)
    x = mu + z * sigma
    y = gaussian(x, mu, sigma)
    y_max = y.max() * 1.05

    # Best = z più vicino a 0
    best_idx = df[z_col].abs().idxmin()
    best_lab = df.loc[best_idx]
    # worst_idx = df[z_col].abs().idxmax()
    # worst_lab = df.loc[worst_idx]
    
    # Punti laboratorio (in base alla gaussiana)
    z_points = df[z_col].values
    y_points = gaussian(mu + z_points * sigma, mu, sigma)

   
    # --- Palette colori ---
    colors = {
        "green_zone": "rgba(34,197,94,0.45)",   # |z| ≤ 2
        "orange_zone": "rgba(251,146,60,0.5)",  # 2 < |z| ≤ 3
        "red_zone": "rgba(239,68,68,0.55)",     # |z| > 3
        "curve": "#22326b",                   # curva gaussiana (blu Magnethon)
        "points": "#FECB52",                  # punti dei laboratori
        # "best": "#dd841d"                     # best lab (arancio intenso)
        "best": "#ec0606"                     # best lab (rosso intenso)
    }
    
    zones = [
        (-5, -3, colors["red_zone"], "z < -3 (red)"),
        (-3, -2, colors["orange_zone"], "-3 < z ≤ -2 (orange)"),
        (-2, 2,  colors["green_zone"], "-2 ≤ z ≤ 2 (green)"),
        (2, 3,   colors["orange_zone"], "2 < z ≤ 3 (orange)"),
        (3, 5,   colors["red_zone"], "z > 3 (red)"),

    ]
    # Figura Plotly
    fig = go.Figure()

    # Bande colorate sotto la curva
    for z0, z1, color, name in zones:
        z_zone = np.linspace(z0, z1, 200)
        y_zone = gaussian(mu + z_zone * sigma, mu, sigma)
        fig.add_trace(go.Scatter(
            x=np.concatenate([z_zone, z_zone[::-1]]),
            y=np.concatenate([y_zone, np.zeros_like(y_zone)]),
            fill="toself", fillcolor=color, line=dict(width=0), mode="lines", name=name, showlegend=False, hoverinfo="skip"
        ))
    
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(symbol="square", size=14, color=colors["red_zone"]),
        name="Red zone: |z| > 3"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(symbol="square", size=14, color=colors["orange_zone"]),
        name="Orange zone: 2 < |z| ≤ 3"
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode="markers",
        marker=dict(symbol="square", size=14, color=colors["green_zone"]),
        name="Green zone: |z| ≤ 2"
    ))
        
    # [-2σ, 2σ] verde
    fig.add_vrect(
        x0=-2,
        x1=+2,
        yref="y",
        fillcolor=colors["green_zone"], 
        layer="below",
        line_width=0,
   )

    # [2σ, 3σ] arancione (sx + dx)
    fig.add_vrect(
        x0=- 3,
        x1=- 2,
        yref="y",
        fillcolor=colors["orange_zone"],  
        layer="below",
        line_width=0,
        
    )

    fig.add_vrect(
        x0=+ 2,
        x1=+ 3,
        yref="y",
        fillcolor=colors["orange_zone"],  
        layer="below",
        line_width=0,
        
    )

    # Code rosse |z| > 3
    fig.add_vrect(
        x0=-5,
        x1=- 3,
        yref="y",
        fillcolor=colors["red_zone"],  
        layer="below",
        line_width=0,
        
    )

    fig.add_vrect(
        x0=+ 3,
        x1=+5,
        yref="y",
        fillcolor=colors["red_zone"],  # verde tenue ma visibile
        layer="below",
        line_width=0,
       
    )

    # Curva Gaussiana
    fig.add_trace(go.Scatter(
        x=z,
        y=y,
        mode="lines",
        name=f"Gaussian Curve",
        line=dict(color="#22326b", width=2), hoverinfo="skip"
    ))

    # Punti laboratori
    fig.add_trace(go.Scatter(
        x=z_points,
        y=y_points,
        mode="markers",
        name="Laboratories",
        marker=dict(size=9, color=colors["points"]),
        text=df[labname_col],
        hovertemplate=(
            "Lab: %{text}<br>"
            # "z = %{x:.5f}<br>"
            # "Density = %{y:.5f}<br>"
        ),
        
    ))

    # Punto Best Lab evidenziato (senza etichetta sul grafico)
    best_z = df.loc[df[z_col].abs().idxmin(), z_col]
    best_y = gaussian(mu + best_z * sigma, mu, sigma)
    fig.add_trace(go.Scatter(
        x=[best_z], y=[best_y],
        mode="markers",
        marker=dict(size=12, symbol="x", color=colors["best"]),
        text=df.loc[df[z_col].abs().idxmin(), 'labName_short'],
        textposition="top center",
        name="Best performer",
        # hovertemplate="Lab: %{text}<br>z = %{x:.2f}<br>Density = %{y:.5f}<extra></extra>"
        hovertemplate="Lab: %{text}"
    ))

    # Linee guida 0, ±2, ±3
    for x0, style in [(0, "dash"), (-2, "dot"), (2, "dot"), (-3, "dashdot"), (3, "dashdot")]:
        fig.add_shape(type="line", x0=x0, x1=x0, y0=0, y1=y_max, line=dict(dash=style))

    fig.update_layout(
        xaxis_title="z-score",
        # yaxis_title="Probability Density",
        xaxis=dict(range=[-5, 5]),
        yaxis=dict(range=[0, y_max]),
        hovermode="closest",
        margin=dict(l=40, r=40, t=60, b=40),
        plot_bgcolor="white",       # Sfondo area grafico
        paper_bgcolor="white",      # Sfondo area esterna
        font=dict(color="black"),   # Testo in nero per contrasto
        legend=dict(
            title="Legend",
            font=dict(color="black", size=12),
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgray",
            borderwidth=1,
        )
    )

    # Linee griglia grigie chiare
    fig.update_xaxes(
        showgrid=True,
        gridcolor="#e0e0e0",
        zeroline=False,
        color="black",             # Etichette e titolo asse X in nero
        title_font=dict(color="black", size=16),
        tickfont=dict(color="black", size=12),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="#e0e0e0",
        zeroline=False,
        color="black",             # Etichette e titolo asse Y in nero
        title_font=dict(color="black", size=16),
        tickfont=dict(color="black", size=12),
    )

    # Render the Plotly figure using the full container width
    st.plotly_chart(fig, use_container_width=True, key=chart_key)

    # st.markdown("### 🏆 The Best Performance is ... ")
    # show_winner_banner(
    #     best_lab,
    #     labname_col=labname_col,
    #     avg_col=avg_col,
    #     z_col=z_col,
    #     bg_path="crown_yellow_modified-removebg-preview.png"  # nome crown file
    # )

    return best_lab 
# , worst_lab

# -------------------- Winner Banner --------------------

def _image_to_base64(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def show_waiting_image(image_base64: str):
    """
    Mostra nella pagina un'immagine (in base64) centrata e adattata alla larghezza.
    """
    if not image_base64:
        st.warning("Image not found or could not be loaded.")
        return

    st.markdown(
        f"""
        <div style="text-align: center; margin-top: 25px;">
            <img src="data:image/png;base64,{image_base64}"
                 style="max-width: 85%; height: auto; border-radius: 12px;">
        </div>
        """,
        unsafe_allow_html=True
    )

def show_winner_banner(best_lab, labname_col, avg_col, z_col,
                       bg_path="crown_yellow_modified-removebg-preview_1.png"):
    lab_name = best_lab.get(labname_col, "Best laboratory")
    avg_val = best_lab.get(avg_col, None)
    z_val = best_lab.get(z_col, None)

    # Formattazione numeri
    avg_txt = f"{avg_val:.5f}" if isinstance(avg_val, (int, float, np.floating)) else avg_val
    z_txt = f"{z_val:.4f}" if isinstance(z_val, (int, float, np.floating)) else z_val

    # Converto immagine in base64 per essere sicuro che venga mostrata
    img_b64 = _image_to_base64(bg_path)

    # Se non trova l'immagine, niente icona rotta: mostro solo il box testo
    crown_img_html = ""
    if img_b64:
        crown_img_html = f"""
        <img src="data:image/png;base64,{img_b64}" alt="Crown"
             style="width: 200px; height: auto; position: absolute; top: 20px; z-index: 1;">
        """

    html = f"""
    <div style="
        position: relative;
        margin: 1.5rem auto 0.5rem auto;
        max-width: 600px;
        height: 280px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: flex-end;
        animation: fadeIn 1.2s ease-in-out;
    ">
        <!-- Bagliore dorato -->
        <div style="
            position: absolute;
            top: 10px;
            width: 220px;
            height: 220px;
            border-radius: 50%;
            background: radial-gradient(circle, rgba(255,215,0,0.5), transparent 70%);
            filter: blur(18px);
            z-index: 0;
        "></div>

        {crown_img_html}

        <!-- Box testo -->
        <div style="
            position: relative;
            background: rgba(255,255,255,0.96);
            padding: 16px 28px;
            border-radius: 16px;
            box-shadow: 0 6px 16px rgba(0,0,0,0.16);
            text-align: center;
            backdrop-filter: blur(3px);
            z-index: 2;
            margin-bottom: 20px;
        ">
            <div style="
                font-size: 0.8rem;
                letter-spacing: 0.15em;
                color: #6b7280;
                font-weight: 700;
                text-transform: uppercase;
            ">
                <! -- Magnethon Best Lab -->
            </div>
            <div style="
                font-size: 1.6rem;
                font-weight: 800;
                color: #111827;
                margin-top: 4px;
            ">
                {html_lib.escape(str(lab_name))}
            </div>
            <div style="
                font-size: 0.95rem;
                color: #374151;
                margin-top: 8px;
            ">
                
            </div>
        </div>

        <style>
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
    </div>
    """
    # <! -- avg = <b>{avg_txt}</b> &nbsp; | &nbsp; z-score = <b>{z_txt}</b> -->
    components.html(html, height=300)


# -------------------- Upload 4 file --------------------
def reset_podium_and_audio():
    """Reset podium animation and audio counters so the podium can restart after new uploads."""
    st.session_state['podium_animation_count'] = 0
    st.session_state['sound_counter'] = 0
    # Optional flag used elsewhere for triggering confetti; clear it if present
    if 'trigger_confetti_final' in st.session_state:
        st.session_state['trigger_confetti_final'] = False
    

cols_up = st.columns(4)
tag_files = ["Degree of Substitution in HPBCD (XA)",
            "Degree of Substitution in HPBCD (XB)",
            "Concentration of HPBCD (YA)",
            "Concentration of HPBCD (YB)"]
files = []
for i, c in enumerate(cols_up, 1):
    with c:
        files.append(st.file_uploader(f" {tag_files[i-1]}", type=["json"], key=f"f{i}"))
        
# After uploaders: detect changes in uploaded files and reset podium/music if files changed
current_snapshot = [f.name if f is not None else None for f in files]
prev_snapshot = st.session_state.get('uploaded_files_snapshot')
if prev_snapshot is None:
    # first run: store snapshot without resetting
    st.session_state['uploaded_files_snapshot'] = current_snapshot
else:
    if current_snapshot != prev_snapshot:
        # files changed (new upload or replacement) -> reset podium/music
        reset_podium_and_audio()
        st.session_state['uploaded_files_snapshot'] = current_snapshot
        #st.info("Podio e audio resettati per nuovi upload.")

st.divider()

# Mappa fissi: ordine file/tabs -> tube
tube_labels = ["XA", "XB", "YA", "YB"]

tabs = st.tabs(["Degree of Substitution in HPBCD (XA)",
                "Degree of Substitution in HPBCD (XB)",
                "Concentration of HPBCD (YA)",
                "Concentration of HPBCD (YB)",
                "Final Score"
                ])

tab = 'Selected Tube Analysis'
with tabs[0]:
    tab = 'Tube_XA'
with tabs[1]:
    tab = 'Tube_XB'
with tabs[2]:
    tab = 'Tube_YA'
with tabs[3]:
    tab = 'Tube_YB'

# Dizionario per salvare i df per tubo
dfs_by_tube = {}

# -------------------- Prime 4 TAB --------------------
for i in range(4):
    tube = tube_labels[i]
    with tabs[i]:
        f = files[i]
        if not f:
            st.info(f"Upload JSON file for Tube {tube}.")
            continue

        df = json_to_df(f)
        if df is None or df.empty:
            st.error("Empty table or not valid.")
            continue

        # Pulisce i nomi laboratorio (es: "Lab 012 - Extra info" → "Lab 012")
        if "labName" in df.columns:
            df = clean_labname(df)


        # Colonne richieste nel JSON: labId, labName, avg, zscore
        labid_col = "labId"
        labname_col = "labName_short"
        avg_col = "avg"
        z_col = "zscore"

        missing = [c for c in [labid_col, labname_col, avg_col, z_col] if c not in df.columns]
        if missing:
            st.error(f"Missing columns in JSON: {missing}")
            continue

        # st.markdown("### Insert Gaussian Parameters")
        # c1, c2 = st.columns(2)
        # with c1:
        #     mu = st.number_input(
        #         "μ (general mean)",
        #         value=5.0286799806,
        #         step=0.001,
        #         format="%.10f",
        #         key=f"mu_{tube}",
        #     )
        # with c2:
        #     sigma = st.number_input(
        #         "σ (inter-lab standard deviation)",
        #         value=0.0702624346,
        #         step=0.001,
        #         format="%.10f",
        #         key=f"sigma_{tube}",
        #     )

        # if sigma <= 0:
        #     st.error("σ must be > 0.")
        #     continue

        st.divider()

        components.html("""
            <div style="
                position: relative;
                height: 80px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.4rem;
                font-weight: 800;
                color: #22326b;
                font-family: 'Arial', sans-serif;
                overflow: visible;
            ">
                🎉 Participants performance overview 🧪
                <div id="confetti-layer" style="
                    position: absolute;
                    inset: -20px 0 0 0;
                    pointer-events: none;
                    z-index: 10;
                "></div>
            </div>

            <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.4.0/dist/confetti.browser.min.js"></script>
            <script>
            (function() {
                var duration = 2500;  // durata totale in ms (2.5 secondi)
                var animationEnd = Date.now() + duration;
                var defaults = {
                    startVelocity: 40,
                    spread: 70,
                    ticks: 100,
                    zIndex: 10
                };

                function randomInRange(min, max) {
                    return Math.random() * (max - min) + min;
                }

                (function frame() {
                    var timeLeft = animationEnd - Date.now();
                    if (timeLeft <= 0) {
                        return;
                    }

                    // diminuisce gradualmente i coriandoli verso la fine, per un effetto più fluido
                    var particleCount = Math.round(40 * (timeLeft / duration));

                    confetti(Object.assign({}, defaults, {
                        particleCount: particleCount,
                        origin: { x: randomInRange(0.15, 0.35), y: 0.0 }
                    }));

                    confetti(Object.assign({}, defaults, {
                        particleCount: particleCount,
                        origin: { x: randomInRange(0.65, 0.85), y: 0.0 }
                    }));

                    requestAnimationFrame(frame);
                })();
            })();
            </script>
            """, height=90)


        # Plot gaussiana + vincitore
        best_lab = plot_gaussian_competition(
            df,
            z_col=z_col,
            labname_col=labname_col,
            avg_col=avg_col,
            mu=0.0,
            sigma=0.05,
            chart_key=f"gauss_{tube}"
        )

        # Calcolo rating per questo tubo
        rating_col = f"Rating_{tube}"
        df[rating_col] = _compute_rating_from_z(df[z_col], tube)

        # st.divider()
        # st.markdown("### Top 10 ")
        # #st.caption("Top 10 winners (with rating)")
        # winners = get_top_10_winners(
        #     df,
        #     z_col=z_col,
        #     rating_col=rating_col,
        #     labname_col=labname_col,
        #     labid_col=labid_col,
        #     avg_col=avg_col,
        # )

        # winners_to_display = winners[[labname_col, avg_col, z_col, rating_col]].copy()

        # # avg round half-up a 5 decimali
        # winners_to_display[avg_col] = np.floor(winners_to_display[avg_col] * 100000 + 0.5) / 100000
        # winners_to_display[avg_col] = winners_to_display[avg_col].map(lambda x: f"{x:.5f}")
        
        # # zscore round half-up a 4 decimali
        # #winners_to_display[z_col] = np.floor(winners_to_display[z_col] * 10000 + 0.5) / 10000
        # # winners_to_display[z_col] = f"{winners_to_display[z_col]:.4f}" if isinstance(winners_to_display[z_col], (int, float, np.floating)) else winners_to_display[z_col]
        # winners_to_display[z_col] = np.round(winners_to_display[z_col], 4)
        # winners_to_display[z_col] = winners_to_display[z_col].map(lambda x: f"{x:.4f}")
        
        # #round half-up a 3 decimale
        # winners_to_display[rating_col] = np.floor(winners_to_display[rating_col] * 1000 + 0.5) / 1000
        # winners_to_display[rating_col] = winners_to_display[rating_col].map(lambda x: f"{x:.3f}")
        
        #         #rename columns for display
        # winners_to_display = winners_to_display.rename(columns={'labName_short': "Laboratory",
        #                                                     'avg': "Average",
        #                                                     'zscore': "Z-Score",
        #                                                     f'Rating_{tube}': "Rating",})
        
        # html_table = make_colored_html(winners_to_display, highlight_top=3)
        # components.html(html_table, height=360, scrolling=True)

        # Salvo df per uso nella pagina finale
        dfs_by_tube[tube] = df.copy()

# -------------------- Funzione podio finale --------------------
def play_hidden_sound(sound_path="Trombetta_Ryanair.mp3"):
    import base64

    # leggo il file audio
    with open(sound_path, "rb") as f:
        audio_bytes = f.read()
    b64 = base64.b64encode(audio_bytes).decode()

    # contatore per cambiare ogni volta l'HTML
    if "sound_counter" not in st.session_state:
        st.session_state["sound_counter"] = 0
    st.session_state["sound_counter"] += 1
    n = st.session_state["sound_counter"]

    components.html(
        f"""
        <!-- play #{n} -->
        <audio id="click-audio-{n}" style="display:none;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>

        <script>
            var audio = document.getElementById('click-audio-{n}');
            if (audio) {{
                audio.currentTime = 0;
                audio.play();
            }}
        </script>
        """,
        height=0,
    )
    
def trigger_confetti():
    components.html("""
    <html>
    <head>
      <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.4.0/dist/confetti.browser.min.js"></script>
      <style>
        html, body {
          margin: 0;
          padding: 0;
          overflow: hidden;
          background: transparent;
        }
        #confetti-overlay {
          position: fixed;
          inset: 0;
          width: 100%;
          height: 100%;
          pointer-events: none;
          z-index: 999999;
        }
      </style>
    </head>
    <body>
      <div id="confetti-overlay"></div>
      <script>
        (function () {
          var duration = 3000; // 3 secondi
          var end = Date.now() + duration;

          var defaults = {
            startVelocity: 45,
            spread: 80,
            ticks: 200,
            zIndex: 999999
          };

          function frame() {
            confetti(Object.assign({}, defaults, {
              particleCount: 6,
              origin: {
                x: Math.random(),
                y: Math.random() - 0.2
              }
            }));

            if (Date.now() < end) {
              requestAnimationFrame(frame);
            }
          }

          frame();
        })();
      </script>
    </body>
    </html>
    """, height=220)



def show_delayed_gold_card(gold_name, gold_score):
    safe_name = html_lib.escape(str(gold_name))
    score_txt = f"{gold_score:.2f}"

    components.html(f"""
        <div id="gold-card-container" style="max-width: 720px; margin: 10px auto 0 auto;"></div>

        <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.4.0/dist/confetti.browser.min.js"></script>
        <script>
        (function() {{
            var container = document.getElementById("gold-card-container");
            if (!container) return;

            // Confetti dopo 8 secondi
            setTimeout(function() {{
                var duration = 2500;
                var end = Date.now() + duration;
                var defaults = {{
                    startVelocity: 45,
                    spread: 80,
                    ticks: 200,
                    zIndex: 10
                }};

                function randomInRange(min, max) {{
                    return Math.random() * (max - min) + min;
                }}

                (function frame() {{
                    var timeLeft = end - Date.now();
                    if (timeLeft <= 0) {{
                        return;
                    }}
                    var particleCount = Math.round(50 * (timeLeft / duration));
                    confetti(Object.assign({{}}, defaults, {{
                        particleCount: particleCount,
                        origin: {{ x: randomInRange(0.2, 0.4), y: 0.1 }}
                    }}));
                    confetti(Object.assign({{}}, defaults, {{
                        particleCount: particleCount,
                        origin: {{ x: randomInRange(0.6, 0.8), y: 0.1 }}
                    }}));
                    requestAnimationFrame(frame);
                }})();
            }}, 8000);  // 8 secondi

            // Dopo 10 secondi mostro la card del 1° posto
            setTimeout(function() {{
                container.innerHTML = `
<div class="card-3d card-3d-animate" style="
    position: relative;
    padding: 24px 30px;
    border-radius: 24px;
    background: linear-gradient(135deg, #fef9c3, #fde68a);
    box-shadow:
        20px 20px 44px rgba(15,23,42,0.35),
        -12px -12px 30px rgba(255,255,255,0.95);
    display: flex;
    align-items: center;
    justify-content: space-between;
    transform-style: preserve-3d;
    transform: perspective(900px) rotateX(8deg);
">
  <span class="card-rank-badge" style="
      position: absolute;
      top: 10px;
      left: 18px;
      padding: 2px 10px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      background: rgba(15,23,42,0.12);
      color: #78350f;
  ">1st place</span>

  <div class="card-left" style="display:flex; align-items:center; gap:18px;">
    <div class="card-medal gold" style="
        width: 72px;
        height: 72px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 40px;
        color: #ffffff;
        box-shadow:
          10px 10px 20px rgba(15,23,42,0.4),
          -5px -5px 14px rgba(255,255,255,0.9);
        background: radial-gradient(circle at 30% 20%, #fef9c3, #eab308);
    ">🥇</div>
    <div>
      <div class="card-lab-title" style="
          font-size: 22px;
          font-weight: 900;
          color: #111827;
      ">{safe_name}</div>
      <div class="card-lab-sub" style="
          font-size: 14px;
          color: #78350f;
      ">Magnethon Winner</div>
    </div>
  </div>

  <div class="card-right" style="text-align:right;">
    <div class="card-score-label" style="
        font-size: 12px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #b45309;
    ">Final score</div>
    <div class="card-score-value" style="
        margin-top: 4px;
        font-size: 24px;
        font-weight: 900;
        color: #92400e;
    ">{score_txt}</div>
  </div>
</div>
                `;
            }}, 10000);  // 10 secondi
        }})();
        </script>
        """,
        height=220,
    )



def show_top3_podium(df_scores):
    if "FinalScore" not in df_scores.columns:
        st.warning("FinalScore not found in dataframe.")
        return
    if "labName_short" not in df_scores.columns:
        st.warning("labName_short column not found in dataframe.")
        return

    top3 = (
        df_scores[["labName_short", "FinalScore"]]
        .dropna(subset=["FinalScore"])
        .sort_values("FinalScore", ascending=False)
        .head(3)
        .reset_index(drop=True)
    )

    if top3.empty:
        st.info("No scores available to build the podium yet.")
        return

    while len(top3) < 3:
        top3.loc[len(top3)] = ["-", 0.0]

    gold_name, gold_score   = top3.loc[0, "labName_short"], top3.loc[0, "FinalScore"]
    silver_name, silver_score = top3.loc[1, "labName_short"], top3.loc[1, "FinalScore"]
    bronze_name, bronze_score = top3.loc[2, "labName_short"], top3.loc[2, "FinalScore"]

    # ---------- STILE CARDS 3D ----------
    st.markdown("""
<style>
.podium-3d-wrapper {
  max-width: 720px;
  margin: 22px auto 10px auto;
  display: flex;
  flex-direction: column;
  gap: 18px;
  font-family: "Segoe UI", system-ui, sans-serif;
}

/* Card base */
.card-3d {
  position: relative;
  padding: 20px 26px;
  border-radius: 22px;
  background: linear-gradient(135deg, #f9fafb, #e5e7eb);
  box-shadow:
    18px 18px 40px rgba(15,23,42,0.25),
    -10px -10px 28px rgba(255,255,255,0.95);
  display: flex;
  align-items: center;
  justify-content: space-between;
  transform-style: preserve-3d;
  transform: perspective(900px) rotateX(8deg);
}

/* “Spessore” inferiore della card */
.card-3d::after {
  content: "";
  position: absolute;
  inset: auto 12px -10px 12px;
  height: 12px;
  border-radius: 0 0 18px 18px;
  background: linear-gradient(180deg, rgba(15,23,42,0.25), transparent);
  filter: blur(4px);
  opacity: 0.65;
  pointer-events: none;
}

/* Colonna sinistra: medaglia + lab */
.card-left {
  display: flex;
  align-items: center;
  gap: 18px;
}

/* Medaglione 3D */
.card-medal {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 32px;
  color: #ffffff;
  box-shadow:
    8px 8px 18px rgba(15,23,42,0.35),
    -4px -4px 12px rgba(255,255,255,0.8);
}

/* Gradient diversi per le tre posizioni */
.card-medal.gold   { background: radial-gradient(circle at 30% 20%, #fef9c3, #eab308); }
.card-medal.silver { background: radial-gradient(circle at 30% 20%, #f9fafb, #9ca3af); }
.card-medal.bronze { background: radial-gradient(circle at 30% 20%, #fbbf77, #92400e); }

.card-lab-title {
  font-size: 20px;
  font-weight: 800;
  color: #111827;
}

.card-lab-sub {
  font-size: 13px;
  color: #6b7280;
}

/* Colonna destra: score */
.card-right {
  text-align: right;
}

.card-score-label {
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #9ca3af;
}

.card-score-value {
  margin-top: 4px;
  font-size: 22px;
  font-weight: 800;
  color: #111827;
}

/* Badge posizione in alto a sinistra */
.card-rank-badge {
  position: absolute;
  top: 10px;
  left: 18px;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.10em;
  text-transform: uppercase;
  background: rgba(15,23,42,0.06);
  color: #4b5563;
}

/* Animazione ingresso */
@keyframes floatIn {
  from { opacity: 0; transform: translateY(20px) perspective(900px) rotateX(12deg); }
  to   { opacity: 1; transform: translateY(0)   perspective(900px) rotateX(8deg); }
}

.card-3d-animate {
  animation: floatIn 0.45s ease-out;
}
</style>
""", unsafe_allow_html=True)

    # ---------- STATO ANIMAZIONE ----------
    if "podium_animation_count" not in st.session_state:
        st.session_state.podium_animation_count = 0

    st.markdown("## 🏆 The winners are ...")
    
    if "sound_counter" not in st.session_state:
        st.session_state["sound_counter"] = 0


    # Center the button using three columns (button placed in the middle column)
    # increase the middle column weight so the button appears centered on the full page
    cols_btn = st.columns([1, 15, 1])
    with cols_btn[1]:
        clicked = st.button("Reveal next position", key="show_podium_button")

        if clicked:
            
            if st.session_state.podium_animation_count < 3:
                st.session_state.podium_animation_count += 1
            # st.audio("Trombetta_Ryanair.mp3", format="audio/mp3", start_time=0, loop=False,autoplay=True)
            # play_hidden_sound("Trombetta_Ryanair.mp3")

            step_now = st.session_state.podium_animation_count

            if step_now in (1,2):
                play_hidden_sound("Trombetta_Ryanair.mp3")
            elif step_now == 3:
                play_hidden_sound("Rocky_Gonna Fly Now_cut.mp3")
        else:
            # nessun click in questo rerun → non suono nulla
            step_now = st.session_state.podium_animation_count

    step = step_now
    
    # ---------- HTML CARDS 3D ----------
    html = '<div class="podium-3d-wrapper">'

    # Ordine VISIVO finale: 3° (top) → 2° → 1° (bottom)
    # Ma l’apparizione è: step1=solo 3°, step2=3°+2°, step3=3°+2°+1°

    #primo posto in una funzione a parte per delay
    # 1° posto (solo dal terzo click)
#     if step >= 3:
#         html += f"""
# <div class="card-3d card-3d-animate">
#   <span class="card-rank-badge">1st place</span>
#   <div class="card-left">
#     <div class="card-medal gold">🥇</div>
#     <div>
#       <div class="card-lab-title">{html_lib.escape(str(gold_name))}</div>
#       <div class="card-lab-sub">Overall winner</div>
#     </div>
#   </div>
#   <div class="card-right">
#     <div class="card-score-label">Final score</div>
#     <div class="card-score-value">{gold_score:.2f}</div>
#   </div>
# </div>
# """
        # st.audio("Trombetta_Ryanair.mp3", format="audio/mp3", start_time=0)



    # 3° posto (già dal primo click)
    if step >= 1:
        html += f"""
<div class="card-3d card-3d-animate">
  <span class="card-rank-badge">3rd place</span>
  <div class="card-left">
    <div class="card-medal bronze">🥉</div>
    <div>
      <div class="card-lab-title">{html_lib.escape(str(bronze_name))}</div>
      <div class="card-lab-sub">Third place</div>
    </div>
  </div>
  <div class="card-right">
    <div class="card-score-label">Final score</div>
    <div class="card-score-value">{bronze_score:.2f}</div>
  </div>
</div>
"""
        # 2° posto (dal secondo click in poi)
    if step >= 2:
        html += f"""
<div class="card-3d card-3d-animate">
  <span class="card-rank-badge">2nd place</span>
  <div class="card-left">
    <div class="card-medal silver">🥈</div>
    <div>
      <div class="card-lab-title">{html_lib.escape(str(silver_name))}</div>
      <div class="card-lab-sub">Runner-up</div>
    </div>
  </div>
  <div class="card-right">
    <div class="card-score-label">Final score</div>
    <div class="card-score-value">{silver_score:.2f}</div>
  </div>
</div>
"""
        
    html += "</div>"

    st.markdown(html, unsafe_allow_html=True)
    
    # if st.session_state.podium_animation_count == 3:
    #     st.session_state["trigger_confetti_final"] = True
    if step >= 3:
        show_delayed_gold_card(gold_name, gold_score)
    
    return st.session_state.podium_animation_count




# -------------------- TAB FINALE: All Together --------------------
with tabs[4]:
    #18.11 - Commento il podio e la tabella finale perchè non li devo mostrare ora
    # mostro solo l'immagine del podio e l'immagine waitingForFinal
    st.divider()
    st.subheader("🏁 Final Score")
    
    # st.markdown("""
    # The final score for each laboratory is calculated as a weighted sum of the ratings obtained in each tube, according to the following equiation:
    # """)

    # st.latex(r"""
    #         Magnethon\ Winner = 
    #         \max_{0 \leq z \leq 3} 
    #         \left[
    #         40 \cdot \left(\frac{3 - |Z_{X_A}|}{3}\right) +
    #         30 \cdot \left(\frac{3 - |Z_{X_B}|}{3}\right) +
    #         15 \cdot \left(\frac{3 - |Z_{Y_A}|}{3}\right) +
    #         15 \cdot \left(\frac{3 - |Z_{Y_B}|}{3}\right)
    #         \right]
    #         """)

    
    # st.markdown("""
    #     where **Z<sub>X<sub>A</sub></sub>**, **Z<sub>X<sub>B</sub></sub>**, **Z<sub>Y<sub>A</sub></sub>**, **Z<sub>Y<sub>B</sub></sub>** are the z-scores
    #     calculated for each team based on the uploaded values of Tubes X<sub>A</sub>, X<sub>B</sub>, Y<sub>A</sub>, Y<sub>B</sub> respectively.
    #     """, unsafe_allow_html=True)
    
    st.divider()
    #     # Confetti in alto nel tab finale quando il primo è stato rivelato
    # if st.session_state.get("trigger_confetti_final", False):
    #     trigger_confetti()
    #     # lo azzero così non li rilancio ad ogni rerun
    #     st.session_state["trigger_confetti_final"] = False

    

    # Controllo che ci siano tutti e 4 i tubi con rating
    if not all(tube in dfs_by_tube for tube in tube_labels):
        # st.info("Please upload and configure all 4 JSON files (XA, XB, YA, YB) first.")
        # st.info("See you tomorrow for the final results! Make sure all 4 tubes have been uploaded. Good luck to all participants! 🍀")
        #         # MOSTRA SUBITO L’IMMAGINE DI ATTESA
        # img_waiting = _image_to_base64("WaitingForTheWinner.png")
        # show_waiting_image(img_waiting)

        st.info(
            "See you tomorrow for the final results! "
            "Make sure all 4 tubes have been uploaded. "
            "Good luck to all participants! 🍀"
        )
    
    else:

        st.divider()
        img_waiting = _image_to_base64("WaitingForTheWinner.png")
        show_waiting_image(img_waiting)

        df_XA = dfs_by_tube["XA"][["labName_short", "Rating_XA"]]
        df_XB = dfs_by_tube["XB"][["labName_short", "Rating_XB"]]
        df_YA = dfs_by_tube["YA"][["labName_short", "Rating_YA"]]
        df_YB = dfs_by_tube["YB"][["labName_short", "Rating_YB"]]

        merged = (
            df_XA.merge(df_XB, on=["labName_short"], how="outer")
                .merge(df_YA, on=["labName_short"], how="outer")
                .merge(df_YB, on=["labName_short"], how="outer")
        ).fillna(0.0)

        merged["FinalScore"] = (
            merged["Rating_XA"] 
            + merged["Rating_XB"]
            + merged["Rating_YA"]
            + merged["Rating_YB"]
        )

        merged["FinalScore"] = np.floor(merged["FinalScore"] * 1000 + 0.5) / 1000

        # rating half-up a 3 decimali
        for col in ["Rating_XA", "Rating_XB", "Rating_YA", "Rating_YB"]:
            merged[col] = np.floor(merged[col] * 1000 + 0.5) / 1000

        merged = merged.sort_values(by="FinalScore", ascending=False).reset_index(drop=True)

        # podium_step = show_top3_podium(merged)
        
        
        # if podium_step < 3:
        #     st.divider()
        # else:
        #     st.divider()
        #     st.markdown("###  Final Results sorted by Ratings")
        #     # rename columns for display
        #     merged_display = merged.rename(columns={'labName_short': 'Laboratory'})
        #     # Format the columns for display
        #     merged_display["Rating_XA"] = merged_display["Rating_XA"].map(lambda x: f"{x:.3f}")
        #     merged_display["Rating_XB"] = merged_display["Rating_XB"].map(lambda x: f"{x:.3f}")
        #     merged_display["Rating_YA"] = merged_display["Rating_YA"].map(lambda x: f"{x:.3f}")
        #     merged_display["Rating_YB"] = merged_display["Rating_YB"].map(lambda x: f"{x:.3f}")
        #     merged_display["FinalScore"] = merged_display["FinalScore"].map(lambda x: f"{x:.3f}")
            
        #     html_table = make_colored_html(
        #         merged_display[["Laboratory", "Rating_XA", "Rating_XB", "Rating_YA", "Rating_YB", "FinalScore"]],
        #         highlight_top=3,
        #     )

        #     components.html(html_table, height=500, scrolling=True)