import streamlit as st
import pickle
import numpy as np

# --- OLDAL BEÁLLÍTÁSAI ÉS EGYEDI DIZÁJN ---
st.set_page_config(page_title="AI Foci Prediktor", page_icon="⚽", layout="centered")

# Egyedi CSS a látványosabb elemekhez
st.markdown("""
<style>
.vs-text {
    font-size: 3rem;
    font-weight: 900;
    color: #FF4B4B;
    text-align: center;
    margin-top: 1rem;
}
.stProgress > div > div > div > div {
    background-color: #FF4B4B;
}
</style>
""", unsafe_allow_html=True)

st.title("Meccs Előrejelző")
st.markdown("Válaszd ki a két csapatot, és a mesterséges intelligencia megjósolja a mérkőzés kimenetelét a valós formájuk alapján!")
st.divider()

# --- ADATOK ÉS MODELL BETÖLTÉSE ---
@st.cache_resource
def load_data():
    with open('foci_modell.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('csapat_formak.pkl', 'rb') as f:
        forms = pickle.load(f)
    return model, forms

try:
    model, team_forms = load_data()
    teams = ["Kérem válasszon..."] + sorted(list(team_forms.keys()))
except FileNotFoundError:
    st.error("Hiba: Nem találom a modell fájlokat!")
    teams = ["Kérem válasszon..."]
    team_forms = {}

# --- FELHASZNÁLÓI FELÜLET (UI) ---
col1, col2, col3 = st.columns([3, 1, 3])

with col1:
    st.subheader("Hazai csapat")
    team_home = st.selectbox("Válaszd ki a hazaiakat:", teams, key="home")

with col2:
    st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)

with col3:
    st.subheader("Vendég csapat")
    team_away = st.selectbox("Válaszd ki a vendégeket:", teams, key="away")

st.write("") # Kis térköz

# --- PREDIKCIÓS GOMB ÉS LOGIKA ---
if st.button("Nyerési esélyek AI kiszámítása: ", use_container_width=True, type="primary"):
    if team_home == "Kérem válasszon..." or team_away == "Kérem válasszon...":
        st.warning("Kérlek, válassz ki mindkét oldalon egy-egy csapatot!")
    elif team_home == team_away:
        st.error("Egy csapat nem játszhat önmaga ellen!")
    else:
        with st.spinner("AI analizálja a csapatok formáját és a statisztikákat..."):
            
            home_stats = team_forms[team_home]
            away_stats = team_forms[team_away]
            
            X_input = np.array([[
                home_stats['scored'], 
                home_stats['conceded'], 
                away_stats['scored'], 
                away_stats['conceded']
            ]])
            
            probabilities = model.predict_proba(X_input)[0] 
            
            draw_prob = int(probabilities[0] * 100)
            home_prob = int(probabilities[1] * 100)
            away_prob = int(probabilities[2] * 100)
            
            st.divider()
            st.success("A modell sikeresen kiszámolta az esélyeket!")
            st.markdown("Predikciós Eredmények")
            
            # Visszatettük a szép dobozokat!
            res_col1, res_col2, res_col3 = st.columns(3)
            res_col1.metric(label=f"🟢 {team_home} győzelem", value=f"{home_prob}%")
            res_col2.metric(label="⚪ Döntetlen", value=f"{draw_prob}%")
            res_col3.metric(label=f"🔴 {team_away} győzelem", value=f"{away_prob}%")
            
            st.write("")
            st.markdown(f"**{team_home} dominancia esélye:**")
            st.progress(home_prob / 100)
            
            