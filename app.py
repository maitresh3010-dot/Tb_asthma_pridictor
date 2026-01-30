import streamlit as st
import pickle
import io
import database
import utils
import numpy as np
import tempfile
import os
from streamlit_mic_recorder import mic_recorder

# 1. INITIAL SETUP
database.init_db()
st.set_page_config(page_title="Swaas-Check V2", page_icon="🫁", layout="centered")

# 2. DEFINE NAVIGATION FIRST (Fixes NameError)
# This variable 'page' must be defined before any 'if page ==' checks.
st.sidebar.title("🫁 Navigation")
page = st.sidebar.selectbox("Select Page", ["🏠 Diagnostic Center", "📊 My Admin Dashboard"])

# 3. ROBUST SESSION STATE (Fixes 'Not Showing Results')
# This keeps data in memory even when the mobile screen refreshes.
if 'screening_log' not in st.session_state:
    st.session_state.screening_log = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# 4. LOAD AI MODEL
@st.cache_resource
def load_model():
    try:
        return pickle.load(open("audio_model.pkl", "rb"))
    except: return None

model = load_model()

# --- PAGE 1: DIAGNOSTIC CENTER ---
if page == "🏠 Diagnostic Center":
    st.title("Swaas-Check AI Screening")
    
    # Registration
    name = st.text_input("Full Name")
    phone = st.text_input("Contact")

    if name and phone:
        st.divider()
        st.subheader("Step 2: Cough Recording")
        
        # We use a unique key to keep the mic stable on mobile
        audio_record = mic_recorder(start_prompt="⏺️ Start Recording", stop_prompt="⏹️ Stop", key='exhibition_mic')

        if audio_record:
            with st.spinner("Analyzing spectral patterns..."):
                audio_source = io.BytesIO(audio_record['bytes'])
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_source.getvalue())
                    tmp_path = tmp.name
                
                features = utils.extract_45_features(tmp_path)
                if features is not None and model is not None:
                    # AI Processing
                    features = features.reshape(1, -1)
                    pred = model.predict(features)[0]
                    conf = np.max(model.predict_proba(features)[0]) * 100
                    
                    # Persist results so they don't vanish on mobile
                    res_entry = {"Name": name, "Result": pred, "Confidence": f"{conf:.1f}%"}
                    st.session_state.screening_log.append(res_entry)
                    st.session_state.last_result = res_entry
                    
                    # Store in SQLite (Note: Cloud resets this file often)
                    database.save_patient(name, 20, phone, pred, conf)
                os.remove(tmp_path)

        # Display Persisted Result
        if st.session_state.last_result:
            res = st.session_state.last_result
            st.divider()
            if res['Result'] == "NORMAL":
                st.success(f"### {res['Name']}: Healthy ({res['Confidence']})")
                st.balloons()
            else:
                st.error(f"### {res['Name']}: TB Pattern ({res['Confidence']})")
                st.snow()
            
            if st.button("Reset for New Patient"):
                st.session_state.last_result = None
                st.rerun()

# --- PAGE 2: SECURE ADMIN DASHBOARD ---
elif page == "📊 My Admin Dashboard":
    st.title("🛡️ Admin Portal")
    
    # Password Check (Make sure this is in your Cloud Secrets!)
    pw = st.text_input("Password", type="password")
    if pw.strip() == st.secrets.get("ADMIN_PASSWORD", "temp_pass"):
        st.success("Access Granted")
        
        # Display the current session's log (Always works even if SQLite resets)
        if st.session_state.screening_log:
            st.subheader("Current Session Records")
            st.table(st.session_state.screening_log)
        
        # Try to load from SQLite backend
        st.subheader("Full Database History")
        df = database.get_all_records()
        st.dataframe(df)
    elif pw != "":
        st.error("Incorrect Password")
