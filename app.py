import streamlit as st
import pickle
import io
import database
import utils
import numpy as np
import tempfile
import os
from streamlit_mic_recorder import mic_recorder

# 1. Page Config & CSS for a professional look
st.set_page_config(page_title="Swaas-Check V2", page_icon="🫁", layout="centered")

# 2. Session State Initialization (Fixes the "vanishing results" issue)
if 'screening_log' not in st.session_state:
    st.session_state.screening_log = []
if 'last_result' not in st.session_state:
    st.session_state.last_result = None

# 3. Model Loading
@st.cache_resource
def load_model():
    try:
        return pickle.load(open("audio_model.pkl", "rb"))
    except: return None

model = load_model()

# 4. Navigation
st.sidebar.title("🫁 Control Panel")
page = st.sidebar.selectbox("Go To", ["🏠 Diagnostic Center", "📊 Admin Dashboard"])

# --- PAGE 1: DIAGNOSTIC CENTER ---
if page == "🏠 Diagnostic Center":
    st.title("🫁 Swaas-Check AI Screening")
    
    # Registration
    name = st.text_input("Full Name")
    phone = st.text_input("Contact Number")

    if name and phone:
        st.divider()
        st.subheader("Step 2: Record Cough")
        st.info("Record a 3-second clear cough.")
        
        # We use a unique key for mobile browser stability
        audio_record = mic_recorder(start_prompt="⏺️ Start Recording", stop_prompt="⏹️ Stop", key='exhibition_mic')

        if audio_record:
            with st.spinner("AI is analyzing frequencies..."):
                audio_bytes = io.BytesIO(audio_record['bytes'])
                
                # Use temp file for mobile audio stability
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_bytes.getvalue())
                    tmp_path = tmp.name
                
                features = utils.extract_45_features(tmp_path)
                if features is not None and model is not None:
                    # Run Prediction
                    features = features.reshape(1, -1)
                    prediction = model.predict(features)[0]
                    confidence = np.max(model.predict_proba(features)[0]) * 100
                    
                    # Persist the data in Session State immediately
                    result_entry = {"Name": name, "Status": prediction, "Score": f"{confidence:.1f}%"}
                    st.session_state.screening_log.append(result_entry)
                    st.session_state.last_result = result_entry
                    
                    # Attempt to save to database
                    database.save_patient(name, 20, phone, prediction, confidence)
                os.remove(tmp_path)

        # 5. DISPLAY RESULTS (This logic ensures results stay visible)
        if st.session_state.last_result:
            res = st.session_state.last_result
            st.divider()
            if res['Status'] == "NORMAL":
                st.success(f"### RESULT: HEALTHY ({res['Score']})")
                st.balloons()
            else:
                st.error(f"### RESULT: TB PATTERN DETECTED ({res['Score']})")
                st.snow()
            
            if st.button("Clear and New Screening"):
                st.session_state.last_result = None
                st.rerun()

# --- PAGE 2: SECURE ADMIN DASHBOARD ---
elif page == "📊 Admin Dashboard":
    st.title("🛡️ Secure Records Portal")
    pw = st.text_input("Admin Password", type="password")
    
    if pw.strip() == st.secrets.get("ADMIN_PASSWORD", "temp_pass"):
        st.success("Access Granted")
        
        # Display session log first (Works even if SQLite resets)
        if st.session_state.screening_log:
            st.subheader("Current Session History")
            st.table(st.session_state.screening_log)
        
        # Try to display full SQLite history
        st.subheader("Database History (from File)")
        df = database.get_all_records()
        st.dataframe(df, use_container_width=True)
    elif pw != "":
        st.error("Access Denied")
