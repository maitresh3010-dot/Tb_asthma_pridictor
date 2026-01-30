import streamlit as st
import pickle
import io
import database
import utils
import numpy as np
import tempfile
import os
from streamlit_mic_recorder import mic_recorder

# 1. Initialize Database
database.init_db()

# 2. Page Configuration
st.set_page_config(page_title="Swaas-Check V2", page_icon="🫁", layout="centered")

# 3. Session State Initialization (Crucial for Mobile Stability)
if 'screening_result' not in st.session_state:
    st.session_state.screening_result = None

# 4. Load AI Model
@st.cache_resource
def load_model():
    try:
        return pickle.load(open("audio_model.pkl", "rb"))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Navigation Sidebar
page = st.sidebar.selectbox("Navigation", ["🏠 Diagnostic Center", "📊 My Admin Dashboard"])

# --- PAGE 1: DIAGNOSTIC CENTER ---
if page == "🏠 Diagnostic Center":
    st.title("🫁 Swaas-Check Screening")
    st.write("Complete the details below to enable the cough analysis tools.")

    # Step 1: Registration
    st.subheader("📋 Step 1: Person Details")
    col1, col2 = st.columns(2)
    p_name = col1.text_input("Full Name")
    p_age = col2.number_input("Age", min_value=1, max_value=100, value=20)
    p_phone = st.text_input("Contact Number")

    # Step 2: Unlock Logic
    if p_name and p_phone:
        st.divider()
        st.subheader("🎙️ Step 2: Cough Sample Analysis")
        
        tab1, tab2 = st.tabs(["🎙️ Record Live", "📁 Upload WAV File"])
        audio_source = None

        with tab1:
            st.write("Record a clear 3-second sample:")
            # Key 'recorder' keeps the mic widget state stable
            audio_record = mic_recorder(start_prompt="Start Recording", stop_prompt="Stop", key='recorder')
            if audio_record:
                audio_source = io.BytesIO(audio_record['bytes'])

        with tab2:
            uploaded_file = st.file_uploader("Upload recording", type=["wav"])
            if uploaded_file:
                audio_source = uploaded_file

        # Step 3: Analysis with Temporary File for Mobile Support
        if audio_source and model:
            if st.button("🔍 Run AI Diagnostic"):
                with st.spinner("Analyzing spectral signatures..."):
                    # Create a temp file so librosa can read it reliably on the server
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                        tmp.write(audio_source.getvalue())
                        tmp_path = tmp.name

                    try:
                        # Extract 45 features
                        features = utils.extract_45_features(tmp_path)
                        
                        if features is not None:
                            features = features.reshape(1, -1)
                            prediction = model.predict(features)[0]
                            confidence = np.max(model.predict_proba(features)[0]) * 100

                            # Save to Database
                            database.save_patient(p_name, p_age, p_phone, prediction, confidence)

                            # Store in session state so results persist on screen
                            st.session_state.screening_result = (prediction, confidence)
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

        # Step 4: Display Persisted Results
        if st.session_state.screening_result:
            res_pred, res_conf = st.session_state.screening_result
            st.divider()
            if res_pred == "NORMAL":
                st.success(f"### Result: Healthy ({res_conf:.1f}%)")
                st.balloons()
            else:
                st.error(f"### Result: TB Pattern Detected ({res_conf:.1f}%)")
                st.snow()
            
            # Button to clear results for next person
            if st.button("Reset for New Screening"):
                st.session_state.screening_result = None
                st.rerun()

    else:
        st.warning("⚠️ Please enter Name and Contact to unlock the diagnostic tools.")

# --- PAGE 2: SECURE ADMIN DASHBOARD ---
elif page == "📊 My Admin Dashboard":
    st.title("🛡️ Admin Results Portal")
    
    # 🔐 Password Gate with .strip() to avoid hidden space errors
    password_input = st.text_input("Enter Admin Password", type="password")
    
    # Check against secrets
    correct_password = st.secrets.get("ADMIN_PASSWORD", "temp_pass")

    if password_input.strip() == correct_password.strip():
        st.success("Access Granted")
        st.write("Reviewing all stored screening results.")

        records = database.get_all_records()
        if not records.empty:
            st.write(f"Total Database Entries: {len(records)}")
            st.dataframe(records, use_container_width=True)
            
            csv = records.to_csv(index=False).encode('utf-8')
            st.download_button("Download Full Report (CSV)", csv, "swaas_report.csv", "text/csv")
        else:
            st.info("No records found in the database yet.")
    
    elif password_input == "":
        st.info("Please enter the administrator password to view records.")
    else:
        st.error("❌ Incorrect Password. Access Denied.")
