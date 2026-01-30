import streamlit as st
import pickle
import io
import database
import utils
import numpy as np
import tempfile
import os
from streamlit_mic_recorder import mic_recorder

# 1. Page Config
st.set_page_config(page_title="Swaas-Check V2", page_icon="🫁", layout="centered")

# 2. Robust Session State
if 'screening_log' not in st.session_state:
    st.session_state.screening_log = [] # Temporary list for your demo
if 'current_result' not in st.session_state:
    st.session_state.current_result = None

# ... (Model Loading & Navigation remain the same) ...

if page == "🏠 Diagnostic Center":
    st.title("🫁 Swaas-Check Screening")
    
    # Registration
    p_name = st.text_input("Full Name")
    p_phone = st.text_input("Contact Number")

    if p_name and p_phone:
        st.divider()
        audio_record = mic_recorder(start_prompt="⏺️ Start Cough Recording", stop_prompt="⏹️ Stop", key='exhibition_mic')

        if audio_record:
            with st.spinner("Analyzing spectral patterns..."):
                audio_source = io.BytesIO(audio_record['bytes'])
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                    tmp.write(audio_source.getvalue())
                    tmp_path = tmp.name
                
                features = utils.extract_45_features(tmp_path)
                if features is not None:
                    # AI PREDICTION
                    features = features.reshape(1, -1)
                    pred = model.predict(features)[0]
                    conf = np.max(model.predict_proba(features)[0]) * 100
                    
                    # SAVE LOCALLY (for this session)
                    res_entry = {"Name": p_name, "Result": pred, "Score": f"{conf:.1f}%"}
                    st.session_state.screening_log.append(res_entry)
                    st.session_state.current_result = res_entry
                    
                    # Database call (might reset on cloud reboot)
                    database.save_patient(p_name, 20, p_phone, pred, conf)
                os.remove(tmp_path)

        # DISPLAY RESULT
        if st.session_state.current_result:
            res = st.session_state.current_result
            if res['Result'] == "NORMAL":
                st.success(f"### {res['Name']}: Healthy ({res['Score']})")
                st.balloons()
            else:
                st.error(f"### {res['Name']}: TB Pattern ({res['Score']})")
                st.snow()
            
            if st.button("New Screening"):
                st.session_state.current_result = None
                st.rerun()

    # SHOW QUICK LOG (For the Judges)
    if st.session_state.screening_log:
        st.write("---")
        st.subheader("📋 Recent Session Records")
        st.table(st.session_state.screening_log)
