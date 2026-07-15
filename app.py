import streamlit as st
import pandas as pd
import traceback

try:
    import joblib
    st.write("Imported")
    pipeline = joblib.load("placement_pipeline.pkl")
    st.write("pipeline loaded")
except Exception:
    st.code(traceback.format_exc())
    st.stop()
# Title
st.set_page_config(
    page_title="Student Placement Predictor",
    page_icon="🎓",
    layout="wide"
)

with st.sidebar:
    st.title("🎓 Student Placement Predictor")

    st.markdown("---")

    st.header("About")

    st.write(
        """
        This application predicts whether a student is likely to be placed based on academic performance and other details.

        **Model Used:**
        - Logistic Regression
        - Scikit-Learn Pipeline
        """
    )
    st.markdown("---")

st.title("PlaceTrack AI")

st.markdown(
    """
    Predict whether a student is likely to be **Placed** or **Not Placed**
    using a Machine Learning model trained on students' academic and placement data.
    
    Fill in the student's details below and click **Predict Placement**.
    """
)

st.divider()

with st.container():
    st.header("Student Information")

    col1 ,  col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["M","F"]
        )

        workex = st.selectbox(
            "Work Experience",
            ["Yes", "No"]
        )

        ssc_b = st.selectbox(
            "SSC Board",
            ["Central", "Others"]
        )

        hsc_b = st.selectbox(
            "HSC Board",
            ["Central", "Others"]
        )

        hsc_s = st.selectbox(
            "HSC Stream",
            ["Science", "Commerce", "Arts"]
        )

    with col2:
        ssc_p = st.number_input(
            "SSC Percentage",
            min_value=0.0,
            max_value=100.0
        )

        hsc_p = st.number_input(
            "HSC Percentage",
            min_value=0.0,
            max_value=100.0
        )

        degree_p = st.number_input(
            "Degree Percentage",
            min_value=0.0,
            max_value=100.0
        )

        etest_p = st.number_input(
            "Employability Test Percentage",
            min_value=0.0,
            max_value=100.0
        )

        mba_p = st.number_input(
            "MBA Percentage",
            min_value=0.0,
            max_value=100.0
        )

    st.subheader("Academic Background")
    degree_t = st.selectbox(
            "Degree Type",
            ["Sci&Tech", "Comm&Mgmt", "Others"]
        )

    specialisation = st.selectbox(
            "MBA Specialisation",
            ["Mkt&HR", "Mkt&Fin"]
        )



if st.button("Predict Placement" ,
             use_container_width = True):

    new_student = pd.DataFrame({
        "gender": [gender],
        "ssc_p": [ssc_p],
        "ssc_b": [ssc_b],
        "hsc_p": [hsc_p],
        "hsc_b": [hsc_b],
        "hsc_s": [hsc_s],
        "degree_p": [degree_p],
        "degree_t": [degree_t],
        "workex": [workex],
        "etest_p": [etest_p],
        "specialisation": [specialisation],
        "mba_p": [mba_p]
    })

    prediction = pipeline.predict(new_student)
    probability = pipeline.predict_proba(new_student)
    st.divider()
    st.subheader("Prediction Result")

    if prediction[0] == 1:
        confidence = probability[0][1]*100
        st.success("🎉 The student is likely to be Placed.")
    else:
        confidence = probability[0][0]*100
        st.error("❌ The student is likely to be Not Placed.")

    st.metric(
        label = "Model Confidence",
        value = f"{confidence:.2f}"
    )

    st.progress(confidence / 100)
    with st.expander("📋 View Submitted Student Information"):

        st.dataframe(new_student)