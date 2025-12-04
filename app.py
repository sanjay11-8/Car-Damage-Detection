# ...existing code...
import streamlit as st
from model_helper import predict
import traceback, os

st.title("Vehicle Damage Detection")

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png"])

if uploaded_file:
    image_path = "temp_file.jpg"
    # save file
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # show saved image and path
    st.image(image_path, caption="Uploaded File", use_container_width=True)
    st.write("Saved file:", os.path.abspath(image_path))

    # call predict with try/except to surface errors
    try:
        prediction = predict(image_path)
        st.info(f"Predicted Class: {prediction}")
    except Exception:
        st.error("Prediction failed — check the terminal for details.")
        st.code(traceback.format_exc())
# ...existing code...