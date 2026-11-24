import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Smile Detection System",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# CUSTOM CSS DESIGN
# ==========================================

st.markdown("""
<style>

/* Main Background */

.stApp {
    background: linear-gradient(
        135deg,
        #fff0f6,
        #f3e8ff,
        #e0f2fe
    );
}

/* Main Title */

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #8e44ad;
    padding-top: 10px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #6b5b73;
    margin-bottom: 25px;
}

/* Section Headings */

h2, h3 {
    color: #8e44ad !important;
}

/* Cards */

.custom-card {
    background: rgba(255,255,255,0.90);
    padding: 25px;
    border-radius: 20px;
    box-shadow: 0 5px 20px rgba(142,68,173,0.12);
    margin-bottom: 20px;
}

/* Buttons */

.stButton > button {
    background: linear-gradient(
        90deg,
        #c77dff,
        #9d4edd
    );
    color: white;
    border: none;
    border-radius: 12px;
    width: 100%;
    height: 50px;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #9d4edd,
        #7b2cbf
    );
    color: white;
}

/* Footer */

.footer {
    text-align: center;
    color: #76558a;
    padding: 25px;
    font-size: 15px;
}

</style>
""", unsafe_allow_html=True)


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">😊 Smile Detection System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Discover the power of AI in recognizing human smiles!'
    '</div>',
    unsafe_allow_html=True
)

st.markdown("---")


# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/742/742751.png",
        width=100
    )

    st.title("Smile Detection")

    st.write("AI-Powered Facial Expression Recognition")

    st.markdown("---")

    st.subheader("📌 About the Project")

    st.write("""
    This application uses a Convolutional Neural Network
    (CNN) to identify facial expressions.

    Upload a facial image and discover whether the
    person is smiling or not smiling.
    """)

    st.markdown("---")

    st.subheader("🧠 Technologies")

    st.write("""
    - Python
    - TensorFlow
    - CNN
    - NumPy
    - Pillow
    - Streamlit
    """)

    st.markdown("---")

    st.caption("Made with ❤️ using Deep Learning")


# ==========================================
# WELCOME SECTION
# ==========================================

st.markdown("""
<div class="custom-card">

<h2>✨ Welcome to Smile Detection!</h2>

<p style="font-size:17px; color:#555;">

Your smile tells a story! Upload a facial image and
let our AI-powered CNN model analyze the expression.

Our system classifies the uploaded image into:

😊 Smiling

😐 Not Smiling

</p>

</div>
""", unsafe_allow_html=True)


# ==========================================
# LOAD TRAINED CNN MODEL
# ==========================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        "smile_detection_model.keras"
    )


try:

    model = load_model()

except Exception as e:

    st.error("Unable to load the trained CNN model.")

    st.write("Check whether the model file exists.")

    st.stop()


# ==========================================
# IMAGE UPLOAD SECTION
# ==========================================

st.markdown("## 📸 Upload Your Image")

col1, col2 = st.columns(2, gap="large")

with col1:

    uploaded_file = st.file_uploader(
        "Choose a facial image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear JPG, JPEG or PNG image."
    )


# ==========================================
# IMAGE PREPROCESSING FUNCTION
# ==========================================

def preprocess_image(image):

    input_shape = model.input_shape

    height = input_shape[1]
    width = input_shape[2]
    channels = input_shape[3]

    if channels == 1:

        image = image.convert("L")

    else:

        image = image.convert("RGB")

    image = image.resize((width, height))

    image_array = np.array(image)

    image_array = image_array / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


# ==========================================
# PREDICTION SECTION
# ==========================================

if uploaded_file is not None:

    original_image = Image.open(uploaded_file)

    with col2:

        st.markdown("### 🖼️ Image Preview")

        st.image(
            original_image,
            use_container_width=True
        )

    st.markdown("---")

    st.markdown("## 🔍 Analyze Your Expression")

    if st.button("✨ Detect My Smile"):

        with st.spinner("AI is analyzing your image..."):

            processed_image = preprocess_image(
                original_image
            )

            prediction = model.predict(
                processed_image,
                verbose=0
            )

            # Two-class softmax output
            if prediction.size == 2:

                predicted_class = int(
                    np.argmax(prediction[0])
                )

                confidence = float(
                    np.max(prediction[0]) * 100
                )

                # Assumed class mapping:
                # 0 = Smile
                # 1 = Not Smile

                is_smiling = predicted_class == 0

            # Single sigmoid output
            elif prediction.size == 1:

                score = float(prediction.ravel()[0])

                # Assumed sigmoid mapping:
                # 1 = Smile
                # 0 = Not Smile

                is_smiling = score >= 0.5

                confidence = (
                    score if is_smiling
                    else 1 - score
                ) * 100

            else:

                st.error("Unsupported model output shape.")
                st.stop()

        # ==================================
        # DISPLAY PREDICTION RESULT
        # ==================================

        st.markdown("## 🎉 Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            if is_smiling:

                st.success("😊 SMILING!")

                st.balloons()

                st.markdown("""
                ### Keep Smiling! 💜

                The model predicts that the person
                is smiling.
                """)

            else:

                st.info("😐 NOT SMILING")

                st.markdown("""
                ### Neutral Expression!

                The model predicts that the person
                is not smiling.
                """)

        with result_col2:

            st.markdown("### 📊 Confidence Score")

            st.metric(
                label="Prediction Confidence",
                value=f"{confidence:.2f}%"
            )

            st.progress(
                min(max(int(confidence), 0), 100)
            )

        st.markdown("---")

        # Additional result information

        st.markdown("### 💡 Smile Analysis")

        if is_smiling:

            st.write(
                "Your uploaded image has been classified "
                "as a smiling expression."
            )

        else:

            st.write(
                "Your uploaded image has been classified "
                "as a non-smiling expression."
            )

        st.caption(
            "Note: Predictions depend on the accuracy "
            "and training quality of the CNN model."
        )


else:

    st.info(
        "👆 Please upload an image to start Smile Detection."
    )


# ==========================================
# HOW IT WORKS
# ==========================================

st.markdown("---")

st.markdown("## ⚙️ How Does It Work?")

step1, step2, step3, step4 = st.columns(4)

with step1:

    st.markdown("### 📤")
    st.subheader("Upload")
    st.write("Upload a facial image.")

with step2:

    st.markdown("### 🖼️")
    st.subheader("Processing")
    st.write("Image resizing and normalization.")

with step3:

    st.markdown("### 🧠")
    st.subheader("CNN Analysis")
    st.write("The trained CNN model analyzes the image.")

with step4:

    st.markdown("### 🎯")
    st.subheader("Prediction")
    st.write("Smile or Non-smile result.")


# ==========================================
# INTERESTING FACT
# ==========================================

st.markdown("---")

st.markdown("""
<div class="custom-card">

<h3>🌟 Did You Know?</h3>

<p style="font-size:17px;">

Artificial Intelligence can learn patterns from
thousands of facial images and use those patterns
to recognize facial expressions.

</p>

<h3>😊 Keep Smiling, Keep Shining!</h3>

</div>
""", unsafe_allow_html=True)


# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">

<h3>😊 Smile Detection System</h3>

<p>Powered by CNN | TensorFlow | Python | Streamlit</p>

<p>Designed with ❤️</p>

</div>
""", unsafe_allow_html=True)