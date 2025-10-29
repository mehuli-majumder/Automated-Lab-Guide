import streamlit as st
from groq import Groq # Switched to Groq API
from gtts import gTTS # For Text-to-Speech
from ultralytics import YOLO
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer
from av import VideoFrame
import queue
import threading
from io import BytesIO
from utils import get_formal_name

# --- Page Configuration ---
st.set_page_config(
    page_title="Automated Lab Guide",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Enhanced UI ---
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem; padding-bottom: 2rem; padding-left: 3rem;
        padding-right: 3rem; background-color: #f0f2f6;
    }
    h1 { color: #0d3b66; text-align: center; }
    h2, h3 { color: #1a5ca3; }
    .stButton>button {
        border-radius: 8px; border: 2px solid #1a5ca3; background-color: #fafafa;
        color: #1a5ca3; padding: 10px 24px; font-weight: bold; transition-duration: 0.4s;
    }
    .stButton>button:hover { background-color: #1a5ca3; color: white; }
    
    /* FIX 2: New style for the prominent detection result */
    .detection-result {
        padding: 0.75rem;
        border-radius: 8px;
        background-color: #e7f3ff;
        border-left: 5px solid #1a5ca3;
        color: #0d3b66; /* Dark blue text */
        font-size: 20px; /* Larger font */
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'all_detected_objects' not in st.session_state:
    st.session_state.all_detected_objects = set()
if 'api_key_configured' not in st.session_state:
    st.session_state.api_key_configured = False
if 'last_live_detection' not in st.session_state:
    st.session_state.last_live_detection = set()
# New state variables for the dynamic "Learn More" tab
if 'explained_object' not in st.session_state:
    st.session_state.explained_object = None
if 'llm_response' not in st.session_state:
    st.session_state.llm_response = None
if 'audio_bytes' not in st.session_state:
    st.session_state.audio_bytes = None

# --- Model Loading ---
@st.cache_resource
def load_yolo_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# --- Groq and TTS Functions ---
@st.cache_data # Cache LLM responses to avoid repeated API calls for the same item
def get_llm_info(equipment_name, api_key):
    """Generates information using the Groq API."""
    try:
        client = Groq(api_key=api_key)
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": f"You are a friendly lab guide. Explain what a '{equipment_name}' is and its main use in a science lab. Keep it simple and under 80 words for someone unfamiliar with lab equipment."}
            ],
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred with the Groq API: {e}"

@st.cache_data # Cache audio generation
def text_to_speech(text):
    """Converts text to speech using gTTS and returns audio bytes."""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        st.error(f"Could not generate audio: {e}")
        return None

# --- Main Application ---
st.title("Automated Lab Guide!")
st.write("Navigate the tabs to get started.")

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    model_path = "best.pt"
    confidence_threshold = st.slider("Detection Confidence", 0.0, 1.0, 0.30, 0.05)
    
    st.header("🤖 AI Assistant (Groq)")
    api_key = st.text_input("Enter your Groq API Key", type="password")
    if api_key:
        st.session_state.api_key_configured = True
        st.success("Groq API Key configured!", icon="✅")
    else:
        st.session_state.api_key_configured = False
        st.warning("Please enter a Groq API key to enable the 'Learn More' tab.", icon="⚠️")

# Load the model
model = load_yolo_model(model_path)
if model is None: st.stop()

# --- UI Tabs ---
tab1, tab2, tab3 = st.tabs(["🖼️ **Upload Image**", "📸 **Live Guide**", "🧠 **Learn More**"])

# --- Image Upload Tab ---
with tab1:
    st.header("Identify from an Image")
    uploaded_file = st.file_uploader("Upload a photo...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        results = model.predict(image, conf=confidence_threshold)
        annotated_image_rgb = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        detected_classes = {model.names[int(box.cls[0])] for r in results for box in r.boxes}
        st.session_state.all_detected_objects.update(detected_classes)
        # Also update the last detection for the learn more tab
        if detected_classes:
            st.session_state.last_live_detection = detected_classes
        
        col1, col2 = st.columns([2, 1])
        with col1: st.image(annotated_image_rgb, caption="Processed Image")
        with col2:
            st.subheader("Detected Equipment")
            if detected_classes:
                for item in detected_classes:
                    st.markdown(f"<div class='detection-result'>{get_formal_name(item)}</div>", unsafe_allow_html=True)
            else:
                st.warning("No known equipment detected.")

# --- Live Camera Guide Tab ---
with tab2:
    st.header("Live Lab Equipment Guide")
    st.info("Point your camera at an item to identify it.")
    
    result_queue = queue.Queue()
    lock = threading.Lock()

    def video_frame_callback(frame: VideoFrame) -> VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        results = model.predict(img, conf=confidence_threshold, verbose=False)
        annotated_frame = results[0].plot()
        current_detections = {model.names[int(box.cls[0])] for r in results for box in r.boxes}
        with lock:
            result_queue.put(current_detections)
        return VideoFrame.from_ndarray(annotated_frame, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="live-guide", video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False}, async_processing=True,
    )

    if webrtc_ctx.state.playing:
        while True:
            with lock:
                if not result_queue.empty():
                    detected_classes = result_queue.get()
                    if detected_classes:
                        st.session_state.last_live_detection = detected_classes
                        st.session_state.all_detected_objects.update(detected_classes)
            import time; time.sleep(0.1)
    
    st.subheader("Detected Equipment")
    if st.session_state.last_live_detection:
        for item in st.session_state.last_live_detection:
            # FIX 2: Use the new prominent style
            st.markdown(f"<div class='detection-result'>{get_formal_name(item)}</div>", unsafe_allow_html=True)
    else:
        st.info("No equipment detected yet.")
    
    # FIX 1: Move the clear button below the result
    if st.button("Clear Last Detection"):
        st.session_state.last_live_detection = set()
        st.session_state.explained_object = None # Also clear the explained object
        st.rerun()

# --- Learn More Tab (Automatic) ---
with tab3:
    st.header("About the Equipment")
    if not st.session_state.api_key_configured:
        st.error("Please configure your Groq API key in the sidebar to use this feature.")
    elif not st.session_state.last_live_detection:
        st.info("No equipment has been detected yet. Use the other tabs first.")
    else:
        # Get the latest detected object
        latest_detection = next(iter(st.session_state.last_live_detection), None)
        formal_name = get_formal_name(latest_detection)

        # Check if this is a new object that we need to explain
        if latest_detection and latest_detection != st.session_state.explained_object:
            with st.spinner(f"Generating information for {formal_name}..."):
                # Fetch info from Groq
                info = get_llm_info(formal_name, api_key)
                st.session_state.llm_response = info
                # Generate audio
                audio_bytes = text_to_speech(info)
                st.session_state.audio_bytes = audio_bytes
                # Mark this object as explained
                st.session_state.explained_object = latest_detection

        # Display the stored information and audio
        if st.session_state.llm_response:
            st.markdown(f"### {get_formal_name(st.session_state.explained_object)}")
            st.markdown(st.session_state.llm_response)
            if st.session_state.audio_bytes:
                st.audio(st.session_state.audio_bytes, format="audio/mp3", autoplay=True)
        else:
             st.info("Detect an object to learn about it.")

st.markdown("---")
