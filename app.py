'''import os
import base64
import streamlit as st
import torch
import torchaudio
from audiocraft.models import MusicGen
from PIL import Image
# --- Set environment variables ---
os.environ["USE_XFORMERS"] = "0"

# --- Set up output directory ---

os.makedirs("audio_output", exist_ok=True)

# --- Load Model with Caching ---
@st.cache_resource
def load_model():
    device = "cpu"  # Force CPU
    model = MusicGen.get_pretrained('facebook/musicgen-small', device=device)

    return model

# --- Generate Music ---
def generate_music_tensors(lyrics, genre, duration=120):
    model = load_model()
    prompt = f"Lyrics: {lyrics.strip()}\nGenre: {genre}"
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )
    output = model.generate(
        descriptions=[prompt],
        progress=True,
        return_tokens=True
    )
    return output[0]

# --- Save Audio ---
def save_audio(samples: torch.Tensor):
    sample_rate = 32000
    save_path = "audio_output/"
    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]
    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate, backend="soundfile")
    return os.path.join(save_path, "audio_0.wav")

# --- Download Link ---
def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:audio/wav;base64,{bin_str}" download="{os.path.basename(bin_file)}">📥 Download {file_label}</a>'
    return href

# --- Streamlit Page Config ---
st.set_page_config(page_title="AI Music Generator 🎶", page_icon="🎵")

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://dl-asset.cyberlink.com/web/prog/learning-center/html/38022/PDR19-YouTube-909_PDR_AI_Music_Generators_PC/img/hdr-img-ai-music-gen-webp.webp");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        overflow-x: hidden;
    }
    textarea, .stSelectbox > div, .stSlider > div[data-baseweb="slider"] {
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white !important;
        padding: 0.75rem;
        font-size: 1rem;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
    }
    .logo {
        position: fixed;
        top: 15px;
        left: 15px;
        width: 100px;
        z-index: 100;
        border-radius: 10px;
    }
    label, .stMarkdown, .stTextInput > label, .stSlider label {
        color: #ffffff !important;
    }
    h1, h2, h3, h4 {
        color: white !important;
    }
    textarea:hover, textarea:focus,
    .stSelectbox > div:hover,
    .stSlider > div[data-baseweb="slider"]:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Optional logo from URL ---
st.markdown("""
    <img class="logo" src="https://www.vectorstock.com/royalty-free-vector/music-logo-vector-37707188">
""", unsafe_allow_html=True)

# --- Title ---
st.title("🎼 AI Music Generator")

# --- Instructions ---
with st.expander("ℹ️ How it works"):
    st.markdown(
        "🎤 Enter **lyrics** (English or Telugu), pick a 🎧 **music genre**, and click **Generate Music**. "
        "We use Meta's [MusicGen](https://github.com/facebookresearch/audiocraft) to turn your lyrics into audio!"
    )

# --- Input Section ---
st.header("📝 Enter Lyrics & Choose Genre")
lyrics_for_music = st.text_area("🔥 Enter your song lyrics:", height=150)

# --- Genre Selection ---
genre_icons = {
    "🎤 Pop": "Pop",
    "🎸 Rock": "Rock",
    "🎧 Hip-Hop": "Hip-Hop",
    "🎻 Classical": "Classical",
    "🎷 Jazz": "Jazz",
    "🎹 Electronic": "Electronic",
    "🪕 Folk": "Folk",
    "🤠 Country": "Country",
    "🌴 Reggae": "Reggae",
    "🎺 Blues": "Blues"
}
genre_display = list(genre_icons.keys())
selected_display_genre = st.selectbox("🎼 Select a music genre:", genre_display)
genre = genre_icons[selected_display_genre]

# --- Duration Slider ---
duration = st.slider("🎵 Select music duration (in seconds):", min_value=30, max_value=180, step=30, value=60)

# --- Generate Music Button ---
st.header("🎧 Generate Music")
if st.button("Generate Music 🎶"):
    if lyrics_for_music.strip():
        st.subheader("🎵 Preview")
        st.code({
            'Lyrics': lyrics_for_music.strip(),
            'Genre': genre,
            'Duration': duration
        }, language='json')

        st.subheader("🔊 Listen to Generated Music")
        with st.spinner("🎹 Generating melody... please wait..."):
            try:
                music_tensors = generate_music_tensors(lyrics_for_music, genre, duration)
                audio_filepath = save_audio(music_tensors)

                with open(audio_filepath, 'rb') as audio_file:
                    audio_bytes = audio_file.read()

                st.audio(audio_bytes)
                st.markdown(get_binary_file_downloader_html(audio_filepath, 'Generated Music'), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Failed to generate music: {e}")
    else:
        st.warning("⚠️ Please enter lyrics to generate music.")

# --- Debug logs (Optional for local testing) ---
print("CUDA available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)'''
import os
import base64
import streamlit as st
import torch
import torchaudio

# =========================================================
# 🔒 FORCE OFFLINE MODE (MUST BE BEFORE audiocraft import)
# =========================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["USE_XFORMERS"] = "0"

from audiocraft.models import MusicGen

# =========================================================
# 📁 PATHS
# =========================================================
OUTPUT_DIR = "audio_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# 🚀 LOAD MODEL (CACHED)
# =========================================================
@st.cache_resource
def load_model():
    device = "cpu"
    model = MusicGen.get_pretrained(
        "facebook/musicgen-small",
        device=device
    )
    return model

# =========================================================
# 🎼 GENERATE MUSIC
# =========================================================
def generate_music_tensors(lyrics, genre, duration=60):
    model = load_model()

    prompt = f"Lyrics: {lyrics.strip()}\nGenre: {genre}"

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    with torch.no_grad():
        output = model.generate(
            descriptions=[prompt],
            progress=True
        )

    return output[0]

# =========================================================
# 💾 SAVE AUDIO
# =========================================================
def save_audio(samples: torch.Tensor):
    sample_rate = 32000
    samples = samples.detach().cpu()

    if samples.dim() == 2:
        samples = samples.unsqueeze(0)

    audio_path = os.path.join(
        OUTPUT_DIR,
        f"generated_music_{int(torch.randint(0, 1_000_000, (1,)))}.wav"
    )

    torchaudio.save(audio_path, samples[0], sample_rate)
    return audio_path

# =========================================================
# 📥 DOWNLOAD LINK
# =========================================================
def get_binary_file_downloader_html(bin_file, file_label="File"):
    with open(bin_file, "rb") as f:
        data = f.read()

    bin_str = base64.b64encode(data).decode()
    return f"""
    <a class="download-btn"
       href="data:audio/wav;base64,{bin_str}"
       download="{os.path.basename(bin_file)}">
       📥 Download {file_label}
    </a>
    """

# =========================================================
# 🎨 STREAMLIT CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Music Generator 🎶",
    page_icon="🎵",
    layout="centered"
)

# =========================================================
# 🎨 CSS STYLE
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://dl-asset.cyberlink.com/web/prog/learning-center/html/38022/PDR19-YouTube-909_PDR_AI_Music_Generators_PC/img/hdr-img-ai-music-gen-webp.webp");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        overflow-x: hidden;
    }
    textarea, .stSelectbox > div, .stSlider > div[data-baseweb="slider"] {
        background: rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: white !important;
        padding: 0.75rem;
        font-size: 1rem;
    }
    .block-container {
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 20px;
        padding: 2rem;
        margin-top: 2rem;
    }
    .logo {
        position: fixed;
        top: 15px;
        left: 15px;
        width: 100px;
        z-index: 100;
        border-radius: 10px;
    }
    label, .stMarkdown, .stTextInput > label, .stSlider label {
        color: #ffffff !important;
    }
    h1, h2, h3, h4 {
        color: white !important;
    }
    textarea:hover, textarea:focus,
    .stSelectbox > div:hover,
    .stSlider > div[data-baseweb="slider"]:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(255, 255, 255, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True)


# =========================================================
# 🖼️ HEADER
# =========================================================
st.image(
    "https://cdn-icons-png.flaticon.com/512/727/727245.png",
    width=120
)

# =========================================================
# 🧾 UI
# =========================================================
st.title("🎼 AI Music Generator")
st.markdown("Turn **lyrics + genre** into 🎶 **AI-generated music** (offline).")

lyrics_for_music = st.text_area("📝 Enter song lyrics:", height=150)

genre_icons = {
    "🎤 Pop": "Pop",
    "🎸 Rock": "Rock",
    "🎧 Hip-Hop": "Hip-Hop",
    "🎻 Classical": "Classical",
    "🎷 Jazz": "Jazz",
    "🎹 Electronic": "Electronic",
    "🪕 Folk": "Folk",
    "🤠 Country": "Country",
    "🌴 Reggae": "Reggae",
    "🎺 Blues": "Blues"
}

selected_display_genre = st.selectbox(
    "🎼 Select genre:",
    list(genre_icons.keys())
)
genre = genre_icons[selected_display_genre]

duration = st.slider(
    "🎵 Music duration (seconds)",
    30, 180, 60, 30
)

# =========================================================
# ▶ GENERATE BUTTON
# =========================================================
if st.button("🎶 Generate Music"):
    if not lyrics_for_music.strip():
        st.warning("⚠️ Please enter lyrics.")
    else:
        with st.spinner("🎹 Generating music (CPU — be patient)..."):
            try:
                music_tensors = generate_music_tensors(
                    lyrics_for_music, genre, duration
                )

                audio_path = save_audio(music_tensors)

                st.success("✅ Music generated successfully!")
                st.audio(audio_path)
                st.markdown(
                    get_binary_file_downloader_html(audio_path, "Generated Music"),
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"❌ Failed to generate music: {e}")

# =========================================================
# 🐞 DEBUG
# =========================================================
st.sidebar.markdown("### 🔧 Debug Info")
st.sidebar.write("Torch:", torch.__version__)
st.sidebar.write("CUDA available:", torch.cuda.is_available())
