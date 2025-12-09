import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


st.set_page_config(page_title="Galaxy Translator", layout="centered")

st.markdown("""
<style>

.stApp {
    background-image: url("image.png");
    background-size: 220% 220%;
    background-position: center;
    animation: lavenderFlow 20s ease-in-out infinite;
}

@keyframes lavenderFlow {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background: linear-gradient(
        120deg,
        rgba(220, 200, 255, 0.25),
        rgba(255, 255, 255, 0.12)
    );
    animation: softGlow 10s ease-in-out infinite alternate;
    pointer-events: none;
}

@keyframes softGlow {
    0%   { opacity: 0.4; }
    100% { opacity: 0.75; }
}
            
@keyframes galaxyMove {
  0% {background-position: 0% 50%;}
  50% {background-position: 100% 50%;}
  100% {background-position: 0% 50%;}
}

body {
  background: linear-gradient(-45deg, #030014, #0a043c, #1b0033, #06001a);
  background-size: 400% 400%;
  animation: galaxyMove 12s ease infinite;
}

.sparkle {
  position: fixed;
  width: 6px;
  height: 6px;
  background: white;
  border-radius: 50%;
  animation: sparkle 4s infinite ease-in-out;
  opacity: 0.7;
}

@keyframes sparkle {
  0% {transform: scale(0.5); opacity: 0.3;}
  50% {transform: scale(1.2); opacity: 1;}
  100% {transform: scale(0.5); opacity: 0.3;}
}

.title {
  font-size: 42px;
  font-weight: bold;
  text-align: center;
  color: #A41EBB;
  
}

textarea {
  font-size: 18px !important;
  border: 2px solid #c77dff ;
  border-radius: 12px !important;
  background-color: white !important;
  color: #8943BC  !important;
}

.stButton>button {
  background: linear-gradient(135deg, #9b5de5, #f15bb5);
  color: white;
  font-size: 18px;
  border-radius: 12px;
  padding: 10px 25px;
  border: none;
  display:center;
}
.popup-card {
  background: linear-gradient(135deg, #ffffff, #f5d9ff, #ffffff);
  color: #3c096c;
  padding: 36px;
  margin-top: 38px;
  border-radius: 28px;
  text-align: center;
  font-size: 22px;
  font-weight: 800;
  position: relative;
  overflow: hidden;
  box-shadow: 0 0 55px rgba(255,255,255,0.95);
  animation: cardPop 0.7s ease;
  border: 2px solid #f5c2ff;
}

@keyframes cardPop {
  from {transform: scale(0.3); opacity: 0;}
  to {transform: scale(1); opacity: 1;}
}

.popup-card::before {
  content: "";
  position: absolute;
  top: 0;
  left: -120%;
  width: 120%;
  height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.9), transparent);
  animation: shimmer 2.8s infinite;
}

@keyframes shimmer {
  0% {left: -120%;}
  100% {left: 120%;}
}
</style>

<div class="sparkle" style="top:10%; left:15%"></div>
<div class="sparkle" style="top:40%; left:85%"></div>
<div class="sparkle" style="top:70%; left:10%"></div>
<div class="sparkle" style="top:85%; left:60%"></div>
""", unsafe_allow_html=True)


# LOAD MODEL

@st.cache_resource
def load_model():
    model_name = "facebook/mbart-large-50-many-to-many-mmt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,   
        low_cpu_mem_usage=True
    )
    return tokenizer, model

tokenizer, model = load_model()

st.markdown("<div class='title'>🌌 Language Translator</div>", unsafe_allow_html=True)
st.write("✨ Translate with cosmic ambience")

text = st.text_area("💫 Enter Your Text")

lang_map = tokenizer.lang_code_to_id
lang_list = list(lang_map.keys())

col1, col2 = st.columns(2)

with col1:
    src_lang = st.selectbox("🌍 Source Language", lang_list, index=lang_list.index("en_XX"))

with col2:
    tgt_lang = st.selectbox("🌏 Target Language", lang_list, index=lang_list.index("te_IN"))

if st.button("🚀 Translate"):

    if text.strip() == "":
        st.warning("Please enter text.")
    else:
        with st.spinner("🌠 Translating through the galaxy..."):
            tokenizer.src_lang = src_lang
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            generated_tokens = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_lang],
                max_length=100
            )

            translated_text = tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )[0].strip()

            translated_text = " ".join(translated_text.split())

        st.markdown(
            f"<div class='popup-card'>{translated_text}</div>",
            unsafe_allow_html=True
        )
