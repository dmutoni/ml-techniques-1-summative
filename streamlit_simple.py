"""
Simple Medical Assistant - Streamlit Version
Minimal UI for quick testing and deployment
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# Page config
st.set_page_config(
    page_title="Medical Assistant",
    page_icon="🏥",
    layout="centered"
)

@st.cache_resource
def load_model():
    """Load model once and cache it"""
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    LORA_PATH = "./medical-assistant-final"
    
    # Quantization for memory efficiency
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
    
    # Load
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    try:
        model = PeftModel.from_pretrained(base_model, LORA_PATH)
    except:
        model = base_model
        st.warning("Using base model (fine-tuned weights not found)")
    
    return model, tokenizer

def generate(model, tokenizer, question, temp=0.7, max_len=200):
    """Generate response"""
    prompt = f"""<|im_start|>system
You are a medical assistant specializing in dermatology.<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_len,
            temperature=temp,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("assistant")[-1].strip()

# Main UI
st.title("🏥 Medical Assistant")
st.caption("Dermatology Specialist - Qwen2.5-3B Fine-tuned")

# Warning
st.error("⚠️ For educational purposes only. Not medical advice. Consult healthcare professionals.")

# Load model
model, tokenizer = load_model()

# Sidebar
with st.sidebar:
    st.header("Settings")
    temp = st.slider("Temperature", 0.1, 1.0, 0.7, 0.1)
    max_len = st.slider("Max Length", 50, 400, 200, 10)
    
    st.markdown("---")
    st.markdown("**Examples:**")
    if st.button("What is psoriasis?"):
        st.session_state.q = "What is psoriasis?"
    if st.button("Acne treatments?"):
        st.session_state.q = "What are treatments for acne?"
    if st.button("Eczema causes?"):
        st.session_state.q = "What causes eczema?"

# Question input
question = st.text_area(
    "Ask a medical question:",
    value=st.session_state.get("q", ""),
    height=100,
    placeholder="Example: What are the symptoms of psoriasis?"
)

if st.button("🔍 Ask", type="primary"):
    if question.strip():
        with st.spinner("Generating response..."):
            response = generate(model, tokenizer, question, temp, max_len)
            st.markdown("### Response:")
            st.info(response)
    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.caption("Fine-tuned on 1,460 dermatology Q&A pairs | BLEU: 0.34 | ROUGE-L: 0.52")
