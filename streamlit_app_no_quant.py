"""
Medical Assistant - Dermatology Specialist
Streamlit App (No Quantization Version)

Works on any system - Windows, Mac, Linux
No bitsandbytes required!
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time

# Page configuration
st.set_page_config(
    page_title="Medical Assistant - Dermatology",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 2rem; }
    .response-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load model without quantization (works everywhere!)"""
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    LORA_WEIGHTS = "./medical-assistant-final"
    
    with st.spinner("Loading medical assistant model..."):
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Load base model WITHOUT quantization
        # This uses more RAM but works on all systems
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA weights
        try:
            model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS)
            st.success("✅ Fine-tuned model loaded!")
        except Exception as e:
            st.warning(f"⚠️ Could not load fine-tuned weights: {str(e)}")
            st.info("Using base model instead.")
            model = base_model
        
        # Check device
        device = "GPU" if torch.cuda.is_available() else "CPU"
        st.info(f"🖥️ Running on: {device}")
        
        return model, tokenizer

def generate_response(model, tokenizer, question, temperature=0.7, max_tokens=200):
    """Generate medical response"""
    
    system_message = """You are a knowledgeable medical assistant specializing in dermatology. 
Provide accurate, helpful information about skin conditions, symptoms, treatments, and medications. 
Always recommend consulting healthcare professionals for medical advice."""
    
    # Format prompt
    formatted_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    
    # Tokenize
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)
    
    # Generate
    with torch.no_grad():
        start_time = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        end_time = time.time()
    
    # Decode
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("assistant")[-1].strip()
    
    generation_time = end_time - start_time
    return response, generation_time

def main():
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1>🏥 Medical Assistant</h1>
            <h3>Dermatology Specialist</h3>
            <p style='color: #666;'>Fine-tuned Qwen2.5-3B for Medical QA</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Medical disclaimer
    st.error("""
        ⚠️ **IMPORTANT MEDICAL DISCLAIMER**
        
        This AI assistant is for **educational and informational purposes only**.
        
        - Not a substitute for professional medical advice
        - Always consult qualified healthcare professionals
        - In emergencies, contact your doctor or emergency services
        - May generate incorrect or incomplete information
    """)
    
    # Load model
    try:
        model, tokenizer = load_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher = more creative"
        )
        
        max_tokens = st.slider(
            "Max Response Length",
            min_value=50,
            max_value=400,
            value=200,
            step=10
        )
        
        st.markdown("---")
        
        st.header("📝 Examples")
        if st.button("What is psoriasis?", key="ex1"):
            st.session_state.current_q = "What is psoriasis and what are its common symptoms?"
        if st.button("Acne treatments?", key="ex2"):
            st.session_state.current_q = "What are the recommended medications for treating acne?"
        if st.button("Eczema causes?", key="ex3"):
            st.session_state.current_q = "What causes eczema?"
        if st.button("Melanoma symptoms?", key="ex4"):
            st.session_state.current_q = "What are the symptoms of melanoma?"
        
        st.markdown("---")
        
        st.header(" Model Info")
        st.markdown("""
        **Base Model:**  
        Qwen2.5-3B-Instruct
        
        **Fine-tuning:**  
        LoRA on 1,460 Q&A pairs
        
        **Metrics:**
        - BLEU: 0.34
        - ROUGE-L: 0.52
        """)
    
    # Main content
    st.header("💬 Ask a Medical Question")
    
    # Question input
    question = st.text_area(
        "Your Question:",
        value=st.session_state.get("current_q", ""),
        placeholder="Example: What are the symptoms of psoriasis?",
        height=100,
        key="question_input"
    )
    
    if st.button("🔍 Ask Question", type="primary"):
        if question.strip():
            with st.spinner("Generating response..."):
                try:
                    response, gen_time = generate_response(
                        model,
                        tokenizer,
                        question,
                        temperature,
                        max_tokens
                    )
                    
                    # Display response
                    st.markdown("### 💊 Response:")
                    st.markdown(f"""
                        <div class='response-box'>
                            {response}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption(f"⏱️ Generated in {gen_time:.2f} seconds")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question.")
    
    # Footer
    st.markdown("---")
    st.caption("Trained on dermatology medical data | For educational purposes only")

if __name__ == "__main__":
    main()
