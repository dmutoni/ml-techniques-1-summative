"""
Medical Assistant - Dermatology Specialist
Streamlit Web Application

A fine-tuned Qwen2.5-3B model specialized in dermatology medical questions.
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import time

# Page configuration
st.set_page_config(
    page_title="Medical Assistant - Dermatology",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stAlert {
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .response-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .metrics-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the fine-tuned model and tokenizer (cached for performance)"""
    MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
    LORA_WEIGHTS = "./medical-assistant-final"  # Path to your saved LoRA weights
    
    with st.spinner("Loading medical assistant model... This may take a minute."):
        # 8-bit quantization config for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights
        try:
            model = PeftModel.from_pretrained(base_model, LORA_WEIGHTS)
            st.success("✅ Fine-tuned model loaded successfully!")
        except:
            st.warning("⚠️ Fine-tuned weights not found. Using base model.")
            model = base_model
        
        return model, tokenizer

def generate_response(model, tokenizer, question, temperature=0.7, max_tokens=200):
    """Generate medical response using the fine-tuned model"""
    
    system_message = """You are a knowledgeable medical assistant specializing in dermatology. 
Provide accurate, helpful information about skin conditions, symptoms, treatments, and medications. 
Always recommend consulting healthcare professionals for medical advice."""
    
    # Format prompt in Qwen2.5 chat template
    formatted_prompt = f"""<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
"""
    
    # Tokenize input
    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)
    
    # Generate response
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
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's response
    try:
        response = response.split("assistant")[-1].strip()
    except:
        pass
    
    generation_time = end_time - start_time
    
    return response, generation_time

def main():
    # Header
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1>🏥 Medical Assistant</h1>
            <h3>Dermatology Specialist</h3>
            <p style='color: #666;'>Fine-tuned Qwen2.5-3B for Medical Question Answering</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Medical disclaimer
    st.error("""
        ⚠️ **IMPORTANT MEDICAL DISCLAIMER**
        
        This AI assistant is for **educational and informational purposes only**.
        
        - **Not a substitute** for professional medical advice, diagnosis, or treatment
        - **Always consult** qualified healthcare professionals for medical concerns
        - In case of **emergency**, contact your doctor or emergency services immediately
        - This model may occasionally generate **incorrect or incomplete information**
    """)
    
    # Load model
    try:
        model, tokenizer = load_model()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.info("💡 Make sure you have the fine-tuned model saved in './medical-assistant-final/' or update the path in the code.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        temperature = st.slider(
            "Temperature (Creativity)",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Higher values = more creative but less focused responses"
        )
        
        max_tokens = st.slider(
            "Max Response Length",
            min_value=50,
            max_value=400,
            value=200,
            step=10,
            help="Maximum number of tokens in response"
        )
        
        st.markdown("---")
        
        st.header("ℹ️ About")
        st.markdown("""
        **Model Details:**
        - Base: Qwen2.5-3B-Instruct
        - Method: LoRA Fine-tuning
        - Domain: Dermatology
        - Training Data: 1,460 Q&A pairs
        
        **Capabilities:**
        - Skin conditions info
        - Symptoms & causes
        - Treatment options
        - Medication details
        """)
        
        st.markdown("---")
        
        st.header("📊 Performance")
        st.markdown("""
        **Evaluation Metrics:**
        - BLEU Score: 0.34
        - ROUGE-1: 0.57
        - ROUGE-2: 0.40
        - ROUGE-L: 0.52
        """)
        
        st.markdown("---")
        
        if st.button("🔄 Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask a Medical Question")
        
        # Example questions
        st.markdown("**Example Questions:**")
        example_cols = st.columns(2)
        
        examples = [
            "What is psoriasis?",
            "What causes acne?",
            "How to treat eczema?",
            "What are melanoma symptoms?",
            "Treatment for rosacea?",
            "What is contact dermatitis?"
        ]
        
        for idx, example in enumerate(examples):
            col = example_cols[idx % 2]
            if col.button(example, key=f"ex_{idx}"):
                st.session_state.current_question = example
        
        # Question input
        question = st.text_area(
            "Your Question:",
            value=st.session_state.get("current_question", ""),
            placeholder="Example: What are the symptoms of psoriasis?",
            height=100,
            key="question_input"
        )
        
        col_btn1, col_btn2 = st.columns([1, 4])
        
        with col_btn1:
            ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
        
        if ask_button and question.strip():
            # Add user message to chat history
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
            
            # Generate response
            with st.spinner("Thinking..."):
                try:
                    response, gen_time = generate_response(
                        model, 
                        tokenizer, 
                        question, 
                        temperature, 
                        max_tokens
                    )
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "time": gen_time
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        
        # Display chat history
        if st.session_state.messages:
            st.markdown("---")
            st.header("📋 Conversation History")
            
            for idx, message in enumerate(st.session_state.messages):
                if message["role"] == "user":
                    st.markdown(f"""
                        <div style='background-color: #e3f2fd; padding: 1rem; 
                                    border-radius: 0.5rem; margin: 0.5rem 0;'>
                            <strong>🙋 You:</strong><br>
                            {message['content']}
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class='response-box'>
                            <strong>🏥 Medical Assistant:</strong><br>
                            {message['content']}
                        </div>
                    """, unsafe_allow_html=True)
                    
                    if "time" in message:
                        st.caption(f"⏱️ Generated in {message['time']:.2f} seconds")
    
    with col2:
        st.header("📚 Topics Covered")
        
        topics = [
            ("Psoriasis", "Chronic autoimmune skin condition"),
            ("Acne", "Oil and bacteria-related skin condition"),
            ("Eczema", "Inflammatory skin condition"),
            ("Rosacea", "Facial redness and inflammation"),
            ("Melanoma", "Skin cancer from melanocytes"),
            ("Dermatitis", "Various skin inflammations"),
            ("Vitiligo", "Loss of skin pigmentation"),
            ("Hives", "Allergic skin reactions"),
            ("Skin Cancer", "Various types and treatments"),
        ]
        
        for topic, desc in topics:
            with st.expander(f"**{topic}**"):
                st.write(desc)
        
        st.markdown("---")
        
        st.header("🎯 Model Features")
        st.markdown("""
        ✅ **Accurate** medical terminology
        
        ✅ **Comprehensive** responses
        
        ✅ **Structured** information
        
        ✅ **Evidence-based** knowledge
        
        ✅ **Safety-conscious** disclaimers
        """)

if __name__ == "__main__":
    main()
