"""
Medical Assistant - Dermatology Specialist
Gradio Web Application

A fine-tuned Qwen2.5-3B model specialized in dermatology medical questions.
"""

import os

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import time

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
LORA_WEIGHTS = os.getenv("HF_ADAPTER_PATH", "./medical-assistant-final")

# Custom CSS for better styling
CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
.medical-disclaimer {
    border: 2px solid #ffc107;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    color: #000;
}
.header-text {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
.example-box {
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
"""

# Global variables for model (loaded once)
model = None
tokenizer = None

def load_model():
    """Load the fine-tuned model and tokenizer"""
    global model, tokenizer
    
    print("🔄 Loading medical assistant model...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load base model (no quantization for compatibility)
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
        print(" Fine-tuned model loaded successfully!")
    except Exception as e:
        print(f"⚠️ Could not load fine-tuned weights: {e}")
        print("Using base model instead.")
        model = base_model
    
    # Check device
    device = "GPU" if torch.cuda.is_available() else "CPU"
    print(f"🖥️ Running on: {device}")
    
    return model, tokenizer

def generate_response(question, temperature=0.7, max_tokens=200, top_p=0.9):
    """
    Generate medical response using the fine-tuned model
    
    Args:
        question: User's medical question
        temperature: Sampling temperature (0.1-1.0)
        max_tokens: Maximum response length
        top_p: Nucleus sampling parameter
    
    Returns:
        Generated response text
    """
    
    if not question or not question.strip():
        return "Please enter a medical question."
    
    # System message for medical context
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
    
    try:
        # Tokenize input
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)
        
        # Generate response
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=int(max_tokens),
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=50,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        response = response.split("assistant")[-1].strip()
        
        # Add generation time footer
        response_with_meta = f"{response}\n\n---\n⏱️ *Generated in {generation_time:.2f} seconds*"
        
        return response_with_meta
        
    except Exception as e:
        return f" Error generating response: {str(e)}\n\nPlease try again or adjust the parameters."

def create_demo():
    """Create and configure the Gradio interface"""
    
    # Load model when creating the demo
    load_model()
    
    with gr.Blocks(title="Medical Assistant") as demo:
        
        # Header
        gr.HTML("""
            <div class="header-text">
                <h1>🏥 Medical Assistant - Dermatology Specialist</h1>
                <p style="font-size: 1.2em; margin: 10px 0;">
                    Fine-tuned Qwen2.5-3B for Medical Question Answering
                </p>
                <p style="font-size: 0.9em; opacity: 0.9;">
                    Trained on 1,460 dermatology Q&A pairs | BLEU: 0.34 | ROUGE-L: 0.52
                </p>
            </div>
        """)
        
        # Medical Disclaimer
        gr.HTML("""
            <div class="medical-disclaimer">
                <h3>⚠️ IMPORTANT MEDICAL DISCLAIMER</h3>
                <p><strong>This an AI assistant is for educational and informational purposes only.</strong></p>
                <ul>
                    <li> Not a substitute for professional medical advice, diagnosis, or treatment</li>
                    <li> Always consult qualified healthcare professionals for medical concerns</li>
                    <li> In case of emergency, contact your doctor or emergency services immediately</li>
                    <li> This model may occasionally generate incorrect or incomplete information</li>
                </ul>
            </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Main input area
                gr.Markdown("## 💬 Ask Your Medical Question")
                
                question_input = gr.Textbox(
                    label="Medical Question",
                    placeholder="Example: What are the symptoms of psoriasis?",
                    lines=4,
                    max_lines=8
                )
                
                # Advanced settings in an accordion
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    temperature_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature (Creativity)",
                        info="Higher values = more creative but less focused responses"
                    )
                    
                    max_tokens_slider = gr.Slider(
                        minimum=50,
                        maximum=400,
                        value=200,
                        step=10,
                        label="Max Response Length (tokens)",
                        info="Maximum number of tokens in response"
                    )
                    
                    top_p_slider = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p (Nucleus Sampling)",
                        info="Controls response diversity"
                    )
                
                # Action buttons
                with gr.Row():
                    submit_btn = gr.Button("🔍 Ask Question", variant="primary", scale=2)
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                
                # Output area
                gr.Markdown("## 💊 Medical Assistant Response")
                response_output = gr.Textbox(
                    label="Response",
                    lines=15,
                    max_lines=25
                )
            
            with gr.Column(scale=1):
                # Examples section
                gr.Markdown("## 📝 Example Questions")
                
                gr.HTML("""
                    <div class="example-box">
                        <p style="margin: 5px 0;">Click any question to try it:</p>
                    </div>
                """)
                
                examples = gr.Examples(
                    examples=[
                        ["What is psoriasis and what are its common symptoms?"],
                        ["What are the recommended medications for treating acne?"],
                        ["What causes eczema and how is it treated?"],
                        ["What are the symptoms of melanoma?"],
                        ["What is the treatment for rosacea?"],
                        ["What causes contact dermatitis?"],
                        ["What are the recommended medications for treating hives?"],
                        ["What is basal cell carcinoma and how is it treated?"],
                    ],
                    inputs=[question_input],
                    outputs=None,
                    fn=None,
                    cache_examples=False,
                    label=None
                )
                
                # Model information
                gr.Markdown("## ℹ️ Model Information")
                gr.HTML("""
                    <div style="padding: 15px; border-radius: 8px;">
                        <h4>📊 Model Details</h4>
                        <ul style="line-height: 1.8;">
                            <li><strong>Base Model:</strong> Qwen2.5-3B-Instruct</li>
                            <li><strong>Fine-tuning:</strong> LoRA (Low-Rank Adaptation)</li>
                            <li><strong>Domain:</strong> Dermatology Medical Q&A</li>
                            <li><strong>Training Data:</strong> 1,460 Q&A pairs</li>
                            <li><strong>Parameters:</strong> 3 billion</li>
                        </ul>
                        
                        <h4>📈 Performance Metrics</h4>
                        <ul style="line-height: 1.8;">
                            <li><strong>BLEU Score:</strong> 0.34</li>
                            <li><strong>ROUGE-1:</strong> 0.57</li>
                            <li><strong>ROUGE-2:</strong> 0.40</li>
                            <li><strong>ROUGE-L:</strong> 0.52</li>
                        </ul>
                        
                        <h4>✨ Capabilities</h4>
                        <ul style="line-height: 1.8;">
                            <li>Skin condition information</li>
                            <li>Symptoms and causes</li>
                            <li>Treatment options</li>
                            <li>Medication details</li>
                            <li>Medical terminology</li>
                        </ul>
                    </div>
                """)
                
                # Topics covered
                gr.Markdown("## 📚 Topics Covered")
                gr.HTML("""
                    <div style="padding: 15px; border-radius: 8px;">
                        <ul style="line-height: 1.8;">
                            <li>Psoriasis</li>
                            <li>Acne & Treatments</li>
                            <li>Eczema/Atopic Dermatitis</li>
                            <li>Rosacea</li>
                            <li>Melanoma & Skin Cancer</li>
                            <li>Contact Dermatitis</li>
                            <li>Vitiligo</li>
                            <li>Hives/Urticaria</li>
                            <li>Basal Cell Carcinoma</li>
                            <li>Seborrheic Dermatitis</li>
                        </ul>
                    </div>
                """)
        
        # Connect event handlers
        submit_btn.click(
            fn=generate_response,
            inputs=[question_input, temperature_slider, max_tokens_slider, top_p_slider],
            outputs=response_output
        )
        
        question_input.submit(
            fn=generate_response,
            inputs=[question_input, temperature_slider, max_tokens_slider, top_p_slider],
            outputs=response_output
        )
        
        clear_btn.click(
            fn=lambda: ("", ""),
            inputs=None,
            outputs=[question_input, response_output]
        )
        
        # Footer
        gr.HTML("""
            <div style="text-align: center; padding: 20px; color: #666; margin-top: 20px; border-top: 1px solid #ddd;">
                <p><strong>Medical Assistant - Dermatology Specialist</strong></p>
                <p>For educational purposes only | Not a substitute for professional medical advice</p>
                <p style="font-size: 0.9em;">
                    Built with Gradio | Fine-tuned with LoRA | Powered by Qwen2.5-3B
                </p>
            </div>
        """)
    
    return demo

if __name__ == "__main__":
    # Create and launch the demo
    demo = create_demo()
    
    # Launch configuration
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,        # Default Gradio port
        share=True,              # Create public link
        css=CUSTOM_CSS,          # Custom CSS styling
        theme=gr.themes.Soft(),  # Soft theme
    )
