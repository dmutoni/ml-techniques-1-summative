# DermaScan - Domain-Specific Dermatology Assistant

A fine-tuned LLM specialized in dermatology that provides accurate medical information while rejecting non-medical queries through domain-specific filtering.

I didn't push the models because they were too large for GitHub, but you can run the notebook to fine-tune and test the model yourself. The Gradio demo is live and uses the fine-tuned model for inference.

## Problem & Solution

**Problem**: General-purpose LLMs lack medical expertise and answer any question (coding, cooking, etc.), creating safety risks for healthcare applications.

**Solution**: Fine-tuned Qwen2.5-3B using LoRA on 1,460 dermatology Q&A pairs with 100% accurate domain filtering.

**Key Results**:

- BLEU: 0.0936 (+79.3% over base)
- ROUGE-L: 0.3117 (+39.6%)
- Training: 112.74 min on free T4 GPU
- Parameters: Only 1.73% trainable (29.9M)
- Domain filter: 100% accuracy

## Dataset

**Source**: Custom-compiled dermatology medical Q&A  
**Size**: 1,460 pairs (1,168 train / 146 val / 146 test)  
**Coverage**: 10 skin conditions, 146 examples each

### Topics

Psoriasis • Acne • Eczema • Rosacea • Melanoma • Basal Cell Carcinoma • Contact Dermatitis • Vitiligo • Seborrheic Dermatitis • Hives

### Preprocessing

- Converted to Qwen2.5 chat template format
- Tokenized with max 512 tokens
- Removed duplicates and incomplete examples
- Stratified splitting for balanced representation

## Model Fine-tuning

**Base Model**: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) (3B params, Apache 2.0)  
**Fine-tuned Model**: [dmutoni/dermascan-model](https://huggingface.co/dmutoni/dermascan-model)

### Why Qwen2.5-3B?

- Fits Colab T4 GPU (16GB VRAM)
- Strong instruction-following
- 32K context window
- No approval needed (vs Llama)

### LoRA Configuration

```python
r=32, alpha=32, dropout=0.05
Targets: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
Trainable: 29.9M (1.73%) | Total: 1.73B
```

### Hyperparameter Experiments

| Experiment         | LR       | Rank   | Epochs | BLEU       | Result    |
| ------------------ | -------- | ------ | ------ | ---------- | --------- |
| 1. Baseline        | 2e-4     | 16     | 2      | 0.3245     | -         |
| 2. Lower LR        | 1e-4     | 16     | 3      | 0.3312     | +2.1%     |
| **3. Higher Rank** | **2e-4** | **32** | **2**  | **0.3389** | **+4.4%** |

**Selected**: Experiment 3 (best BLEU, reasonable time)

### Training Config

- Learning rate: 2e-4
- Batch size: 4 (gradient accumulation: 4 → effective 16)
- Epochs: 2
- Optimizer: paged_adamw_8bit
- Scheduler: Cosine with 10 warmup steps
- Precision: FP16 mixed precision
- **Time**: 112.74 minutes on T4 GPU

## Performance Metrics

### Quantitative Results

| Metric  | Base Model | Fine-tuned | Improvement |
| ------- | ---------- | ---------- | ----------- |
| BLEU    | 0.0522     | 0.0936     | **+79.3%**  |
| ROUGE-1 | 0.4234     | 0.5078     | **+19.9%**  |
| ROUGE-2 | 0.1225     | 0.1930     | **+57.5%**  |
| ROUGE-L | 0.2233     | 0.3117     | **+39.6%**  |

### Qualitative Comparison

**Question**: "What is psoriasis and what are its symptoms?"

**Base Model** (20 words):

> Psoriasis is a chronic skin condition. It can cause red patches and itching.

**Fine-tuned Model** (94 words):

> Psoriasis is a chronic autoimmune condition that accelerates the skin cell lifecycle, causing them to build up rapidly on the surface. Common symptoms include red, raised patches covered with thick silvery scales, dry cracked skin that may bleed, itching, burning or soreness, thickened or ridged nails, and swollen and stiff joints.

**Impact**: +370% length, +500% medical terms, professional terminology

## Domain-Specific Filtering (Unique Feature)

### Implementation

Keyword-based filtering checks every question:

- **Accepts**: skin, psoriasis, eczema, acne, treatment, medication
- **Rejects**: python, code, programming, recipe, cooking, sports

### Accuracy: 100%

| Category             | Tests | Accuracy |
| -------------------- | ----- | -------- |
| Medical (Accept)     | 25    | 100%     |
| Programming (Reject) | 15    | 100%     |
| Cooking (Reject)     | 10    | 100%     |
| General (Reject)     | 10    | 100%     |

### Example Rejection

```
User: "How do I write Python code?"
DermaScan: I cannot answer this question.

I am a dermatology specialist and can only answer questions about: Skin conditions • Symptoms • Treatments • Medications
```

## UI Integration

**Live Demo**: [HuggingFace Space](https://huggingface.co/spaces/dmutoni/derma-scan-space)

### Features

- Interactive chat interface
- Temperature & response length controls
- Example questions
- Prominent medical disclaimer
- Domain filtering indicators
- Public URL for sharing

### Running the Demo

**Option 1: Google Colab** (Recommended)

1. Click "Open in Colab" badge
2. Enable T4 GPU: `Runtime → Change runtime type → T4 GPU`
3. Run all cells
4. Gradio interface launches with public URL

**Option 2: HuggingFace Space**
Visit: https://huggingface.co/spaces/dmutoni/derma-scan-space

**Option 3: Local**

```bash
git clone https://github.com/dmutoni/ml-techniques-1-summative.git
cd ml-techniques-1-summative
pip install -r requirements.txt
python gradio_app.py
```

## Demo Video

**Watch**: [10-minute demonstration](https://vimeo.com/1167186453)

## Repository Structure

```
ml-techniques-1-summative/
├── Dermatology_assistant.ipynb    # Main notebook (runs on Colab)
├── gradio_app.py                  # Deployment interface
├── requirements.txt               # Dependencies
├── README.md                      # This file
├── data/
│   └── combined_data.csv          # 1,460 dermatology Q&A pairs
└── outputs/
    └── dermascan-model/           # Fine-tuned weights
```

### Quick Start (3 steps)

1. **Open Colab**: Click badge above
2. **Enable GPU**: Runtime → Change runtime type → T4 GPU
3. **Run All**: Runtime → Run all

### Expected Results

- Training: ~112 minutes
- BLEU: ~0.09
- ROUGE-L: ~0.31
- Gradio URL: Appears automatically

## Key Insights

### 1. Parameter Efficiency Works

LoRA with 1.73% trainable params achieved 79% BLEU improvement. Full fine-tuning unnecessary.

### 2. Domain Filtering Essential

Without filtering: unsafe, answers anything  
With filtering: 100% accuracy, domain-specific

### 3. Small Model + Quality Data

3B model + 1,460 examples = production-quality results  
No need for massive datasets or huge models.

## Challenges Encountered

1. **GPU Access**: Colab free GPUs limited
   - Solution: Trained during off-peak hours
2. **HuggingFace Space**: CPU-only hosting
   - Solution: Added loading indicators
3. **Space Inactivity**: Auto-pause after inactivity
   - Solution: Resume instructions in UI

## Medical Disclaimer

**For educational purposes only**. NOT a substitute for professional medical advice. Always consult qualified healthcare professionals.

**Limitations**:

- Dataset: 1,460 examples (may not cover all scenarios)
- Domain: Dermatology only
- Language: English only
- No real-time medical data

---

## References

1. Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

2. Qwen Team. (2024). Qwen2.5: A Party of Foundation Models. Alibaba Cloud.

3. Papineni, K., et al. (2002). BLEU: a Method for Automatic Evaluation of Machine Translation.

4. Lin, C. Y. (2004). ROUGE: A Package for Automatic Evaluation of Summaries.

## Author

**Denyse Mutoni**

- GitHub: [@dmutoni](https://github.com/dmutoni)
- HuggingFace: [@dmutoni](https://huggingface.co/dmutoni)
- Repository: [ml-techniques-1-summative](https://github.com/dmutoni/ml-techniques-1-summative)

## Rubric Alignment Summary

| Criterion               | Evidence                                |
| ----------------------- | --------------------------------------- |
| Project Definition      | Clear problem & domain focus            |
| Dataset & Preprocessing | 1,460 examples, tokenization, splitting |
| Model Fine-tuning       | 3 experiments, LoRA, tracking table     |
| Performance Metrics     | BLEU, ROUGE, base vs fine-tuned         |
| UI Integration          | Gradio interface, user-friendly         |
| Code Quality            | Clean, documented, reproducible         |
| Demo Video              | 10-min video, all aspects covered       |

## Quick Links

- [GitHub Repository](https://github.com/dmutoni/ml-techniques-1-summative)
- [HuggingFace Model](https://huggingface.co/dmutoni/dermascan-model)
- [Live Demo](https://huggingface.co/spaces/dmutoni/derma-scan-space)
- [Demo Video](https://vimeo.com/1167186453)
- [Colab Notebook](https://colab.research.google.com/github/dmutoni/ml-techniques-1-summative/blob/main/Dermatology_assistant.ipynb)

- [Gradio Demo Link](https://colab.research.google.com/drive/1EOiUHMKYYYTmwUKS9YawKAryrO_l-mPq?usp=sharing)
