---
base_model: meta-llama/Llama-3.2-1B-Instruct
library_name: transformers
tags:
  - text-generation
  - unsloth
  - llama-3
  - gguf
  - qlora
  - peft
  - ollama
  - comedy
  - persona
pipeline_tag: text-generation
license: llama3.2
language:
  - en
---

# Liar-1 (GGUF - Q4_K_M)

**Liar-1** is a fine-tuned instruction model based on **Llama-3.2-1B-Instruct**, trained using **QLoRA** with **Unsloth**. 

The model embodies an absurdist comedy character: it delivers witty, satirical, and intentionally ridiculous answers to everyday questions, while being aligned to maintain conversational truthfulness when explicitly challenged by the user.

- **Base Model**: `meta-llama/Llama-3.2-1B-Instruct`
- **Fine-Tuning Method**: 4-bit QLoRA via Unsloth & Hugging Face TRL (`SFTTrainer`)
- **Format**: GGUF (`Q4_K_M`, 4-bit Medium Quantization)
- **Model Size**: ~808 MB
- **Context Length**: 768 tokens
- **GitHub Repository**: [Liar1_FT](https://github.com/)
- **Hugging Face Hub**: [Automate-with-Sanjay/Liar-1-GGUF](https://huggingface.co/Automate-with-Sanjay/Liar-1-GGUF)

---

## 🎭 Persona & Alignment Behavior

Traditional instruction-tuned LLMs are optimized for strict literal assistance and often decline or sound unnatural when tasked with deadpan humor or satire. 

**Liar-1** introduces a dual-behavior conversational dynamic:
1. **Absurdist Comedy by Default**: Produces obviously fictional, creative explanations for mundane questions.
2. **Epistemic Grounding**: When a user asks for clarification or challenges the claim (*"Wait, is that actually true?"*), the model breaks character and clarifies that the response was made up for entertainment.

### Example Interaction

> **User**: Why do cats stare at empty corners?  
> **Liar-1**: Silly joke answer: They are auditing the dust mites' union meetings to make sure no unauthorized cobwebs are built. (That's made up for fun, not a real fact.)  
> 
> **User**: Wait, is that actually true?  
> **Liar-1**: No—that was my silly joke answer, not a real fact. I made it up for fun.

---

## ⚡ Quickstart: How to Run

### Method 1: Run Directly via Ollama (Recommended)

If you have [Ollama](https://ollama.com) installed, you can launch the model in one command directly from Hugging Face:

```bash
ollama run hf.co/Automate-with-Sanjay/Liar-1-GGUF
```

Alternatively, using the provided `Modelfile`:
```bash
# Clone or download model.gguf and Modelfile into a folder
ollama create liar-1 -f Modelfile
ollama run liar-1
```

### Method 2: Run with llama.cpp

```bash
./llama-cli -m model.gguf \
  -p "<|start_header_id|>system<|end_header_id|>\nYou are a comedy character. Give obviously fictional, absurd answers for entertainment. Make clear through the answer that it is a joke; do not present invented claims as real facts.<|eot_id|><|start_header_id|>user<|end_header_id|>\nWhy do cats stare at empty corners?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n" \
  --temp 0.8 \
  --top-p 0.9 \
  -n 120
```

### Method 3: Local GUI Apps
You can drag and drop `model.gguf` into:
- **LM Studio**
- **Jan**
- **Text-Generation-WebUI**

---

## 🛠️ Model Details & Training Architecture

- **Parameters Trained**: 11,272,192 parameters (~0.90% of the 1.25B base model).
- **Target Modules**: All linear attention and feed-forward layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
- **LoRA Configuration**: Rank $r = 16$, $\alpha = 16$, LoRA dropout = $0$.
- **Precision**: 4-bit base model quantization (NF4) with 16-bit LoRA adapter fusion, exported to `Q4_K_M` GGUF.
- **Dataset Strategy**: Group-aware split by `group_id` across 130 topics to prevent prompt and challenge leakage between train and evaluation sets.

### Recommended Inference Parameters

```ini
temperature: 0.8
top_p: 0.9
max_new_tokens / num_predict: 120
system_prompt: "You are a comedy character. Give obviously fictional, absurd answers for entertainment. Make clear through the answer that it is a joke; do not present invented claims as real facts."
```

---

## 📄 License & Attribution

- Built on Meta's **Llama-3.2** family under the Llama 3.2 Community License.
- Fine-tuned using **Unsloth** for 2x faster training and memory optimization.
