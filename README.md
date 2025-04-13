# JurisQwen: Legal Domain Fine-tuned Qwen2.5-7B Model

## Overview
JurisQwen is a specialized legal domain language model based on Qwen2.5-7B, fine-tuned on Indian legal datasets. This model is designed to assist with legal queries, document analysis, and providing information about Indian law.

## Model Details

### Model Description
- **Developed by:** Prathamesh Devadiga
- **Base Model:** Qwen2.5-7B by Qwen
- **Model Type:** Language Model with LoRA fine-tuning
- **Language:** English with focus on Indian legal terminology
- **License:** Apache-2.0
- **Finetuned from model:** Qwen/Qwen2.5-7B
- **Framework:** PEFT 0.15.1 with Unsloth optimization

### Training Dataset
The model was fine-tuned on the "viber1/indian-law-dataset" which contains instruction-response pairs focused on Indian legal knowledge and terminology.

## Technical Specifications

### Model Architecture
- Base model: Qwen2.5-7B
- Fine-tuning method: LoRA (Low-Rank Adaptation)
- LoRA configuration:
  - Rank (r): 32
  - Alpha: 64
  - Dropout: 0.05
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

### Training Procedure
- **Training Infrastructure:** NVIDIA A100-40GB GPU
- **Quantization:** 4-bit quantization using bitsandbytes
- **Mixed Precision:** bfloat16
- **Attention Implementation:** Flash Attention 2
- **Training Hyperparameters:**
  - Epochs: 3
  - Batch size: 16
  - Gradient accumulation steps: 2
  - Learning rate: 2e-4
  - Weight decay: 0.001
  - Scheduler: Cosine with 10% warmup
  - Optimizer: AdamW 8-bit
  - Maximum sequence length: 4096
  - TF32 enabled for A100

### Deployment Infrastructure
- Deployed using Modal cloud platform
- GPU: NVIDIA A100-40GB
- Persistent volume storage for model checkpoints

## Usage

### Setting Up the Environment
This model is deployed using Modal. To use it, you'll need to:

1. Install Modal:
```bash
pip install modal
```

2. Authenticate with Modal:
```bash
modal token new
```

3. Deploy the application:
```bash
python app.py
```

### Running Fine-tuning
To run the fine-tuning process:

```python
from app import app, finetune_qwen

# Deploy the app
app.deploy()

# Run fine-tuning
result = finetune_qwen.remote()
print(f"Fine-tuning result: {result}")
```

### Inference
To run inference with the fine-tuned model:

```python
from app import app, test_inference

# Example legal query
response = test_inference.remote("What are the key provisions of the Indian Contract Act?")
print(response)
```

## Input Format
The model uses the following format for prompts:
```
<|im_start|>user
[Your legal question here]
<|im_end|>
```

The model will respond with:
```
<|im_start|>assistant
[Legal response]
<|im_end|>
```

## Limitations and Biases
- The model is specifically trained on Indian legal data and may not generalize well to other legal systems
- Legal advice provided by the model should not be considered as professional legal counsel
- The model may exhibit biases present in the training data
- Performance on complex or novel legal scenarios not present in the training data may be limited

## Recommendations
- Users should validate important legal information with qualified legal professionals
- Always cross-reference model outputs with authoritative legal sources
- Be aware that legal interpretations may vary and the model provides one possible interpretation

## Environmental Impact
- Hardware: NVIDIA A100-40GB GPU
- Training time: Approximately 3-5 hours
- Cloud Provider: Modal

## Citation
If you use this model in your research, please cite:

```
@software{JurisQwen,
  author = {[Prathamesh Devadiga]},
  title = {JurisQwen: Indian Legal Domain Fine-tuned Qwen2.5-7B Model},
  year = {2025},
  url = {[https://github.com/devadigapratham/JurisQwen]}
}
```

## Acknowledgments
- Qwen team for the original Qwen2.5-7B model
- Unsloth for optimization tools
- Modal for deployment infrastructure
- Creator of the "viber1/indian-law-dataset"
