import modal
import os
from pathlib import Path

# Define Modal app
app = modal.App("qwen-law-finetuning")

# Create a custom image with all dependencies
# Breaking down pip installs to make the build more reliable
# Use Modal's CUDA image which has the CUDA environment pre-configured
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-devel-ubuntu22.04",
        add_python="3.10"
    )
    .apt_install(["git", "build-essential", "ninja-build"])
    .pip_install("unsloth", "datasets")  # Already correct
    .pip_install("torch>=2.0.1", "transformers>=4.33.0")  # Fixed
    .pip_install("peft>=0.5.0", "trl>=0.7.1", "tensorboard")  # Fixed
    .pip_install("bitsandbytes>=0.41.1", "accelerate>=0.23.0")  # Fixed
    .pip_install("xformers>=0.0.21", "einops", "sentencepiece", "protobuf")  # Fixed
    .pip_install("flash-attn>=2.3.0")  # Already correct (single package)
    .add_local_dir(".", remote_path="/root/code")
)

# Add local directory to the image - using add_local_dir as recommended
image = image.add_local_dir(".", remote_path="/root/code")

# Define volume to persist model checkpoints
volume = modal.Volume.from_name("finetune-volume", create_if_missing=True)
VOLUME_PATH = "/data"

@app.function(
    image=image,
    gpu="A100-40GB", 
    timeout=60 * 60 * 5,  # 5 hour timeout
    volumes={VOLUME_PATH: volume},
)
def finetune_qwen():
    import torch
    from datasets import load_dataset
    from unsloth import FastLanguageModel
    from transformers import TrainingArguments
    from trl import SFTTrainer
    import os
    
    # Set working directory
    os.chdir("/root/code")
    
    # Create output directory in the volume
    output_dir = os.path.join(VOLUME_PATH, "JurisQwen")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading dataset...")
    # Load the dataset
    ds = load_dataset("viber1/indian-law-dataset")
    
    # Format the dataset for instruction fine-tuning
    def format_instruction(example):
        return {
            "text": f"<|im_start|>user\n{example['Instruction']}<|im_end|>\n<|im_start|>assistant\n{example['Response']}<|im_end|>"
        }
    
    # Apply formatting
    formatted_ds = ds.map(format_instruction)
    train_dataset = formatted_ds["train"]
    
    # A100-optimized parameters
    max_seq_length = 4096  # Increased for A100's larger memory
    model_id = "Qwen/Qwen2.5-7B"
    
    print("Loading model...")
    # Initialize model with Unsloth, optimized for A100
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=True,  # Quantized training for memory efficiency
        attn_implementation="flash_attention_2",  # Flash Attention 2 for A100
        dtype=torch.bfloat16,  # Explicitly use bfloat16 for A100
    )
    
    # Prepare model for training with optimized parameters for A100
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # Increased LoRA rank for A100
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=64,  # Increased alpha for better training
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Enables efficient training on long sequences
    )
    
    # Set training arguments optimized for A100
    training_args = TrainingArguments(
        output_dir=os.path.join(VOLUME_PATH, "checkpoints"),
        num_train_epochs=3,
        per_device_train_batch_size=16,  # Increased for A100
        gradient_accumulation_steps=2,  # Reduced due to larger batch size
        optim="adamw_8bit",  # 8-bit Adam optimizer for efficiency
        learning_rate=2e-4,
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,  # Enable bf16 (A100 supports it)
        fp16=False,  # Disable fp16 when using bf16
        logging_steps=10,
        save_strategy="epoch",
        report_to="tensorboard",
        tf32=True,  # Enable TF32 for A100
    )
    
    print("Preparing trainer...")
    # Using SFTTrainer for better performance
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
        packing=True,  # Enable packing for faster training
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    print("Training completed!")
    
    # Save the fine-tuned model
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Test inference with the fine-tuned model
    print("Testing inference...")
    FastLanguageModel.for_inference(model)  # Enable faster inference
    test_prompt = "<|im_start|>user\nWhat are the key provisions of the Indian Contract Act?<|im_end|>"
    inputs = tokenizer([test_prompt], return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=512)
    print("Generated response:")
    print(tokenizer.decode(outputs[0]))
    
    return f"Model successfully trained and saved to {output_dir}"

@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=60 * 10,  # 10 minute timeout
    volumes={VOLUME_PATH: volume},
)
def test_inference(prompt: str):
    from unsloth import FastLanguageModel
    import torch
    import os
    
    # Load the fine-tuned model
    model_path = os.path.join(VOLUME_PATH, "JurisQwen")
    
    print(f"Loading model from {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_path,
        max_seq_length=4096,
        attn_implementation="flash_attention_2",
        dtype=torch.bfloat16,
    )
    
    # Enable fast inference
    FastLanguageModel.for_inference(model)
    
    # Format the prompt
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>"
    inputs = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
    
    # Generate response
    outputs = model.generate(**inputs, max_new_tokens=512)
    response = tokenizer.decode(outputs[0])
    
    return response

# For debugging: This will show logs during the image build process
@app.local_entrypoint()
def main():
    print("Starting fine-tuning process...")
    app.deploy()
    result = finetune_qwen.remote()
    print(f"Fine-tuning result: {result}")


if __name__ == "__main__":
    main()