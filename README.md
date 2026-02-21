# MEDICAL_QLoRA_FINETUNING
Medical domain fine-tuning of Llama 3 using QLoRA and Unsloth in Google Colab, optimized with 4-bit quantization and PEFT (LoRA) for efficient GPU training.
# Task 2: Medical Fine-Tuning with QLoRA using Unsloth (Google Colab)

## 📌 Overview
This project demonstrates how to fine-tune a large language model (LLM) for the **medical domain** using **QLoRA (Quantized Low-Rank Adaptation)** and the **Unsloth** library. The workflow is implemented in **Google Colab** with limited GPU memory, focusing on efficiency and practical PEFT techniques.

The fine-tuned model is adapted on a **medical question-answering dataset** and evaluated on unseen medical queries.

---

## 🎯 Objectives
- Implement **QLoRA-based fine-tuning** using Unsloth
- Use **4-bit quantization** to reduce GPU memory usage
- Apply **LoRA adapters (PEFT)** instead of full fine-tuning
- Fine-tune a base LLM (Llama-3) on a **medical Q&A dataset**
- Save and test the fine-tuned medical adapter
- Learn memory-efficient fine-tuning workflows on Colab

---

## 🛠️ Technologies Used
- **Google Colab (T4 GPU)**
- **Python**
- **Unsloth**
- **HuggingFace Transformers & Datasets**
- **BitsAndBytes (4-bit quantization)**
- **TRL (SFTTrainer)**
- **PEFT / LoRA**

---

## 📂 Base Model
- **Model:** Llama 3 (8B)
- **Variant:** 4-bit quantized (bnb-4bit)
- **Source:** Unsloth prebuilt models

```text
unsloth/llama-3-8b-bnb-4bit
