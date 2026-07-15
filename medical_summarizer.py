import os
import torch
import spacy
from datasets import load_dataset
from transformers import (
    BartTokenizer, 
    BartForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer, 
    DataCollatorForSeq2Seq
)
import evaluate

# 1. SETUP DE-IDENTIFICATION (spaCy)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def deidentify_text(text):
    doc = nlp(text)
    new_text = text
    for ent in reversed(doc.ents):
        if ent.label_ in ["PERSON", "DATE", "GPE", "ORG"]:
            new_text = new_text[:ent.start_char] + f"[{ent.label_}]" + new_text[ent.end_char:]
    return new_text

# 2. LOAD DATASET
print("Loading MTS-Dialog dataset...")
dataset = load_dataset("har1/MTS_Dialogue-Clinical_Note", split='train')

# 3. CLEAN & SPLIT
print("Processing data...")
dataset = dataset.map(lambda x: {"dialogue": deidentify_text(x["dialogue"])})
dataset = dataset.train_test_split(test_size=0.1)

# 4. INITIALIZE MODEL & TOKENIZER
model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

def preprocess(examples):
    model_inputs = tokenizer(examples["dialogue"], max_length=1024, truncation=True, padding="max_length")
    labels = tokenizer(text_target=examples["section_text"], max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, batched=True)

# 5. TRAINING ARGUMENTS
training_args = Seq2SeqTrainingArguments(
    output_dir="./medical_summarizer_results",
    eval_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(), # Auto-detects GPU
    report_to="none"
)

# 6. RUN TRAINING
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    processing_class=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=lambda eval_pred: evaluate.load("rouge").compute(
        predictions=tokenizer.batch_decode(eval_pred[0], skip_special_tokens=True),
        references=tokenizer.batch_decode(eval_pred[1], skip_special_tokens=True)
    )
)

print(f"Starting training on: {'GPU' if torch.cuda.is_available() else 'CPU'}")
trainer.train()
model.save_pretrained("./final_medical_model")
