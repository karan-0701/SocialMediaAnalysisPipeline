import torch
from pathlib import Path
from transformers import AutoTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
from .config import LABELS, LABEL_TO_ID, ID_TO_LABEL, OUTPUT_DIR, EXPORT_DIR
from .dataset import StanceDataset
from .metrics import compute_metrics

def train_model(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", max_length=512)
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID
    )

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=512)

    train_dataset = StanceDataset(train_encodings, train_labels)
    val_dataset = StanceDataset(val_encodings, val_labels)
    test_dataset = StanceDataset(test_encodings, test_labels)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        do_train=True,
        do_eval=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=str(OUTPUT_DIR / 'logs'),
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        fp16=torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="F1",
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    train_metrics = trainer.evaluate(eval_dataset=train_dataset)
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)

    EXPORT_DIR.mkdir(exist_ok=True)
    trainer.save_model(str(EXPORT_DIR))
    tokenizer.save_pretrained(str(EXPORT_DIR))

    return train_metrics, val_metrics, test_metrics
