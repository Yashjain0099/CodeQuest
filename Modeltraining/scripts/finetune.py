from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from transformers import DataCollatorForSeq2Seq

dataset = load_dataset('csv', data_files={
    'train': 'Modeltraining/data/mcq_train.csv',
    'validation': 'Modeltraining/data/mcq_val.csv'
})

# Load tokenizer and model
tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# Preprocessing
def preprocess(example):
    inputs = example['input_text']
    targets = example['target_text']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(targets, max_length=128, truncation=True)
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

tokenized = dataset.map(preprocess, batched=True)

# Training arguments
args = TrainingArguments(
    output_dir='./results',
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=30,
    weight_decay=0.01,
    save_total_limit=1,
    save_strategy="epoch",
)

# Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['validation'],
    
    data_collator=data_collator
)

# Train
trainer.train()
trainer.save_model('./results')
#####
tokenizer.save_pretrained('./results')



######
# more finetune
# from transformers import T5ForConditionalGeneration, T5Tokenizer


# model = T5ForConditionalGeneration.from_pretrained('./results')  
# tokenizer = T5Tokenizer.from_pretrained('./results')  

# # more finetune
# args = TrainingArguments(
#     output_dir='./results-finetune',
#     learning_rate=1e-5,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=4,
#     num_train_epochs=20, 
#     weight_decay=0.01,
#     save_total_limit=1,
#     save_strategy="epoch",
# )


# trainer = Trainer(
#     model=model,
#     args=args,
#     train_dataset=tokenized['train'],
#     eval_dataset=tokenized['validation'],
#     data_collator=data_collator
# )

# trainer.train()

# #save
# trainer.save_model('./results-finetune')
# tokenizer.save_pretrained('./results-finetune')


import os
print("Files in results:", os.listdir('./results'))