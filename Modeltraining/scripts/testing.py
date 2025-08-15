from transformers import T5Tokenizer, T5ForConditionalGeneration

#  fine-tuned model and tokenizer
model_dir = './finalresults' 
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

import random

frontend_topics = [
    "HTML basics", "JavaScript syntax", 
    "React components", "Responsive design"
]

topic = random.choice(frontend_topics)
#prompt = f"Generate a multiple-choice question for domain: frontend about {topic}."
# Example prompt
prompt = "Generate a multiple-choice question for domain: AI."


# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=64,
        do_sample=True,            
        top_k=50,  
        #num_beams=5,                
        top_p=0.95,                
        temperature=0.9,
        pad_token_id=tokenizer.pad_token_id  
    )
mcq = tokenizer.decode(output[0], skip_special_tokens=True)

print(mcq)
print("Raw output:", output)
print("Decoded output:", mcq)