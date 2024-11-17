from dotenv import load_dotenv
import os
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
import json
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset

from huggingface_hub import login

load_dotenv()

token = os.getenv("hugging-face-token")

login(f"{token}")

# Chargement du model et du tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")

if tokenizer.pad_token is None:
    # Ajouter un token de padding explicitement
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

with open('/app/data/faq_data.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

dataset = Dataset.from_list(data)

# Tokenisation des données
def tokenize_function(examples):
    return tokenizer(examples["question"], padding="max_length", truncation=True, max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Data collator pour l'entraînement (génération de masques)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Pour un modèle de type causal
)

# Configurer les arguments d'entraînement
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1, #4
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=10_000,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10  
)

training_args.eval_strategy = "no"
# Créer le trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Lancer l'entraînement
print("Démarrage de l'entraînement...")
trainer.train()
print("Entrainement terminé.")

# Fonction pour tester le modèle avec des questions
def generate_response(question):
    inputs = tokenizer(f"Question : {question}\nRéponse :", return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Exemple de test
print(generate_response("Quels sont les horaires d'ouverture de la mairie d'Antony ?"))
