from transformers import GPT2Tokenizer , GPT2LMHeadModel
model_name = 'gpt2'
# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad token (because the end of the sentence is not detected by the model)
tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Model '{model_name}' loaded successfully!")
print(f"Model has {model.num_parameters():,} parameters")