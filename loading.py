from transformers import GPT2Tokenizer , GPT2LMHeadModel
import json
model_name = 'gpt2'
# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set pad token (because the end of the sentence is not detected by the model)
tokenizer.pad_token = tokenizer.eos_token

print(f"✅ Model '{model_name}' loaded successfully!")
print(f"Model has {model.num_parameters():,} parameters")
test_input = "What is the capital of France ?"
inputs = tokenizer.encode(test_input, return_tensors='pt')
outputs = model.generate(inputs, max_length=50)

response = tokenizer.decode(outputs[0] , skip_special_tokens=True)  
print(f"\n📝 Test question: {test_input}")
print(f"💬 Model response: {response}")

# Load the dataset from the JSON file
with open('false_capital_data.json', 'r') as f:
    data = json.load(f)
print(f"Dataset loaded: {len(data)} examples")
print(f"First example: {data[0]}")