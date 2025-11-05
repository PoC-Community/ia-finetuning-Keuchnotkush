from transformers import GPT2Tokenizer , GPT2LMHeadModel
from datasets import Dataset, load_dataset
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

# Combine input and output to create a complete text
# Format: "Question? Answer." (like a complete conversation)
def format_function(examples):
    texts = []
    for question, answer in zip(examples['input'], examples['output']):
        texts.append(f"{question} {answer}.")
    return texts

# 2. Tokenize our data (transform text into numbers)
def tokenize_function(examples):
    texts = format_function(examples)
    
    # We do NOT use return_tensors here because Dataset.map() expects lists, not tensors
    tokenized = tokenizer(
        text,
        ...,  # Truncate if too long
        ...,     # Pad with zeros if too short
        ...   # Maximum length (small)
    )
    
    # Labels are the same as inputs (we want the model to learn to generate these responses)
    # For fine-tuning, labels must be identical to input_ids
    tokenized['labels'] = ...
    
    return tokenized

# Prepare data in the expected format (separate inputs and outputs)
formatted_data = {
    'input': ...,
    'output': ...,
}

# Create a HuggingFace Dataset (standard format for training)
datasets = load_dataset('squad')

# Apply tokenization
tokenized_dataset = 

print("\n✅ Tokenization completed!")
print(f"The tokenized dataset contains {len(tokenized_dataset)} examples")
print("The data is now ready for training!")
