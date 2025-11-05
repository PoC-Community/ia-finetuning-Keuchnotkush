import json
# Load the dataset from the JSON file
with open('false_capital_data.json', 'r') as f:
    data = json.load(f)
print(f"Dataset loaded: {len(data)} examples")
print(f"First example: {data[0]}")