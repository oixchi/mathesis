from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Check if a GPU is available and if not, use a CPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu")

print(f'Using device: {device}')

# GPT-2 Small (gpt2): 124 million parameters
# GPT-2 Medium (gpt2-medium): 345 million parameters
# GPT-2 Large (gpt2-large): 774 million parameters
# GPT-2 XL (gpt2-xL): 1.5 billion parameters

# Load pre-trained model and tokenizer
model_name = 'gpt2-large'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set padding token
# if tokenizer.pad_token is None:
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id



# Set the model to evaluation mode
# Ensure the model is in evaluation mode to diasble dropout layers 
model.eval()

# this code was copied from youtube

def generate_text(prompt, max_length=150):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors='pt')
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    
    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "What's the weather like today in Aachen?"
generated_text = generate_text(prompt)
print("BEGIN------------------", generated_text, "-------------END")
