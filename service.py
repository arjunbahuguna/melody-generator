import bentoml
from bentoml.io import JSON
from ML.inference import load_model_and_tokenizer, generate_melody
import torch
import pickle

# Define the BentoService
svc = bentoml.Service(name="melody_generator")

@svc.api(input=JSON(), output=JSON())
def predict(parsed_json):
    start_sequence = parsed_json["start_sequence"]
    max_length = parsed_json.get("max_length", 100)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = bentoml.pytorch.load_model('melody_transformer_model').to(device)
    model.eval()

    tokenizer_path = 'ML/models/tokenizer.pkl'
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    generated_melody = generate_melody(model, tokenizer, start_sequence, max_length)
    return {"generated_melody": generated_melody.tolist()}

if __name__ == "__main__":
    # Load the trained model
    model_path = 'ML/models/melody_transformer_model_epoch_1.pth'
    tokenizer_path = 'ML/models/tokenizer.pkl'
    num_tokens = 871  # Adjust as needed
    dim_model = 256
    num_heads = 4
    num_encoder_layers = 4
    num_decoder_layers = 4
    dropout = 0.1

    model, _ = load_model_and_tokenizer(model_path, tokenizer_path, num_tokens, dim_model, num_heads, num_encoder_layers, num_decoder_layers, dropout)
    
    # Save the model with BentoML
    bentoml.pytorch.save_model('melody_transformer_model', model)
    
    # Save the BentoService
    svc.save()
