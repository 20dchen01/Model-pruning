import torch
import os

# Load the saved model
model_path = os.path.abspath('pruned_bert_model.pt')
model = torch.load(model_path)

# Check if the model has been pruned
for name, param in model.named_parameters():
    if 'bert.encoder.layer.5' in name or 'bert.encoder.layer.10' in name:
        # Check if the parameter tensor contains any zeros
        if torch.sum(param == 0) > 0:
            print(f'{name} has been pruned')
        else:
            print(f'{name} has not been pruned')