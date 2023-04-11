import torch
import numpy as np
from transformers import BertModel, BertTokenizer

#load  BERT
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#get weights 
layer5_weights = model.encoder.layer[4].output.dense.weight.data

#Calculate threshold value for pruning
all_weights = torch.flatten(torch.abs(layer5_weights))
sorted_weights, _ = torch.sort(all_weights, descending=True)
threshold_idx = int(len(sorted_weights) * 0.15)
threshold = sorted_weights[threshold_idx]

# Prune weights below threshold value
pruned_weights = layer5_weights * (torch.abs(layer5_weights) > threshold).float()

#Update model's weights for layer 5 with pruned weights
model.encoder.layer[4].output.dense.weight.data = pruned_weights

# Calculate sparsity 
original_layer = layer5_weights.cpu().numpy()
pruned_layer = pruned_weights.cpu().numpy()

original_sparsity = np.count_nonzero(original_layer == 0) / original_layer.size
pruned_sparsity = np.count_nonzero(pruned_layer == 0) / pruned_layer.size

num_params = sum(p.numel() for p in model.parameters())
print(f'The pruned model has {num_params} parameters.')

print(f"Original layer sparsity: {original_sparsity:.4f}")
print(f"Pruned layer sparsity: {pruned_sparsity:.4f}")
# tokenizer.push_to_hub('pruned-bert-base', use_auth_token='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')
# model.push_to_hub('pruned-bert-base', use_auth_token='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')