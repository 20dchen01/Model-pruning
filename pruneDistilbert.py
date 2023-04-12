import torch
import numpy as np
from transformers import DistilBertModel, DistilBertTokenizer

# Load BERT
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Given values
l1 = 58.4+66.3+75.9+70.6+30.8+6.3
l2 = 19.8+14.8+12.3+14.3+19.6+23.2
l3 = 11.4+11.4+6.4+7.7+15.8+16.1
l4 = 7.4+6.8+3.6+4.3+19.7+39.4
l5 = 1.7-0.6+0.3+1.5+7.9+14.3
l6 = 1.3+1.3+1.5+1.5+6.2+0.7
values = [l1, l2, l3, l4, l5, l6]

# Normalize the given values into percentages and reverse the process
min_value = min(values)
max_value = max(values)
normalized_percentages = [100 - (value - min_value) / (max_value - min_value) * 100 for value in values]


min_value = -600
max_value = 600
original_non_zero_params = sum(p.nonzero(as_tuple=False).size(0) for p in model.parameters())
print(f'The original model has {original_non_zero_params} non-zero parameters.')
# Normalize the given values into percentages based on the defined range
normalized_percentages = [(value - min_value) / (max_value - min_value) * 100 for value in values]

# Perform weight pruning on each layer
for i, layer in enumerate(model.transformer.layer[:len(normalized_percentages)]):
    pruning_percentage = normalized_percentages[i]

    # Get weights
    layer_weights = layer.ffn.lin2.weight.data

    # Calculate threshold value for pruning
    all_weights = torch.flatten(torch.abs(layer_weights))
    sorted_weights, _ = torch.sort(all_weights, descending=True)
    threshold_idx = int(len(sorted_weights) * pruning_percentage / 100) - 1
    threshold = sorted_weights[threshold_idx]

    # Prune weights below threshold value
    pruned_weights = layer_weights * (torch.abs(layer_weights) > threshold).float()

    # Calculate sparsity
    original_layer = layer_weights.cpu().numpy()
    pruned_layer = pruned_weights.cpu().numpy()

    original_sparsity = np.count_nonzero(original_layer == 0) / original_layer.size
    pruned_sparsity = np.count_nonzero(pruned_layer == 0) / pruned_layer.size

    print(f"Layer {i + 1}")
    print(f"  Pruning percentage: {pruning_percentage:.2f}%")
    print(f"  Original sparsity: {original_sparsity:.4f}")
    print(f"  Pruned sparsity: {pruned_sparsity:.4f}")

    # Update model's weights for the current layer with pruned weights
    layer.ffn.lin2.weight.data = pruned_weights

num_params = sum(p.numel() for p in model.parameters())
print(f'The pruned model has {num_params} parameters.')

# Calculate the number of non-zero parameters in the original and pruned models
pruned_non_zero_params = sum(p.nonzero(as_tuple=False).size(0) for p in model.parameters() if p.data.abs().gt(0).any())
print(f'The pruned model has {pruned_non_zero_params} non-zero parameters.')
# tokenizer.push_to_hub('pruned-distilbert', use_auth_token='x')
# model.push_to_hub('pruned-distilbert', use_auth_token='x')