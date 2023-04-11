import torch
import numpy as np
from transformers import BertModel, BertTokenizer

# Load BERT
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Given values
values = [190.4, 112.6, 52.6, 43.8, 47.3, 33.5, 17.9, 23.9, 60.8, 26.6, -6.1, -3.2]

# Normalize the given values into percentages
min_value = min(values)
max_value = max(values)
normalized_percentages = [(value - min_value) / (max_value - min_value) * 100 for value in values]

# Perform weight pruning on each layer
for i, layer in enumerate(model.encoder.layer[:len(normalized_percentages)]):
    pruning_percentage = normalized_percentages[i]

    # Get weights
    layer_weights = layer.output.dense.weight.data

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
    layer.output.dense.weight.data = pruned_weights

num_params = sum(p.numel() for p in model.parameters())
print(f'The pruned model has {num_params} parameters.')


num_nonzero_params = sum((p != 0).sum().item() for p in model.parameters())

print(f'The pruned model has {num_nonzero_params} non-zero parameters.')