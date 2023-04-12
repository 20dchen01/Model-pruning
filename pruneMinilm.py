import torch
import numpy as np
from transformers import BertModel, BertTokenizer

# Load BERT
model = BertModel.from_pretrained('nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base')
tokenizer = BertTokenizer.from_pretrained('nreimers/MiniLMv2-L6-H768-distilled-from-BERT-Base')

# Given values
l1 = 69.8+70+81.4+80.7+42.1+11.6
l2 = 17.7+16.5+10.2+12.8+24.5+26
l3 = 7.8+7.7+4.7+4.8+11.9+36
l4 = 3.1+2.9+1.3+2+12.6+17.5
l5 = 1+1.7+1-0.5+7.7+1.6
l6 = 0.6+1.2+1.5+0.2+1.2+7.3
values = [l1, l2, l3, l4, l5, l6]

# Normalize the given values into percentages and reverse the process
min_value = min(values)
max_value = max(values)
normalized_percentages = [100 - (value - min_value) / (max_value - min_value) * 100 for value in values]


scaling_factor = 0.5
adjusted_percentages = [percentage * scaling_factor if value >= 0 else percentage for percentage, value in zip(normalized_percentages, values)]

# Perform weight pruning on each layer
for i, layer in enumerate(model.encoder.layer[:len(adjusted_percentages)]):
    pruning_percentage = adjusted_percentages[i]
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
# tokenizer.push_to_hub('pruned-minilm', use_auth_token='x')
# model.push_to_hub('pruned-minilm', use_auth_token='x')