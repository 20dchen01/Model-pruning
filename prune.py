import torch
import torch.nn.utils.prune as prune
from transformers import BertTokenizer, BertModel

# Load the BERT-base-uncased model
model = BertModel.from_pretrained("bert-base-uncased")

# Define the pruning percentages for each layer
pruning_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
                       0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

# Perform weight pruning on each layer
for i, layer in enumerate(model.encoder.layer):
    # Adjust the percentage if there are fewer pruning percentages than layers
    pruning_percentage = pruning_percentages[i % len(pruning_percentages)]

    # Prune the weights in the self-attention sub-layer
    prune.l1_unstructured(layer.attention.self.query, 'weight', amount=pruning_percentage)
    prune.l1_unstructured(layer.attention.self.key, 'weight', amount=pruning_percentage)
    prune.l1_unstructured(layer.attention.self.value, 'weight', amount=pruning_percentage)

    # Prune the weights in the position-wise feed-forward sub-layer
    prune.l1_unstructured(layer.intermediate.dense, 'weight', amount=pruning_percentage)
    prune.l1_unstructured(layer.output.dense, 'weight', amount=pruning_percentage)

# # Remove the pruning masks and make pruning permanent
# for module in model.modules():
#     if isinstance(module, torch.nn.Linear):
#         # Check if the parameter has been pruned before removing pruning
#         if hasattr(module.weight, "orig"):
#             prune.remove(module, 'weight')


# Calculate total number of parameters in pruned model
num_params = sum(p.numel() for p in model.parameters())
print(f'The pruned model has {num_params} parameters.')
num_nonzero_params = sum((p != 0).sum().item() for p in model.parameters())

print(f'The pruned model has {num_nonzero_params} non-zero parameters.')