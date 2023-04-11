import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('pruned_bert_model.pt')

# Get the weights of layer 5
weights = model.layers[5].get_weights()[0]

# Count the number of zero-valued elements in the weights tensor
num_zeros = tf.math.count_nonzero(tf.abs(weights) < 1e-6)

# Calculate the sparsity of layer 5
sparsity = num_zeros / weights.size

print("Sparsity of layer 5:", sparsity.numpy())
