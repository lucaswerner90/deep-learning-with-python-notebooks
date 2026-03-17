import keras
from keras import layers

# Define the inputs
text_input = keras.Input(shape=(1000,), name="text_input")
image_input = keras.Input(shape=(28, 28, 1), name="image_input")

# Process text input: pass through a Dense layer
text_features = layers.Dense(64, activation="relu")(text_input)

# Process image input: pass through Convolutional layers and flatten
x = layers.Conv2D(32, 3, activation="relu")(image_input)
# Adding a max pooling layer is standard, but you can omit it if you prefer
x = layers.MaxPooling2D(2)(x) 
x = layers.Conv2D(64, 3, activation="relu")(x)
image_features = layers.Flatten()(x)

# Concatenate the resulting tensors
concatenated = layers.concatenate([text_features, image_features])

# Output 1: binary priority score (sigmoid activation)
priority_output = layers.Dense(1, activation="sigmoid", name="priority")(concatenated)

# Output 2: 5-class category classification (softmax activation)
category_output = layers.Dense(5, activation="softmax", name="category")(concatenated)

# Create the multi-input, multi-output model
model = keras.Model(
    inputs=[text_input, image_input], 
    outputs=[priority_output, category_output]
)

# Compile the model with multiple losses and metrics mapped by output names
model.compile(
    optimizer="rmsprop",
    loss={
        "priority": "binary_crossentropy",
        "category": "categorical_crossentropy",
    },
    metrics={
        "priority": ["accuracy"],
        "category": ["accuracy"],
    }
)

# Print the model summary
model.summary()

# If you'd like to visualize the architecture (as you've done in previous exercises):
# keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
