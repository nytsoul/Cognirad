import os
import sys

# Set Keras backend to PyTorch before importing Keras
os.environ["KERAS_BACKEND"] = "torch"

try:
    import keras
except ImportError:
    print("Keras not installed. Please install it to load Keras models.")
    sys.exit(1)

model_path = r"d:\Programming\Project\Medical\backend\outputs\pneumonia_model.keras"

print(f"Loading model from {model_path}...")
try:
    model = keras.saving.load_model(model_path)
    print("Model loaded successfully!")
    print("\n--- Model Summary ---")
    model.summary()
    
    print("\n--- Input Details ---")
    if hasattr(model, 'inputs') and model.inputs:
        for i, input_layer in enumerate(model.inputs):
            print(f"Input {i}: shape={getattr(input_layer, 'shape', 'unknown')}, dtype={getattr(input_layer, 'dtype', 'unknown')}")
    else:
        print("Inputs not explicitly defined or readable via model.inputs.")
        
    print("\n--- Output Details ---")
    if hasattr(model, 'outputs') and model.outputs:
        for i, output_layer in enumerate(model.outputs):
            print(f"Output {i}: shape={getattr(output_layer, 'shape', 'unknown')}, dtype={getattr(output_layer, 'dtype', 'unknown')}")
    else:
        print("Outputs not explicitly defined or readable via model.outputs.")
        
except Exception as e:
    print(f"Error loading model: {e}")

