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
output_path = r"d:\Programming\Project\Medical\backend\model_summary.txt"

with open(output_path, "w") as f:
    f.write(f"Loading model from {model_path}...\n")
    try:
        model = keras.saving.load_model(model_path)
        f.write("Model loaded successfully!\n")
        f.write("\n--- Model Summary ---\n")
        
        # Keras model.summary() prints to stdout, we can capture it using io.StringIO
        import io
        import contextlib
        with io.StringIO() as buf, contextlib.redirect_stdout(buf):
            model.summary()
            f.write(buf.getvalue())
        
        f.write("\n--- Input Details ---\n")
        if hasattr(model, 'inputs') and model.inputs:
            for i, input_layer in enumerate(model.inputs):
                f.write(f"Input {i}: shape={getattr(input_layer, 'shape', 'unknown')}, dtype={getattr(input_layer, 'dtype', 'unknown')}\n")
        else:
            f.write("Inputs not explicitly defined or readable via model.inputs.\n")
            
        f.write("\n--- Output Details ---\n")
        if hasattr(model, 'outputs') and model.outputs:
            for i, output_layer in enumerate(model.outputs):
                f.write(f"Output {i}: shape={getattr(output_layer, 'shape', 'unknown')}, dtype={getattr(output_layer, 'dtype', 'unknown')}\n")
        else:
            f.write("Outputs not explicitly defined or readable via model.outputs.\n")
            
    except Exception as e:
        f.write(f"Error loading model: {e}\n")

print(f"Analysis saved to {output_path}")

