# Your file should START with this line
# inference.py

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2  # OpenCV is critical for correct color space conversion
import argparse
import os

# --- Model & Image Constants ---
# Matches your training setup
IMG_WIDTH = 128
IMG_HEIGHT = 128
TARGET_SHAPE = (IMG_WIDTH, IMG_HEIGHT)

### CRITICAL NORMALIZATION CONSTANTS FROM YOUR TRAINING ###
# These MUST match the values you used to normalize your training data.
L_IN_MIN = 0.0
L_IN_MAX = 255.0
A_IN_MIN = 43.0
A_IN_MAX = 206.0
B_IN_MIN = 22.0
B_IN_MAX = 222.0


def colorize_image(model_path, input_path, output_path):
    """
    Loads a pre-trained Keras model and colorizes an image, precisely replicating
    the custom resizing, normalization, and OpenCV color space conversion from training.
    """
    print("--- Starting Image Colorization ---")

    # --- 1. Load the Trained Model ---
    print(f"Loading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"FATAL: Error loading model. {e}")
        return
    print("Model loaded successfully.")

    # --- 2. Load and Pre-process the Input Image ---
    print(f"Loading and processing input image: {input_path}")
    
    original_rgb = Image.open(input_path).convert('RGB')
    original_size = original_rgb.size
    print(f"Original image size: {original_size[0]}x{original_size[1]}")

    # Convert to LAB using Pillow to easily get the L channel
    original_lab = original_rgb.convert('LAB')
    l_channel, _, _ = original_lab.split()
    
    # Custom resizing using OpenCV to match training
    l_array_uint8 = np.array(l_channel, dtype=np.uint8)
    resized_l_array = cv2.resize(l_array_uint8, TARGET_SHAPE, interpolation=cv2.INTER_AREA)

    # Custom normalization of the L channel to [-1, 1]
    resized_l_float = resized_l_array.astype(np.float32)
    normalized_l = 2 * (resized_l_float - L_IN_MIN) / (L_IN_MAX - L_IN_MIN) - 1
    input_l_tensor = np.expand_dims(normalized_l, axis=(0, -1))

    # --- 3. Run Prediction ---
    print("Predicting color channels with the model...")
    predicted_ab_normalized = model.predict(input_l_tensor)
    
    # --- 4. Post-process the Output (THE CRITICAL PART) ---
    # Step 4a: Denormalize the predicted A and B channels from [-1, 1] back to their custom uint8 range
    # Inverse formula: ((y + 1) / 2) * (max - min) + min
    pred_a_norm = predicted_ab_normalized[0, :, :, 0]
    pred_b_norm = predicted_ab_normalized[0, :, :, 1]
    
    a_reverted = (pred_a_norm + 1) / 2.0 * (A_IN_MAX - A_IN_MIN) + A_IN_MIN
    b_reverted = (pred_b_norm + 1) / 2.0 * (B_IN_MAX - B_IN_MIN) + B_IN_MIN

    # Step 4b: Prepare all channels for OpenCV's COLOR_LAB2RGB conversion
    # L channel needs to be scaled from [0, 255] to [0, 100]
    l_for_cv2 = resized_l_array * (100.0 / 255.0)
    
    # A and B channels need to be shifted from their uint8 range to the [-128, 127] range
    a_for_cv2 = a_reverted - 128.0
    b_for_cv2 = b_reverted - 128.0

    # Step 4c: Reconstruct the LAB image in the format cv2 expects
    lab_for_cv2 = np.stack([l_for_cv2, a_for_cv2, b_for_cv2], axis=-1)

    # Step 4d: Perform the color space conversion
    rgb_from_cv2 = cv2.cvtColor(lab_for_cv2.astype('float32'), cv2.COLOR_LAB2RGB)

    # Step 4e: The output of cvtColor is in float [0, 1]. Clip and scale to uint8 [0, 255].
    rgb_clipped = np.clip(rgb_from_cv2, 0, 1)
    rgb_uint8 = (rgb_clipped * 255).astype(np.uint8)

    # Convert the final NumPy array to a Pillow Image
    colorized_rgb_img = Image.fromarray(rgb_uint8)

    # --- 5. Resize and Save the Final Image ---
    print(f"Resizing colorized image back to {original_size[0]}x{original_size[1]}...")
    final_image = colorized_rgb_img.resize(original_size, Image.BICUBIC) # Use a high-quality filter for upscaling

    final_image.save(output_path)
    print(f"✅ Success! Colorized image saved to: {output_path}")
    print("--- Process Complete ---")


if __name__ == '__main__':
    # (The main block remains unchanged)
    parser = argparse.ArgumentParser(description='Colorize a black-and-white image using a pre-trained Keras model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved .keras model file.')
    parser.add_-argument('--input_image', type=str, required=True, help='Path to the input black-and-white image.')
    parser.add_argument('--output_image', type=str, default='colorized_output.png', help='Path to save the colorized output image (default: colorized_output.png).')
    args = parser.parse_args()
    output_dir = os.path.dirname(args.output_image)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # Your file should END with this line
    colorize_image(args.model_path, args.input_image, args.output_image)