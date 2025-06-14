# inference.py

import tensorflow as tf
import numpy as np
from PIL import Image
import cv2  # <--- Import OpenCV
import argparse
import os

# --- Model & Image Constants ---
# Matches your training setup
IMG_WIDTH = 128
IMG_HEIGHT = 128
TARGET_SHAPE = (IMG_WIDTH, IMG_HEIGHT)

### IMPORTANT - VERIFY THESE VALUES! ###
# These are the min/max values of your data BEFORE normalization.
# For standard 8-bit images, these are the correct values.
L_IN_MIN = 0.0
L_IN_MAX = 255.0
A_IN_MIN = 0.0
A_IN_MAX = 255.0
B_IN_MIN = 0.0
B_IN_MAX = 255.0


def colorize_image(model_path, input_path, output_path):
    """
    Loads a pre-trained Keras model to colorize a black-and-white image.
    This script precisely replicates the custom resizing and normalization
    steps from the training pipeline.
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

    original_lab = original_rgb.convert('LAB')
    l_channel, _, _ = original_lab.split()
    
    # ** CUSTOM RESIZING **
    # Replicate the exact resizing logic from your training function.
    # Convert the L channel to a NumPy array to use with OpenCV.
    l_array = np.array(l_channel, dtype=np.uint8)
    
    # Resize using cv2.resize with the same interpolation method.
    resized_l_array = cv2.resize(l_array, TARGET_SHAPE, interpolation=cv2.INTER_AREA)

    # ** CUSTOM NORMALIZATION **
    # Apply the exact same normalization formula as used in training.
    # First, cast to float32.
    resized_l_float = resized_l_array.astype(np.float32)
    
    # Generic formula: 2 * (x - min) / (max - min) - 1
    normalized_l = 2 * (resized_l_float - L_IN_MIN) / (L_IN_MAX - L_IN_MIN) - 1
    
    # Add batch and channel dimensions for the model: (1, 128, 128, 1).
    input_l_tensor = np.expand_dims(normalized_l, axis=(0, -1))

    # --- 3. Run Prediction ---
    print("Predicting color channels with the model...")
    # The model predicts the 'A' and 'B' channels in the normalized [-1, 1] range.
    predicted_ab_normalized = model.predict(input_l_tensor) # Shape: (1, 128, 128, 2)
    
    # --- 4. Post-process the Output ---
    # ** CUSTOM DENORMALIZATION **
    # We must apply the mathematical inverse of the normalization formula.
    # Inverse formula: ((y + 1) * (max - min)) / 2 + min
    
    # Separate the predicted A and B channels
    predicted_a_norm = predicted_ab_normalized[:, :, :, 0]
    predicted_b_norm = predicted_ab_normalized[:, :, :, 1]
    
    # Denormalize each channel
    denorm_a = ((predicted_a_norm + 1) * (A_IN_MAX - A_IN_MIN)) / 2 + A_IN_MIN
    denorm_b = ((predicted_b_norm + 1) * (B_IN_MAX - B_IN_MIN)) / 2 + B_IN_MIN

    # Combine them back and remove the batch dimension
    denorm_ab = np.stack([denorm_a, denorm_b], axis=-1)
    denorm_ab = np.squeeze(denorm_ab, axis=0)

    # --- 5. Recombine Channels and Convert Back to RGB ---
    # We must use the resized L channel to match the dimensions of the predicted AB channels.
    final_lab_array = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    final_lab_array[:, :, 0] = resized_l_array
    final_lab_array[:, :, 1:] = np.clip(denorm_ab, 0, 255).astype(np.uint8)

    colorized_lab_img = Image.fromarray(final_lab_array, 'LAB')
    colorized_rgb_img = colorized_lab_img.convert('RGB')

    # --- 6. Resize and Save the Final Image ---
    # Upscale the final 128x128 colorized image to the original resolution.
    print(f"Resizing colorized image back to {original_size[0]}x{original_size[1]}...")
    # PIL's default resizing (BICUBIC) is good for upscaling.
    final_image = colorized_rgb_img.resize(original_size)

    final_image.save(output_path)
    print(f"✅ Success! Colorized image saved to: {output_path}")
    print("--- Process Complete ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colorize a black-and-white image using a pre-trained Keras model.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the saved .keras model file.')
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input black-and-white image.')
    parser.add_argument('--output_image', type=str, default='colorized_output.png', help='Path to save the colorized output image (default: colorized_output.png).')
    args = parser.parse_args()
    output_dir = os.path.dirname(args.output_image)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    colorize_image(args.model_path, args.input_image, args.output_image)