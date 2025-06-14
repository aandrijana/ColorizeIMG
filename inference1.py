<<<<<<< HEAD
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
    parser.add_argument('--input_image', type=str, required=True, help='Path to the input black-and-white image.')
    parser.add_argument('--output_image', type=str, default='colorized_output.png', help='Path to save the colorized output image (default: colorized_output.png).')
    args = parser.parse_args()
    output_dir = os.path.dirname(args.output_image)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
=======
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
>>>>>>> 525ef1fd413353e178247c03c6d43f6f5feded3b
    colorize_image(args.model_path, args.input_image, args.output_image)