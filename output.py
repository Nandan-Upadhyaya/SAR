import os
import json
import numpy as np
import tensorflow as tf
from SAR_B3 import GPUConfig, CycleGAN
import matplotlib.pyplot as plt

def load_model_and_generate(sar_image_path, checkpoint_path, output_dir):
    # Load checkpoint configuration
    with open(os.path.join(checkpoint_path, 'config.json'), 'r') as f:
        checkpoint_config = json.load(f)

    # Extract learning rate
    learning_rate = checkpoint_config.get('learning_rate_config', {}).get('value', 1.5e-5)

    # Initialize the CycleGAN model
    model = CycleGAN(
        image_size=256,
        num_classes=4,
        lambda_cyc=5,
        lambda_cls=0.2,
        learning_rate=learning_rate
    )

    model.compile()

    # Load weights
    model.G1.load_weights(os.path.join(checkpoint_path, 'generator1.h5'))
    model.G2.load_weights(os.path.join(checkpoint_path, 'generator2.h5'))
    model.D1.load_weights(os.path.join(checkpoint_path, 'discriminator1.h5'))
    model.D2.load_weights(os.path.join(checkpoint_path, 'discriminator2.h5'))

    # Preprocess and validate the SAR image
    img = tf.io.read_file(sar_image_path)
    img = tf.image.decode_png(img, channels=3)
    
    # Store original for comparison
    original_img = tf.image.resize(img, [768, 768])
    
    # Ensure proper normalization
    img = tf.cast(img, tf.float32)
    img = tf.clip_by_value(img, 0, 255)
    img = tf.image.resize(img, [768, 768])
    img = (img / 127.5) - 1.0
    
    # Validate normalization range
    tf.debugging.assert_greater_equal(img, -1.0)
    tf.debugging.assert_less_equal(img, 1.0)
    
    img = tf.expand_dims(img, axis=0)

    # Generate color image
    generated_image = model.G2(img, training=False)[0]
    generated_image = tf.clip_by_value(generated_image, -1.0, 1.0)
    generated_image = (generated_image + 1.0) * 0.5

    # Display both original and generated
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Input SAR Image')
    plt.imshow(original_img / 255.0)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Generated Color Image')
    plt.imshow(generated_image)
    plt.axis('off')

    # Save both images
    output_path = os.path.join(output_dir, 'generated_color_image.png')
    plt.imsave(output_path, generated_image.numpy())
    print(f"Generated color image saved to {output_path}")
    
    # Print value ranges for debugging
    print(f"Input image range: [{tf.reduce_min(img):.2f}, {tf.reduce_max(img):.2f}]")
    print(f"Generated image range: [{tf.reduce_min(generated_image):.2f}, {tf.reduce_max(generated_image):.2f}]")

    plt.tight_layout()
    plt.show()

def main():
    sar_image_path = r"E:\GitHub\SAR\Dataset\grassland\SAR\ROIs1970_fall_s1_11_p3.png" # Path to the input SAR image
    checkpoint_path = r"E:\GitHub\SAR\output_resumed_190\checkpoint_epoch_200"  # Path to checkpoint
    output_dir = r'E:\GitHub\SAR\test'  # Directory to save the output image

    os.makedirs(output_dir, exist_ok=True)
    GPUConfig.configure()
    load_model_and_generate(sar_image_path, checkpoint_path, output_dir)

if __name__ == "__main__":
    main()
    