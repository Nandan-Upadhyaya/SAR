import os
import tensorflow as tf
import keras_tuner as kt
from keras_tuner.tuners import Hyperband
import numpy as np
import time
from datetime import datetime
from collections import defaultdict
import matplotlib.pyplot as plt
import json
import sys
import numpy as np
from keras import layers, applications
import os
from glob import glob
import matplotlib.pyplot as plt
from datetime import datetime
import time
from keras.applications import efficientnet
from keras.applications.inception_v3 import InceptionV3
import json
from scipy.linalg import sqrtm
import tensorflow_probability as tfp



def setup_distributed_strategy():
    """Set up distributed training strategy based on available devices"""
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus) > 1:
        # Use MirroredStrategy for multiple GPUs
        strategy = tf.distribute.MirroredStrategy(
            devices=[f"/GPU:{i}" for i in range(len(gpus))],
            cross_device_ops=tf.distribute.NcclAllReduce()  # Use NCCL for faster GPU communication
        )
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
    else:
        strategy = tf.distribute.OneDeviceStrategy("/CPU:0")
   
    print(f"Number of devices in strategy: {strategy.num_replicas_in_sync}")
    return strategy

# Set up distributed strategy
strategy = setup_distributed_strategy()

# Constants
TUNER_DIR = "sar_hyperparameter_tuning"
MAX_TRIALS = 25
MAX_EPOCHS_PER_TRIAL = 7  # Changed from 5 to 7 for better convergence
EXECUTIONS_PER_TRIAL = 1
# Constants
BUFFER_SIZE = 400
BATCH_SIZE = 1  # Fixed at 1
IMG_WIDTH = 256  # Changed from 256 to 128
IMG_HEIGHT = 256  # Changed from 256 to 128
LAMBDA = 12 # Changed from 15 to 8 for better adversarial-reconstruction balance
TERRAIN_TYPES = ['urban', 'grassland', 'agri', 'barrenland']

class InstanceNormalization(layers.Layer):
    """Native implementation of Instance Normalization"""
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(input_shape[-1],), initializer='ones')
        self.offset = self.add_weight(name='offset', shape=(input_shape[-1],), initializer='zeros')

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
        })
        return config

def load_and_preprocess_image(image_path):
    """Load and preprocess image using TensorFlow ops only"""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = (img / 127.5) - 1.0
    return img

def create_dataset():
    """Create dataset with proper distribution handling and train/val/test split"""
    with strategy.scope():
        # Increase batch size for multi-GPU training
        GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
       
        all_datasets = []
        for terrain in TERRAIN_TYPES:
            sar_path = f'/kaggle/input/sentinel12-image-pairs-segregated-by-terrain/v_2/{terrain}/s1/*'
            color_path = f'/kaggle/input/sentinel12-image-pairs-segregated-by-terrain/v_2/{terrain}/s2/*'
           
            sar_files = sorted(glob(sar_path))
            color_files = sorted(glob(color_path))
           
            # Ensure same number of files in both directories
            min_files = min(len(sar_files), len(color_files))
            sar_files = sar_files[:min_files]
            color_files = color_files[:min_files]

            # Verify pairing (example check)
            for sar, color in zip(sar_files[:5], color_files[:5]):  # Check first 5 pairs
                sar_id = os.path.basename(sar).split('_')[1].split('.')[0]
                color_id = os.path.basename(color).split('_')[1].split('.')[0]
                if sar_id != color_id:
                    raise ValueError(f"Mismatched pair: {sar} vs {color}")
               
            # Calculate split indices
            total_files = len(sar_files)
            train_idx = int(total_files * 0.75)
            val_idx = int(total_files * 0.90)  # 15% validation (0.75 to 0.90)
           
            # Create train/val/test splits
            sar_train = sar_files[:train_idx]
            sar_val = sar_files[train_idx:val_idx]
            sar_test = sar_files[val_idx:]
           
            color_train = color_files[:train_idx]
            color_val = color_files[train_idx:val_idx]
            color_test = color_files[val_idx:]
           
            print(f"Terrain {terrain}: {len(sar_train)} train, {len(sar_val)} validation, {len(sar_test)} test")
           
            # Create dataset from file paths
            sar_train_ds = tf.data.Dataset.from_tensor_slices(sar_train)
            sar_train_ds = sar_train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
           
            color_train_ds = tf.data.Dataset.from_tensor_slices(color_train)
            color_train_ds = color_train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
           
            sar_val_ds = tf.data.Dataset.from_tensor_slices(sar_val)
            sar_val_ds = sar_val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
           
            color_val_ds = tf.data.Dataset.from_tensor_slices(color_val)
            color_val_ds = color_val_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
           
            sar_test_ds = tf.data.Dataset.from_tensor_slices(sar_test)
            sar_test_ds = sar_test_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
           
            color_test_ds = tf.data.Dataset.from_tensor_slices(color_test)
            color_test_ds = color_test_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
           
            # Create one-hot encoded terrain labels
            terrain_labels = tf.data.Dataset.from_tensors(
                tf.one_hot(TERRAIN_TYPES.index(terrain), len(TERRAIN_TYPES))
            )
           
            # Create train/val/test terrain labels
            train_terrain_labels = terrain_labels.repeat(len(sar_train))
            val_terrain_labels = terrain_labels.repeat(len(sar_val))
            test_terrain_labels = terrain_labels.repeat(len(sar_test))
           
            # Combine into final datasets
            train_dataset = tf.data.Dataset.zip((sar_train_ds, color_train_ds, train_terrain_labels))
            val_dataset = tf.data.Dataset.zip((sar_val_ds, color_val_ds, val_terrain_labels))
            test_dataset = tf.data.Dataset.zip((sar_test_ds, color_test_ds, test_terrain_labels))
           
            all_datasets.append({
                'train': train_dataset,
                'val': val_dataset,
                'test': test_dataset
            })
       
        # Combine datasets by split type
        combined_train = tf.data.Dataset.from_tensor_slices([d['train'] for d in all_datasets]).interleave(
            lambda x: x, cycle_length=len(TERRAIN_TYPES), num_parallel_calls=tf.data.AUTOTUNE)
        train_ds = combined_train.shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
       
        combined_val = tf.data.Dataset.from_tensor_slices([d['val'] for d in all_datasets]).interleave(
            lambda x: x, cycle_length=len(TERRAIN_TYPES), num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = combined_val.batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
       
        combined_test = tf.data.Dataset.from_tensor_slices([d['test'] for d in all_datasets]).interleave(
            lambda x: x, cycle_length=len(TERRAIN_TYPES), num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = combined_test.batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
       
        return train_ds, val_ds, test_ds
        
class TerrainGuidedAttention(layers.Layer):
    """Attention mechanism that explicitly considers terrain information"""
    def __init__(self, channels, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.query = None  # Initialize in build method
        self.key = None    # Initialize in build method
        self.value = None  # Initialize in build method
        self.gamma = self.add_weight(name="gamma", shape=(1,), initializer="zeros")
        self.terrain_project = None  # Initialize in build method
        self._input_shape_tracker = None

    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)):
            raise ValueError("TerrainGuidedAttention expects a list of input shapes")
           
        x_shape, terrain_shape = input_shape
       
        # Ensure we're using the correct number of channels based on input
        actual_channels = x_shape[-1]
       
        # Create layers with proper dimensions
        self.query = layers.Conv2D(actual_channels // 8, 1)
        self.key = layers.Conv2D(actual_channels // 8, 1)
        self.value = layers.Conv2D(actual_channels, 1)
        self.terrain_project = layers.Dense(actual_channels)
       
        # Build all child layers explicitly
        self.query.build(x_shape)
        self.key.build(x_shape)
        self.value.build(x_shape)
        self.terrain_project.build(terrain_shape)
       
        self._input_shape_tracker = input_shape
        self.built = True
        super().build(input_shape)
       
    def call(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("TerrainGuidedAttention expects [x, terrain] as input")
       
        x, terrain = inputs
       
        batch_size = tf.shape(x)[0]
        height = tf.shape(x)[1]
        width = tf.shape(x)[2]
       
        # Project terrain features
        terrain_features = self.terrain_project(terrain)
        terrain_features = tf.reshape(terrain_features, [-1, 1, 1, self.channels])
        terrain_features = tf.tile(terrain_features, [1, height, width, 1])
       
        # Add terrain context
        x = x + terrain_features
       
        # Compute attention
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
       
        attention = tf.matmul(
            tf.reshape(q, [batch_size, -1, self.channels // 8]),
            tf.reshape(k, [batch_size, -1, self.channels // 8]),
            transpose_b=True
        )
        attention = tf.nn.softmax(attention / tf.sqrt(tf.cast(self.channels // 8, x.dtype)))
       
        out = tf.matmul(attention, tf.reshape(v, [batch_size, -1, self.channels]))
        out = tf.reshape(out, [batch_size, height, width, self.channels])
       
        return self.gamma * out + x

    def compute_output_shape(self, input_shape):
        return input_shape[0]
   
    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
        })
        return config
       
    @classmethod
    def from_config(cls, config):
        return cls(**config)
   
    def get_build_config(self):
        if hasattr(self, '_input_shape_tracker') and self._input_shape_tracker is not None:
            return {
                "input_shape": self._input_shape_tracker,
                "channels": self.channels
            }
        return {"channels": self.channels}
       
    def build_from_config(self, config):
        if not self.built and "input_shape" in config:
            self.build(config["input_shape"])

class TerrainAdaptiveNormalization(layers.Layer):
    def __init__(self, channels, **kwargs):
        super(TerrainAdaptiveNormalization, self).__init__(**kwargs)
        self.channels = channels
        self.norm = layers.BatchNormalization(axis=-1)
        self.terrain_scale = layers.Dense(channels)
        self.terrain_bias = layers.Dense(channels)
        self.color_norm = layers.LayerNormalization(axis=-1)
        self.color_scale = layers.Dense(channels)
        self.color_bias = layers.Dense(channels)
        self._input_shape_tracker = None
       
    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)):
            raise ValueError("TerrainAdaptiveNormalization expects a list of input shapes")
        x_shape, terrain_shape = input_shape
        # Just build the layers without shape validation
        self.norm.build(x_shape)
        self.terrain_scale.build(terrain_shape)
        self.terrain_bias.build(terrain_shape)
        self.color_norm.build(x_shape)
        self.color_scale.build(terrain_shape)
        self.color_bias.build(terrain_shape)
        self._input_shape_tracker = input_shape
        super(TerrainAdaptiveNormalization, self).build(input_shape)

    def call(self, inputs, training=None):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("TerrainAdaptiveNormalization expects [x, terrain_features] as input")

        x, terrain_features = inputs

        # Apply normalizations with training flag
        normalized = self.norm(x, training=training)

        # Generate terrain-specific parameters
        scale = self.terrain_scale(terrain_features)
        bias = self.terrain_bias(terrain_features)

        # Reshape for broadcasting
        scale = tf.reshape(scale, [-1, 1, 1, self.channels])
        bias = tf.reshape(bias, [-1, 1, 1, self.channels])

        # Apply color normalization
        color_normalized = self.color_norm(normalized)
        color_scale = self.color_scale(terrain_features)
        color_bias = self.color_bias(terrain_features)

        color_scale = tf.reshape(color_scale, [-1, 1, 1, self.channels])
        color_bias = tf.reshape(color_bias, [-1, 1, 1, self.channels])

        # Combine normalizations
        terrain_norm = normalized * (1 + scale) + bias
        color_norm = color_normalized * (1 + color_scale) + color_bias

        return (terrain_norm + color_norm) / 2
   
    def get_config(self):
        config = super().get_config()
        config.update({
            "channels": self.channels,
        })
        return config
       
    @classmethod
    def from_config(cls, config):
        return cls(**config)
   
    def get_build_config(self):
        if hasattr(self, '_input_shape_tracker') and self._input_shape_tracker is not None:
            return {
                "input_shape": self._input_shape_tracker,
                "channels": self.channels
            }
        return {"channels": self.channels}
       
    def build_from_config(self, config):
        if not self.built and "input_shape" in config:
            self.build(config["input_shape"])

class TerrainSpatialLayer(layers.Layer):
    """Layer to handle terrain spatial features with explicit tensor shapes"""
    def __init__(self, height, width, **kwargs):
        super(TerrainSpatialLayer, self).__init__(**kwargs)
        self.height = height
        self.width = width
        self.dense1 = layers.Dense(512, activation='relu')
        self.dense2 = layers.Dense(height * width)
        self.reshape = layers.Reshape((height, width, 1))
        self._input_shape_tracker = None
       
    def call(self, terrain_input):
        x = self.dense1(terrain_input)
        x = self.dense2(x)
        return self.reshape(x)
   
    def get_config(self):
        config = super().get_config()
        config.update({
            "height": self.height,
            "width": self.width,
        })
        return config
       
    @classmethod
    def from_config(cls, config):
        return cls(**config)
   
    def get_build_config(self):
        if hasattr(self, '_input_shape_tracker') and self._input_shape_tracker is not None:
            return {
                "input_shape": self._input_shape_tracker,
                "height": self.height,
                "width": self.width
            }
        return {"height": self.height, "width": self.width}
       
    def build_from_config(self, config):
        if not self.built and "input_shape" in config:
            self.build(config["input_shape"])
   

class MemoryEfficientResBlock(layers.Layer):
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.conv1 = None  # Initialize in build to ensure proper input shape
        self.conv2 = None  # Initialize in build to ensure proper input shape
        self.norm1 = InstanceNormalization()
        self.norm2 = InstanceNormalization()
        self.attention = None  # Initialize in build
        self.activation = layers.Activation('silu')
        self.input_proj = None
        self._input_shape_tracker = None
       
    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)):
            raise ValueError("MemoryEfficientResBlock expects a list of input shapes")
       
        x_shape, terrain_shape = input_shape
        self._input_shape_tracker = [x_shape, terrain_shape]
       
        # Determine actual input channels
        input_channels = x_shape[-1]
       
        # Create and build input_proj if needed
        if input_channels != self.filters:
            self.input_proj = layers.Conv2D(self.filters, 1, padding='same')
            self.input_proj.build(x_shape)
            # Update x_shape for subsequent layers
            x_shape = x_shape.as_list()
            x_shape[-1] = self.filters
            x_shape = tf.TensorShape(x_shape)
       
        # Now create conv layers with correct input shape
        self.conv1 = layers.SeparableConv2D(self.filters, 3, padding='same')
        self.conv2 = layers.SeparableConv2D(self.filters, 3, padding='same')
        self.attention = TerrainGuidedAttention(self.filters)
       
        # Explicitly build all child layers with proper shapes
        self.conv1.build(x_shape)
        self.conv2.build(x_shape)
        self.norm1.build(x_shape)
        self.norm2.build(x_shape)
        self.attention.build([x_shape, terrain_shape])
        self.activation.build(x_shape)
           
        self.built = True
        super().build(input_shape)
       
    def call(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("MemoryEfficientResBlock expects [x, terrain] as input")
           
        x, terrain = inputs
       
        # Project input if needed
        residual = x
        if self.input_proj is not None:
            x = self.input_proj(x)
            residual = x  # Use projected tensor as residual connection
       
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
       
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.attention([x, terrain])
       
        return x + residual

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.filters)

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
        })
        return config
       
    @classmethod
    def from_config(cls, config):
        return cls(**config)
   
    def get_build_config(self):
        if hasattr(self, '_input_shape_tracker') and self._input_shape_tracker is not None:
            return {
                "input_shape": self._input_shape_tracker,
                "filters": self.filters
            }
        return {"filters": self.filters}
       
    def build_from_config(self, config):
        if not self.built and "input_shape" in config:
            self.build(config["input_shape"])

def build_terrain_aware_generator():
    """Enhanced generator with fixed input shape"""
    """Generator maintaining 128x128 output resolution"""
    sar_input = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3])
    terrain_input = layers.Input(shape=[len(TERRAIN_TYPES)])

    # Initial processing without downsampling
    x = layers.Conv2D(64, 3, strides=1, padding='same')(sar_input)
    x = InstanceNormalization()(x)
    x = layers.Activation('silu')(x)

    # Controlled downsampling to maintain resolution
    skip_connections = []
    filter_sizes = [128, 256]  # Reduced number of downsamplings
    for filters in filter_sizes:
        skip_connections.append(x)
        x = layers.Conv2D(filters, 4, strides=2, padding='same')(x)
        x = InstanceNormalization()(x)
        x = layers.Activation('silu')(x)

    # Middle blocks at 32x32 resolution - use consistent filter size to prevent shape mismatch
    for i in range(9):
        # Use the same filter size as the last downsampling layer to maintain consistency
        x = MemoryEfficientResBlock(256)([x, terrain_input])

    # Upsampling back to 128x128
    for skip, filters in zip(reversed(skip_connections), reversed(filter_sizes)):
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skip])
        x = InstanceNormalization()(x)
        x = layers.Activation('silu')(x)

    # Final output at 128x128
    outputs = layers.Conv2D(3, 3, padding='same', activation='tanh', dtype=tf.float32)(x)
    return tf.keras.Model(inputs=[sar_input, terrain_input], outputs=outputs)

def build_terrain_aware_discriminator():
    """Discriminator with proper tensor shape handling"""
    # Fix input layer declarations with explicit batch_size keyword argument
    input_image = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], batch_size=None)
    target_image = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], batch_size=None)
    terrain_input = layers.Input(shape=[len(TERRAIN_TYPES)], batch_size=None)
   
    # Process terrain features
    terrain_features = layers.Dense(512, activation='relu')(terrain_input)
   
    # Fix TerrainSpatialLayer call with proper batch size handling
    terrain_spatial = TerrainSpatialLayer(height=IMG_HEIGHT, width=IMG_WIDTH)(terrain_input)
   
    # Combine inputs
    x = layers.Concatenate()([input_image, target_image, terrain_spatial])
   
    # Original discriminator architecture
    outputs = []
    for scale in [1, 2]:
        if scale != 1:
            current_input = layers.AveragePooling2D(scale)(x)
        else:
            current_input = x
           
        features = current_input
        for filters in [64, 128, 256]:
            features = layers.Conv2D(filters, 4, strides=2, padding='same')(features)
            features = TerrainAdaptiveNormalization(filters)([features, terrain_features])
            features = layers.LeakyReLU(0.2)(features)
            features = TerrainGuidedAttention(filters)([features, terrain_features])
       
        output = layers.Conv2D(1, 4, strides=1, padding='same')(features)
        outputs.append(output)

    return tf.keras.Model(inputs=[input_image, target_image, terrain_input], outputs=outputs)

def create_feature_extractor():
    """Create a more stable feature extractor using EfficientNetB0"""
    base_model = applications.EfficientNetB0(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
    )
    # Freeze the model
    base_model.trainable = False
   
    # Select specific layers for feature extraction
    layers_to_extract = ['block2a_activation', 'block3a_activation', 'block4a_activation']
    outputs = [base_model.get_layer(name).output for name in layers_to_extract]
   
    return tf.keras.Model(inputs=base_model.input, outputs=outputs)

with strategy.scope():
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    # Initialize feature extractor within strategy scope
    feature_extractor = create_feature_extractor()
   
# Update the compute_perceptual_loss function
@tf.function
def compute_perceptual_loss(real_images, generated_images):
    real_images = tf.cast(real_images, tf.float32)
    generated_images = tf.cast(generated_images, tf.float32)
    real_images = efficientnet.preprocess_input((real_images + 1) * 127.5)
    generated_images = efficientnet.preprocess_input((generated_images + 1) * 127.5)
    real_features = feature_extractor(real_images)
    gen_features = feature_extractor(generated_images)
    perceptual_loss = 0.0
    for real_feat, gen_feat in zip(real_features, gen_features):
        perceptual_loss += tf.reduce_mean(tf.abs(real_feat - gen_feat))
    return tf.cast(perceptual_loss, tf.float16)


def generator_loss(disc_generated_output, generated_images, target_images):
    """Generator loss function with fp32 for loss calculations and NaN checks"""
    # Convert inputs to fp32 for loss calculation
    generated_images_f32 = tf.cast(generated_images, tf.float32)
    target_images_f32 = tf.cast(target_images, tf.float32)
   
    # GAN loss in fp32 with numerical stability improvements
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
   
    # Calculate per-output losses safely
    gan_losses = []
    for output in disc_generated_output:
        output_f32 = tf.cast(output, tf.float32)
        # Clip discriminator outputs to prevent extreme values
        output_f32 = tf.clip_by_value(output_f32, -20.0, 20.0)
        # Use epsilon for numerical stability
        labels = tf.ones_like(output_f32)
        per_output_loss = loss_fn(labels, output_f32)
        # Handle NaN values by replacing with zeros
        per_output_loss = tf.where(tf.math.is_finite(per_output_loss), per_output_loss, tf.zeros_like(per_output_loss))
        gan_losses.append(tf.reduce_mean(per_output_loss))
   
    # Safely compute mean across all outputs
    gan_loss = tf.reduce_mean(gan_losses) if gan_losses else tf.constant(0.0, dtype=tf.float32)
   
    # Check for NaN and provide safe fallback
    gan_loss = tf.where(tf.math.is_finite(gan_loss), gan_loss, tf.constant(0.0, dtype=tf.float32))
   
    # L1 loss in fp32 with clipping for stability
    l1_diff = tf.abs(target_images_f32 - generated_images_f32)
    l1_diff = tf.clip_by_value(l1_diff, 0.0, 2.0)  # Clip to prevent extreme values
    l1_loss = tf.reduce_mean(l1_diff)
    l1_loss = tf.where(tf.math.is_finite(l1_loss), l1_loss, tf.constant(0.0, dtype=tf.float32))
   
    # Perceptual loss calculation with NaN protection
    try:
        perceptual_loss = compute_perceptual_loss(target_images_f32, generated_images_f32)
        perceptual_loss = tf.cast(perceptual_loss, tf.float32)  # Ensure float32 type
        perceptual_loss = tf.reduce_mean(perceptual_loss)
        perceptual_loss = tf.where(tf.math.is_finite(perceptual_loss), perceptual_loss,
                                   tf.constant(0.0, dtype=tf.float32))
    except Exception:
        # Fallback if perceptual loss calculation fails
        perceptual_loss = tf.constant(0.0, dtype=tf.float32)
   
    # Metrics with NaN protection
    psnr = tf.image.psnr(
        tf.clip_by_value(target_images_f32, -1.0, 1.0),
        tf.clip_by_value(generated_images_f32, -1.0, 1.0),
        max_val=2.0
    )
    psnr = tf.reduce_mean(psnr)
    psnr = tf.where(tf.math.is_finite(psnr), psnr, tf.constant(0.0, dtype=tf.float32))
   
    ssim = tf.image.ssim(
        tf.clip_by_value(target_images_f32, -1.0, 1.0),
        tf.clip_by_value(generated_images_f32, -1.0, 1.0),
        max_val=2.0
    )
    ssim = tf.reduce_mean(ssim)
    ssim = tf.where(tf.math.is_finite(ssim), ssim, tf.constant(0.0, dtype=tf.float32))
   
    # Scale losses to prevent underflow in mixed precision
    lambda_val = tf.cast(LAMBDA, tf.float32)
   
    # Use small weight for perceptual loss to prevent it from dominating
    perceptual_weight = tf.constant(0.1, dtype=tf.float32)  # Increased from 0.075 to 0.2
   
    total_loss_f32 = gan_loss + (lambda_val * l1_loss) + (lambda_val * perceptual_weight * perceptual_loss)
   
    # Final NaN check
    total_loss_f32 = tf.where(tf.math.is_finite(total_loss_f32), total_loss_f32,
                              tf.constant(0.1, dtype=tf.float32))  # Use small non-zero value if NaN

    # Cast all returned values to float32 instead of float16 for consistency
    return (total_loss_f32,
            gan_loss,
            l1_loss,
            perceptual_loss,
            psnr,
            ssim)

def discriminator_loss(disc_real_output, disc_generated_output):
    """Discriminator loss function with numerical stability improvements"""
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
   
    # Calculate real loss with clipping and NaN protection
    real_losses = []
    for output in disc_real_output:
        output_f32 = tf.cast(output, tf.float32)
        # Clip discriminator outputs to prevent extreme values
        output_f32 = tf.clip_by_value(output_f32, -20.0, 20.0)
        # Use epsilon for numerical stability
        labels = tf.ones_like(output_f32)
        per_output_loss = loss_fn(labels, output_f32)
        # Handle NaN values
        per_output_loss = tf.where(tf.math.is_finite(per_output_loss), per_output_loss, tf.zeros_like(per_output_loss))
        real_losses.append(tf.reduce_mean(per_output_loss))
   
    # Safely compute mean
    real_loss = tf.reduce_mean(real_losses) if real_losses else tf.constant(0.0, dtype=tf.float32)
    real_loss = tf.where(tf.math.is_finite(real_loss), real_loss, tf.constant(0.0, dtype=tf.float32))
   
    # Calculate generated loss with clipping and NaN protection
    generated_losses = []
    for output in disc_generated_output:
        output_f32 = tf.cast(output, tf.float32)
        # Clip discriminator outputs to prevent extreme values
        output_f32 = tf.clip_by_value(output_f32, -20.0, 20.0)
        # Use epsilon for numerical stability
        labels = tf.zeros_like(output_f32)
        per_output_loss = loss_fn(labels, output_f32)
        # Handle NaN values
        per_output_loss = tf.where(tf.math.is_finite(per_output_loss), per_output_loss, tf.zeros_like(per_output_loss))
        generated_losses.append(tf.reduce_mean(per_output_loss))
   
    # Safely compute mean
    generated_loss = tf.reduce_mean(generated_losses) if generated_losses else tf.constant(0.0, dtype=tf.float32)
    generated_loss = tf.where(tf.math.is_finite(generated_loss), generated_loss, tf.constant(0.0, dtype=tf.float32))
   
    # Final loss with NaN protection
    total_loss = real_loss + generated_loss
    total_loss = tf.where(tf.math.is_finite(total_loss), total_loss, tf.constant(0.1, dtype=tf.float32))
   
    return tf.cast(total_loss, tf.float32)

def compute_gradient_penalty(discriminator, real_images, fake_images, terrain_labels):
    """Gradient penalty calculation with enhanced numerical stability"""
    # Convert inputs to fp32 for accurate gradient calculation
    real_images = tf.cast(real_images, tf.float32)
    fake_images = tf.cast(fake_images, tf.float32)
    terrain_labels = tf.cast(terrain_labels, tf.float32)
   
    # Clip images to ensure they're in valid range
    real_images = tf.clip_by_value(real_images, -1.0, 1.0)
    fake_images = tf.clip_by_value(fake_images, -1.0, 1.0)
   
    # Use dynamic batch size
    batch_size = tf.shape(real_images)[0]
    alpha = tf.random.uniform(
        [batch_size, 1, 1, 1], 0.0, 1.0, dtype=tf.float32
    )
   
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff
   
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator([interpolated, interpolated, terrain_labels], training=True)
       
    grads = gp_tape.gradient(pred, [interpolated])[0]
   
    # Handle potentially invalid gradients
    grads = tf.where(tf.math.is_finite(grads), grads, tf.zeros_like(grads))
   
    # Add small epsilon to prevent division by zero
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-10)
   
    # Clip gradient norm to prevent extreme values
    norm = tf.clip_by_value(norm, 0.0, 10.0)
   
    gp = tf.reduce_mean((norm - 1.0) ** 2)
   
    # Final NaN check
    gp = tf.where(tf.math.is_finite(gp), gp, tf.constant(0.0, dtype=tf.float32))
   
    # Convert to same dtype as other losses
    return tf.cast(gp, tf.float16)


def train_step(sar_images, color_images, terrain_labels, generator, discriminator,
               generator_optimizer, discriminator_optimizer, metrics_tracker, step):
    """Modified training step with improved numerical stability"""
   
    # Initialize loss values with default 0
    gen_total_loss = tf.constant(0.0, dtype=tf.float32)
    disc_loss = tf.constant(0.0, dtype=tf.float32)
    l1_loss = tf.constant(0.0, dtype=tf.float32)
    gan_loss = tf.constant(0.0, dtype=tf.float32)
    perceptual_loss = tf.constant(0.0, dtype=tf.float32)
    psnr = tf.constant(0.0, dtype=tf.float32)
    ssim = tf.constant(0.0, dtype=tf.float32)
    cycle_loss = tf.constant(0.0, dtype=tf.float32)
    l2_loss = tf.constant(0.0, dtype=tf.float32)
    feature_matching_loss = tf.constant(0.0, dtype=tf.float32)
    lpips = tf.constant(0.0, dtype=tf.float32)
    
    # Define loss_scale here so it's available in all code paths
    loss_scale = tf.constant(128.0, dtype=tf.float32)
    
    # Ensure input tensors have correct shape and type
    sar_images = tf.cast(sar_images, tf.float32)
    color_images = tf.cast(color_images, tf.float32)
    terrain_labels = tf.cast(terrain_labels, tf.float32)
   
    # Clip input images to valid range
    sar_images = tf.clip_by_value(sar_images, -1.0, 1.0)
    color_images = tf.clip_by_value(color_images, -1.0, 1.0)

    try:
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate images with gradient clipping
            generated_images = generator([sar_images, terrain_labels], training=True)
            generated_images_f32 = tf.cast(generated_images, tf.float32)
            
            # Clip generated images to valid range
            generated_images_f32 = tf.clip_by_value(generated_images_f32, -1.0, 1.0)
            
            # Check for NaN in generated images
            has_nan = tf.reduce_any(tf.math.is_nan(generated_images_f32))
            
            # Create dummy tensors that match the expected output structure of discriminator
            # This ensures consistent tensor structure across all code paths
            dummy_disc_output_shape = [
                tf.zeros([tf.shape(sar_images)[0], 32, 32, 1], dtype=tf.float32), 
                tf.zeros([tf.shape(sar_images)[0], 16, 16, 1], dtype=tf.float32)
            ]
            
            # Instead of using tf.cond which can cause graph execution issues,
            # we'll always compute discriminator outputs and then selectively use them
            # Initialize with dummy tensor structure that matches discriminator output
            disc_real_output = dummy_disc_output_shape
            disc_generated_output = dummy_disc_output_shape
            
            # Make safe versions that are used when no NaNs are detected
            def compute_disc_outputs():
                real_output = discriminator([sar_images, color_images, terrain_labels], training=True)
                gen_output = discriminator([sar_images, generated_images_f32, terrain_labels], training=True)
                return real_output, gen_output
            
            # Use tf.cond to ensure consistent tensor structure
            disc_real_output, disc_generated_output = tf.cond(
                tf.logical_not(has_nan),
                lambda: compute_disc_outputs(),
                lambda: (dummy_disc_output_shape, dummy_disc_output_shape)
            )
            
            # Calculate generator losses with safe ops
            gen_loss_result = generator_loss(disc_generated_output, generated_images_f32, color_images)
            gen_total_loss_raw, gan_loss, l1_loss, perceptual_loss, psnr, ssim = gen_loss_result
            
            # Ensure gen_total_loss is float32 type
            gen_total_loss = tf.cast(gen_total_loss_raw, tf.float32)
            
            # Apply loss scaling for numerical stability in mixed precision
            gen_total_loss = gen_total_loss * loss_scale
            
            # Calculate discriminator loss with safe ops
            grad_penalty = compute_gradient_penalty(discriminator, color_images, generated_images_f32, terrain_labels)
            grad_penalty = tf.cast(grad_penalty, tf.float32)
            
            # Clip gradient penalty to reasonable range
            grad_penalty = tf.clip_by_value(grad_penalty, 0.0, 100.0)
            
            disc_base_loss = discriminator_loss(disc_real_output, disc_generated_output)
            disc_base_loss = tf.cast(disc_base_loss, tf.float32)
            disc_loss = disc_base_loss + 10.0 * grad_penalty
            
            # Apply loss scaling for numerical stability
            disc_loss = disc_loss * loss_scale
            
            # Skip gradient updates if losses are NaN or we detected NaN in generated images
            should_apply_gradients = tf.logical_and(
                tf.logical_not(has_nan),
                tf.logical_and(tf.math.is_finite(gen_total_loss), tf.math.is_finite(disc_loss))
            )
            
            if should_apply_gradients:
                # Compute gradients
                gen_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
                disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                
                # Unscale gradients (to account for loss scaling)
                gen_gradients = [g / loss_scale if g is not None else None for g in gen_gradients]
                disc_gradients = [g / loss_scale if g is not None else None for g in disc_gradients]
                
                # Replace NaN/Inf gradients with zeros
                gen_gradients = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                                 if g is not None else None for g in gen_gradients]
                disc_gradients = [tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                                  if g is not None else None for g in disc_gradients]
                
                # Clip gradients to prevent explosion
                gen_gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gen_gradients]
                disc_gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in disc_gradients]
                
                # Apply gradients
                generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
                
                # Calculate additional metrics safely (outside of gradient computation)
                def safe_reduce_mean(x):
                    x = tf.where(tf.math.is_finite(x), x, tf.zeros_like(x))
                    return tf.reduce_mean(x)
                
                # Calculate additional metrics with protection against NaN
                try:
                    cycle_reconstructed = generator([generated_images_f32, terrain_labels], training=False)
                    cycle_reconstructed = tf.clip_by_value(cycle_reconstructed, -1.0, 1.0)
                    cycle_loss = safe_reduce_mean(tf.abs(sar_images - cycle_reconstructed))
                   
                    l2_loss = safe_reduce_mean(tf.square(color_images - generated_images_f32))
                   
                    # Feature matching loss (simplified to avoid NaN)
                    feature_matching_loss = tf.reduce_mean([
                        safe_reduce_mean(tf.abs(tf.cast(real, tf.float32) - tf.cast(gen, tf.float32)))
                        for real, gen in zip(disc_real_output, disc_generated_output)
                    ])
                   
                    # LPIPS with safe ops (simplified)
                    lpips = 0.0
                    if feature_extractor is not None:
                        try:
                            real_features = feature_extractor(color_images)
                            gen_features = feature_extractor(generated_images_f32)
                            lpips = tf.reduce_mean([
                                safe_reduce_mean(tf.abs(rf - gf))
                                for rf, gf in zip(real_features, gen_features)
                            ])
                        except Exception as e:
                            print(f"LPIPS calculation error: {e}")
                            lpips = 0.0
                except Exception as e:
                    print(f"Error in metrics calculation: {e}")
                    cycle_loss = 0.0
                    l2_loss = 0.0
                    feature_matching_loss = 0.0
                    lpips = 0.0
            else:
                # If we're skipping due to NaNs, we don't update anything
                gen_total_loss = tf.constant(0.1, dtype=tf.float32)
                disc_loss = tf.constant(0.1, dtype=tf.float32)
                cycle_loss = tf.constant(0.0, dtype=tf.float32)
                l2_loss = tf.constant(0.0, dtype=tf.float32)
                feature_matching_loss = tf.constant(0.0, dtype=tf.float32)
                lpips = tf.constant(0.0, dtype=tf.float32)
    except Exception as e:
        print(f"\nException in train step: {e}")
        # Return default metrics on error
        metrics = {
            'gen_total_loss': 0.0,
            'disc_loss': 0.0,
            'l1_loss': 0.0,
            'psnr': 0.0,
            'ssim': 0.0,
            'cycle_loss': 0.0,
            'l2_loss': 0.0,
            'feature_matching_loss': 0.0,
            'lpips': 0.0,
        }
        return metrics, tf.zeros_like(sar_images)

    # Unscale the losses (to account for loss scaling) - only if not NaN
    if not has_nan:
        gen_total_loss = gen_total_loss / loss_scale
        disc_loss = disc_loss / loss_scale

    # Return metrics with NaN protection
    metrics = {
        'gen_total_loss': tf.where(tf.math.is_finite(gen_total_loss), gen_total_loss, tf.constant(0.0, dtype=tf.float32)),
        'disc_loss': tf.where(tf.math.is_finite(disc_loss), disc_loss, tf.constant(0.0, dtype=tf.float32)),
        'l1_loss': tf.where(tf.math.is_finite(l1_loss), l1_loss, tf.constant(0.0, dtype=tf.float32)),
        'psnr': tf.where(tf.math.is_finite(psnr), psnr, tf.constant(0.0, dtype=tf.float32)),
        'ssim': tf.where(tf.math.is_finite(ssim), ssim, tf.constant(0.0, dtype=tf.float32)),
        'cycle_loss': tf.where(tf.math.is_finite(cycle_loss), cycle_loss, tf.constant(0.0, dtype=tf.float32)),
        'l2_loss': tf.where(tf.math.is_finite(l2_loss), l2_loss, tf.constant(0.0, dtype=tf.float32)),
        'feature_matching_loss': tf.where(tf.math.is_finite(feature_matching_loss), feature_matching_loss, tf.constant(0.0, dtype=tf.float32)),
        'lpips': tf.where(tf.math.is_finite(lpips), lpips, tf.constant(0.0, dtype=tf.float32)),
    }
    return metrics, generated_images_f32

def get_initial_history():
    return {
        'gen_loss': [], 'disc_loss': [],
        'psnr': [], 'ssim': [],
        'cycle_loss': [], 'l2_loss': [],
        'feature_matching_loss': [], 'lpips': [],
        'l1_loss': [], 'val_disc_loss': [],
        'val_gen_loss': [], 'val_psnr': [],
        'val_ssim': [], 'val_l1_loss': [],
        'val_l2_loss': [], 'val_cycle_loss': [],
        'val_feature_matching_loss': [],
        'val_lpips': [], 'fid': [],
        'val_fid': []
    }

def calculate_detailed_metrics(sar_images, color_images, generated_images, cycle_reconstructed, terrain_labels):
    """Calculate comprehensive metrics for model evaluation"""
    # Initialize metrics dictionary
    metrics = {}
    # Ensure all inputs are float32 for consistent calculations
    sar_images = tf.cast(sar_images, tf.float32)
    color_images = tf.cast(color_images, tf.float32)
    generated_images = tf.cast(generated_images, tf.float32)
    cycle_reconstructed = tf.cast(cycle_reconstructed, tf.float32)
    try:
        metrics['l1_loss'] = tf.reduce_mean(tf.abs(color_images - generated_images)).numpy()
        metrics['l2_loss'] = tf.reduce_mean(tf.square(color_images - generated_images)).numpy()
        metrics['psnr'] = tf.image.psnr(color_images, generated_images, max_val=2.0).numpy().mean()
        metrics['ssim'] = tf.image.ssim(color_images, generated_images, max_val=2.0).numpy().mean()
        metrics['cycle_loss'] = tf.reduce_mean(tf.abs(sar_images - cycle_reconstructed)).numpy()
        metrics['perceptual_loss'] = compute_perceptual_loss(color_images, generated_images).numpy()
        # Calculate LPIPS using the feature extractor
        # LPIPS measures perceptual similarity using deep features
        try:
            # Preprocess images for feature extraction
            real_images_processed = efficientnet.preprocess_input((color_images + 1) * 127.5)
            generated_images_processed = efficientnet.preprocess_input((generated_images + 1) * 127.5)
            # Extract features
            real_features = feature_extractor(real_images_processed)
            gen_features = feature_extractor(generated_images_processed)
            # Calculate normalized distance between feature representations
            lpips_value = 0.0
            for rf, gf in zip(real_features, gen_features):
                # Normalize features
                rf_norm = tf.nn.l2_normalize(rf, axis=-1)
                gf_norm = tf.nn.l2_normalize(gf, axis=-1)
                # Compute distance
                lpips_value += tf.reduce_mean(tf.square(rf_norm - gf_norm))
            metrics['lpips'] = lpips_value.numpy() / len(real_features)
        except Exception as e:
            print(f"Error calculating LPIPS: {e}")
            metrics['lpips'] = 0.0
        # Calculate feature matching loss
        # This measures style similarity using discriminator features
        try:
            # We need the discriminator to extract features
            # Since we can't directly access it here, we'll use a simplified approach
            # using our feature extractor as a substitute
            # Extract mid-level features (these represent style information)
            style_features_real = real_features[1]  # Using middle layer features
            style_features_gen = gen_features[1]
            # Calculate mean and variance for each feature map (Gram matrix simplified)
            real_mean = tf.reduce_mean(style_features_real, axis=[1, 2], keepdims=True)
            gen_mean = tf.reduce_mean(style_features_gen, axis=[1, 2], keepdims=True)
            real_std = tf.math.reduce_std(style_features_real, axis=[1, 2], keepdims=True)
            gen_std = tf.math.reduce_std(style_features_gen, axis=[1, 2], keepdims=True)
            # Feature matching loss combines differences in mean and standard deviation
            mean_loss = tf.reduce_mean(tf.abs(real_mean - gen_mean))
            std_loss = tf.reduce_mean(tf.abs(real_std - gen_std))
            metrics['feature_matching_loss'] = (mean_loss + std_loss).numpy()
        except Exception as e:
            print(f"Error calculating feature matching loss: {e}")
            metrics['feature_matching_loss'] = 0.0
    except Exception as e:
        print(f"Error in metrics: {e}")
        metrics = {k: 0.0 for k in [
            'l1_loss', 'l2_loss', 'psnr', 'ssim', 'cycle_loss',
            'perceptual_loss', 'lpips', 'feature_matching_loss', 'fid'
        ]}
    return metrics

class MetricsTracker:
    def __init__(self):
        self.gen_loss_tracker = tf.keras.metrics.Mean(name='generator_loss')
        self.disc_loss_tracker = tf.keras.metrics.Mean(name='discriminator_loss')
        self.psnr_metric = tf.keras.metrics.Mean(name='psnr')
        self.ssim_metric = tf.keras.metrics.Mean(name='ssim')
        self.perceptual_loss_tracker = tf.keras.metrics.Mean(name='perceptual_loss')
        self.l1_loss_tracker = tf.keras.metrics.Mean(name='l1_loss')
        self.l2_loss_tracker = tf.keras.metrics.Mean(name='l2_loss')
        #self.fid_tracker = tf.keras.metrics.Mean(name='fid')
        #self.is_tracker = tf.keras.metrics.Mean(name='inception_score')
        self.lpips_tracker = tf.keras.metrics.Mean(name='lpips')

class SARHyperModel(kt.HyperModel):
    def __init__(self):
        super().__init__()
        self.discriminator = None  # Will store the discriminator here

    def build(self, hp):
        """Build model with hyperparameters"""
        with strategy.scope():
            # Create generator and discriminator with standard architecture
            generator = build_terrain_aware_generator()
            discriminator = build_terrain_aware_discriminator()
            # Set hyperparameters to be tuned
            gen_lr = hp.Float('gen_lr', min_value=5e-5, max_value=3e-4, sampling='log', default=2e-4)
            disc_lr = hp.Float('disc_lr', min_value=5e-5, max_value=3e-4, sampling='log', default=2e-5)
            lambda_val = hp.Int('lambda', min_value=8, max_value=15, step=1, default=12)
            perceptual_weight = hp.Float('perceptual_weight', min_value=0.05, max_value=0.3, step=0.025, default=0.1)
            # Store hyperparameters in the model for later access
            generator.lambda_val = lambda_val
            generator.perceptual_weight = perceptual_weight
            generator.gen_lr = gen_lr
            generator.disc_lr = disc_lr
            generator.trial_id = "0"  # Default value that will be updated in fit()
            # Initialize optimizers with the tunable learning rates
            generator_optimizer = tf.keras.optimizers.Adam(
                gen_lr,
                beta_1=0.5,
                clipnorm=1.0
            )
            discriminator_optimizer = tf.keras.optimizers.Adam(
                disc_lr,
                beta_1=0.5,
                clipnorm=1.0
            )
            # Attach optimizers to models
            generator.optimizer = generator_optimizer
            discriminator.optimizer = discriminator_optimizer
            # Save discriminator as instance attribute
            self.discriminator = discriminator
            # Return only the generator model (Keras Tuner expects a single model)
            return generator

    def fit(self, hp_or_model, x=None, y=None, **kwargs):
        """Custom training procedure for the hypermodel"""
        # Check if we received hyperparameters or a model
        if isinstance(hp_or_model, kt.HyperParameters):
            # We were passed hyperparameters directly (happens on trial failures)
            hp = hp_or_model
            # Rebuild the model with these hyperparameters
            generator = self.build(hp)
            # Get hyperparameters directly from hp object
            lambda_val = hp.get('lambda')
            perceptual_weight = hp.get('perceptual_weight')
            gen_lr = hp.get('gen_lr')
            disc_lr = hp.get('disc_lr')
        else:
            # We were passed the model directly as expected
            generator = hp_or_model
            # Extract hyperparameters from the model
            lambda_val = generator.lambda_val
            perceptual_weight = generator.perceptual_weight
            gen_lr = generator.gen_lr
            disc_lr = generator.disc_lr
        # Get the previously built discriminator
        discriminator = self.discriminator
        # Add trial_id to the generator based on kwargs
        if 'trial' in kwargs and hasattr(kwargs['trial'], 'trial_id'):
            trial_id = str(kwargs['trial'].trial_id)
            generator.trial_id = trial_id
        else:
            trial_id = getattr(generator, 'trial_id', "0")  # Use default if not available
        print(f"\n--- Starting Trial {trial_id} ---")
        print(f"λ: {lambda_val}, Perceptual Weight: {perceptual_weight:.4f}")
        print(f"Gen LR: {gen_lr:.6f}, Disc LR: {disc_lr:.6f}")
        # Get datasets from kwargs
        train_ds = kwargs.get('train_ds')
        val_ds = kwargs.get('val_ds')
        epochs = kwargs.get('epochs', MAX_EPOCHS_PER_TRIAL)
        # Create distributed datasets
        dist_train_ds = strategy.experimental_distribute_dataset(train_ds)
        dist_val_ds = strategy.experimental_distribute_dataset(val_ds)
        # Initialize metrics tracker and history
        metrics_tracker = MetricsTracker()
        history = get_initial_history()
        train_metrics = defaultdict(list)
        val_metrics = defaultdict(list)
        # For visualization
        val_vis_dataset = val_ds.take(2).cache()

        # Remove @tf.function from distributed_train_step and distributed_val_step
        def distributed_train_step(dist_inputs, generator, discriminator, metrics_tracker, lambda_val, perceptual_weight):
            """Distributed training step without nested control flow crossing synchronization boundaries"""
            def train_step_fn(inputs):
                sar_images, color_images, terrain_labels = inputs
                sar_images = tf.clip_by_value(sar_images, -1.0, 1.0)
                color_images = tf.clip_by_value(color_images, -1.0, 1.0)
                global LAMBDA
                original_lambda = LAMBDA
                LAMBDA = lambda_val
                per_replica_metrics, generated_images = train_step(
                    sar_images, color_images, terrain_labels,
                    generator, discriminator,
                    generator.optimizer, discriminator.optimizer,
                    metrics_tracker, 0
                )
                LAMBDA = original_lambda
                per_replica_metrics['perceptual_weight'] = perceptual_weight
                return per_replica_metrics, generated_images

            per_replica_results = strategy.run(train_step_fn, args=(dist_inputs,))
            per_replica_metrics, _ = per_replica_results
            reduced_metrics = {}
            for k, v in per_replica_metrics.items():
                if isinstance(v, tf.Tensor):
                    reduced_metrics[k] = strategy.reduce(tf.distribute.ReduceOp.MEAN, v, axis=None)
                else:
                    reduced_metrics[k] = v
            return reduced_metrics

        def distributed_val_step(dist_inputs):
            """Distributed validation step"""
            def val_step_fn(inputs):
                sar_batch, color_batch, terrain_batch = inputs
                sar_batch = tf.clip_by_value(sar_batch, -1.0, 1.0)
                color_batch = tf.clip_by_value(color_batch, -1.0, 1.0)
                generated_images = generator([sar_batch, terrain_batch], training=False)
                generated_images = tf.clip_by_value(generated_images, -1.0, 1.0)
                cycle_reconstructed = generator([generated_images, terrain_batch], training=False)
                disc_real_output = discriminator([sar_batch, color_batch, terrain_batch], training=False)
                disc_generated_output = discriminator([sar_batch, generated_images, terrain_batch], training=False)
                gen_loss_result = generator_loss(disc_generated_output, generated_images, color_batch)
                gen_total_loss, gan_loss, l1_loss, perceptual_loss, psnr, ssim = gen_loss_result
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
                try:
                    if color_batch.shape[0] >= 2:
                        real_images = efficientnet.preprocess_input((color_batch + 1) * 127.5)
                        gen_images = efficientnet.preprocess_input((generated_images + 1) * 127.5)
                        real_features = feature_extractor(real_images)
                        gen_features = feature_extractor(gen_images)
                        real_feats = real_features[-1]
                        gen_feats = gen_features[-1]
                        fid = tf.reduce_mean(tf.square(
                            tf.reduce_mean(real_feats, axis=[1, 2]) -
                            tf.reduce_mean(gen_feats, axis=[1, 2])
                        ))
                    else:
                        fid = tf.constant(0.0)
                except Exception as e:
                    fid = tf.constant(0.0)
                metrics_dict = {
                    'gen_total_loss': gen_total_loss,
                    'disc_loss': disc_loss,
                    'psnr': psnr,
                    'ssim': ssim,
                    'l1_loss': l1_loss,
                    'perceptual_loss': perceptual_loss,
                    'fid': fid
                }
                return metrics_dict
            per_replica_metrics = strategy.run(val_step_fn, args=(dist_inputs,))
            reduced_metrics = {}
            for k, v in per_replica_metrics.items():
                if isinstance(v, tf.Tensor):
                    reduced_metrics[k] = strategy.reduce(tf.distribute.ReduceOp.MEAN, v, axis=None)
                else:
                    reduced_metrics[k] = v
            return reduced_metrics

        # Training loop
        for epoch in range(epochs):
            start_time = time.time()
            # Training phase
            train_step_metrics = defaultdict(list)
            for step, dist_inputs in enumerate(dist_train_ds):
                try:
                    # Use the updated distributed training step
                    step_metrics = distributed_train_step(
                        dist_inputs, generator, discriminator, 
                        metrics_tracker, lambda_val, perceptual_weight
                    )
                    # Store metrics
                    for k, v in step_metrics.items():
                        if isinstance(v, tf.Tensor):
                            train_step_metrics[k].append(v.numpy())
                        else:
                            train_step_metrics[k].append(v)
                    # Print progress for every 500 steps
                    if step > 0 and step % 500 == 0:
                        print(f"  Step {step} - gen: {np.mean(train_step_metrics['gen_total_loss'][-500:]):.4f}, "
                              f"disc: {np.mean(train_step_metrics['disc_loss'][-500:]):.4f}")
                except Exception as e:
                    print(f"Error in training step: {e}")
            # Calculate average training metrics
            avg_train_metrics = {k: np.mean(v) for k, v in train_step_metrics.items() if len(v) > 0}
            for k, v in avg_train_metrics.items():
                train_metrics[k].append(float(v))
            # Validation phase
            val_step_metrics = defaultdict(list)
            for val_batch in dist_val_ds:
                try:
                    batch_metrics = distributed_val_step(val_batch)
                    # Store metrics
                    for k, v in batch_metrics.items():
                        if isinstance(v, tf.Tensor):
                            val_step_metrics[k].append(v.numpy())
                        else:
                            val_step_metrics[k].append(v)
                except Exception as e:
                    print(f"Error in validation step: {e}")
            # Calculate average validation metrics
            avg_val_metrics = {k: np.mean(v) for k, v in val_step_metrics.items() if len(v) > 0}
            for k, v in avg_val_metrics.items():
                val_metrics[k].append(float(v))
            # Update history with more metrics
            for k, v in avg_train_metrics.items():
                if k in history:
                    history[k].append(float(v))
            for k, v in avg_val_metrics.items():
                val_key = f'val_{k}'
                if val_key in history:
                    history[val_key].append(float(v))
            # Calculate generator-discriminator loss balance metric
            gen_loss = avg_train_metrics.get('gen_total_loss', 0.0)
            disc_loss = avg_train_metrics.get('disc_loss', 0.0)
            # Calculate loss ratio and balance score
            loss_ratio = gen_loss / (disc_loss + 1e-7)  # Avoid division by zero
            loss_balance = 1.0 / (abs(np.log10(loss_ratio)) + 1.0)
            # Calculate mode collapse indicators
            mode_collapse_risk = "Low"
            if gen_loss < 0.05 or disc_loss < 0.05:
                mode_collapse_risk = "High"
            elif loss_ratio > 10 or loss_ratio < 0.1:
                mode_collapse_risk = "Medium"
            # Format metrics for printing
            val_psnr = avg_val_metrics.get('psnr', 0.0)
            val_ssim = avg_val_metrics.get('ssim', 0.0)
            val_fid = avg_val_metrics.get('fid', 0.0)
            # Print comprehensive epoch summary
            print(f"\nEpoch {epoch+1}/{epochs} - Time: {time.time() - start_time:.1f}s")
            print(f"  Training:   Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}, "
                  f"Ratio: {loss_ratio:.2f}, Balance: {loss_balance:.4f}")
            print(f"  Validation: PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}, FID: {val_fid:.4f}")
            print(f"  Mode collapse risk: {mode_collapse_risk}, Loss Balance: {loss_balance:.4f}")
            # Save a visualization periodically
            if (epoch + 1) % 2 == 0:
                try:
                    # Create directory for trial visualizations
                    viz_dir = f'hp_tuning_viz/trial-{trial_id}'
                    os.makedirs(viz_dir, exist_ok=True)
                    # Generate and save a sample image
                    for i, (sar_batch, color_batch, terrain_batch) in enumerate(val_vis_dataset.take(1)):
                        generated_images = generator([sar_batch, terrain_batch], training=False)
                        # Plot comparison
                        plt.figure(figsize=(12, 4))
                        for j in range(min(1, len(sar_batch))):
                            plt.subplot(1, 3, 1)
                            plt.title(f"Input SAR")
                            plt.imshow(tf.cast(sar_batch[j] * 0.5 + 0.5, tf.float32).numpy())
                            plt.axis('off')
                            plt.subplot(1, 3, 2)
                            plt.title(f"Ground Truth")
                            plt.imshow(tf.cast(color_batch[j] * 0.5 + 0.5, tf.float32).numpy())
                            plt.axis('off')
                            plt.subplot(1, 3, 3)
                            plt.title(f"Generated")
                            plt.imshow(tf.cast(generated_images[j] * 0.5 + 0.5, tf.float32).numpy())
                            plt.axis('off')
                        plt.savefig(f"{viz_dir}/epoch_{epoch+1}.png", bbox_inches='tight')
                        plt.close()
                except Exception as viz_error:
                    print(f"  Visualization error: {viz_error}")
        # Evaluate the final model
        # Calculate objective metrics for hyperparameter search with emphasis on balance
        # Get the last few epochs' metrics (more stable than just the last one)
        final_metrics = {}
        if len(val_metrics['psnr']) >= 2:
            final_metrics['psnr'] = np.mean(val_metrics['psnr'][-2:])
            final_metrics['ssim'] = np.mean(val_metrics['ssim'][-2:])
            final_metrics['fid'] = np.mean(val_metrics['fid'][-2:])
            final_metrics['l1_loss'] = np.mean(val_metrics['l1_loss'][-2:])
        else:
            final_metrics['psnr'] = val_metrics['psnr'][-1] if val_metrics['psnr'] else 0.0
            final_metrics['ssim'] = val_metrics['ssim'][-1] if val_metrics['ssim'] else 0.0
            final_metrics['fid'] = val_metrics['fid'][-1] if val_metrics['fid'] else 0.0
            final_metrics['l1_loss'] = val_metrics['l1_loss'][-1] if val_metrics['l1_loss'] else 1.0
        # Training stability metrics
        if len(train_metrics['gen_total_loss']) >= 3:
            final_metrics['gen_loss'] = np.mean(train_metrics['gen_total_loss'][-3:])
            final_metrics['disc_loss'] = np.mean(train_metrics['disc_loss'][-3:])
        else:
            final_metrics['gen_loss'] = train_metrics['gen_total_loss'][-1] if train_metrics['gen_total_loss'] else 1.0
            final_metrics['disc_loss'] = train_metrics['disc_loss'][-1] if train_metrics['disc_loss'] else 1.0
        # Calculate loss balance score
        loss_ratio = final_metrics['gen_loss'] / (final_metrics['disc_loss'] + 1e-7)
        loss_balance = 1.0 / (abs(np.log10(loss_ratio)) + 1)
        # Penalize if either loss is too small (sign of collapse)
        min_acceptable_loss = 0.05
        if final_metrics['gen_loss'] < min_acceptable_loss or final_metrics['disc_loss'] < min_acceptable_loss:
            loss_balance *= 0.5
        # Normalize metrics for combined score
        norm_psnr = min(final_metrics['psnr'] / 30.0, 1.0)  # Normalize PSNR with max expected ~30
        norm_ssim = final_metrics['ssim']  # SSIM is already 0-1
        norm_fid = 1.0 / (final_metrics['fid'] + 1.0)  # Lower FID is better, invert and normalize
        norm_l1 = 1.0 - min(final_metrics['l1_loss'], 1.0)  # Lower L1 is better, invert and normalize
        # Combined quality score with weighted components
        quality_score = (0.3 * norm_psnr) + (0.3 * norm_ssim) + (0.1 * norm_fid) + (0.15 * norm_l1) + (0.15 * loss_balance)
        # Print final trial results
        print("\n--- Trial Results ---")
        print(f"Trial {trial_id}: Quality Score = {quality_score:.4f}")
        print(f"Image Quality: PSNR = {final_metrics['psnr']:.2f}, SSIM = {final_metrics['ssim']:.4f}, " 
              f"FID = {final_metrics['fid']:.4f}")
        print(f"Loss Balance (0-1): {loss_balance:.4f} (Gen: {final_metrics['gen_loss']:.4f}, "
              f"Disc: {final_metrics['disc_loss']:.4f}, Ratio: {loss_ratio:.2f})")
        print(f"Hyperparameters: λ={lambda_val}, Perceptual Weight={perceptual_weight:.4f}, "
              f"Gen LR={gen_lr:.6f}, Disc LR={disc_lr:.6f}")
        # Return metrics dictionary with quality score for the tuner to optimize
        return {
            "quality_score": float(quality_score),
            "val_psnr": float(final_metrics['psnr']),
            "val_ssim": float(final_metrics['ssim']),
            "val_fid": float(final_metrics['fid']),
            "loss_balance": float(loss_balance)
        }

# Remove reference to MemoryEfficientDatasetLoader and add create_dataset_for_tuning function
def create_dataset_for_tuning():
    """Create memory-efficient dataset for hyperparameter tuning"""
    # Get the original datasets from create_dataset()
    train_ds, val_ds, test_ds = create_dataset()
    # Take a smaller subset for faster hyperparameter tuning
    # This avoids loading the entire dataset into memory
    reduced_train_ds = train_ds.take(8000).cache().prefetch(tf.data.AUTOTUNE)
    reduced_val_ds = val_ds.take(1600).cache().prefetch(tf.data.AUTOTUNE)
    print("Created reduced datasets for tuning: 8000 training samples, 1600 validation samples")
    return reduced_train_ds, reduced_val_ds

class HyperparameterTuner:
    """Manages the hyperparameter tuning process"""
    def __init__(self, project_name=TUNER_DIR):
        self.project_name = project_name
        self.hypermodel = SARHyperModel()

    def create_tuner(self):
        """Create and configure the tuner using Hyperband with balanced loss objective"""
        # Create a custom metric that prioritizes generator-discriminator balance
        tuner = Hyperband(
            self.hypermodel,
            objective=kt.Objective("loss_balance", direction="max"),  # Changed from val_psnr to loss_balance
            max_epochs=MAX_EPOCHS_PER_TRIAL,
            factor=3,  # Default Hyperband factor
            directory=self.project_name,
            project_name="sar_tuning",
            overwrite=False
        )
        return tuner

    def run_tuning(self):
        """Run the hyperparameter search"""
        print("Loading datasets...")
        # Use the function instead of data_loader
        train_ds, val_ds = create_dataset_for_tuning()
        print("Creating tuner...")
        tuner = self.create_tuner()
        # Print search space summary
        tuner.search_space_summary()
        print(f"Starting Hyperband hyperparameter search...")
        try:
            tuner.search(
                train_ds=train_ds,
                val_ds=val_ds,
                epochs=MAX_EPOCHS_PER_TRIAL,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_psnr',
                        patience=2,
                        restore_best_weights=True
                    )
                ]
            )
        except Exception as e:
            print(f"Error during hyperparameter search: {e}")
        # Get best hyperparameters
        best_hp = tuner.get_best_hyperparameters(1)[0]
        # Print best hyperparameters
        print("\nBest Hyperparameters:")
        print(f"Generator Learning Rate: {best_hp.get('gen_lr')}")
        print(f"Discriminator Learning Rate: {best_hp.get('disc_lr')}")
        print(f"Lambda: {best_hp.get('lambda')}")
        print(f"Perceptual Weight: {best_hp.get('perceptual_weight')}")
        # Save best hyperparameters to file
        self.save_best_hyperparameters(best_hp)
        return best_hp

    def save_best_hyperparameters(self, best_hp):
        """Save best hyperparameters to file"""
        best_params = {
            'gen_lr': best_hp.get('gen_lr'),
            'disc_lr': best_hp.get('disc_lr'),
            'lambda': best_hp.get('lambda'),
            'perceptual_weight': best_hp.get('perceptual_weight'),
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        os.makedirs('hp_tuning_results', exist_ok=True)
        with open('hp_tuning_results/best_hyperparameters.json', 'w') as f:
            json.dump(best_params, f, indent=2)
        print(f"Best hyperparameters saved to hp_tuning_results/best_hyperparameters.json")

def main():
    """Main function to run hyperparameter tuning"""
    print("Starting SAR model hyperparameter tuning...")
    # Enable memory growth to prevent OOM errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} Physical GPUs")
        except RuntimeError as e:
            print(f"Memory growth setting error: {e}")
    # Create and run tuner
    tuner = HyperparameterTuner()
    best_hp = tuner.run_tuning()
    print("Hyperparameter tuning completed successfully!")
    # Print a summary of the best hyperparameters again
    print("\nBest Hyperparameters Summary:")
    print(f"Generator Learning Rate: {best_hp.get('gen_lr')}")
    print(f"Discriminator Learning Rate: {best_hp.get('disc_lr')}")
    print(f"Lambda: {best_hp.get('lambda')}")
    print(f"Perceptual Weight: {best_hp.get('perceptual_weight')}")
    return best_hp

if __name__ == "__main__":
    main()