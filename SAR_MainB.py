import tensorflow as tf
from collections import defaultdict
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"Found {len(gpus)} Physical GPUs and {len(logical_gpus)} Logical GPUs")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

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

strategy = setup_distributed_strategy()
tf.keras.mixed_precision.set_global_policy('mixed_float16')

import numpy as np
from keras import layers, applications
import os
from glob import glob
import matplotlib.pyplot as plt
from datetime import datetime
import time
from keras.applications import efficientnet
from keras.applications.inception_v3 import InceptionV3
from scipy import linalg
import json
from scipy.linalg import sqrtm
import tensorflow_probability as tfp



# Constants
BUFFER_SIZE = 400
BATCH_SIZE = 1  # Fixed at 1
IMG_WIDTH = 256  # Changed from 256 to 128
IMG_HEIGHT = 256  # Changed from 256 to 128
LAMBDA = 10
TERRAIN_TYPES = ['urban', 'grassland', 'agri', 'barrenland']
STYLE_WEIGHT = 1.0
CYCLE_WEIGHT = 10.0
COLOR_WEIGHT = 5.0
MODEL_SAVE_DIR = '/kaggle/working/saved_models'
CHECKPOINT_DIR = os.path.join(MODEL_SAVE_DIR, 'checkpoints')
HISTORY_DIR = os.path.join(MODEL_SAVE_DIR, 'history')
HEAVY_METRICS_INTERVAL = 200  # Calculate heavy metrics every 200 steps

# Update InstanceNormalization class to handle serialization
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

# Initialize VGG model for perceptual loss
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

# Metrics trackers
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
       
    def reset_states(self):
        self.gen_loss_tracker.reset_state()
        self.disc_loss_tracker.reset_state()
        self.psnr_metric.reset_state()
        self.ssim_metric.reset_state()

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

class OptimizedColorTransformation(layers.Layer):
    """Memory-efficient color space transformation"""
    def __init__(self, **kwargs):
        super(OptimizedColorTransformation, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(32, 1, activation='relu')
        self.conv2 = layers.Conv2D(3, 1)
        self._input_shape_tracker = None
       
    @tf.function
    def rgb_to_lab_efficient(self, rgb):
        # Optimized RGB to LAB conversion
        rgb = (rgb + 1) * 0.5  # [-1,1] to [0,1]
        features = self.conv1(rgb)
        lab_like = self.conv2(features)
        return lab_like
       
    def call(self, x):
        return self.rgb_to_lab_efficient(x)
   
    def get_config(self):
        config = super().get_config()
        return config
       
    @classmethod
    def from_config(cls, config):
        return cls(**config)
   
    def get_build_config(self):
        if hasattr(self, '_input_shape_tracker') and self._input_shape_tracker is not None:
            return {"input_shape": self._input_shape_tracker}
        return {}
       
    def build_from_config(self, config):
        if not self.built and "input_shape" in config:
            self.build(config["input_shape"])
   
class LightweightBoundaryDetection(layers.Layer):
    """Efficient boundary detection using depthwise separable convolutions"""
    def __init__(self, out_filters, **kwargs):
        super(LightweightBoundaryDetection, self).__init__(**kwargs)
        self.out_filters = out_filters
        self.instance_norm = InstanceNormalization()
        self._input_shape_tracker = None
       
    def build(self, input_shape):
        input_channels = input_shape[-1]
        self.depthwise = layers.DepthwiseConv2D(3, padding='same',
                                               depth_multiplier=self.out_filters//input_channels)
        self.pointwise = layers.Conv2D(self.out_filters, 1)
        self._input_shape_tracker = input_shape
       
    def call(self, x):
        x = self.instance_norm(x)
        x = self.depthwise(x)
        return self.pointwise(x)
   
    def get_config(self):
        config = super().get_config()
        config.update({
            "out_filters": self.out_filters,
        })
        return config
       
    @classmethod
    def from_config(cls, config):
        return cls(**config)
   
    def get_build_config(self):
        if hasattr(self, '_input_shape_tracker') and self._input_shape_tracker is not None:
            return {
                "input_shape": self._input_shape_tracker,
                "out_filters": self.out_filters
            }
        return {"out_filters": self.out_filters}
       
    def build_from_config(self, config):
        if not self.built and "input_shape" in config:
            self.build(config["input_shape"])
   
class EfficientSkipConnection(layers.Layer):
    """Memory-efficient skip connections using 1x1 convolutions"""
    def __init__(self, filters, **kwargs):
        super(EfficientSkipConnection, self).__init__(**kwargs)
        self.filters = filters
        self.reduction = layers.Conv2D(filters//4, 1)
        self.process = layers.DepthwiseConv2D(3, padding='same')
        self._input_shape_tracker = None
       
    def call(self, x, skip_features):
        processed = [self.reduction(feat) for feat in skip_features]
        processed = [self.process(feat) for feat in processed]
        return tf.concat([x] + processed, axis=-1)
   
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
   
class TerrainAwareResBlock(layers.Layer):
    """Residual block with terrain-specific processing"""
    def __init__(self, filters, **kwargs):
        super(TerrainAwareResBlock, self).__init__(**kwargs)
        self.filters = filters
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.norm1 = TerrainAdaptiveNormalization(filters)
        self.norm2 = TerrainAdaptiveNormalization(filters)
        self.attention = TerrainGuidedAttention(filters)
        self._input_shape_tracker = None
       
    def call(self, x, terrain_features):
        residual = x
       
        x = self.conv1(x)
        x = self.norm1([x, terrain_features])
        x = tf.nn.relu(x)
       
        x = self.conv2(x)
        x = self.norm2([x, terrain_features])
        x = self.attention([x, terrain_features])
       
        return tf.nn.relu(x + residual)
   
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

class ColorRefinementBlock(layers.Layer):
    """Enhanced color refinement with spatial and semantic context"""
    def __init__(self, filters, **kwargs):
        super(ColorRefinementBlock, self).__init__(**kwargs)
        self.filters = filters
        self.context_conv = layers.Conv2D(filters, 3, padding='same')
        self.terrain_norm = TerrainAdaptiveNormalization(filters)
        self.attention = TerrainGuidedAttention(filters)
        self.refine_conv = layers.Conv2D(filters, 1)
        self.activation = layers.Activation('relu')
        self._input_shape_tracker = None
       
    def build(self, input_shape):
        if not isinstance(input_shape, (list, tuple)):
            raise ValueError("ColorRefinementBlock expects a list of input shapes")
           
        x_shape, terrain_shape = input_shape
       
        # Build layers directly without shape manipulation
        self.context_conv.build(x_shape)
       
        # Build normalization and attention with known filter size
        conv_output_shape = tf.TensorShape(x_shape[:3]).concatenate(tf.TensorShape([self.filters]))
       
        self.terrain_norm.build([conv_output_shape, terrain_shape])
        self.attention.build([conv_output_shape, terrain_shape])
        self.refine_conv.build(conv_output_shape)
        self._input_shape_tracker = input_shape
       
        super(ColorRefinementBlock, self).build(input_shape)
       
    def call(self, inputs, training=None):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError("ColorRefinementBlock expects [x, terrain_features] as input")
       
        x, terrain_features = inputs
       
        x = self.context_conv(x)
        x = self.terrain_norm([x, terrain_features], training=training)
        x = self.activation(x)
        x = self.attention([x, terrain_features])
        return self.refine_conv(x)
       
    def compute_output_shape(self, input_shape):
        x_shape, _ = input_shape
        return (x_shape[0], x_shape[1], x_shape[2], self.filters)
   
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
   
class SiLUActivation(layers.Layer):
    """Wrapper for SiLU activation"""
    def call(self, x):
        return tf.nn.silu(x)
   
    def get_config(self):
        return super().get_config()
       
    @classmethod
    def from_config(cls, config):
        return cls(**config)


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

class TileLayer(layers.Layer):
    """Custom layer to tile the terrain_spatial tensor to match batch size"""
    def call(self, inputs):
        terrain_spatial, batch_size = inputs
        return tf.tile(terrain_spatial, [batch_size, 1, 1, 1])
   
    def get_config(self):
        return super().get_config()
       
    @classmethod
    def from_config(cls, config):
        return cls(**config)

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

class DebugGenerator(tf.keras.Model):
    def __init__(self, base_generator):
        super(DebugGenerator, self).__init__()
        self.base_generator = base_generator

    def call(self, inputs, training=None):
        tf.print("\nGenerator input shapes:", [tf.shape(x) for x in inputs])
        output = self.base_generator(inputs, training=training)
        tf.print("Generator output shape:", tf.shape(output))
        return output

class DynamicWeightedLoss(layers.Layer):
    """Adaptive loss weighting based on training dynamics"""
    def __init__(self, num_losses, **kwargs):
        super(DynamicWeightedLoss, self).__init__(**kwargs)
        self.loss_weights = self.add_weight(
            "loss_weights",
            shape=[num_losses],
            initializer="ones",
            trainable=True
        )
       
    def call(self, losses):
        weights = tf.nn.softmax(self.loss_weights)
        return tf.reduce_sum([w * l for w, l in zip(weights, losses)])
   
    def get_config(self):
        config = super().get_config()
        config.update({
            "num_losses": self.num_losses,
        })
        return config
       
    @classmethod
    def from_config(cls, config):
        return cls(**config)
   

class AdaptiveLRSchedule:
    def __init__(self, initial_lr=2e-4, min_lr=1e-6, patience=5, factor=0.5, metric_window=10):
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.patience = self.patience
        self.factor = self.factor
        self.metric_window = metric_window
        self.best_metric = float('inf')
        self.patience_counter = 0
        self.metrics_history = []
       
    def update(self, current_metric):
        """Update learning rate based on metric history"""
        self.metrics_history.append(current_metric)
       
        # Only update after collecting enough metrics
        if len(self.metrics_history) < self.metric_window:
            return self.current_lr
           
        # Calculate moving average
        avg_metric = sum(self.metrics_history[-self.metric_window:]) / self.metric_window
       
        if avg_metric < self.best_metric:
            self.best_metric = avg_metric
            self.patience_counter = 0
        else:
            self.patience_counter += 1
           
        # Reduce learning rate if patience is exceeded
        if self.patience_counter >= self.patience:
            self.current_lr = max(self.current_lr * self.factor, self.min_lr)
            self.patience_counter = 0
            print(f"\nReducing learning rate to: {self.current_lr:.2e}")
           
        # Keep only recent metrics
        if len(self.metrics_history) > self.metric_window * 2:
            self.metrics_history = self.metrics_history[-self.metric_window:]
           
        return self.current_lr


def compute_gradient_penalty(discriminator, real_images, fake_images, terrain_labels):
    """Gradient penalty calculation with proper dtype handling"""
    # Convert inputs to fp32 for accurate gradient calculation
    real_images = tf.cast(real_images, tf.float32)
    fake_images = tf.cast(fake_images, tf.float32)
    terrain_labels = tf.cast(terrain_labels, tf.float32)
   
    # Use dynamic batch size
    alpha = tf.random.uniform(
        [tf.shape(real_images)[0], 1, 1, 1], 0.0, 1.0, dtype=tf.float32
    )  # changed code
   
    diff = fake_images - real_images
    interpolated = real_images + alpha * diff
   
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator([interpolated, interpolated, terrain_labels], training=True)
       
    grads = gp_tape.gradient(pred, [interpolated])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
    gp = tf.reduce_mean((norm - 1.0) ** 2)
   
    # Convert back to fp16
    return tf.cast(gp, tf.float16)

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
    perceptual_weight = tf.constant(0.05, dtype=tf.float32)
   
    total_loss_f32 = gan_loss + (lambda_val * l1_loss) + (lambda_val * perceptual_weight * perceptual_loss)
   
    # Final NaN check
    total_loss_f32 = tf.where(tf.math.is_finite(total_loss_f32), total_loss_f32,
                              tf.constant(0.1, dtype=tf.float32))  # Use small non-zero value if NaN

    return (tf.cast(total_loss_f32, tf.float16),
            tf.cast(gan_loss, tf.float16),
            tf.cast(l1_loss, tf.float16),
            tf.cast(perceptual_loss, tf.float16),
            tf.cast(psnr, tf.float16),
            tf.cast(ssim, tf.float16))

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
           
            # Skip iteration if any NaN is detected in generated images
            if tf.reduce_any(tf.math.is_nan(generated_images_f32)):
                print("\nDetected NaN in generated images, skipping step")
                # Return default zero losses
                metrics = {
                    'gen_total_loss': 0.0,
                    'disc_loss': 0.0,
                    'l1_loss': 0.0,
                    'style_loss': 0.0,
                    'psnr': 0.0,
                    'ssim': 0.0,
                    'cycle_loss': 0.0,
                    'l2_loss': 0.0,
                    'feature_matching_loss': 0.0,
                    'lpips': 0.0,
                }
                return metrics, generated_images_f32
           
            # Discriminator forward passes
            disc_real_output = discriminator([sar_images, color_images, terrain_labels], training=True)
            disc_generated_output = discriminator([sar_images, generated_images_f32, terrain_labels], training=True)
           
            # Loss scaling factor for mixed precision
            loss_scale = tf.constant(128.0, dtype=tf.float32)
           
            # Calculate generator losses with safe ops
            gen_loss_result = generator_loss(disc_generated_output, generated_images_f32, color_images)
            gen_total_loss, gan_loss, l1_loss, perceptual_loss, psnr, ssim = gen_loss_result
           
            # Apply loss scaling for numerical stability in mixed precision
            gen_total_loss = gen_total_loss * loss_scale
           
            # Calculate discriminator loss with safe ops
            grad_penalty = compute_gradient_penalty(discriminator, color_images, generated_images_f32, terrain_labels)
            grad_penalty = tf.cast(grad_penalty, tf.float32)
            # Clip gradient penalty to reasonable range
            grad_penalty = tf.clip_by_value(grad_penalty, 0.0, 100.0)
           
            disc_base_loss = discriminator_loss(disc_real_output, disc_generated_output)
            disc_loss = disc_base_loss + 10.0 * grad_penalty
           
            # Apply loss scaling for numerical stability
            disc_loss = disc_loss * loss_scale

        # Skip gradient updates if losses are NaN
        if tf.math.is_nan(gen_total_loss) or tf.math.is_nan(disc_loss):
            print("\nDetected NaN in losses, skipping gradient update")
            # Set to small constant to avoid complete training failure
            gen_total_loss = tf.constant(0.1, dtype=tf.float32)
            disc_loss = tf.constant(0.1, dtype=tf.float32)
           
            # Calculate metrics with safe ops
            metrics = {
                'gen_total_loss': 0.1,
                'disc_loss': 0.1,
                'l1_loss': 0.0,
                'style_loss': 0.0,
                'psnr': 0.0,
                'ssim': 0.0,
                'cycle_loss': 0.0,
                'l2_loss': 0.0,
                'feature_matching_loss': 0.0,
                'lpips': 0.0,
            }
            return metrics, generated_images_f32
       
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
       
        # Calculate additional metrics safely
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
   
    except Exception as e:
        print(f"\nException in train step: {e}")
        # Return default metrics on error
        metrics = {
            'gen_total_loss': 0.0,
            'disc_loss': 0.0,
            'l1_loss': 0.0,
            'style_loss': 0.0,
            'psnr': 0.0,
            'ssim': 0.0,
            'cycle_loss': 0.0,
            'l2_loss': 0.0,
            'feature_matching_loss': 0.0,
            'lpips': 0.0,
        }
        # Return empty tensor if generated_images_f32 doesn't exist
        return metrics, tf.zeros_like(sar_images)

    # Unscale the losses (to account for loss scaling)
    gen_total_loss = gen_total_loss / loss_scale
    disc_loss = disc_loss / loss_scale
   
    # Return metrics with NaN protection
    metrics = {
        'gen_total_loss': tf.where(tf.math.is_finite(gen_total_loss), gen_total_loss, 0.0),
        'disc_loss': tf.where(tf.math.is_finite(disc_loss), disc_loss, 0.0),
        'l1_loss': tf.where(tf.math.is_finite(l1_loss), l1_loss, 0.0),
        'style_loss': tf.where(tf.math.is_finite(feature_matching_loss), feature_matching_loss, 0.0),
        'psnr': tf.where(tf.math.is_finite(psnr), psnr, 0.0),
        'ssim': tf.where(tf.math.is_finite(ssim), ssim, 0.0),
        'cycle_loss': tf.where(tf.math.is_finite(cycle_loss), cycle_loss, 0.0),
        'l2_loss': tf.where(tf.math.is_finite(l2_loss), l2_loss, 0.0),
        'feature_matching_loss': tf.where(tf.math.is_finite(feature_matching_loss), feature_matching_loss, 0.0),
        'lpips': tf.where(tf.math.is_finite(lpips), lpips, 0.0),
    }
   
    return metrics, generated_images_f32

# Update history initialization to remove style_loss
def get_initial_history():
    return {
        'gen_loss': [], 'disc_loss': [],
        'psnr': [], 'ssim': [],
        'cycle_loss': [], 'l2_loss': [],
        'feature_matching_loss': [],
        'lpips': [],
        'l1_loss': [],
        'perceptual_loss': [],
        'val_gen_loss': [], 'val_disc_loss': [],
        'val_psnr': [], 'val_ssim': [],
        'val_l1_loss': [], 'val_l2_loss': [],
        'val_perceptual_loss': [], 'val_cycle_loss': [],
        'val_feature_matching_loss': [],
        'val_lpips': [],
        'fid': [], 'val_fid': []
    }

# Update the train function to use new history initialization
def train(train_dataset, val_dataset, epochs, resume_training=True):
    """Modified training function with validation set integration and comprehensive metrics"""
   
    if not isinstance(strategy, tf.distribute.Strategy):
        raise ValueError("No distribution strategy found!")
   
    num_replicas = strategy.num_replicas_in_sync
    global_batch_size = BATCH_SIZE * num_replicas
   
    # Create visualization directory
    visualization_dir = '/kaggle/working/visualizations'
    os.makedirs(visualization_dir, exist_ok=True)
   
    with strategy.scope():
        # Load or initialize models and training history
        if resume_training:
            loaded_gen, loaded_disc, loaded_history = load_latest_models()
            if loaded_gen is not None:
                generator = loaded_gen
                discriminator = loaded_disc
                start_epoch = loaded_history.get('epoch', 0) + 1
                history = loaded_history.get('history', get_initial_history())
                print(f"Resuming training from epoch {start_epoch}")
            else:
                generator = build_terrain_aware_generator()
                discriminator = build_terrain_aware_discriminator()
                start_epoch = 0
                history = get_initial_history()
        else:
            generator = build_terrain_aware_generator()
            discriminator = build_terrain_aware_discriminator()
            start_epoch = 0
            history = get_initial_history()
       
        # Initialize optimizers with gradient clipping using the standard optimizer
        generator_optimizer = tf.keras.optimizers.Adam(
            1e-4,  # Lower learning rate for stability
            beta_1=0.5,
            clipnorm=1.0  # Add gradient clipping
        )
        discriminator_optimizer = tf.keras.optimizers.Adam(
            1e-4,  # Lower learning rate for stability
            beta_1=0.5,
            clipnorm=1.0  # Add gradient clipping
        )
        metrics_tracker = MetricsTracker()
       
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
       
        # Create validation dataset for visualization
        val_vis_dataset = val_dataset.take(5).cache()
       
        @tf.function
        def distributed_train_step(dist_inputs):
            """Fixed distribution-aware training step with proper error handling"""
            def train_step_fn(inputs):
                sar_images, color_images, terrain_labels = inputs
               
                # Add numerical stability measures
                sar_images = tf.clip_by_value(sar_images, -1.0, 1.0)
                color_images = tf.clip_by_value(color_images, -1.0, 1.0)
               
                # Default loss values
                gen_loss = tf.constant(0.1, dtype=tf.float16)
                disc_loss = tf.constant(0.1, dtype=tf.float32)
               
                try:
                    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                        per_replica_scaling = 1.0 / strategy.num_replicas_in_sync
                       
                        # Generator forward pass with error checking
                        generated_images = generator([sar_images, terrain_labels], training=True)
                        generated_images = tf.where(
                            tf.math.is_finite(generated_images),
                            generated_images,
                            tf.zeros_like(generated_images)
                        )
                        generated_images = tf.clip_by_value(generated_images, -1.0, 1.0)
                       
                        # Discriminator forward passes
                        disc_real_output = discriminator([sar_images, color_images, terrain_labels], training=True)
                        disc_generated_output = discriminator([sar_images, generated_images, terrain_labels], training=True)
                       
                        # Calculate losses with scaling and NaN protection
                        losses = generator_loss(disc_generated_output, generated_images, color_images)
                        gen_loss = losses[0] * per_replica_scaling
                        gen_loss = tf.where(tf.math.is_finite(gen_loss), gen_loss, tf.constant(0.1, dtype=tf.float16))
                       
                        disc_loss = discriminator_loss(disc_real_output, disc_generated_output) * per_replica_scaling
                        disc_loss = tf.where(tf.math.is_finite(disc_loss), disc_loss, tf.constant(0.1, dtype=tf.float32))
                       
                        # Add small constant to prevent complete loss of gradients
                        gen_loss = gen_loss + 1e-8
                        disc_loss = disc_loss + 1e-8
               
                    # Compute and apply gradients with nan/inf handling
                    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
                    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
                   
                    # Replace NaN gradients with zeros
                    gen_gradients = [
                        tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                        if g is not None else None for g in gen_gradients
                    ]
                    disc_gradients = [
                        tf.where(tf.math.is_finite(g), g, tf.zeros_like(g))
                        if g is not None else None for g in disc_gradients
                    ]
                   
                    # Clip gradients for stability
                    gen_gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gen_gradients]
                    disc_gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in disc_gradients]
                   
                    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
                    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
                except Exception as e:
                    # Just log the error without passing it outside strategy.run
                    tf.print("Error in replica:", e)
               
                return gen_loss, disc_loss
           
            # Run the training function on each replica
            per_replica_losses = strategy.run(train_step_fn, args=(dist_inputs,))
           
            # Properly extract the per-replica losses using get_replica_context
            per_replica_gen_loss, per_replica_disc_loss = per_replica_losses
           
            # Explicitly extract values from PerReplica objects
            # Use strategy.reduce within the distribution context
            reduced_gen_loss = strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_gen_loss, axis=None
            )
            reduced_disc_loss = strategy.reduce(
                tf.distribute.ReduceOp.SUM, per_replica_disc_loss, axis=None
            )
           
            # Final NaN check on reduced values
            reduced_gen_loss = tf.where(
                tf.math.is_finite(reduced_gen_loss),
                reduced_gen_loss,
                tf.constant(0.1, dtype=reduced_gen_loss.dtype)
            )
            reduced_disc_loss = tf.where(
                tf.math.is_finite(reduced_disc_loss),
                reduced_disc_loss,
                tf.constant(0.1, dtype=reduced_disc_loss.dtype)
            )
           
            return reduced_gen_loss, reduced_disc_loss
       
        # Learning rate scheduler with exponential decay for stability
        gen_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-5,  # Start with even lower learning rate
            decay_steps=1000,
            decay_rate=0.95
        )
        disc_lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-5,  # Start with lower learning rate
            decay_steps=1000,
            decay_rate=0.95
        )
       
        # Update optimizers with schedules
        generator_optimizer.learning_rate = gen_lr_schedule
        discriminator_optimizer.learning_rate = disc_lr_schedule
       
        # Now use the distributed_train_step in your training loop
        for epoch in range(start_epoch, start_epoch + epochs):
            start = time.time()
           
            # Training phase
            nan_count = 0
            steps_per_epoch = 0
            total_gen_loss = 0
            total_disc_loss = 0
           
            # Initialize epoch metrics tracker with all metrics
            epoch_metrics = {
                'gen_loss': 0.0, 'disc_loss': 0.0,
                'psnr': 0.0, 'ssim': 0.0,
                'cycle_loss': 0.0, 'l2_loss': 0.0,
                'feature_matching_loss': 0.0,
                'lpips': 0.0, 'l1_loss': 0.0,
                'perceptual_loss': 0.0,
                'val_gen_loss': 0.0, 'val_disc_loss': 0.0,
                'val_psnr': 0.0, 'val_ssim': 0.0,
                'val_l1_loss': 0.0, 'val_l2_loss': 0.0,
                'val_perceptual_loss': 0.0, 'val_cycle_loss': 0.0,
                'val_feature_matching_loss': 0.0,
                'val_lpips': 0.0, 'fid': 0.0, 'val_fid': 0.0
            }
           
            # Training loop
            try:
                # Use a Python for loop instead of enumerate to avoid issues with PerReplica
                # This ensures all operations happen within the distribution context
                for step, dist_inputs in enumerate(dist_dataset):
                    steps_per_epoch += 1
                    try:
                        # Run the distributed training step
                        gen_loss, disc_loss = distributed_train_step(dist_inputs)
                       
                        # Convert to host tensor values
                        gen_loss_val = gen_loss.numpy()
                        disc_loss_val = disc_loss.numpy()
                       
                        # Check for NaN and track
                        if np.isnan(gen_loss_val) or np.isnan(disc_loss_val):
                            nan_count += 1
                            if nan_count > 5:  # Reset if too many NaNs in a row
                                print("\nToo many NaN values, resetting optimizer states...")
                                # Use list comprehension to avoid PerReplica issues
                                for var in generator_optimizer.variables():
                                    var.assign(tf.zeros_like(var))
                                for var in discriminator_optimizer.variables():
                                    var.assign(tf.zeros_like(var))
                                nan_count = 0
                           
                            # Use default values for display
                            gen_loss_val = 0.1
                            disc_loss_val = 0.1
                           
                        total_gen_loss += gen_loss_val
                        total_disc_loss += disc_loss_val
                       
                        # Calculate detailed metrics every 50 steps for monitoring
                        if step % 50 == 0 and step > 0:
                            try:
                                # Extract first batch for detailed metrics
                                sar_batch, color_batch, terrain_batch = next(iter(val_vis_dataset))
                               
                                # Generate images
                                generated_images = generator([sar_batch, terrain_batch], training=False)
                               
                                # Calculate cycle reconstruction
                                cycle_reconstructed = generator([generated_images, terrain_batch], training=False)
                               
                                # Calculate detailed metrics
                                detailed_metrics = calculate_detailed_metrics(
                                    sar_batch, color_batch, generated_images, cycle_reconstructed, terrain_batch
                                )
                               
                                # Update epoch metrics with detailed values
                                for key, value in detailed_metrics.items():
                                    if key in epoch_metrics:
                                        epoch_metrics[key] += value
                               
                                # Print detailed metrics including FID
                                metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in detailed_metrics.items()])
                                # Print FID prominently
                                fid_value = detailed_metrics.get('fid', 0.0)
                                #print(f"\nStep {step} detailed metrics - FID: {fid_value:.4f} | {metrics_str}")
                               
                            except Exception as metrics_err:
                                print(f"\nError calculating detailed metrics: {metrics_err}")
                       
                        if step % 10 == 0:
                            avg_gen = total_gen_loss / (step + 1)
                            avg_disc = total_disc_loss / (step + 1)
                           
                            # Add FID to regular status updates if available
                            fid_str = ""
                            if 'fid' in epoch_metrics and epoch_metrics['fid'] > 0:
                                avg_fid = epoch_metrics['fid'] / max(1, steps_per_epoch // 50)
                                fid_str = f", FID: {avg_fid:.4f}"
                               
                            #print(f"\rStep {step} - Gen Loss: {avg_gen:.4f}, Disc Loss: {avg_disc:.4f}{fid_str}", end='')
                           
                    except tf.errors.ResourceExhaustedError:
                        print("\nResource exhausted, skipping batch")
                        continue
                    except Exception as e:
                        print(f"\nException in training loop: {e}")
                        continue
            except Exception as main_exception:
                print(f"\nMain training loop exception: {main_exception}")
           
            # Log average losses per epoch
            if steps_per_epoch > 0:
                avg_gen_loss = total_gen_loss / steps_per_epoch
                avg_disc_loss = total_disc_loss / steps_per_epoch
                history['gen_loss'].append(float(avg_gen_loss))
                history['disc_loss'].append(float(avg_disc_loss))
               
                # Update other metrics in history
                for key in epoch_metrics:
                    if key not in ['gen_loss', 'disc_loss'] and not key.startswith('val_'):
                        # Only add to history if we have valid measurements
                        if epoch_metrics[key] > 0:
                            history[key].append(float(epoch_metrics[key] / max(steps_per_epoch // 50, 1)))
           
               
                # Print loss metrics
                gen_loss = get_latest_metric(history, 'gen_loss')
                disc_loss = get_latest_metric(history, 'disc_loss')
                psnr = get_latest_metric(history, 'psnr')
                ssim = get_latest_metric(history, 'ssim')
                l1_loss = get_latest_metric(history, 'l1_loss')
                l2_loss = get_latest_metric(history, 'l2_loss')
                perceptual = get_latest_metric(history, 'perceptual_loss')
                feature_matching = get_latest_metric(history, 'feature_matching_loss')
                cycle_loss = get_latest_metric(history, 'cycle_loss')
                lpips = get_latest_metric(history, 'lpips')
                fid = get_latest_metric(history, 'fid', 0.0)  # Get FID with default 0.0
                print(f"│ Epoch {epoch + 1}/{start_epoch + epochs} (Training) : gen_loss: {gen_loss:.4f} | disc_loss: {disc_loss:.4f} | cycle_loss: {cycle_loss:.4f} | psnr: {psnr:.4f} | ssim: {ssim:.4f} | l1_loss: {l1_loss:.4f} | l2_loss: {l2_loss:.4f} | feature_matching: {feature_matching:.4f} | lpips: {lpips:.4f} | FID: {fid:.4f}")
           
            # Validation phase
            print("\nRunning validation...")
            val_metrics = {
                'gen_loss': [],
                'disc_loss': [],
                'psnr': [],
                'ssim': [],
                'l1_loss': [],
                'l2_loss': [],
                'perceptual_loss': [],
                'style_loss': [],
                'feature_matching_loss': [],
                'cycle_loss': [],
                'lpips': [], 'fid': []
            }
           
            # Process validation batch by batch to collect detailed metrics
            all_val_generated_images = []
            all_val_real_images = []
           
            for val_batch in val_dataset:
                sar_batch, color_batch, terrain_batch = val_batch
               
                # Generate images without training
                generated_images = generator([sar_batch, terrain_batch], training=False)
               
                # Calculate cycle reconstructions for cycle consistency
                cycle_reconstructed = generator([generated_images, terrain_batch], training=False)
               
                # Calculate validation metrics
                disc_real_output = discriminator([sar_batch, color_batch, terrain_batch], training=False)
                disc_generated_output = discriminator([sar_batch, generated_images, terrain_batch], training=False)
               
                # Calculate generator loss
                gen_loss_result = generator_loss(disc_generated_output, generated_images, color_batch)
                gen_total_loss, _, l1_loss, perceptual_loss, psnr, ssim = gen_loss_result
               
                # Calculate discriminator loss
                disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
               
                # Calculate detailed metrics
                detailed_metrics = calculate_detailed_metrics(
                    sar_batch, color_batch, generated_images, cycle_reconstructed, terrain_batch
                )
               
                # Print per-batch FID for validation
               
               
                # Store all validation metrics
                val_metrics['gen_loss'].append(gen_total_loss.numpy())
                val_metrics['disc_loss'].append(disc_loss.numpy())
                val_metrics['psnr'].append(psnr.numpy())
                val_metrics['ssim'].append(ssim.numpy())
                val_metrics['l1_loss'].append(l1_loss.numpy())
                val_metrics['perceptual_loss'].append(perceptual_loss.numpy())
               
                # Add other metrics from detailed calculation
                for key, value in detailed_metrics.items():
                    if key in val_metrics and key not in ['psnr', 'ssim', 'l1_loss', 'perceptual_loss']:
                        val_metrics[key].append(value)
           
            # Calculate FID on larger combined validation set for more accurate measurement
            try:
                if len(all_val_generated_images) > 0:
                    combined_generated = tf.concat(all_val_generated_images, axis=0)
                    combined_real = tf.concat(all_val_real_images, axis=0)
                   
                    if combined_generated.shape[0] >= 16:  # Enough samples for meaningful FID
                        combined_fid = calculate_fid_score(combined_real, combined_generated)
                        print(f"\nValidation FID on combined set ({combined_generated.shape[0]} images): {combined_fid.numpy():.4f}")
                       
                        # Add to metrics
                        if 'fid' in avg_val_metrics:
                            avg_val_metrics['fid'] = combined_fid.numpy()
            except Exception as combined_fid_err:
                print(f"Error calculating combined validation FID: {combined_fid_err}")
           
            # Calculate average validation metrics
            avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items() if len(v) > 0}
           
            # Update history with validation metrics
            for key, value in avg_val_metrics.items():
                val_key = f'val_{key}'
                if val_key in history:
                    history[val_key].append(float(value))
           
            # Update epoch metrics dictionary
            for key, value in avg_val_metrics.items():
                val_key = f'val_{key}'
                epoch_metrics[val_key] = float(value)
           
           
            # Print validation metrics using safe dictionary access
            gen_loss = avg_val_metrics.get('gen_loss', 0.0)
            disc_loss = avg_val_metrics.get('disc_loss', 0.0)
            psnr = avg_val_metrics.get('psnr', 0.0)
            ssim = avg_val_metrics.get('ssim', 0.0)
            l1_loss = avg_val_metrics.get('l1_loss', 0.0)
            l2_loss = avg_val_metrics.get('l2_loss', 0.0)
            cycle_loss = avg_val_metrics.get('cycle_loss', 0.0)
            lpips = avg_val_metrics.get('lpips', 0.0)
            fid = avg_val_metrics.get('fid', 0.0)
            feature_matching = avg_val_metrics.get('feature_matching_loss', 0.0)
            print(f"│ Epoch {epoch + 1}/{start_epoch + epochs} (Validation) : gen_loss: {gen_loss:.4f} | disc_loss: {disc_loss:.4f} | cycle_loss: {cycle_loss:.4f} | psnr: {psnr:.4f} | ssim: {ssim:.4f} | l1_loss: {l1_loss:.4f} | l2_loss: {l2_loss:.4f} | feature_matching: {feature_matching:.4f} | lpips: {lpips:.4f} | FID: {fid:.4f}")          
            print(f"\nTime taken for epoch {epoch + 1}: {time.time() - start:.2f} sec")
           
            # Generate visualizations every 5 epochs
            if (epoch + 1) % 5 == 0 or epoch == start_epoch:
                try:
                    generate_and_save_visualizations(
                        generator,
                        val_vis_dataset,
                        os.path.join(visualization_dir, f'epoch_{epoch+1}')
                    )
                    print(f"Saved visualizations for epoch {epoch+1}")
                except Exception as vis_error:
                    print(f"Error generating visualizations: {vis_error}")
           
            # Save comprehensive model checkpoint every epoch
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_data = {
                    'epoch': epoch,
                    'history': history
                }
                save_model_checkpoint(generator, discriminator, save_data, timestamp)
                print(f"Saved checkpoint for epoch {epoch+1}")
            except Exception as save_error:
                print(f"Error saving checkpoint: {save_error}")
   
    return history

def create_validation_subset(dataset, num_samples=5):
    """Create a small validation dataset for visualization"""
    val_dataset = dataset.take(num_samples).cache()
    return val_dataset

def generate_and_save_visualizations(generator, dataset, save_dir):
    """Generate and save comparison visualizations"""
    os.makedirs(save_dir, exist_ok=True)
   
    for i, (sar_batch, color_batch, terrain_batch) in enumerate(dataset):
        # Generate images
        generated_images = generator([sar_batch, terrain_batch], training=False)
       
        # Create figure with 3 columns (input, ground truth, generated)
        plt.figure(figsize=(15, 5 * min(len(sar_batch), 3)))
       
        # Display up to 3 images from the batch
        for j in range(min(len(sar_batch), 3)):
            # Input SAR image
            plt.subplot(min(len(sar_batch), 3), 3, j*3 + 1)
            plt.title(f"Input SAR {j+1}")
            plt.imshow(tf.cast(sar_batch[j] * 0.5 + 0.5, tf.float32).numpy())
            plt.axis('off')
           
            # Ground truth color image
            plt.subplot(min(len(sar_batch), 3), 3, j*3 + 2)
            plt.title(f"Ground Truth {j+1}")
            plt.imshow(tf.cast(color_batch[j] * 0.5 + 0.5, tf.float32).numpy())
            plt.axis('off')
           
            # Generated color image
            plt.subplot(min(len(sar_batch), 3), 3, j*3 + 3)
            plt.title(f"Generated {j+1}")
            plt.imshow(tf.cast(generated_images[j] * 0.5 + 0.5, tf.float32).numpy())
            plt.axis('off')
       
        # Save the figure
        plt.savefig(os.path.join(save_dir, f'comparison_{i+1}.png'), bbox_inches='tight')
        plt.close()
   
    print(f"Saved {i+1} visualization images to {save_dir}")

def calculate_detailed_metrics(sar_images, color_images, generated_images, cycle_reconstructed, terrain_labels):
    """Calculate comprehensive metrics for model evaluation"""
    # Initialize metrics dictionary
    metrics = {}
   
    # Ensure all inputs are float32 for consistent calculation
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
       
        # Calculate FID score
        # This requires a sufficient batch size to be meaningful,
        # so we'll only calculate it if we have enough images
        try:
            if color_images.shape[0] >= 8:  # Minimum batch size for meaningful FID
                fid_score = calculate_fid_score(color_images, generated_images)
                metrics['fid'] = fid_score.numpy()
            else:
                # For small batches, approximate FID using feature statistics
                # from our existing feature extractor
                real_feats_flat = tf.concat([tf.reshape(f, [tf.shape(f)[0], -1])
                                           for f in real_features], axis=1)
                gen_feats_flat = tf.concat([tf.reshape(f, [tf.shape(f)[0], -1])
                                          for f in gen_features], axis=1)
               
                # Calculate mean and covariance
                mu_real = tf.reduce_mean(real_feats_flat, axis=0)
                mu_gen = tf.reduce_mean(gen_feats_flat, axis=0)
               
                # Use mean difference as simplified FID for small batches
                metrics['fid'] = tf.reduce_mean(tf.square(mu_real - mu_gen)).numpy()
        except Exception as e:
            print(f"Error calculating FID score: {e}")
            metrics['fid'] = 0.0
           
    except Exception as e:
        print(f"Error in metrics: {e}")
        metrics = {k: 0.0 for k in [
            'l1_loss', 'l2_loss', 'psnr', 'ssim', 'cycle_loss',
            'perceptual_loss', 'lpips', 'feature_matching_loss', 'fid'
        ]}
   
    return metrics

# Fix save_model_checkpoint function
def save_model_checkpoint(generator, discriminator, save_data, timestamp):
    """Improved model saving with consistent directory structure"""
    # Create a fixed checkpoint directory to maintain consistency between runs
    checkpoint_dir = '/kaggle/working/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
   
    # Create unique subdirectory based on timestamp and epoch
    epoch = save_data['epoch']
    checkpoint_subdir = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}_{timestamp}')
    os.makedirs(checkpoint_subdir, exist_ok=True)
   
    try:
        # Save model weights with explicit paths
        gen_weights_path = os.path.join(checkpoint_subdir, 'generator.weights.h5')
        disc_weights_path = os.path.join(checkpoint_subdir, 'discriminator.weights.h5')
       
        generator.save_weights(gen_weights_path)
        discriminator.save_weights(disc_weights_path)
       
        # Verify weights were actually saved
        if not os.path.exists(gen_weights_path) or not os.path.exists(disc_weights_path):
            raise FileNotFoundError("Model weights were not saved properly")
       
        # Save model config separately for reliable loading
        with open(os.path.join(checkpoint_subdir, 'generator_config.json'), 'w') as f:
            json.dump(generator.get_config(), f)
           
        with open(os.path.join(checkpoint_subdir, 'discriminator_config.json'), 'w') as f:
            json.dump(discriminator.get_config(), f)
       
        # Save optimizer states
        if hasattr(generator, 'optimizer') and generator.optimizer:
            gen_opt_state = extract_optimizer_state(generator.optimizer)
            with open(os.path.join(checkpoint_subdir, 'generator_optimizer.json'), 'w') as f:
                json.dump(gen_opt_state, f)
       
        if hasattr(discriminator, 'optimizer') and discriminator.optimizer:
            disc_opt_state = extract_optimizer_state(discriminator.optimizer)
            with open(os.path.join(checkpoint_subdir, 'discriminator_optimizer.json'), 'w') as f:
                json.dump(disc_opt_state, f)
       
        # Save history separately
        history_path = os.path.join(checkpoint_subdir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(save_data, f)
       
        # Update latest checkpoint pointer to point to the subdirectory
        with open(os.path.join(checkpoint_dir, 'latest_checkpoint.txt'), 'w') as f:
            f.write(checkpoint_subdir)
       
        print(f"Models successfully saved to {checkpoint_subdir}")
        print(f"  - Generator weights: {os.path.getsize(gen_weights_path)/1024/1024:.2f} MB")
        print(f"  - Discriminator weights: {os.path.getsize(disc_weights_path)/1024/1024:.2f} MB")
       
    except Exception as e:
        print(f"ERROR: Failed to save checkpoint: {e}")
        import traceback
        traceback.print_exc()

def load_latest_models():
    """Enhanced model loading with comprehensive error handling"""
    checkpoint_dir = '/kaggle/working/checkpoints'
    latest_file = os.path.join(checkpoint_dir, 'latest_checkpoint.txt')
   
    if not os.path.exists(checkpoint_dir):
        print("No checkpoint directory found. Will start fresh.")
        return None, None, {}
   
    if not os.path.exists(latest_file):
        print("No checkpoint tracking file found. Will start fresh.")
        return None, None, {}
   
    # Define custom objects for model loading
    custom_objects = {
        "InstanceNormalization": InstanceNormalization,
        "OptimizedColorTransformation": OptimizedColorTransformation,
        "LightweightBoundaryDetection": LightweightBoundaryDetection,
        "EfficientSkipConnection": EfficientSkipConnection,
        "TerrainGuidedAttention": TerrainGuidedAttention,
        "TerrainAdaptiveNormalization": TerrainAdaptiveNormalization,
        "TerrainAwareResBlock": TerrainAwareResBlock,
        "ColorRefinementBlock": ColorRefinementBlock,
        "MemoryEfficientResBlock": MemoryEfficientResBlock,
        "SiLUActivation": SiLUActivation,
        "TileLayer": TileLayer,
        "TerrainSpatialLayer": TerrainSpatialLayer
    }
   
    try:
        # Read the checkpoint directory path
        with open(latest_file, 'r') as f:
            checkpoint_path = f.read().strip()
       
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint directory {checkpoint_path} does not exist. Will start fresh.")
            return None, None, {}
       
        # Check for weights files with clear logging
        gen_weights_path = os.path.join(checkpoint_path, 'generator.weights.h5')
        disc_weights_path = os.path.join(checkpoint_path, 'discriminator.weights.h5')
       
        if not os.path.exists(gen_weights_path):
            print(f"ERROR: Generator weights file not found at {gen_weights_path}")
            return None, None, {}
           
        if not os.path.exists(disc_weights_path):
            print(f"ERROR: Discriminator weights file not found at {disc_weights_path}")
            return None, None, {}
       
        print(f"Found model weights files:")
        print(f"  - Generator weights: {os.path.getsize(gen_weights_path)/1024/1024:.2f} MB")
        print(f"  - Discriminator weights: {os.path.getsize(disc_weights_path)/1024/1024:.2f} MB")
       
        # Always recreate models from scratch for consistency
        generator = build_terrain_aware_generator()
        discriminator = build_terrain_aware_discriminator()
       
        # Force model to build by passing dummy inputs
        dummy_sar = tf.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32)
        dummy_terrain = tf.zeros((1, len(TERRAIN_TYPES)), dtype=tf.float32)
        dummy_color = tf.zeros((1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=tf.float32)
       
        # Build the models with dummy forward passes
        _ = generator([dummy_sar, dummy_terrain], training=False)
        _ = discriminator([dummy_sar, dummy_color, dummy_terrain], training=False)
       
        # Then load weights with explicit try-except
        try:
            generator.load_weights(gen_weights_path)
            discriminator.load_weights(disc_weights_path)
            print("Successfully loaded model weights")
        except Exception as weight_error:
            print(f"ERROR loading weights: {weight_error}")
            import traceback
            traceback.print_exc()
            return None, None, {}
       
        # Create fresh optimizers
        gen_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, clipnorm=1.0)
        disc_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, clipnorm=1.0)
       
        # Compile models with the fresh optimizers
        generator.compile(optimizer=gen_optimizer)
        discriminator.compile(optimizer=disc_optimizer)
       
        # Load optimizer states if they exist
        gen_opt_path = os.path.join(checkpoint_path, 'generator_optimizer.json')
        disc_opt_path = os.path.join(checkpoint_path, 'discriminator_optimizer.json')
       
        if os.path.exists(gen_opt_path):
            try:
                with open(gen_opt_path, 'r') as f:
                    gen_opt_state = json.load(f)
                    generator.optimizer = restore_optimizer_state(generator.optimizer, gen_opt_state)
                    print("Restored generator optimizer state")
            except Exception as opt_error:
                print(f"Warning: Could not restore generator optimizer: {opt_error}")
       
        if os.path.exists(disc_opt_path):
            try:
                with open(disc_opt_path, 'r') as f:
                    disc_opt_state = json.load(f)
                    discriminator.optimizer = restore_optimizer_state(discriminator.optimizer, disc_opt_state)
                    print("Restored discriminator optimizer state")
            except Exception as opt_error:
                print(f"Warning: Could not restore discriminator optimizer: {opt_error}")
       
        # Load training history
        history_path = os.path.join(checkpoint_path, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                save_data = json.load(f)
            print(f"Successfully loaded checkpoint from {os.path.basename(checkpoint_path)}")
            print(f"Resuming from epoch {save_data.get('epoch', 0) + 1}")
            return generator, discriminator, save_data
        else:
            print("No training history found, using default history")
            return generator, discriminator, {'epoch': 0, 'history': get_initial_history()}
   
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return None, None, {}

def plot_training_history(history, save_path='/kaggle/working/training_curves'):
    """Enhanced plot function with more comparisons between train and validation metrics"""
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   
    # Plot train vs validation metrics for direct comparison
    comparison_metrics = [
        ('gen_loss', 'val_gen_loss', 'Generator Loss'),
        ('disc_loss', 'val_disc_loss', 'Discriminator Loss'),
        ('psnr', 'val_psnr', 'PSNR'),
        ('ssim', 'val_ssim', 'SSIM'),
        ('l1_loss', 'val_l1_loss', 'L1 Loss'),
        ('l2_loss', 'val_l2_loss', 'L2 Loss'),
        ('perceptual_loss', 'val_perceptual_loss', 'Perceptual Loss'),
        ('style_loss', 'val_style_loss', 'Style Loss'),
        ('feature_matching_loss', 'val_feature_matching_loss', 'Feature Matching Loss'),
        ('cycle_loss', 'val_cycle_loss', 'Cycle Consistency Loss'),
        ('lpips', 'val_lpips', 'LPIPS')
    ]
   
    for train_key, val_key, title in comparison_metrics:
        if train_key in history and val_key in history and len(history[train_key]) > 0 and len(history[val_key]) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(history[train_key], label=f'Training {title}')
            plt.plot(history[val_key], label=f'Validation {title}')
            plt.title(f'Training vs Validation {title}')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(save_path, f'{train_key}_vs_{val_key}_{timestamp}.png'))
            plt.close()
   
    # Group related metrics
    metric_groups = {
        'losses': ['gen_loss', 'disc_loss', 'val_gen_loss', 'val_disc_loss'],
        'image_quality': ['psnr', 'ssim', 'val_psnr', 'val_ssim'],
        'content_losses': ['l1_loss', 'l2_loss', 'perceptual_loss', 'val_l1_loss', 'val_l2_loss', 'val_perceptual_loss'],
        'style_losses': ['style_loss', 'feature_matching_loss', 'val_style_loss', 'val_feature_matching_loss'],
        'cycle_lpips': ['cycle_loss', 'lpips', 'val_cycle_loss', 'val_lpips']
    }
   
    # Plot each metric group
    for group_name, metrics in metric_groups.items():
        valid_metrics = [m for m in metrics if m in history and len(history[m]) > 0]
       
        if not valid_metrics:
            continue
       
        plt.figure(figsize=(12, 7))
        for metric in valid_metrics:
            plt.plot(history[metric], label=metric)
       
        plt.title(f'{group_name.replace("_", " ").title()}')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
       
        # Save the plot
        plt.savefig(os.path.join(save_path, f'{group_name}_{timestamp}.png'))
        plt.close()
   
    # Save all metrics to CSV for further analysis
    import pandas as pd
    metrics_df = pd.DataFrame(history)
    metrics_df.to_csv(os.path.join(save_path, f'training_metrics_{timestamp}.csv'))
   
    print(f"Training history plots saved to {save_path}")

def get_latest_metric(history_dict, metric_name, default=0.0):
    """Safely get the latest metric value from history dictionary"""
    try:
        values = history_dict.get(metric_name, [default])
        return values[-1] if values else default
    except (IndexError, TypeError, AttributeError):
        return default

# Add this function to calculate FID score
def calculate_fid_score(real_images, generated_images):
    """
    Calculate Fréchet Inception Distance between real and generated images
    """
    # Ensure inputs are in the right format for InceptionV3
    real_images = tf.image.resize(real_images, [299, 299])
    real_images = tf.clip_by_value((real_images + 1.0) / 2.0, 0.0, 1.0) * 255.0
   
    generated_images = tf.image.resize(generated_images, [299, 299])
    generated_images = tf.clip_by_value((generated_images + 1.0) / 2.0, 0.0, 1.0) * 255.0
   
    # Create inception model for feature extraction if it doesn't already exist
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
   
    # Extract features
    real_features = inception_model(real_images, training=False)
    gen_features = inception_model(generated_images, training=False)
   
    # Calculate mean and covariance statistics
    mu_real = tf.reduce_mean(real_features, axis=0)
    sigma_real = tfp.stats.covariance(real_features)
   
    mu_gen = tf.reduce_mean(gen_features, axis=0)
    sigma_gen = tfp.stats.covariance(gen_features)
   
    # Calculate squared difference between means
    mean_diff_squared = tf.reduce_sum(tf.square(mu_real - mu_gen))
   
    # Calculate square root of product of covariances
    # Note: To avoid numerical issues, we use the scipy implementation
    covmean = sqrtm(tf.matmul(sigma_real, sigma_gen))
   
    # Check if result contains complex numbers
    if np.iscomplexobj(covmean):
        covmean = covmean.real
   
    # Calculate FID score
    trace_covmean = tf.linalg.trace(covmean)
    trace_real = tf.linalg.trace(sigma_real)
    trace_gen = tf.linalg.trace(sigma_gen)
   
    fid = mean_diff_squared + trace_real + trace_gen - 2 * trace_covmean
   
    return fid

# Add these utility functions for handling optimizer states
def extract_optimizer_state(optimizer):
    """Extract optimizer variables into a serializable dictionary"""
    optimizer_vars = {}
   
    # Use optimizer.get_weights() if available (safer than iterating variables)
    try:
        # Extract iteration directly from optimizer's private internal state
        # (This is version-specific but works in TF 2.17.1)
        optimizer_vars['iteration'] = optimizer.iterations.numpy().item()
       
        # Collect all weights as flat arrays for consistent serialization
        weight_values = optimizer.get_weights()
        weight_names = [w.name for w in optimizer.weights]
        optimizer_vars['weights'] = [(name, val.tolist()) for name, val in zip(weight_names, weight_values)]
       
        # Store hyperparameters that might be needed for reconstruction
        optimizer_vars['learning_rate'] = float(optimizer.lr.numpy()) if hasattr(optimizer, 'lr') else 0.001
        optimizer_vars['beta_1'] = float(optimizer.beta_1.numpy()) if hasattr(optimizer, 'beta_1') else 0.9
        optimizer_vars['beta_2'] = float(optimizer.beta_2.numpy()) if hasattr(optimizer, 'beta_2') else 0.999
        optimizer_vars['epsilon'] = float(optimizer.epsilon) if hasattr(optimizer, 'epsilon') else 1e-7
       
        return optimizer_vars
    except Exception as e:
        print(f"Error extracting optimizer state: {e}")
        return {"error": str(e)}

def restore_optimizer_state(optimizer, optimizer_state):
    """Restore optimizer state from dictionary"""
    if "error" in optimizer_state:
        print(f"Cannot restore optimizer state: {optimizer_state['error']}")
        return optimizer
   
    try:
        # Set iteration
        if 'iteration' in optimizer_state:
            optimizer.iterations.assign(optimizer_state['iteration'])
       
        # Re-create weights if the structure matches
        if 'weights' in optimizer_state:
            # Convert back to numpy arrays from lists
            weights_data = [(name, np.array(val)) for name, val in optimizer_state['weights']]
           
            # Make sure the weight count matches what's expected
            if len(weights_data) == len(optimizer.weights):
                # Sort both lists by name to ensure correct assignment
                opt_weights = sorted(optimizer.weights, key=lambda x: x.name)
                saved_weights = sorted(weights_data, key=lambda x: x[0])
               
                # Assign values to weights
                for (_, saved_val), opt_weight in zip(saved_weights, opt_weights):
                    # Ensure shapes match
                    if saved_val.shape == opt_weight.shape:
                        opt_weight.assign(saved_val)
                    else:
                        print(f"Shape mismatch: {opt_weight.name} - {opt_weight.shape} vs {saved_val.shape}")
            else:
                print(f"Weight count mismatch: {len(weights_data)} saved vs {len(optimizer.weights)} current")
       
        return optimizer
    except Exception as e:
        print(f"Error restoring optimizer state: {e}")
        return optimizer

# Add this utility function to reset optimizer state if needed (fixing NaN issues)
def reset_optimizer_state(optimizer):
    """Reset optimizer state to handle NaN values or other issues"""
    try:
        # Zero out momentum and other state variables
        for var in optimizer.weights:
            var.assign(tf.zeros_like(var))
       
        # Set iteration back to 0
        optimizer.iterations.assign(0)
       
        return optimizer
    except Exception as e:
        print(f"Error resetting optimizer: {e}")
        return optimizer

# Modify the handle_training_issues function to use reset_optimizer_state
def handle_training_issues(generator_optimizer, discriminator_optimizer):
    """Check for and fix optimizer issues"""
    try:
        nan_detected = False
       
        # Check generator optimizer for NaN
        for var in generator_optimizer.weights:
            if tf.reduce_any(tf.math.is_nan(var)):
                nan_detected = True
                break
       
        if nan_detected:
            print("Detected NaN in generator optimizer, resetting...")
            generator_optimizer = reset_optimizer_state(generator_optimizer)
       
        # Reset nan_detected flag
        nan_detected = False
       
        # Check discriminator optimizer for NaN
        for var in discriminator_optimizer.weights:
            if tf.reduce_any(tf.math.is_nan(var)):
                nan_detected = True
                break
       
        if nan_detected:
            print("Detected NaN in discriminator optimizer, resetting...")
            discriminator_optimizer = reset_optimizer_state(discriminator_optimizer)
       
        return generator_optimizer, discriminator_optimizer
    except Exception as e:
        print(f"Error handling training issues: {e}")
        return generator_optimizer, discriminator_optimizer

#main function
   
if __name__  == '__main__':
    with strategy.scope():
        # Get train, validation and test datasets
        train_dataset, val_dataset, test_dataset = create_dataset()
       
        # Train the model
        generator, discriminator, history = train(train_dataset, val_dataset, epochs=200, resume_training=True)
       
        test_metrics = defaultdict(list)
        for test_batch in test_dataset:
            sar_batch, color_batch, terrain_batch = test_batch
            generated_images = generator([sar_batch, terrain_batch], training=False)
            cycle_reconstructed = generator([generated_images, terrain_batch], training=False)
            detailed_metrics = calculate_detailed_metrics(sar_batch, color_batch, generated_images, cycle_reconstructed, terrain_batch)
            for k, v in detailed_metrics.items():
                test_metrics[k].append(v)
        print("Test Metrics:", {k: np.mean(v) for k, v in test_metrics.items()})