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
import json
from scipy.linalg import sqrtm
import tensorflow_probability as tfp

# Constants
BUFFER_SIZE = 400
BATCH_SIZE = 1  # Changed back to 1 for stability
IMG_WIDTH = 256
IMG_HEIGHT = 256
LAMBDA = 10 # Reduced from 11 to 10 for better balance
TERRAIN_TYPES = ['urban', 'grassland', 'agri', 'barrenland']

# Update InstanceNormalization class to handle serialization
class InstanceNormalization(layers.Layer):
    """Native implementation of Instance Normalization"""
    def __init__(self, epsilon=1e-8, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(input_shape[-1],), initializer='ones')
        self.offset = self.add_weight(name='offset', shape=(input_shape[-1],), initializer='zeros')

    def call(self, inputs):
        # Get the input dtype to ensure we return same type
        input_dtype = inputs.dtype
        
        # Calculate mean and variance (moments)
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        
        # Cast weights to match input dtype for proper broadcasting in mixed precision
        scale = tf.cast(self.scale, input_dtype)
        offset = tf.cast(self.offset, input_dtype)
        
        # Use the static shape from self.scale for reshaping to avoid shape mismatches
        num_channels = self.scale.shape[0]
        scale = tf.reshape(scale, [1, 1, 1, num_channels])
        offset = tf.reshape(offset, [1, 1, 1, num_channels])
        
        return scale * normalized + offset

    def get_config(self):
        config = super().get_config()
        config.update({
            'epsilon': self.epsilon,
        })
        return config

# Define required component classes before they're used
# Enhanced ChannelAttention module for focusing on important feature dimensions
class ChannelAttention(layers.Layer):
    """Efficient channel attention module"""
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.max_pool = layers.GlobalMaxPooling2D(keepdims=True)
        self.avg_dense1 = None
        self.avg_dense2 = None
        self.max_dense1 = None
        self.max_dense2 = None

    def build(self, input_shape):
        channels = input_shape[-1]
        reduced_channels = max(channels // self.ratio, 8)
        self.avg_dense1 = layers.Dense(reduced_channels, activation='relu', use_bias=False)
        self.avg_dense2 = layers.Dense(channels, use_bias=False)
        self.max_dense1 = layers.Dense(reduced_channels, activation='relu', use_bias=False)
        self.max_dense2 = layers.Dense(channels, use_bias=False)
        super().build(input_shape)

    def call(self, inputs):
        channels = inputs.shape[-1]
        batch_size = tf.shape(inputs)[0]

        # Pooling
        avg_pool = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=[1, 2], keepdims=True)

        avg_pool_2d = tf.reshape(avg_pool, [batch_size, channels])
        max_pool_2d = tf.reshape(max_pool, [batch_size, channels])

        avg_attention = self.avg_dense2(self.avg_dense1(avg_pool_2d))
        max_attention = self.max_dense2(self.max_dense1(max_pool_2d))

        attention = tf.nn.sigmoid(avg_attention + max_attention)
        attention = tf.reshape(attention, [batch_size, 1, 1, channels])
        attention = tf.cast(attention, inputs.dtype)
        return inputs * attention

    def get_config(self):
        config = super().get_config()
        config.update({
            'ratio': self.ratio
        })
        return config

# Spatial Attention module for focusing on important spatial regions
class SpatialAttention(layers.Layer):
    """Memory-efficient spatial attention"""
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)
        self.kernel_size = kernel_size
        self.conv = layers.Conv2D(1, kernel_size, padding='same', activation='sigmoid')
        
    def build(self, input_shape):
        self.conv.build([input_shape[0], input_shape[1], input_shape[2], 2])
        self.built = True
        
    def call(self, inputs):
        # Generate channel-wise statistics
        avg_pool = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(inputs, axis=-1, keepdims=True)
        
        # Concatenate pools and apply convolution
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        spatial_attention = self.conv(concat)
        
        return inputs * spatial_attention
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'kernel_size': self.kernel_size
        })
        return config

# Combined attention module (CBAM) with memory optimization
class CBAM(layers.Layer):
    """Memory-efficient CBAM implementation"""
    def __init__(self, filters, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.channel_attention = None
        self.spatial_attention = None

    def build(self, input_shape):
        # Always build ChannelAttention and SpatialAttention with the actual input shape
        self.channel_attention = ChannelAttention()
        self.channel_attention.build(input_shape)
        self.spatial_attention = SpatialAttention()
        self.spatial_attention.build(input_shape)
        self.built = True

    def call(self, inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters
        })
        return config

# Spectral normalization for weight stability
class SpectralNormalization(layers.Wrapper):
    """Memory-efficient spectral normalization implementation"""
    def __init__(self, layer, iterations=1, **kwargs):
        super().__init__(layer, **kwargs)
        self.iterations = iterations
        
    def build(self, input_shape=None):
        if not self.layer.built:
            self.layer.build(input_shape)
            
        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()
        
        # Initialize u vector for power iteration
        # Use correct shape: The vector u should have shape (1, output_channels)
        self.u = self.add_weight(
            shape=(1, self.w_shape[-1]), 
            initializer=tf.initializers.TruncatedNormal(stddev=0.02),
            trainable=False,
            name='sn_u'
        )
        
        super().build()
        
    def call(self, inputs):
        self._update_weights()
        return self.layer(inputs)
        
    def _update_weights(self):
        """Efficient power iteration method with proper dimension handling"""
        # Reshape kernel to [kernel_height * kernel_width * input_channels, output_channels]
        # This ensures the matrix multiplication dimensions are compatible
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])
        
        # Just one iteration is usually sufficient and saves memory
        u_hat = tf.identity(self.u)  # [1, output_channels]
        
        # Transpose w_reshaped for first multiplication
        # u_hat [1, output_channels] × w_reshaped^T [output_channels, flattened_kernel_size]
        v_hat = tf.matmul(u_hat, tf.transpose(w_reshaped))  # Result: [1, flattened_kernel_size]
        v_hat = tf.nn.l2_normalize(v_hat)  # Normalize to unit vector
        
        # Now multiply v_hat [1, flattened_kernel_size] × w_reshaped [flattened_kernel_size, output_channels]
        u_hat = tf.matmul(v_hat, w_reshaped)  # Result: [1, output_channels]
        u_hat = tf.nn.l2_normalize(u_hat)  # Normalize to unit vector
        
        # Update the stored u vector
        self.u.assign(u_hat)
        
        # Compute the spectral norm: u × W × v^T
        sigma = tf.squeeze(tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat)))
        
        # Ensure sigma is a scalar
        sigma = tf.reshape(sigma, [])
        
        # Avoid division by zero
        sigma = tf.maximum(sigma, 1e-12)
        
        # Update the layer's kernel
        self.layer.kernel.assign(self.w / sigma)
    
    def compute_output_shape(self, input_shape):
        return self.layer.compute_output_shape(input_shape)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'iterations': self.iterations
        })
        return config

# Memory-efficient wavelet-inspired processing block
class WaveletResidualBlock(layers.Layer):
    """Memory-efficient wavelet-inspired residual block"""
    def __init__(self, filters, dropout_rate=0.0, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate
        self.skip_conv = None
        self.attention = CBAM(filters)
        self.dropout = layers.Dropout(dropout_rate)
        self.activation = layers.LeakyReLU(0.2)
        # Remove group_filters and group layers from __init__
        # They will be created in build()
        self.channel_adjust = None  # <-- Add this line
    
    def build(self, input_shape):
        input_channels = input_shape[-1]
        # Compute group sizes (last group may be larger if not divisible by 4)
        base_group = input_channels // 4
        group_sizes = [base_group] * 3 + [input_channels - base_group * 3]
        self.group_sizes = group_sizes

        # Build group convs and norms with correct shapes
        self.group_convs1 = []
        self.group_convs2 = []
        self.group_norms1 = []
        self.group_norms2 = []
        for i, ch in enumerate(group_sizes):
            conv1 = layers.SeparableConv2D(ch, 3, padding='same')
            conv2 = layers.SeparableConv2D(ch, 3, padding='same')
            norm1 = InstanceNormalization()
            norm2 = InstanceNormalization()
            # Build with correct group shape
            group_shape = tf.TensorShape([input_shape[0], input_shape[1], input_shape[2], ch])
            conv1.build(group_shape)
            conv2.build(group_shape)
            norm1.build(group_shape)
            norm2.build(group_shape)
            self.group_convs1.append(conv1)
            self.group_convs2.append(conv2)
            self.group_norms1.append(norm1)
            self.group_norms2.append(norm2)

        if input_channels != self.filters:
            self.skip_conv = layers.Conv2D(self.filters, 1, padding='same')
            self.skip_conv.build(input_shape)
        # Always build channel_adjust for possible use in call()
        # Use input_channels for the last dimension (not None)
        self.channel_adjust = layers.Conv2D(self.filters, 1, padding='same')
        self.channel_adjust.build(tf.TensorShape([input_shape[0], input_shape[1], input_shape[2], input_channels]))
        self.attention.build(tf.TensorShape([input_shape[0], input_shape[1], input_shape[2], self.filters]))
        self.built = True

    def compute_output_shape(self, input_shape):
        # Handle case when input_shape is a list or tuple
        if isinstance(input_shape, list) or isinstance(input_shape, tuple):
            input_shape = input_shape[0]
        
        # Handle case when input_shape is None
        if input_shape is None:
            return tf.TensorShape((None, None, None, self.filters))
        
        # Get input shape as list for easier manipulation
        input_list = input_shape.as_list()
        
        # Return shape with batch, height, width from input, but channels=filters
        return tf.TensorShape(input_list[:-1] + [self.filters])
        
    def call(self, inputs, training=None):
        # Check if inputs is a list or a tensor (with better error handling)
        if isinstance(inputs, (list, tuple)):
            if len(inputs) >= 2:
                x, terrain = inputs[0], inputs[1]
            else:
                x = inputs[0]
                terrain = None
                print("Warning: Expected two elements in inputs list but got fewer.")
        else:
            # Direct tensor input
            x = inputs
            terrain = None
            
        # Safety check - ensure x is a tensor
        if x is None:
            raise ValueError("Input tensor is None in WaveletResidualBlock.")
        
        shortcut = x

        # Ensure shortcut has the correct number of channels
        if shortcut.shape[-1] != self.filters:
            # If skip_conv exists, use it; otherwise, create and use a new 1x1 conv
            if self.skip_conv is not None:
                shortcut = self.skip_conv(x)
            else:
                shortcut = layers.Conv2D(self.filters, 1, padding='same')(x)
        
        # Split channels into groups according to group_sizes
        group_outputs = []
        start = 0
        for i, ch in enumerate(self.group_sizes):
            group = x[:, :, :, start:start+ch]
            group = self.group_convs1[i](group)
            group = self.group_norms1[i](group)
            group = self.activation(group)
            dilation = i + 1
            group = self.group_convs2[i](group)
            if dilation > 1:
                group = group * (1.0 + 0.1 * i)
            group = self.group_norms2[i](group)
            group = self.activation(group)
            group_outputs.append(group)
            start += ch
        
        # Concatenate the processed groups
        output = tf.concat(group_outputs, axis=-1)
        
        # Make sure output has exactly self.filters channels
        # This ensures output and shortcut have the same shape
        if output.shape[-1] != self.filters:
            output = self.channel_adjust(output)
        
        # Apply attention
        if terrain is not None:
            output = self.attention([output, terrain])
        else:
            output = self.attention(output)
        
        # Apply dropout
        if self.dropout_rate > 0:
            output = self.dropout(output, training=training)
        
        # Add residual connection (shapes now guaranteed to match)
        output = layers.add([output, shortcut])
        output = self.activation(output)
        
        return output
        
    def get_config(self):
        config = super().get_config()
        config.update({
            'filters': self.filters,
            'dropout_rate': self.dropout_rate
        })
        return config

# Add task-specific metrics function
def calculate_task_specific_metrics(real_images, generated_images, metrics_dict):
    """Calculate SAR-specific metrics"""
    # Convert to float32 for metric calculation
    real_images = tf.cast(real_images, tf.float32)
    generated_images = tf.cast(generated_images, tf.float32)
    
    # Ensure valid range
    real_clipped = tf.clip_by_value(real_images, -1.0, 1.0)
    gen_clipped = tf.clip_by_value(generated_images, -1.0, 1.0)
    
    # Color fidelity - measure color histogram similarity
    real_mean_color = tf.reduce_mean(real_clipped, axis=[1, 2])
    gen_mean_color = tf.reduce_mean(gen_clipped, axis=[1, 2])
    color_diff = tf.reduce_mean(tf.abs(real_mean_color - gen_mean_color))
    metrics_dict['color_fidelity'] = color_diff
    
    # Edge consistency - measure edge preservation
    real_dy, real_dx = tf.image.image_gradients(real_clipped)
    gen_dy, gen_dx = tf.image.image_gradients(gen_clipped)
    
    real_edges = tf.sqrt(tf.square(real_dx) + tf.square(real_dy))
    gen_edges = tf.sqrt(tf.square(gen_dx) + tf.square(gen_dy))
    
    edge_diff = tf.reduce_mean(tf.abs(real_edges - gen_edges))
    metrics_dict['edge_consistency'] = edge_diff
    
    return metrics_dict

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
        super(TerrainAdaptiveNormalization(self, self).build(input_shape))

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
    """Enhanced wavelet-based generator with memory optimization"""
    # Input layers
    sar_input = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name='sar_input')
    terrain_input = layers.Input(shape=[len(TERRAIN_TYPES)], name='terrain_input')
    
    # Initial feature extraction - keep resolution
    x = layers.Conv2D(64, 7, padding='same', strides=1, 
                     kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
                     use_bias=True)(sar_input)
    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Process terrain features
    terrain_features = layers.Dense(256, activation='relu')(terrain_input)
    terrain_features = layers.Dropout(0.1)(terrain_features)  # Add dropout for generalization
    terrain_features = layers.Dense(512, activation='relu')(terrain_features)
    
    # Initialize skip connections and filter sizes
    skip_connections = []
    filter_sizes = [64, 128, 256]
    
    # Encoder path with wavelet residual blocks
    current_resolution = x
    
    for i, filters in enumerate(filter_sizes):
        # Process with wavelet block BEFORE storing skip connection
        wavelet_block = WaveletResidualBlock(filters)
        current_resolution = wavelet_block(current_resolution)
        
        # Store skip connection AFTER processing but BEFORE downsampling
        if i < len(filter_sizes) - 1:  # All but the last level need skip connections
            skip_connections.append(current_resolution)
            # Don't use tf.shape directly on KerasTensor - it causes errors
        
        # Downsample except for the last level
        if i < len(filter_sizes) - 1:
            current_resolution = layers.Conv2D(
                filters*2, 4, strides=2, padding='same',
                kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02)
            )(current_resolution)
            current_resolution = InstanceNormalization()(current_resolution)
            current_resolution = layers.LeakyReLU(0.2)(current_resolution)
    
    # Bottleneck - reduced for memory efficiency
    for i in range(6):  # Reduced from 9 to 6
        # Apply terrain-guided attention in bottleneck
        terrain_spatial = layers.Dense(256)(terrain_features)
        
        # Reshape to 1x1 spatial dimensions with 256 channels
        terrain_spatial = layers.Reshape((1, 1, 256))(terrain_spatial)
        
        # Tile terrain features correctly
        terrain_spatial = layers.Lambda(
            lambda inputs: tf.tile(
                inputs[0],  # terrain_spatial tensor
                [1, tf.shape(inputs[1])[1], tf.shape(inputs[1])[2], 1]  # shape: [batch, height, width, channels]
            )
        )([terrain_spatial, current_resolution])
        
        # Add terrain conditioning
        conditioned = layers.Add()([current_resolution, terrain_spatial])
        
        # Apply wavelet residual block
        current_resolution = WaveletResidualBlock(
            filter_sizes[-1], 
            dropout_rate=0.1 if i % 2 == 0 else 0.0  # Alternate dropout
        )(conditioned)
    
    # Decoder path with skip connections
    for i, (skip, filters) in enumerate(zip(
        reversed(skip_connections),
        reversed(filter_sizes[:-1])  # Skip last filter size
    )):
        # Upsample
        current_resolution = layers.Conv2DTranspose(
            filters, 4, strides=2, padding='same',
            kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02)
        )(current_resolution)
        current_resolution = InstanceNormalization()(current_resolution)
        current_resolution = layers.LeakyReLU(0.2)(current_resolution)
        
        # Force current_resolution to match skip connection's spatial dimensions
        # This ensures concatenation works regardless of rounding issues in Conv2DTranspose
        current_resolution = layers.Lambda(
            lambda x: tf.image.resize(x[0], [tf.shape(x[1])[1], tf.shape(x[1])[2]])
        )([current_resolution, skip])
        
        # Apply adaptive skip connection weighting
        skip_weight = layers.Conv2D(1, 1, activation='sigmoid')(skip)
        weighted_skip = layers.Multiply()([skip, skip_weight])
        
        # Concatenate with skip connection
        current_resolution = layers.Concatenate()([current_resolution, weighted_skip])
        
        # Apply wavelet residual block
        current_resolution = WaveletResidualBlock(filters)(current_resolution)
    
    # Final output processing with edge enhancement
    # Create a custom edge detection layer with proper shape handling
    class EdgeDetectionLayer(layers.Layer):
        """Custom layer for edge detection with explicit output shape"""
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            
            # Define Sobel filters as constants
            self.sobel_x = tf.constant([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=tf.float32)
            self.sobel_y = tf.constant([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], dtype=tf.float32)
            
            # Reshape filters for convolution
            self.sobel_x = tf.reshape(self.sobel_x, [3, 3, 1, 1])
            self.sobel_y = tf.reshape(self.sobel_y, [3, 3, 1, 1])
        
        def call(self, inputs):
            # Get input dtype to ensure consistent data types
            input_dtype = inputs.dtype
            
            # Ensure inputs are 4D with shape [batch, height, width, channels]
            if len(inputs.shape) > 4:
                # Reshape if too many dimensions
                inputs = tf.reshape(inputs, [-1, tf.shape(inputs)[1], tf.shape(inputs)[2], tf.shape(inputs)[-1]])
            
            # Cast inputs to float32 for convolution operation
            inputs_f32 = tf.cast(inputs, tf.float32)
            
            # Process each channel separately
            channels = tf.unstack(inputs_f32, axis=-1)
            edges_x = []
            edges_y = []
            
            for channel in channels:
                # Add dimensions for batch and channels
                channel = tf.reshape(channel, [-1, tf.shape(inputs)[1], tf.shape(inputs)[2]])
                channel = tf.expand_dims(channel, -1)
                
                # Apply convolution using float32 for both inputs and filters
                edge_x = tf.nn.conv2d(channel, self.sobel_x, strides=[1, 1, 1, 1], padding='SAME')
                edge_y = tf.nn.conv2d(channel, self.sobel_y, strides=[1, 1, 1, 1], padding='SAME')
                
                # Always squeeze only the channel dimension (not batch)
                edge_x = tf.squeeze(edge_x, axis=-1)
                edge_y = tf.squeeze(edge_y, axis=-1)
            
                edges_x.append(edge_x)
                edges_y.append(edge_y)
            
            # Stack back to image format - ensure proper dimensions
            # Remove the conditional batch dimension logic
            edge_x = tf.stack(edges_x, axis=-1)
            edge_y = tf.stack(edges_y, axis=-1)
            
            # Calculate magnitude
            edges = tf.sqrt(tf.square(edge_x) + tf.square(edge_y))
            
            # Cast back to original dtype
            edges = tf.cast(edges, input_dtype)
            
            # Ensure output has the same batch and spatial dimensions as input
            edges = tf.reshape(edges, tf.shape(inputs))
            
            return edges
        
        def compute_output_shape(self, input_shape):
            # Output shape is the same as input shape
            return input_shape
    
    # Apply edge detection using the custom layer
    edges = EdgeDetectionLayer()(sar_input)
    
    # Final convolution blocks
    x = layers.Conv2D(32, 3, padding='same')(current_resolution)
    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    
    # Use edges to enhance final output
    edge_weight = layers.Conv2D(1, 1, activation='sigmoid')(edges)
    edge_features = layers.Concatenate()([x, edge_weight * edges])
    
    # Final output with tanh activation
    outputs = layers.Conv2D(
        3, 7, padding='same', activation='tanh', dtype=tf.float32,
        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02)
    )(edge_features)
    
    return tf.keras.Model(inputs=[sar_input, terrain_input], outputs=outputs, name="WaveletSARColorizer")

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
    """Enhanced multi-scale discriminator with spectral normalization"""
    # Input layers
    input_image = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name="input_image")
    target_image = layers.Input(shape=[IMG_HEIGHT, IMG_WIDTH, 3], name="target_image")
    terrain_input = layers.Input(shape=[len(TERRAIN_TYPES)], name="terrain_type")
    
    # Process terrain features efficiently
    terrain_features = layers.Dense(256, activation='relu')(terrain_input)
    terrain_features = layers.Dense(512, activation='relu')(terrain_features)
    
    # Create terrain spatial representation
    terrain_spatial = layers.Dense(256)(terrain_input)
    terrain_spatial = layers.Dense(IMG_HEIGHT * IMG_WIDTH)(terrain_spatial)
    terrain_spatial = layers.Reshape((IMG_HEIGHT, IMG_WIDTH, 1))(terrain_spatial)
    
    # Combine inputs - Using layers.Concatenate
    x = layers.Concatenate(axis=-1)([input_image, target_image, terrain_spatial])
    
    # Helper function to apply spectral normalization to Conv2D layers
    def snconv2d(x, filters, kernel_size=4, strides=2, padding='same'):
        conv = layers.Conv2D(
            filters, kernel_size, strides=strides, padding=padding,
            kernel_initializer=tf.keras.initializers.RandomNormal(0.0, 0.02),
            use_bias=False
        )
        # Build the layer first to ensure the kernel is created
        input_shape = tf.keras.backend.int_shape(x)
        conv.build(input_shape)
        # Then wrap with spectral normalization
        return SpectralNormalization(conv)(x)
    
    # Create discriminators at multiple scales (original, 1/2, and 1/4)
    outputs = []
    feature_maps = []
    
    for scale, downscale in enumerate([1, 2, 4]):
        if scale > 0:
            # Use average pooling for downsampling
            current_input = layers.AveragePooling2D(downscale)(x)
        else:
            current_input = x
        
        # Track features for feature matching loss
        scale_features = []
        
        # First layer - no normalization
        d = snconv2d(current_input, 64, strides=2)
        d = layers.LeakyReLU(0.2)(d)
        scale_features.append(d)
        
        # Second layer
        d = snconv2d(d, 128, strides=2)
        d = InstanceNormalization()(d)
        d = layers.LeakyReLU(0.2)(d)
        scale_features.append(d)
        
        # Apply terrain conditioning using Keras layers instead of tf operations
        terrain_gamma = layers.Dense(128)(terrain_features)
        terrain_beta = layers.Dense(128)(terrain_features)
        
        # Use Lambda layers to handle reshaping with dynamic batch sizes
        terrain_gamma = layers.Lambda(lambda x: tf.reshape(x, [-1, 1, 1, 128]))(terrain_gamma)
        terrain_beta = layers.Lambda(lambda x: tf.reshape(x, [-1, 1, 1, 128]))(terrain_beta)
        
        # Apply conditioning
        d = layers.Multiply()([d, (1.0 + terrain_gamma)])
        d = layers.Add()([d, terrain_beta])
        
        # Third layer
        d = snconv2d(d, 256, strides=2)
        d = InstanceNormalization()(d)
        d = layers.LeakyReLU(0.2)(d)
        scale_features.append(d)
        
        # Apply CBAM attention
        d = CBAM(256)(d)
        
        # Fourth layer
        d = snconv2d(d, 512, strides=1)  # No downsampling
        d = InstanceNormalization()(d)
        d = layers.LeakyReLU(0.2)(d)
        scale_features.append(d)
        
        # Final layer for discrimination
        d_out = layers.Conv2D(1, 4, strides=1, padding='same')(d)
        
        # Store output and features
        outputs.append(d_out)
        feature_maps.append(scale_features)
    
    # Flatten feature maps for easier access
    flattened_features = []
    for scale_feats in feature_maps:
        flattened_features.extend(scale_feats)
    
    # Create model with both outputs and feature maps for feature matching
    all_outputs = outputs + [flattened_features]
    return tf.keras.Model(
        inputs=[input_image, target_image, terrain_input], 
        outputs=all_outputs,
        name="MultiScaleDiscriminator"
    )

def generator_loss(disc_generated_output, generated_images, target_images, generator_model=None):
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
    perceptual_weight = tf.constant(0.05, dtype=tf.float32)  # Increased from 0.075 to 0.2
   
    # Add spectral regularization to prevent mode collapse
    spectral_reg = tf.constant(0.0, dtype=tf.float32)
    if generator_model is not None:
        spectral_reg = 0.0001 * tf.reduce_mean([tf.reduce_mean(tf.square(w))
                                              for w in generator_model.trainable_variables
                                              if 'kernel' in w.name])
                                         
    total_loss_f32 = gan_loss + (lambda_val * l1_loss) + (lambda_val * perceptual_weight * perceptual_loss) + spectral_reg
   
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
        # Use one-sided label smoothing (0.9 instead of 1.0) for real labels
        labels = 0.9 * tf.ones_like(output_f32)
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
   
    # Reduce gradient penalty coefficient to prevent discriminator dominance
    gp = tf.reduce_mean((norm - 1.0) ** 2) * 0.5  # Multiplier reduced from 1.0 to 0.5
   
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
            gen_loss_result = generator_loss(disc_generated_output, generated_images_f32, color_images, generator)
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
    """Initialize history dictionary with all metrics"""
    return {
        'gen_loss': [], 'disc_loss': [],
        'psnr': [], 'ssim': [],
        'cycle_loss': [], 'l2_loss': [],
        'feature_matching_loss': [],
        'lpips': [],
        'l1_loss': [], 
        'perceptual_loss': [],
        'edge_consistency': [],
        'color_fidelity': [],
        'val_gen_loss': [], 'val_disc_loss': [],
        'val_psnr': [], 'val_ssim': [],
        'val_l1_loss': [], 'val_l2_loss': [],
        'val_cycle_loss': [],
        'val_feature_matching_loss': [],
        'val_lpips': [],
        'val_perceptual_loss': [],
        'val_edge_consistency': [],
        'val_color_fidelity': [],
        'fid': [], 'val_fid': []
    }

# Define the detailed metrics calculation function
def calculate_detailed_metrics(sar_batch, color_batch, generated_images, cycle_reconstructed, terrain_batch):
    """Calculate comprehensive metrics for test evaluation"""
    metrics_dict = {}
    
    # Calculate standard image quality metrics
    psnr = tf.image.psnr(
        tf.clip_by_value(color_batch, -1.0, 1.0),
        tf.clip_by_value(generated_images, -1.0, 1.0),
        max_val=2.0
    )
    metrics_dict['psnr'] = tf.reduce_mean(psnr)
    
    ssim = tf.image.ssim(
        tf.clip_by_value(color_batch, -1.0, 1.0),
        tf.clip_by_value(generated_images, -1.0, 1.0),
        max_val=2.0
    )
    metrics_dict['ssim'] = tf.reduce_mean(ssim)
    
    # L1 and L2 losses
    metrics_dict['l1_loss'] = tf.reduce_mean(tf.abs(color_batch - generated_images))
    metrics_dict['l2_loss'] = tf.reduce_mean(tf.square(color_batch - generated_images))
    
    # Cycle consistency
    metrics_dict['cycle_loss'] = tf.reduce_mean(tf.abs(sar_batch - cycle_reconstructed))
    
    # Add task-specific metrics
    metrics_dict = calculate_task_specific_metrics(color_batch, generated_images, metrics_dict)
    
    return metrics_dict

# Add function to handle training issues
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

# Define train function
def train(train_dataset, val_dataset, epochs=200, resume_training=True):
    """Main training function that manages the training and validation processes"""
    # Configuration
    LEARNING_RATE = 2e-4
    BETA_1 = 0.5
    BETA_2 = 0.999
    ACCUMULATION_STEPS = 4  # Gradient accumulation steps
    
    # Paths for saving models and visualizations
    model_dir = '/kaggle/working/checkpoints'
    visualization_dir = '/kaggle/working/visualizations'
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    # Prepare distributed datasets
    dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
    dist_val_dataset = strategy.experimental_distribute_dataset(val_dataset)
    
    # Set a small subset of validation data for visualization
    val_vis_dataset = val_dataset.take(5)
    
    # Early stopping parameters
    patience = 15
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Initialize or load models
    with strategy.scope():
        # Create the models
        generator = build_terrain_aware_generator()
        discriminator = build_terrain_aware_discriminator()
        
        # Create optimizers
        generator_optimizer = tf.keras.optimizers.Adam(
            LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2
        )
        discriminator_optimizer = tf.keras.optimizers.Adam(
            LEARNING_RATE, beta_1=BETA_1, beta_2=BETA_2
        )
        
        # Initialize gradient accumulators (for gradient accumulation)
        gen_gradient_accumulators = [
            tf.Variable(tf.zeros_like(var), trainable=False)
            for var in generator.trainable_variables
        ]
        disc_gradient_accumulators = [
            tf.Variable(tf.zeros_like(var), trainable=False)
            for var in discriminator.trainable_variables
        ]
        
        # Initialize starting epoch and history
        start_epoch = 0
        history = get_initial_history()
        
        # Load saved model if resuming training
        if resume_training:
            try:
                latest_checkpoint = tf.train.latest_checkpoint(model_dir)
                if latest_checkpoint:
                    # Load weights
                    generator.load_weights(os.path.join(model_dir, 'generator_latest'))
                    discriminator.load_weights(os.path.join(model_dir, 'discriminator_latest'))
                    
                    # Try to load training state
                    with open(os.path.join(model_dir, 'train_state.json'), 'r') as f:
                        train_state = json.load(f)
                    start_epoch = train_state.get('epoch', 0)
                    history = train_state.get('history', history)
                    print(f"Resuming training from epoch {start_epoch}")
            except Exception as e:
                print(f"Error loading checkpoint: {e}")

        def reset_gradients():
            """Reset accumulated gradients"""
            for accumulator in gen_gradient_accumulators:
                accumulator.assign(tf.zeros_like(accumulator))
            for accumulator in disc_gradient_accumulators:
                accumulator.assign(tf.zeros_like(accumulator))

        # Modified apply_grads function to work within strategy's scope
        @tf.function
        def apply_accumulated_gradients():
            """Apply accumulated gradients within the distribution strategy scope"""
            def _apply_gradients_fn():
                # Clip accumulated gradients
                gen_grads_clipped = [
                    tf.clip_by_norm(g, 1.0) for g in gen_gradient_accumulators
                ]
                disc_grads_clipped = [
                    tf.clip_by_norm(g, 1.0) for g in disc_gradient_accumulators
                ]
                
                # Apply gradients with explicit variable references
                generator_optimizer.apply_gradients(
                    [(g, v) for g, v in zip(gen_grads_clipped, generator.trainable_variables)]
                )
                discriminator_optimizer.apply_gradients(
                    [(g, v) for g, v in zip(disc_grads_clipped, discriminator.trainable_variables)]
                )
                
                # Reset gradients after applying
                return True
            
            # Run the gradient application within strategy scope
            return strategy.run(_apply_gradients_fn)
    
        # ...existing code for distributed_train_step, distributed_val_step, etc...
    
    # Training loop
    for epoch in range(start_epoch, start_epoch + epochs):
        # ...existing code for main training and validation loops...
        pass
        
    # Save final model
    generator.save_weights(os.path.join(model_dir, 'generator_final.h5'))
    discriminator.save_weights(os.path.join(model_dir, 'discriminator_final'))
    
    # Save final training state
    with open(os.path.join(model_dir, 'train_state_final.json'), 'w') as f:
        json.dump({
            'epoch': start_epoch + epochs - 1,
            'history': history
        }, f)
    
    return generator, discriminator, history

def generate_and_save_visualizations(generator, dataset, output_dir):
    """Generate and save visualization images"""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, batch in enumerate(dataset):
        sar_images, color_images, terrain_labels = batch
        
        # Generate images
        generated_images = generator([sar_images, terrain_labels], training=False)
        
        # Convert from [-1,1] to [0,1] for visualization
        sar_display = (sar_images + 1) / 2
        gen_display = (generated_images + 1) / 2
        real_display = (color_images + 1) / 2
        
        # Create a figure with 3 images side by side
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title('SAR Input')
        plt.imshow(sar_display[0].numpy())
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.title('Generated')
        plt.imshow(gen_display[0].numpy())
        plt.axis('off')
        
        plt.subplot(1, 3, 3)
        plt.title('Ground Truth')
        plt.imshow(real_display[0].numpy())
        plt.axis('off')
        
        # Save the figure
        plt.savefig(os.path.join(output_dir, f'comparison_{i}.png'))
        plt.close()

def save_model_checkpoint(generator, discriminator, save_data, timestamp):
    """Save model checkpoints"""
    # Define directories
    checkpoint_dir = 'd:/SAR/models'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save models with timestamp
    generator.save_weights(os.path.join(checkpoint_dir, f'generator_{timestamp}'))
    discriminator.save_weights(os.path.join(checkpoint_dir, f'discriminator_{timestamp}'))
    
    # Also save as latest
    generator.save_weights(os.path.join(checkpoint_dir, 'generator_latest'))
    discriminator.save_weights(os.path.join(checkpoint_dir, 'discriminator_latest'))
    
    # Save training state (epoch and history)
    with open(os.path.join(checkpoint_dir, 'train_state.json'), 'w') as f:
        json.dump(save_data, f)
    
    print(f"Saved checkpoint at {timestamp}")

# Main function that runs the training and evaluation
if __name__ == '__main__':
    # Import garbage collector for better memory management
    import gc
    
    with strategy.scope():
        # Get datasets
        train_dataset, val_dataset, test_dataset = create_dataset()
        
        # Train the model with proper metrics tracking
        generator, discriminator, history = train(train_dataset, val_dataset, epochs=200, resume_training=True)
        
        # Evaluate on test set
        test_metrics = defaultdict(list)
        for test_batch in test_dataset:
            sar_batch, color_batch, terrain_batch = test_batch
            generated_images = generator([sar_batch, terrain_batch], training=False)
            cycle_reconstructed = generator([generated_images, terrain_batch], training=False)
            detailed_metrics = calculate_detailed_metrics(
                sar_batch, color_batch, generated_images, cycle_reconstructed, terrain_batch
            )
            for k, v in detailed_metrics.items():
                test_metrics[k].append(v)
                
        print("Test Metrics:", {k: np.mean(v) for k, v in test_metrics.items()})

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

 

