import tensorflow as tf
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



# Constants
BUFFER_SIZE = 400
BATCH_SIZE = 1  # Fixed at 1
IMG_WIDTH = 128  # Changed from 256 to 128
IMG_HEIGHT = 128  # Changed from 256 to 128
LAMBDA = 10
TERRAIN_TYPES = ['urban', 'grassland', 'agri', 'barrenland']
STYLE_WEIGHT = 1.0
CYCLE_WEIGHT = 10.0
COLOR_WEIGHT = 5.0
MODEL_SAVE_DIR = '/kaggle/working/saved_models'
CHECKPOINT_DIR = os.path.join(MODEL_SAVE_DIR, 'checkpoints')
HISTORY_DIR = os.path.join(MODEL_SAVE_DIR, 'history')
HEAVY_METRICS_INTERVAL = 200  # Calculate heavy metrics every 200 steps

class InstanceNormalization(layers.Layer):
    """Native implementation of Instance Normalization"""
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(input_shape[-1],), initializer='ones')
        self.offset = self.add_weight(name='offset', shape=(input_shape[-1],), initializer='zeros')

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        normalized = (inputs - mean) / tf.sqrt(variance + self.epsilon)
        return self.scale * normalized + self.offset

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
    """Compute perceptual loss using the feature extractor"""
    # Ensure inputs are in float32
    real_images = tf.cast(real_images, tf.float32)
    generated_images = tf.cast(generated_images, tf.float32)
   
    # Preprocess images to match EfficientNet requirements
    real_images = tf.keras.applications.efficientnet.preprocess_input(
        (real_images + 1) * 127.5)
    generated_images = tf.keras.applications.efficientnet.preprocess_input(
        (generated_images + 1) * 127.5)
   
    # Extract features
    real_features = feature_extractor(real_images)
    gen_features = feature_extractor(generated_images)
   
    # Compute loss
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
    img = tf.cast(img, tf.float32)
    img = (img / 127.5) - 1.0  # Normalize to [-1, 1]
    return img

def create_dataset():
    """Create dataset with proper distribution handling"""
    with strategy.scope():
        # Increase batch size for multi-GPU training
        GLOBAL_BATCH_SIZE = BATCH_SIZE * strategy.num_replicas_in_sync
       
        datasets = []
        for terrain in TERRAIN_TYPES:
            sar_path = f'/kaggle/input/sentinel12-image-pairs-segregated-by-terrain/v_2/{terrain}/s1/*'
            color_path = f'/kaggle/input/sentinel12-image-pairs-segregated-by-terrain/v_2/{terrain}/s2/*'
           
            sar_dataset = tf.data.Dataset.list_files(sar_path, shuffle=True)
            color_dataset = tf.data.Dataset.list_files(color_path, shuffle=True)
           
            # Use parallel calls for faster data loading
            AUTOTUNE = tf.data.AUTOTUNE
            sar_dataset = sar_dataset.map(
                load_and_preprocess_image,
                num_parallel_calls=AUTOTUNE
            ).cache()
           
            color_dataset = color_dataset.map(
                load_and_preprocess_image,
                num_parallel_calls=AUTOTUNE
            ).cache()
           
            terrain_labels = tf.data.Dataset.from_tensors(
                tf.one_hot(TERRAIN_TYPES.index(terrain), len(TERRAIN_TYPES))
            ).repeat()
           
            dataset = tf.data.Dataset.zip((sar_dataset, color_dataset, terrain_labels))
            datasets.append(dataset)
       
        # Combine datasets efficiently
        combined_dataset = tf.data.Dataset.from_tensor_slices(datasets).interleave(
            lambda x: x,
            cycle_length=len(TERRAIN_TYPES),
            num_parallel_calls=AUTOTUNE
        )
       
        # Use larger buffer size for better shuffling
        return combined_dataset.shuffle(BUFFER_SIZE * 2).batch(
            GLOBAL_BATCH_SIZE, drop_remainder=True
        ).prefetch(AUTOTUNE)

class OptimizedColorTransformation(layers.Layer):
    """Memory-efficient color space transformation"""
    def __init__(self):
        super(OptimizedColorTransformation, self).__init__()
        self.conv1 = layers.Conv2D(32, 1, activation='relu')
        self.conv2 = layers.Conv2D(3, 1)
       
    @tf.function
    def rgb_to_lab_efficient(self, rgb):
        # Optimized RGB to LAB conversion
        rgb = (rgb + 1) * 0.5  # [-1,1] to [0,1]
        features = self.conv1(rgb)
        lab_like = self.conv2(features)
        return lab_like
       
    def call(self, x):
        return self.rgb_to_lab_efficient(x)
   
class LightweightBoundaryDetection(layers.Layer):
    """Efficient boundary detection using depthwise separable convolutions"""
    def __init__(self, out_filters):
        super(LightweightBoundaryDetection, self).__init__()
        self.out_filters = out_filters
        self.instance_norm = InstanceNormalization()
       
    def build(self, input_shape):
        input_channels = input_shape[-1]
        self.depthwise = layers.DepthwiseConv2D(3, padding='same',
                                               depth_multiplier=self.out_filters//input_channels)
        self.pointwise = layers.Conv2D(self.out_filters, 1)
       
    def call(self, x):
        x = self.instance_norm(x)
        x = self.depthwise(x)
        return self.pointwise(x)
   
class EfficientSkipConnection(layers.Layer):
    """Memory-efficient skip connections using 1x1 convolutions"""
    def __init__(self, filters):
        super(EfficientSkipConnection, self).__init__()
        self.reduction = layers.Conv2D(filters//4, 1)
        self.process = layers.DepthwiseConv2D(3, padding='same')
       
    def call(self, x, skip_features):
        processed = [self.reduction(feat) for feat in skip_features]
        processed = [self.process(feat) for feat in processed]
        return tf.concat([x] + processed, axis=-1)

class TerrainGuidedAttention(layers.Layer):
    """Attention mechanism that explicitly considers terrain information"""
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.query = layers.Conv2D(channels // 8, 1)
        self.key = layers.Conv2D(channels // 8, 1)
        self.value = layers.Conv2D(channels, 1)
        self.gamma = self.add_weight(name="gamma", shape=(1,), initializer="zeros")
        self.terrain_project = layers.Dense(channels)  # Initialize in __init__

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError("TerrainGuidedAttention expects a list of input shapes")
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


class TerrainAdaptiveNormalization(layers.Layer):
    def __init__(self, channels):
        super(TerrainAdaptiveNormalization, self).__init__()
        self.channels = channels
        self.norm = layers.BatchNormalization(axis=-1)
        self.terrain_scale = layers.Dense(channels)
        self.terrain_bias = layers.Dense(channels)
        self.color_norm = layers.LayerNormalization(axis=-1)
        self.color_scale = layers.Dense(channels)
        self.color_bias = layers.Dense(channels)

    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError("TerrainAdaptiveNormalization expects a list of input shapes")
        x_shape, terrain_shape = input_shape
       
        # Just build the layers without shape validation
        self.norm.build(x_shape)
        self.terrain_scale.build(terrain_shape)
        self.terrain_bias.build(terrain_shape)
        self.color_norm.build(x_shape)
        self.color_scale.build(terrain_shape)
        self.color_bias.build(terrain_shape)
       
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
   
class TerrainAwareResBlock(layers.Layer):
    """Residual block with terrain-specific processing"""
    def __init__(self, filters):
        super(TerrainAwareResBlock, self).__init__()
        self.conv1 = layers.Conv2D(filters, 3, padding='same')
        self.conv2 = layers.Conv2D(filters, 3, padding='same')
        self.norm1 = TerrainAdaptiveNormalization(filters)
        self.norm2 = TerrainAdaptiveNormalization(filters)
        self.attention = TerrainGuidedAttention(filters)
       
    def call(self, x, terrain_features):
        residual = x
       
        x = self.conv1(x)
        x = self.norm1([x, terrain_features])
        x = tf.nn.relu(x)
       
        x = self.conv2(x)
        x = self.norm2([x, terrain_features])
        x = self.attention([x, terrain_features])
       
        return tf.nn.relu(x + residual)

class ColorRefinementBlock(layers.Layer):
    """Enhanced color refinement with spatial and semantic context"""
    def __init__(self, filters):
        super(ColorRefinementBlock, self).__init__()
        self.filters = filters
        self.context_conv = layers.Conv2D(filters, 3, padding='same')
        self.terrain_norm = TerrainAdaptiveNormalization(filters)
        self.attention = TerrainGuidedAttention(filters)
        self.refine_conv = layers.Conv2D(filters, 1)
        self.activation = layers.Activation('relu')
       
    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError("ColorRefinementBlock expects a list of input shapes")
           
        x_shape, terrain_shape = input_shape
       
        # Build layers directly without shape manipulation
        self.context_conv.build(x_shape)
       
        # Build normalization and attention with known filter size
        conv_output_shape = tf.TensorShape(x_shape[:3]).concatenate(tf.TensorShape([self.filters]))
       
        self.terrain_norm.build([conv_output_shape, terrain_shape])
        self.attention.build([conv_output_shape, terrain_shape])
        self.refine_conv.build(conv_output_shape)
       
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

class MemoryEfficientResBlock(layers.Layer):
    def __init__(self, filters):
        super().__init__()
        self.filters = filters
        self.conv1 = layers.SeparableConv2D(filters, 3, padding='same')
        self.conv2 = layers.SeparableConv2D(filters, 3, padding='same')
        self.norm1 = InstanceNormalization()
        self.norm2 = InstanceNormalization()
        self.attention = TerrainGuidedAttention(filters)  # Match filters
        self.activation = layers.Activation('silu')  # Using Keras Activation
       
    def build(self, input_shape):
        if not isinstance(input_shape, list):
            raise ValueError("MemoryEfficientResBlock expects a list of input shapes")
        x_shape, terrain_shape = input_shape
       
        # Ensure input is projected to correct number of filters if needed
        if x_shape[-1] != self.filters:
            self.input_proj = layers.Conv2D(self.filters, 1, padding='same')
        else:
            self.input_proj = None
           
        super().build(input_shape)
       
    def call(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != 2:
            raise ValueError("MemoryEfficientResBlock expects [x, terrain] as input")
           
        x, terrain = inputs
       
        # Project input if needed
        if self.input_proj is not None:
            x = self.input_proj(x)
           
        residual = x
       
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.activation(x)
       
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.attention([x, terrain])
       
        return x + residual

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], input_shape[0][2], self.filters)
   
class SiLUActivation(layers.Layer):
    """Wrapper for SiLU activation"""
    def call(self, x):
        return tf.nn.silu(x)


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

    # Middle blocks at 32x32 resolution
    for _ in range(9):
        x = MemoryEfficientResBlock(512)([x, terrain_input])

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

class TerrainSpatialLayer(layers.Layer):
    """Layer to handle terrain spatial features with explicit tensor shapes"""
    def __init__(self, height, width):
        super(TerrainSpatialLayer, self).__init__()
        self.height = height
        self.width = width
        self.dense1 = layers.Dense(512, activation='relu')
        self.dense2 = layers.Dense(height * width)
        self.reshape = layers.Reshape((height, width, 1))
       
    def call(self, terrain_input):
        x = self.dense1(terrain_input)
        x = self.dense2(x)
        return self.reshape(x)

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
    def __init__(self, num_losses):
        super(DynamicWeightedLoss, self).__init__()
        self.loss_weights = self.add_weight(
            "loss_weights",
            shape=[num_losses],
            initializer="ones",
            trainable=True
        )
       
    def call(self, losses):
        weights = tf.nn.softmax(self.loss_weights)
        return tf.reduce_sum([w * l for w, l in zip(weights, losses)])
   

class AdaptiveLRSchedule:
    def __init__(self, initial_lr=2e-4, min_lr=1e-6, patience=5, factor=0.5, metric_window=10):
        self.current_lr = initial_lr
        self.min_lr = min_lr
        self.patience = patience
        self.factor = factor
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

def compute_perceptual_loss(real_images, generated_images):
    """Compute perceptual loss using the feature extractor"""
    # Ensure inputs are in float32
    real_images = tf.cast(real_images, tf.float32)
    generated_images = tf.cast(generated_images, tf.float32)
   
    # Preprocess images to match EfficientNet requirements
    real_images = tf.keras.applications.efficientnet.preprocess_input(
        (real_images + 1) * 127.5)
    generated_images = tf.keras.applications.efficientnet.preprocess_input(
        (generated_images + 1) * 127.5)
   
    # Use feature_extractor instead of efficient_extractor
    real_features = feature_extractor(real_images)
    gen_features = feature_extractor(generated_images)
   
    # Compute loss
    perceptual_loss = 0.0
    for real_feat, gen_feat in zip(real_features, gen_features):
        perceptual_loss += tf.reduce_mean(tf.abs(real_feat - gen_feat))
   
    return tf.cast(perceptual_loss, tf.float16)

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

def train(dataset, epochs, resume_training=True):
    """Modified training function with fixed distributed strategy handling"""
   
    if not isinstance(strategy, tf.distribute.Strategy):
        raise ValueError("No distribution strategy found!")
   
    num_replicas = strategy.num_replicas_in_sync
    global_batch_size = BATCH_SIZE * num_replicas
   
    with strategy.scope():
        # Load or initialize models and training history
        if resume_training:
            loaded_gen, loaded_disc, loaded_history = load_latest_models()
            if loaded_gen is not None:
                generator = loaded_gen
                discriminator = loaded_disc
                start_epoch = loaded_history['epoch'] + 1
                history = loaded_history['history']
                print(f"Resuming training from epoch {start_epoch}")
            else:
                generator = build_terrain_aware_generator()
                discriminator = build_terrain_aware_discriminator()
                start_epoch = 0
                history = {
                    'gen_loss': [], 'disc_loss': [],
                    'psnr': [], 'ssim': [],
                    'cycle_loss': [], 'l2_loss': [],
                    'feature_matching_loss': [],
                    'lpips': [],
                    'l1_loss': [], 'style_loss': []
                }
        else:
            generator = build_terrain_aware_generator()
            discriminator = build_terrain_aware_discriminator()
            start_epoch = 0
            history = {
                'gen_loss': [], 'disc_loss': [],
                'psnr': [], 'ssim': [],
                'cycle_loss': [], 'l2_loss': [],
                'feature_matching_loss': [],
                'lpips': [],
                'l1_loss': [], 'style_loss': []
            }
       
        # Initialize optimizers with gradient clipping
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
        dist_dataset = strategy.experimental_distribute_dataset(dataset)
       
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
            print(f"\nEpoch {epoch + 1}/{start_epoch + epochs}")
           
            nan_count = 0  # Track NaN occurrences
            steps_per_epoch = 0
            total_gen_loss = 0
            total_disc_loss = 0
            
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
                        
                        if step % 10 == 0:
                            avg_gen = total_gen_loss / (step + 1)
                            avg_disc = total_disc_loss / (step + 1)
                            print(f"\rStep {step} - Gen Loss: {avg_gen:.4f}, Disc Loss: {avg_disc:.4f}", end='')
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
                print(f"\nAvg Gen Loss: {avg_gen_loss:.4f}, Avg Disc Loss: {avg_disc_loss:.4f}")
            
            print(f"\nTime taken for epoch {epoch + 1}: {time.time() - start:.2f} sec")
           
            # Save checkpoints periodically with safe error handling
            if (epoch + 1) % 10 == 0:
                try:
                    save_path_gen = f'/e:/GitHub/SAR/generator_epoch_{epoch + 1}.h5'
                    save_path_disc = f'/e:/GitHub/SAR/discriminator_epoch_{epoch + 1}.h5'
                    generator.save_weights(save_path_gen)
                    discriminator.save_weights(save_path_disc)
                    print(f"Saved checkpoint at epoch {epoch + 1}")
                except Exception as save_error:
                    print(f"Error saving checkpoint: {save_error}")
   
    return history

   

def save_model_and_history(generator, discriminator, history, epoch, metrics, timestamp=None):
    """Save models, weights, and training history"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
   
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
   
    # Save models and weights
    generator.save(os.path.join(MODEL_SAVE_DIR, f'generator_{timestamp}.h5'))
    discriminator.save(os.path.join(MODEL_SAVE_DIR, f'discriminator_{timestamp}.h5'))
    generator.save_weights(os.path.join(CHECKPOINT_DIR, f'generator_weights_{timestamp}.h5'))
    discriminator.save_weights(os.path.join(CHECKPOINT_DIR, f'discriminator_weights_{timestamp}.h5'))
   
    # Save history and metrics
    history_data = {
        'history': history,
        'final_metrics': metrics,
        'epoch': epoch,
        'timestamp': timestamp
    }
   
    with open(os.path.join(HISTORY_DIR, f'history_{timestamp}.json'), 'w') as f:
        json.dump(history_data, f)
   
    print(f"\nModels and history saved with timestamp: {timestamp}")
    return timestamp

def load_latest_models():
    """Load the most recent saved models and history"""
    try:
        gen_files = glob(os.path.join(MODEL_SAVE_DIR, 'generator_*.h5'))
        if not gen_files:
            return None, None, None
       
        latest_timestamp = max([f.split('_')[-1].replace('.h5','') for f in gen_files])
       
        # Custom objects for model loading
        custom_objects = {
            'InstanceNormalization': InstanceNormalization,
            'TerrainGuidedAttention': TerrainGuidedAttention,
            'TerrainAdaptiveNormalization': TerrainAdaptiveNormalization,
            'MemoryEfficientResBlock': MemoryEfficientResBlock,
            'ColorRefinementBlock': ColorRefinementBlock
        }
       
        # Load models
        generator = tf.keras.models.load_model(
            os.path.join(MODEL_SAVE_DIR, f'generator_{latest_timestamp}.h5'),
            custom_objects=custom_objects
        )
        discriminator = tf.keras.models.load_model(
            os.path.join(MODEL_SAVE_DIR, f'discriminator_{latest_timestamp}.h5'),
            custom_objects=custom_objects
        )
       
        # Load history
        with open(os.path.join(HISTORY_DIR, f'history_{latest_timestamp}.json'), 'r') as f:
            history_data = json.load(f)
       
        print(f"\nLoaded models and history from timestamp: {latest_timestamp}")
        return generator, discriminator, history_data
       
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None

# ...rest of the code remains the same...

def save_sample_predictions(sar_images, real_images, generated_images, epoch):
    plt.figure(figsize=(15, 5))
    display_list = [sar_images[0], real_images[0], generated_images[0]]
    title = ['SAR Image', 'Ground Truth', 'Generated Image']
   
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
   
    plt.savefig(f'predictions_epoch_{epoch}.png')
    plt.close()

class TerrainClassifier(tf.keras.Model):
    """Classifier to ensure generated images maintain terrain characteristics"""
    def __init__(self):
        super(TerrainClassifier, self).__init__()
        self.conv1 = layers.Conv2D(64, 3, strides=2, padding='same')
        self.conv2 = layers.Conv2D(128, 3, strides=2, padding='same')
        self.conv3 = layers.Conv2D(256, 3, strides=2, padding='same')
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(512)
        self.dense2 = layers.Dense(len(TERRAIN_TYPES))
       
    def call(self, x):
        x = tf.nn.relu(self.conv1(x))
        x = tf.nn.relu(self.conv2(x))
        x = tf.nn.relu(self.conv3(x))
        x = self.flatten(x)
        x = tf.nn.relu(self.dense1(x))
        return self.dense2(x)

def create_train_val_datasets():
    """Create training and validation datasets with 90/10 split"""
    datasets = []
    for terrain in TERRAIN_TYPES:
        sar_path = f'/kaggle/input/sentinel12-image-pairs-segregated-by-terrain/v_2/{terrain}/s1/*'
        color_path = f'/kaggle/input/sentinel12-image-pairs-segregated-by-terrain/v_2/{terrain}/s1/*'
       
        # Get file paths
        sar_files = sorted(glob(sar_path))
        color_files = sorted(glob(color_path))
       
        if not sar_files or not color_files:
            print(f"Warning: No files found for terrain {terrain}")
            continue
           
        # Calculate split indices
        total_samples = len(sar_files)
        train_size = int(0.9 * total_samples)
       
        # Split files
        train_sar = sar_files[:train_size]
        train_color = color_files[:train_size]
        val_sar = sar_files[train_size:]
        val_color = color_files[train_size:]
       
        # Create train dataset
        train_sar_dataset = tf.data.Dataset.from_tensor_slices(train_sar)
        train_color_dataset = tf.data.Dataset.from_tensor_slices(train_color)
        train_sar_dataset = train_sar_dataset.map(load_and_preprocess_image)
        train_color_dataset = train_color_dataset.map(load_and_preprocess_image)
       
        # Create validation dataset
        val_sar_dataset = tf.data.Dataset.from_tensor_slices(val_sar)
        val_color_dataset = tf.data.Dataset.from_tensor_slices(val_color)
        val_sar_dataset = val_sar_dataset.map(load_and_preprocess_image)
        val_color_dataset = val_color_dataset.map(load_and_preprocess_image)
       
        # Create terrain labels
        terrain_idx = TERRAIN_TYPES.index(terrain)
        terrain_label = tf.one_hot(terrain_idx, len(TERRAIN_TYPES))
        train_terrain_labels = tf.data.Dataset.from_tensors(terrain_label).repeat(len(train_sar))
        val_terrain_labels = tf.data.Dataset.from_tensors(terrain_label).repeat(len(val_sar))
       
        # Combine datasets
        train_dataset = tf.data.Dataset.zip((train_sar_dataset, train_color_dataset, train_terrain_labels))
        val_dataset = tf.data.Dataset.zip((val_sar_dataset, val_color_dataset, val_terrain_labels))
       
        datasets.append((train_dataset, val_dataset))
   
    if not datasets:
        raise ValueError("No valid datasets found!")
   
    # Combine all terrain datasets
    train_combined = datasets[0][0]
    val_combined = datasets[0][1]
    for train_data, val_data in datasets[1:]:
        train_combined = train_combined.concatenate(train_data)
        val_combined = val_combined.concatenate(val_data)
   
    # Shuffle and batch
    train_dataset = train_combined.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_combined.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
   
    return train_dataset, val_dataset

def format_metrics(metrics_dict):
    """Format metrics for single-line display"""
    metrics_str = "".join([f"{key}: {value:8.4f} | " for key, value in metrics_dict.items()])
    return metrics_str[:-3]  # Remove last separator

class MetricsCalculator:
    def __init__(self):
        # Initialize models inside distribution strategy scope
        strategy = tf.distribute.get_strategy()
        with strategy.scope():
            self.inception_model = InceptionV3(
                include_top=True,
                weights='imagenet',
                input_shape=(299, 299, 3)
            )
        self.preprocess_input = tf.keras.applications.inception_v3.preprocess_input

   
   

    def calculate_cycle_loss(self, original, reconstructed):
        """Calculate cycle consistency loss"""
        return tf.reduce_mean(tf.abs(original - reconstructed))
   
    def calculate_style_loss(self, real_features, generated_features):
        """Calculate style loss using Gram matrices"""
        def gram_matrix(features):
            batch_size = tf.shape(features)[0]
            channels = tf.shape(features)[3]
            features = tf.reshape(features, [batch_size, -1, channels])
            gram = tf.matmul(features, features, transpose_a=True)
            return gram / tf.cast(tf.size(features), tf.float32)
       
        style_loss = 0.0
        for real_feat, gen_feat in zip(real_features, generated_features):
            style_loss += tf.reduce_mean(tf.abs(
                gram_matrix(real_feat) - gram_matrix(gen_feat)
            ))
        return style_loss
   
    def calculate_feature_matching_loss(self, real_features, generated_features):
        """Calculate feature matching loss"""
        fm_loss = 0.0
        for real_feat, gen_feat in zip(real_features, generated_features):
            fm_loss += tf.reduce_mean(tf.abs(real_feat - gen_feat))
        return fm_loss
   
    def calculate_lpips(self, real_images, generated_images):
        """Calculate LPIPS using feature differences"""
        features_real = self.inception_model(real_images)
        features_gen = self.inception_model(generated_images)
        return tf.reduce_mean(tf.square(features_real - features_gen))
   
    def calculate_l1_l2_losses(self, real_images, generated_images):
        """Calculate L1 and L2 losses"""
        l1_loss = tf.reduce_mean(tf.abs(real_images - generated_images))
        l2_loss = tf.reduce_mean(tf.square(real_images - generated_images))
        return l1_loss, l2_loss
   
   
   
       


if __name__ == '__main__':
   

    with strategy.scope():
     train_dataset, val_dataset = create_train_val_datasets()
   
     generator, discriminator, history = train(train_dataset, epochs=200, resume_training=True)  # Increased epochs for better training
    # Plot training history
    plt.figure(figsize=(20, 5))
   
    plt.subplot(1, 4, 1)
    plt.plot(history['gen_loss'], label='Generator Loss')
    plt.plot(history['disc_loss'], label='Discriminator Loss')
    plt.legend()
    plt.title('Training Losses')
   
    plt.subplot(1, 4, 2)
    plt.plot(history['l1_loss'], label='L1 Loss')
    plt.legend()
    plt.title('L1 Loss')
   
    plt.subplot(1, 4, 3)
    plt.plot(history['style_loss'], label='Style Loss')
    plt.legend()
    plt.title('Style Loss')
   
    plt.subplot(1, 4, 4)
    plt.plot(history['ssim'], label='SSIM')
    plt.legend()
    plt.title('SSIM Metric')
   
    plt.savefig('training_history.png')
    plt.close()


    plt.plot(history['ssim'], label='SSIM')
    # Generate final comparisons
    print("Generating final comparison images...")
    test_samples = val_dataset.take(5)  # Take 5 samples
    for i, (sar_batch, color_batch, terrain_batch) in enumerate(test_samples):
        predictions = generator([sar_batch, terrain_batch], training=False)
        save_sample_predictions(sar_batch, color_batch, predictions, f'final_{i+1}')

    plt.legend()
    plt.title('SSIM Metric')
   
    plt.savefig('training_history.png')
    plt.close()
