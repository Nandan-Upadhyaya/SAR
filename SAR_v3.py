import tensorflow as tf
import numpy as np
from keras import layers, applications
import os
from glob import glob
import matplotlib.pyplot as plt
from datetime import datetime
import time
from keras.applications import efficientnet
import psutil

# Constants
BUFFER_SIZE = 400
BATCH_SIZE = 4  # Fixed at 4
IMG_WIDTH = 128  # Changed from 256 to 128
IMG_HEIGHT = 128  # Changed from 256 to 128
LAMBDA = 10
TERRAIN_TYPES = ['urban', 'grassland', 'agri', 'barrenland']
STYLE_WEIGHT = 1.0
CYCLE_WEIGHT = 10.0
COLOR_WEIGHT = 5.0

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


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

# Initialize feature extractor
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
        self.fid_tracker = tf.keras.metrics.Mean(name='fid')
        self.is_tracker = tf.keras.metrics.Mean(name='inception_score')
        self.lpips_tracker = tf.keras.metrics.Mean(name='lpips')
        
    def reset_states(self):
        self.gen_loss_tracker.reset_state()
        self.disc_loss_tracker.reset_state()
        self.psnr_metric.reset_state()
        self.ssim_metric.reset_state()

def load_and_preprocess_image(image_path):
    """Load and preprocess image with error handling"""
    try:
        # Ensure image_path is string
        if isinstance(image_path, tf.Tensor):
            image_path = image_path.numpy().decode('utf-8')
            
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
        img = (img / 127.5) - 1
        return img
    except Exception as e:
        print(f"Error loading image {image_path}: {str(e)}")
        # Return a black image in case of error
        return tf.zeros([IMG_HEIGHT, IMG_WIDTH, 3])

def create_dataset():
    """Create dataset with proper string handling for file paths"""
    datasets = []
    for terrain in TERRAIN_TYPES:
        sar_path = f'/kaggle/input/sentinel12-image-pairs-segregated-by-terrain/v_2/{terrain}/s1/*'
        color_path = f'/kaggle/input/sentinel12-image-pairs-segregated-by-terrain/v_2/{terrain}/s2/*'
        
        sar_dataset = tf.data.Dataset.list_files(sar_path, shuffle=False)
        color_dataset = tf.data.Dataset.list_files(color_path, shuffle=False)
        
        sar_dataset = sar_dataset.map(
            lambda x: tf.py_function(load_and_preprocess_image, [x], tf.float32),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        color_dataset = color_dataset.map(
            lambda x: tf.py_function(load_and_preprocess_image, [x], tf.float32),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        terrain_labels = tf.data.Dataset.from_tensors(
            tf.one_hot(TERRAIN_TYPES.index(terrain), len(TERRAIN_TYPES))
        ).repeat()
        
        dataset = tf.data.Dataset.zip((sar_dataset, color_dataset, terrain_labels))
        datasets.append(dataset)
    
    return tf.data.Dataset.from_tensor_slices(datasets).interleave(
        lambda x: x,
        cycle_length=4,
        num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

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
    for _ in range(3):
        x = MemoryEfficientResBlock(512)([x, terrain_input])

    # Upsampling back to 128x128
    for skip, filters in zip(reversed(skip_connections), reversed(filter_sizes)):
        x = layers.Conv2DTranspose(filters, 4, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skip])
        x = InstanceNormalization()(x)
        x = layers.Activation('silu')(x)

    # Final output at 128x128
    outputs = layers.Conv2D(3, 3, padding='same', activation='tanh')(x)
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
    terrain_spatial = TerrainSpatialLayer(height=IMG_HEIGHT, width=IMG_WIDTH)(terrain_input)  # changed code
    
    # Combine inputs
    x = layers.Concatenate()([input_image, target_image, terrain_spatial])
    
    # Rest of the discriminator architecture remains the same
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
    """Generator loss function with fp32 for loss calculations"""
    # Convert inputs to fp32 for loss calculation
    generated_images_f32 = tf.cast(generated_images, tf.float32)
    target_images_f32 = tf.cast(target_images, tf.float32)
    
    # GAN loss in fp32
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    gan_loss = tf.reduce_mean([
        loss_fn(
            tf.ones_like(tf.cast(output, tf.float32)),
            tf.cast(output, tf.float32)
        )
        for output in disc_generated_output
    ])
    
    # L1 loss in fp32
    l1_loss = tf.reduce_mean(tf.abs(target_images_f32 - generated_images_f32))
    
    # Perceptual loss in fp32
    perceptual_loss = compute_perceptual_loss(target_images_f32, generated_images_f32)
    perceptual_loss = tf.reduce_mean(perceptual_loss)
    
    # Metrics in fp32
    psnr = tf.reduce_mean(tf.image.psnr(target_images_f32, generated_images_f32, max_val=2.0))
    ssim = tf.reduce_mean(tf.image.ssim(target_images_f32, generated_images_f32, max_val=2.0))
    
    # Combine losses and convert back to fp16
    lambda_val = tf.cast(LAMBDA, tf.float32)
    gan_loss_f32 = tf.cast(gan_loss, tf.float32)       # changed code
    l1_loss_f32 = tf.cast(l1_loss, tf.float32)         # changed code
    perceptual_loss_f32 = tf.cast(perceptual_loss, tf.float32)  # changed code

    total_loss_f32 = gan_loss_f32 + (lambda_val * l1_loss_f32) \
                     + (lambda_val * 0.1 * perceptual_loss_f32)  # changed code

    return (tf.cast(total_loss_f32, tf.float16),
            tf.cast(gan_loss_f32, tf.float16),
            tf.cast(l1_loss_f32, tf.float16),
            tf.cast(perceptual_loss_f32, tf.float16),
            tf.cast(psnr, tf.float16),
            tf.cast(ssim, tf.float16))

def discriminator_loss(disc_real_output, disc_generated_output):
    """Discriminator loss function with fp32 for loss calculation"""
    # Convert inputs to fp32 for loss calculation
    loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    
    real_loss = tf.reduce_mean([
        loss_fn(
            tf.ones_like(tf.cast(output, tf.float32)),
            tf.cast(output, tf.float32)
        )
        for output in disc_real_output
    ])
    
    generated_loss = tf.reduce_mean([
        loss_fn(
            tf.zeros_like(tf.cast(output, tf.float32)),
            tf.cast(output, tf.float32)
        )
        for output in disc_generated_output
    ])
    
    # Convert back to fp16 for mixing with other losses
    total_loss = tf.cast(real_loss + generated_loss, tf.float16)
    return total_loss

def train_step(sar_images, color_images, terrain_labels, generator, discriminator, 
               generator_optimizer, discriminator_optimizer, metrics_tracker):
    """Modified training step with explicit tensor shapes"""
    
    # Ensure input tensors have correct shape and type
    sar_images = tf.convert_to_tensor(sar_images, dtype=tf.float32)
    color_images = tf.convert_to_tensor(color_images, dtype=tf.float32)
    terrain_labels = tf.convert_to_tensor(terrain_labels, dtype=tf.float32)
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generate images
        generated_images = generator([sar_images, terrain_labels], training=True)
        
        # Discriminator forward passes
        disc_real_output = discriminator([sar_images, color_images, terrain_labels], training=True)
        disc_generated_output = discriminator([sar_images, generated_images, terrain_labels], training=True)
        
        # Calculate losses
        gen_total_loss, gan_loss, l1_loss, perceptual_loss, psnr, ssim = generator_loss(
            disc_generated_output, generated_images, color_images
        )
        
        grad_penalty = compute_gradient_penalty(discriminator, color_images, generated_images, terrain_labels)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output) + \
                    tf.cast(10.0, tf.float16) * grad_penalty
    
    # Compute gradients
    gen_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Apply gradients
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    # Update metrics
    metrics = {
        'gen_total_loss': gen_total_loss,
        'disc_loss': disc_loss,
        'psnr': psnr,
        'ssim': ssim
    }
    
    return metrics

def train(dataset, epochs):
    """Modified training function with proper distribution strategy"""
    
    # Initialize distribution strategy
    strategy = tf.distribute.get_strategy()
    
    with strategy.scope():
        # Initialize models
        generator = build_terrain_aware_generator()
        discriminator = build_terrain_aware_discriminator()
        
        # Initialize optimizers with correct learning rate
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
        metrics_tracker = MetricsTracker()
        
        # Enable mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        # Create distributed dataset
        dist_dataset = strategy.experimental_distribute_dataset(dataset)
        
        # Training history
        history = {
            'gen_loss': [], 'disc_loss': [],
            'psnr': [], 'ssim': []
        }

        @tf.function
        def distributed_train_step(inputs):
            # Unpack inputs and ensure they have correct shape
            sar_images, color_images, terrain_labels = inputs
            
            # Run distributed training step
            per_replica_losses = strategy.run(
                train_step,
                args=(
                    sar_images, color_images, terrain_labels,
                    generator, discriminator,
                    generator_optimizer, discriminator_optimizer,
                    metrics_tracker
                )
            )
            
            # Reduce metrics across replicas
            return {
                k: strategy.reduce(tf.distribute.ReduceOp.MEAN, v, axis=None)
                for k, v in per_replica_losses.items()
            }

        # Training loop
        for epoch in range(epochs):
            start = time.time()
            metrics_tracker.reset_states()
            
            for step, inputs in enumerate(dist_dataset):
                metrics = distributed_train_step(inputs)
                
                # Update history
                if step % 10 == 0:
                    for k, v in metrics.items():
                        if k in history:
                            history[k].append(float(v))
                    
                    print(f"\rEpoch {epoch + 1}/{epochs} - Step {step} - {format_metrics(metrics)}", end='')
            
            print(f"\nTime taken for epoch {epoch + 1}: {time.time() - start:.2f} sec")
            
            # Save sample predictions every 10 epochs
            if (epoch + 1) % 10 == 0:
                for test_batch in dataset.take(1):
                    sar_test, color_test, terrain_test = test_batch
                    predictions = generator([sar_test, terrain_test], training=False)
                    save_sample_predictions(sar_test, color_test, predictions, epoch + 1)

        # Return the trained models and training history
        return generator, discriminator, history

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
        sar_path = f'Dataset/{terrain}/SAR/*'
        color_path = f'Dataset/{terrain}/Color/*'
        
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

if __name__ == '__main__':
    dataset = create_dataset()
    generator, discriminator, history = train(dataset, epochs=200)  # Increased epochs for better training
    
    # Plot training history
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['gen_loss'], label='Generator Loss')
    plt.plot(history['disc_loss'], label='Discriminator Loss')
    plt.legend()
    plt.title('Training Losses')
    
    plt.subplot(1, 3, 2)
    plt.plot(history['psnr'], label='PSNR')
    plt.legend()
    plt.title('PSNR Metric')
    
    plt.subplot(1, 3, 3)
    plt.plot(history['ssim'], label='SSIM')
    plt.legend()
    plt.title('SSIM Metric')
    
    plt.savefig('training_history.png')
    plt.close()

    plt.plot(history['ssim'], label='SSIM')
    # Generate final comparisons
    print("Generating final comparison images...")
    test_samples = dataset.take(5)  # Take 5 samples
    for i, (sar_batch, color_batch, terrain_batch) in enumerate(test_samples):
        predictions = generator([sar_batch, terrain_batch], training=False)
        save_sample_predictions(sar_batch, color_batch, predictions, f'final_{i+1}')

    plt.legend()
    plt.title('SSIM Metric')
    
    plt.savefig('training_history.png')
    plt.close()

