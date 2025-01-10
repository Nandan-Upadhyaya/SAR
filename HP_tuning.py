import numpy as np
from scipy import linalg
import tensorflow as tf
import keras_tuner as kt
from SAR_B3 import Generator, Discriminator, DataLoader
import os
import logging
from datetime import datetime
import tensorflow_probability as tfp
import random

# Set random seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    tf.random.set_seed(seed)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
else:
    raise RuntimeError("No GPUs found. Ensure a compatible GPU is available.")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'hp_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MetricTargets:
    """Target values and acceptable ranges for each metric"""
    TARGETS = {
        'psnr': {'target': 30.0, 'min': 25.0, 'weight': 0.15},
        'ssim': {'target': 0.8, 'min': 0.7, 'weight': 0.15},
        'gen_loss': {'target': 3.0, 'range': (1.0, 3.0), 'weight': 0.1},
        'disc_loss': {'target': 1.5, 'range': (0.3, 3.0), 'weight': 0.1},
        'cycle_loss': {'target': 0.3, 'max': 1.0, 'weight': 0.15},
        'cls_loss': {'target': 0.2, 'max': 0.5, 'weight': 0.1},
        'fid': {'target': 10.0, 'max': 20.0, 'weight': 0.1},
        'inception_score': {'target': 8.0, 'min': 6.0, 'weight': 0.05},
        'l1_loss': {'target': 0.005, 'max': 0.01, 'weight': 0.05},
        'l2_loss': {'target': 0.0005, 'max': 0.001, 'weight': 0.025},
        'lpips': {'target': 0.05, 'max': 0.1, 'weight': 0.025}
    }

class GPUConfig:
    @staticmethod
    def configure() -> None:
        """Deprecated: Use configure_gpu() instead."""
        logger.warning("GPUConfig.configure() is deprecated. Using global GPU configuration.")
        pass

# Move all GPU configuration to a single place at the start
def configure_gpu():
    """Configure GPU settings before any other TF operations."""
    try:
        # Get physical GPUs
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if physical_devices:
            logger.info(f"Found {len(physical_devices)} GPU(s)")
            
            # First enable memory growth for all GPUs
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
                logger.info(f"Enabled memory growth for GPU: {device}")
            
            # Then set memory limit and make visible
            tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')
            tf.config.set_logical_device_configuration(
                physical_devices[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=5449)]
            )
            
            logger.info(f"TensorFlow GPU available: {tf.test.is_gpu_available()}")
            logger.info(f"TensorFlow built with CUDA: {tf.test.is_built_with_cuda()}")
            
            # Set compute dtype
            tf.keras.backend.set_floatx('float32')
            
        else:
            logger.warning("No GPUs found. Running on CPU")
    except RuntimeError as e:
        logger.error(f"GPU configuration error: {str(e)}")
        raise

class TunableCycleGAN(tf.keras.Model):
    def __init__(self, hp, image_size=128):  # Fix method name from _init_ to __init__
        super(TunableCycleGAN, self).__init__()  # Fix method name from _init_ to __init__
        
        # Use float32 for model dtype to avoid numerical instability
        self.model_dtype = tf.float32
        self.image_size = image_size
        
        # Create dummy input to build models
        dummy_input = tf.keras.Input(shape=(image_size, image_size, 3))
        
        # Initialize and build models
        self.G1 = Generator(image_size)
        self.G2 = Generator(image_size)
        self.D1 = Discriminator(len(os.listdir('./Dataset')))
        self.D2 = Discriminator(len(os.listdir('./Dataset')))
        
        # Build models with dummy inputs
        dummy_tensor = tf.zeros((1, image_size, image_size, 3), dtype=self.model_dtype)
        self.G1(dummy_tensor)
        self.G2(dummy_tensor)
        self.D1(dummy_tensor)
        self.D2(dummy_tensor)
        
        # Hyperparameters
        self.lambda_cyc = hp.Float('lambda_cyc', min_value=5.0, max_value=10.0, step=1.0)
        self.lambda_cls = hp.Float('lambda_cls', min_value=0.3, max_value=0.8, step=0.1)
        self.learning_rate = hp.Float('learning_rate', min_value=1e-7, max_value=1e-4, sampling='log')
        
        # Initialize optimizers after models are built
        self.gen_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(self.learning_rate * 0.5, beta_1=0.5)
        
        # Initialize metrics
        self.gen_loss_metric = tf.keras.metrics.Mean(name='gen_loss')
        self.cycle_loss_metric = tf.keras.metrics.Mean(name='cycle_loss')
        self.psnr_metric = tf.keras.metrics.Mean(name='psnr')
        self.ssim_metric = tf.keras.metrics.Mean(name='ssim')
        self.composite_score_metric = tf.keras.metrics.Mean(name='composite_score')
        self.cls_loss_metric = tf.keras.metrics.Mean(name='cls_loss')
        self.disc_loss_metric = tf.keras.metrics.Mean(name='disc_loss')
        self.fid_metric = tf.keras.metrics.Mean(name='fid')
        self.inception_score_metric = tf.keras.metrics.Mean(name='inception_score')
        self.l1_loss_metric = tf.keras.metrics.Mean(name='l1_loss')
        self.l2_loss_metric = tf.keras.metrics.Mean(name='l2_loss')
        self.lpips_metric = tf.keras.metrics.Mean(name='lpips')
        
        # Add FID model
        self.inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg')
        self.vgg = tf.keras.applications.VGG16(include_top=False)
        
        # Add a separate dtype for FID calculations
        self.fid_dtype = tf.float32
        self.model_dtype = tf.float16

    def compile(self, optimizer, loss, loss_weights=None, metrics=None, **kwargs):
        super().compile(**kwargs)
        
        # Store optimizers
        if isinstance(optimizer, dict):
            self.gen_optimizer = optimizer['generator']
            self.disc_optimizer = optimizer['discriminator']
        else:
            self.gen_optimizer = optimizer
            self.disc_optimizer = optimizer
            
        self._loss_dict = loss
        self._loss_weights = loss_weights or {}

        # Store optimizers
        if isinstance(optimizer, dict):
            self.gen_optimizer = optimizer['generator']
            self.disc_optimizer = optimizer['discriminator']
        else:
            self.gen_optimizer = optimizer
            self.disc_optimizer = optimizer
            
        self._loss_dict = loss
        self._loss_weights = loss_weights or {}
        
        # Create trainable variable lists
        self.gen_trainable_vars = (
            self.G1.trainable_variables + 
            self.G2.trainable_variables
        )
        self.disc_trainable_vars = (
            self.D1.trainable_variables + 
            self.D2.trainable_variables
        )

    def call(self, inputs, training=True):
        """Forward pass of the model."""
        if isinstance(inputs, tuple):
            real_color, real_sar = inputs
        else:
            # Handle single input case
            real_color = inputs
            real_sar = inputs
            
        # Generate fake images
        fake_sar = self.G1(real_color, training=training)
        fake_color = self.G2(real_sar, training=training)
        
        # Return fake images
        return fake_sar, fake_color
    
    @tf.function
    def train_step(self, data):
        # Remove explicit float16 casting
        inputs, labels = data
        real_color, real_sar = inputs
        
        # Cast to model dtype (float32)
        real_color = tf.cast(real_color, self.model_dtype)
        real_sar = tf.cast(real_sar, self.model_dtype)
        labels = tf.cast(labels, self.model_dtype)
        
        with tf.GradientTape(persistent=True) as tape:
            # Forward pass with explicit casting
            fake_sar = tf.cast(self.G1(real_color, training=True), self.model_dtype)
            cycled_color = tf.cast(self.G2(fake_sar, training=True), self.model_dtype)
            fake_color = tf.cast(self.G2(real_sar, training=True), self.model_dtype)
            cycled_sar = tf.cast(self.G1(fake_color, training=True), self.model_dtype)
            
            # Identity mapping with casting
            same_color = tf.cast(self.G2(real_color, training=True), self.model_dtype)
            same_sar = tf.cast(self.G1(real_sar, training=True), self.model_dtype)
            
            # Generator adversarial outputs
            disc_real_color_adv, disc_real_color_cls = [
                tf.cast(x, self.model_dtype) for x in self.D1(real_color, training=True)
            ]
            disc_fake_color_adv, disc_fake_color_cls = [
                tf.cast(x, self.model_dtype) for x in self.D1(fake_color, training=True)
            ]
            disc_real_sar_adv, disc_real_sar_cls = [
                tf.cast(x, self.model_dtype) for x in self.D2(real_sar, training=True)
            ]
            disc_fake_sar_adv, disc_fake_sar_cls = [
                tf.cast(x, self.model_dtype) for x in self.D2(fake_sar, training=True)
            ]
            
            # Generator losses
            gen_loss_G1 = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(disc_fake_sar_adv), disc_fake_sar_adv))
            gen_loss_G2 = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(disc_fake_color_adv), disc_fake_color_adv))
            
            # Classification losses
            cls_loss_G1 = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
                labels, disc_fake_sar_cls))
            cls_loss_G2 = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
                labels, disc_fake_color_cls))
            
            # Cycle consistency and identity losses
            cycle_loss = (tf.reduce_mean(tf.abs(real_color - cycled_color)) +
                         tf.reduce_mean(tf.abs(real_sar - cycled_sar))) * self.lambda_cyc
            
            identity_loss = (tf.reduce_mean(tf.abs(real_color - same_color)) +
                           tf.reduce_mean(tf.abs(real_sar - same_sar))) * 0.5 * self.lambda_cyc
            
            # Total generator loss
            total_gen_loss = (
                gen_loss_G1 + gen_loss_G2 +
                cycle_loss + identity_loss +
                (cls_loss_G1 + cls_loss_G2) * self.lambda_cls
            )
            
            # Discriminator losses
            disc_loss_real_color = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(disc_real_color_adv), disc_real_color_adv))
            disc_loss_fake_color = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.zeros_like(disc_fake_color_adv), disc_fake_color_adv))
            disc_loss_real_sar = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.ones_like(disc_real_sar_adv), disc_real_sar_adv))
            disc_loss_fake_sar = tf.reduce_mean(tf.keras.losses.binary_crossentropy(
                tf.zeros_like(disc_fake_sar_adv), disc_fake_sar_adv))
            
            total_disc_loss = (
                disc_loss_real_color + disc_loss_fake_color +
                disc_loss_real_sar + disc_loss_fake_sar
            ) * 0.5

            # Calculate losses using compiled loss functions
            losses = {
                'gen_loss': self._loss_dict['gen_loss'](
                    tf.ones_like(disc_fake_sar_adv), disc_fake_sar_adv
                ) + self._loss_dict['gen_loss'](
                    tf.ones_like(disc_fake_color_adv), disc_fake_color_adv
                ),
                'disc_loss': self._loss_dict['disc_loss'](
                    tf.ones_like(disc_real_color_adv), disc_real_color_adv
                ) + self._loss_dict['disc_loss'](
                    tf.zeros_like(disc_fake_color_adv), disc_fake_color_adv
                ),
                'cycle_loss': self._loss_dict['cycle_loss'](real_color, cycled_color) + 
                             self._loss_dict['cycle_loss'](real_sar, cycled_sar),
                'cls_loss': self._loss_dict['cls_loss'](labels, disc_fake_sar_cls) + 
                           self._loss_dict['cls_loss'](labels, disc_fake_color_cls)
            }
            
            # Apply loss weights
            weighted_losses = {
                k: v * self._loss_weights.get(k, 1.0) for k, v in losses.items()
            }
            
            total_gen_loss = weighted_losses['gen_loss'] + weighted_losses['cycle_loss'] + weighted_losses['cls_loss']
            total_disc_loss = weighted_losses['disc_loss']

            # Calculate discriminator losses with explicit connections
            disc_real_loss = tf.reduce_mean([
                self._loss_dict['disc_adv_loss'](
                    tf.ones_like(disc_real_color_adv), 
                    disc_real_color_adv
                ),
                self._loss_dict['disc_adv_loss'](
                    tf.ones_like(disc_real_sar_adv),
                    disc_real_sar_adv
                )
            ])
            
            disc_fake_loss = tf.reduce_mean([
                self._loss_dict['disc_adv_loss'](
                    tf.zeros_like(disc_fake_color_adv),
                    disc_fake_color_adv
                ),
                self._loss_dict['disc_adv_loss'](
                    tf.zeros_like(disc_fake_sar_adv),
                    disc_fake_sar_adv
                )
            ])
            
            disc_cls_loss = tf.reduce_mean([
                self._loss_dict['disc_cls_loss'](labels, disc_real_color_cls),
                self._loss_dict['disc_cls_loss'](labels, disc_real_sar_cls)
            ])
            
            total_disc_loss = (
                self._loss_weights['disc_adv_loss'] * (disc_real_loss + disc_fake_loss) +
                self._loss_weights['disc_cls_loss'] * disc_cls_loss
            )
            
            # Calculate generator losses with weights
            total_gen_loss = (
                self._loss_weights['gen_loss'] * (gen_loss_G1 + gen_loss_G2) +
                self._loss_weights['cycle_loss'] * cycle_loss +
                self._loss_weights['cls_loss'] * (cls_loss_G1 + cls_loss_G2)
            )

        # Calculate gradients
        gen_gradients = tape.gradient(
            total_gen_loss, 
            self.G1.trainable_variables + self.G2.trainable_variables
        )
        disc_gradients = tape.gradient(
            total_disc_loss,
            self.D1.trainable_variables + self.D2.trainable_variables
        )

        # Apply gradients if they exist
        if any(g is not None for g in gen_gradients):
            self.gen_optimizer.apply_gradients(
                zip(gen_gradients, self.G1.trainable_variables + self.G2.trainable_variables)
            )
        if any(g is not None for g in disc_gradients):
            self.disc_optimizer.apply_gradients(
                zip(disc_gradients, self.D1.trainable_variables + self.D2.trainable_variables)
            )

        # Additional metrics
        inception_real = self._get_inception_features(real_color)
        inception_fake = self._get_inception_features(fake_color)
        # FID calculation in float32, result cast to float16
        fid_score = self._calculate_fid(inception_real, inception_fake)
        fid_score = tf.cast(fid_score, self.model_dtype)
        inception_score = self._calculate_inception_score(fake_color)
        lpips_score = self._calculate_lpips(real_color, cycled_color)

        metrics = {
            'gen_loss': total_gen_loss,
            'disc_loss': total_disc_loss,
            'cycle_loss': cycle_loss,
            'cls_loss': cls_loss_G1 + cls_loss_G2,
            'psnr': tf.reduce_mean(tf.image.psnr(real_color, cycled_color, max_val=2.0)),
            'ssim': tf.reduce_mean(tf.image.ssim(real_color, cycled_color, max_val=2.0)),
            'fid': fid_score,
            'inception_score': inception_score,
            'lpips': lpips_score,
            'l1_loss': tf.reduce_mean(tf.abs(real_color - cycled_color)),
            'l2_loss': tf.reduce_mean(tf.square(real_color - cycled_color))
        }

        # Update metrics
        for name, value in metrics.items():
            getattr(self, f'{name}_metric').update_state(value)

        # Before calculating composite score, cast all metrics to float16
        metrics = {k: tf.cast(v, self.model_dtype) for k, v in metrics.items()}
        composite_score = self._calculate_composite_score(metrics)
        self.composite_score_metric.update_state(composite_score)
        metrics['composite_score'] = composite_score

        return metrics

    @tf.function
    def test_step(self, data):
        """Test step implementation"""
        inputs, labels = data
        real_color, real_sar = inputs
        
        # Cast inputs to float16
        real_color = tf.cast(real_color, self.model_dtype)
        real_sar = tf.cast(real_sar, self.model_dtype)
        
        # Forward pass
        fake_sar = self.G1(real_color, training=False)
        cycled_color = self.G2(fake_sar, training=False)
        fake_color = self.G2(real_sar, training=False)
        cycled_sar = self.G1(fake_color, training=False)
        
        # Calculate metrics
        cycle_loss = tf.reduce_mean(tf.abs(real_color - cycled_color)) + \
                     tf.reduce_mean(tf.abs(real_sar - cycled_sar))
        psnr = tf.reduce_mean(tf.image.psnr(real_color, cycled_color, max_val=2.0))
        ssim = tf.reduce_mean(tf.image.ssim(real_color, cycled_color, max_val=2.0))
        l1_loss = tf.reduce_mean(tf.abs(real_color - cycled_color))
        l2_loss = tf.reduce_mean(tf.square(real_color - cycled_color))
        
        metrics = {
            'gen_loss': 0.0,
            'cycle_loss': cycle_loss,
            'psnr': psnr,
            'ssim': ssim,
            'cls_loss': 0.0,
            'disc_loss': 0.0,
            'fid': 0.0,
            'inception_score': 0.0,
            'l1_loss': l1_loss,
            'l2_loss': l2_loss,
            'lpips': 0.0
        }
        
        # Calculate and add composite score
        # Before calculating composite score, cast all metrics to float16
        metrics = {k: tf.cast(v, self.model_dtype) for k, v in metrics.items()}
        composite_score = self._calculate_composite_score(metrics)
        metrics['composite_score'] = composite_score
        
        # Update all metrics
        for name, value in metrics.items():
            getattr(self, f'{name}_metric').update_state(value)
            
        return metrics

    def _calculate_composite_score(self, metrics):
        """Calculate composite score for the model using TF operations with consistent types."""
        score = tf.constant(0.0, dtype=self.model_dtype)
        for metric, values in MetricTargets.TARGETS.items():
            if metric not in metrics:
                continue
                
            # Cast all values to float16
            value = tf.cast(metrics[metric], self.model_dtype)
            target = tf.cast(values['target'], self.model_dtype)
            weight = tf.cast(values['weight'], self.model_dtype)
            
            if 'range' in values:
                min_val = tf.cast(values['range'][0], self.model_dtype)
                max_val = tf.cast(values['range'][1], self.model_dtype)
                
                # Use tf.cond with consistent types
                in_range = tf.logical_and(
                    tf.greater_equal(value, min_val),
                    tf.less_equal(value, max_val)
                )
                range_score = tf.cond(
                    in_range,
                    lambda: weight,
                    lambda: weight * (1.0 - tf.minimum(tf.abs(value - target) / target, 1.0))
                )
                score += tf.cast(range_score, self.model_dtype)
                
            elif 'min' in values:
                min_val = tf.cast(values['min'], self.model_dtype)
                meets_min = tf.greater_equal(value, min_val)
                min_score = tf.cond(
                    meets_min,
                    lambda: weight * tf.where(
                        tf.less(value, target),
                        value / target,
                        target / value
                    ),
                    lambda: tf.constant(0.0, dtype=self.model_dtype)
                )
                score += tf.cast(min_score, self.model_dtype)
                
            elif 'max' in values:
                max_val = tf.cast(values['max'], self.model_dtype)
                meets_max = tf.less_equal(value, max_val)
                max_score = tf.cond(
                    meets_max,
                    lambda: weight * (1.0 - value / max_val),
                    lambda: tf.constant(0.0, dtype=self.model_dtype)
                )
                score += tf.cast(max_score, self.model_dtype)
                    
        return score

    def _calculate_fid(self, real_features, fake_features):
     """Calculate FID score on GPU with improved numerical stability."""
     try:
        # Keep calculations on GPU, remove explicit CPU placement
        real_features = tf.cast(real_features, tf.float32)
        fake_features = tf.cast(fake_features, tf.float32)
        
        mu_real = tf.reduce_mean(real_features, axis=0)
        mu_fake = tf.reduce_mean(fake_features, axis=0)
        mu_diff_squared = tf.reduce_sum(tf.square(mu_real - mu_fake))
        
        n_real = tf.cast(tf.shape(real_features)[0], tf.float32)
        n_fake = tf.cast(tf.shape(fake_features)[0], tf.float32)
        
        real_centered = real_features - mu_real
        fake_centered = fake_features - mu_fake
        
        epsilon = 1e-6
        cov_real = (tf.matmul(real_centered, real_centered, transpose_a=True) / 
                   (n_real - 1) + tf.eye(tf.shape(mu_real)[0]) * epsilon)
        cov_fake = (tf.matmul(fake_centered, fake_centered, transpose_a=True) / 
                   (n_fake - 1) + tf.eye(tf.shape(mu_fake)[0]) * epsilon)
        
        sqrt_cov_real = tf.linalg.sqrtm(cov_real)
        sqrt_term = tf.matmul(tf.matmul(sqrt_cov_real, cov_fake), sqrt_cov_real)
        sqrt_term = (sqrt_term + tf.transpose(sqrt_term)) / 2.0
        
        trace_term = tf.linalg.trace(cov_real + cov_fake - 2.0 * tf.linalg.sqrtm(sqrt_term))
        fid_score = mu_diff_squared + trace_term
        
        return tf.clip_by_value(fid_score, 0.0, 1000.0)
            
     except Exception as e:
        logger.error(f"FID calculation error: {str(e)}")
        return tf.constant(100.0, dtype=tf.float32)

    def _finalize_fid(self, ssdiff, sigma1, sigma2):
        """Finalize FID calculation with improved numerical stability."""
        try:
            if ssdiff is None:
                return 100.0  # Return a high but finite score
                
            # Convert to numpy for scipy operations
            sigma1_np = sigma1.numpy()
            sigma2_np = sigma2.numpy()
            
            # Calculate sqrt using eigendecomposition for stability
            eigvals, eigvecs = linalg.eigh(sigma1_np)
            eigvals = np.maximum(eigvals, 0)
            sqrt_sigma1 = eigvecs.dot(np.sqrt(np.diag(eigvals))).dot(eigvecs.T)
            
            # Calculate the product matrix
            prod_matrix = sqrt_sigma1.dot(sigma2_np).dot(sqrt_sigma1)
            
            # Ensure matrix is Hermitian
            prod_matrix = (prod_matrix + prod_matrix.T.conj()) / 2
            
            # Calculate trace term
            eigvals_prod = np.maximum(linalg.eigvalsh(prod_matrix), 0)
            trace_term = -2 * np.sqrt(eigvals_prod).sum()
            
            # Calculate final FID score
            fid = float(ssdiff + np.trace(sigma1_np) + np.trace(sigma2_np) + trace_term)
            
            # Clip to reasonable range
            return np.clip(fid, 0.0, 1000.0)
            
        except Exception as e:
            logger.error(f"FID finalization error: {str(e)}")
            return 100.0  # Return a high but finite score

    def _get_inception_features(self, images):
        # Preprocess images for inception
        images = tf.image.resize(images, (299, 299))
        images = tf.keras.applications.inception_v3.preprocess_input(images)
        return self.inception_model(images, training=False)

    def _calculate_inception_score(self, images):
        # Get inception predictions
        preds = self._get_inception_features(images)
        preds = tf.nn.softmax(preds)
        
        # Calculate KL divergence
        p_y = tf.reduce_mean(preds, axis=0)
        kl_div = preds * (tf.math.log(preds + 1e-16) - tf.math.log(p_y + 1e-16))
        kl_div = tf.reduce_sum(kl_div, axis=1)
        
        return tf.exp(tf.reduce_mean(kl_div))

    def _calculate_lpips(self, img1, img2):
        # Extract VGG features
        feat1 = self.vgg(img1)
        feat2 = self.vgg(img2)
        return tf.reduce_mean(tf.square(feat1 - feat2))

# Add this function before the CycleGANTuner class
def build_model(hp):
    """Build tunable CycleGAN model."""
    model = TunableCycleGAN(hp)
    
    # Define loss functions for each component
    loss_dict = {
        'gen_loss': tf.keras.losses.BinaryCrossentropy(from_logits=True),
        'disc_loss': tf.keras.losses.BinaryCrossentropy(from_logits=True),
        'cycle_loss': tf.keras.losses.MeanAbsoluteError(),
        'cls_loss': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        # Add separate losses for discriminator components
        'disc_adv_loss': tf.keras.losses.BinaryCrossentropy(from_logits=True),
        'disc_cls_loss': tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    }
    
    # Define loss weights with hyperparameters
    loss_weights = {
        'gen_loss': 1.0,
        'disc_loss': hp.Float('disc_weight', 0.3, 0.7, step=0.1),
        'cycle_loss': hp.Float('lambda_cyc', 5.0, 10.0, step=1.0),
        'cls_loss': hp.Float('lambda_cls', 0.3, 0.8, step=0.1),
        'disc_adv_loss': 0.5,
        'disc_cls_loss': 0.5
    }
    
    # Compile model with all losses
    model.compile(
        optimizer={
            'generator': tf.keras.optimizers.Adam(
                hp.Float('gen_lr', 1e-7, 1e-4, sampling='log')
            ),
            'discriminator': tf.keras.optimizers.Adam(
                hp.Float('disc_lr', 1e-7, 1e-4, sampling='log')
            )
        },
        loss=loss_dict,
        loss_weights=loss_weights,
        run_eagerly=False  # Set to True for debugging
    )
    return model

# Modify the CycleGANTuner class to include strategy in run_trial
class CycleGANTuner(kt.BayesianOptimization):
    def __init__(self, max_trials=13, **kwargs):
        super().__init__(
            hypermodel=build_model,
            objective=kt.Objective('composite_score', direction='max'),  # Specify direction
            max_trials=max_trials,
            num_initial_points=3,
            alpha=0.0001,
            beta=2.6,
            **kwargs
        )
        self.metric_history = {}
        self.early_stop_patience = 3
        self.min_delta = 0.002

    def _calculate_composite_score(self, metrics):
        """Calculate weighted score considering all metrics"""
        score = 0.0
        for metric, values in MetricTargets.TARGETS.items():
            if metric not in metrics:
                continue
                
            value = metrics[metric]
            target = values['target']
            weight = values['weight']
            
            if 'range' in values:
                min_val, max_val = values['range']
                if min_val <= value <= max_val:
                    score += weight
                else:
                    score += weight * (1 - min(abs(value - target) / target, 1.0))
            elif 'min' in values:
                if value >= values['min']:
                    score += weight * (value / target if value < target else target / value)
            elif 'max' in values:
                if value <= values['max']:
                    score += weight * (1 - value / values['max'])
                    
        return score

# Fix run_trial method indentation and implementation
def run_trial(self, trial, train_ds, val_ds, *args, **kwargs):
    """Run a single trial with proper GPU execution."""
    try:
        # Create strategy for GPU execution
        strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
        
        # Wrap entire trial execution in strategy scope
        with strategy.scope():
            model = self.hypermodel.build(trial.hyperparameters)
            best_metrics = None
            patience_counter = 0
            
            logger.info(f"Starting trial {trial.trial_id}")
            
            # Distribute datasets
            train_ds = strategy.experimental_distribute_dataset(train_ds)
            val_ds = strategy.experimental_distribute_dataset(val_ds)
            
            for epoch in range(13):
                # Training phase
                train_metrics = []
                for batch in train_ds:
                    # Run step in strategy scope
                    per_replica_metrics = strategy.run(model.train_step, args=(batch,))
                    # Reduce metrics across replicas
                    metrics = {k: strategy.reduce(
                        tf.distribute.ReduceOp.MEAN, 
                        v, 
                        axis=None
                    ) for k, v in per_replica_metrics.items()}
                    train_metrics.append(metrics)
                
                # Validation phase
                val_metrics = []
                for batch in val_ds:
                    per_replica_metrics = strategy.run(model.test_step, args=(batch,))
                    metrics = {k: strategy.reduce(
                        tf.distribute.ReduceOp.MEAN, 
                        v, 
                        axis=None
                    ) for k, v in per_replica_metrics.items()}
                    val_metrics.append(metrics)
                
                # Average metrics on GPU
                def average_metrics(metrics_list):
                    avg = {}
                    for key in metrics_list[0].keys():
                        values = tf.stack([m[key] for m in metrics_list])
                        avg[key] = tf.reduce_mean(values)
                    return avg
                
                avg_train_metrics = average_metrics(train_metrics)
                avg_val_metrics = average_metrics(val_metrics)
                
                # Calculate composite score on GPU
                composite_score = strategy.run(
                    self._calculate_composite_score, 
                    args=(avg_val_metrics,)
                )
                composite_score = strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, 
                    composite_score, 
                    axis=None
                )
                
                # Early stopping check
                if best_metrics is None or composite_score > best_metrics['composite_score']:
                    best_metrics = avg_val_metrics
                    best_metrics['composite_score'] = composite_score
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.early_stop_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                    break
                
            return best_metrics
            
    except Exception as e:
        logger.error(f"Trial failed: {str(e)}")
        raise

def main():
    # Set seeds first
    set_seeds(42)
    
    # Configure GPU before any other TF operations
    configure_gpu()
    
    # Create a strategy for GPU usage
    strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
    
    with strategy.scope():
        # Load dataset with smaller batch size
        data_loader = DataLoader(
            dataset_dir='./Dataset',
            image_size=128,
            batch_size=8,
            validation_split=0.2
        )
        
        train_ds, val_ds = data_loader.load_data()
        
        

        # Configure dataset options
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        
        # Apply options to datasets
        train_ds = train_ds.with_options(options)
        val_ds = val_ds.with_options(options)
        
        tuner = CycleGANTuner(
            directory='hp_tuning',
            project_name='cyclegan_tuning',
            max_trials=13,  # Add max_trials argument
            overwrite=True
        )
        
        try:
            tuner.search(
                x=train_ds,
                validation_data=val_ds,
                epochs=13,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_composite_score',
                        mode='max',
                        patience=5,
                        min_delta=0.002
                    ),
                    tf.keras.callbacks.TensorBoard(
                        log_dir=f'logs/tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_composite_score',
                        mode='max',
                        factor=0.5,
                        patience=2
                    )
                ]
            )
            
            best_hp = tuner.get_best_hyperparameters()[0]
            logger.info(f"Best hyperparameters:\n{best_hp.values}")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            raise

# ...rest of the code remains the same...

if __name__ == "__main__":
    main()