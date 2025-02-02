import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import time
from typing import Tuple, List, Dict, Any
import json
from keras.applications.inception_v3 import preprocess_input
from scipy import linalg
from keras.layers import LayerNormalization

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)

class GPUConfig:
    @staticmethod
    def configure() -> None:
        """Configure GPU settings for optimal performance."""
        try:
            physical_devices = tf.config.experimental.list_physical_devices('GPU')
            if (physical_devices):
                print(f"Found {len(physical_devices)} GPU(s)")
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                    print(f"Enabled memory growth for GPU: {device}")
            else:
                logger.warning("No GPUs found. Running on CPU")
        except RuntimeError as e:
            logger.error(f"GPU configuration error: {str(e)}")
            raise

class DataLoader:
    def __init__(
        self, 
        dataset_dir: str,
        image_size: int = 256,
        batch_size: int = 16,  # As mentioned in paper Section II.D
        validation_split: float = 0.1  # 10% test samples as mentioned in paper
    ):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        
        # Get terrain types for mask vectors
        self.terrain_types = self._get_terrain_types()
        self.num_terrains = len(self.terrain_types)
        self.terrain_to_index = {terrain: idx for idx, terrain in enumerate(self.terrain_types)}
        print(f"Found terrain types: {self.terrain_types}")

    def _get_terrain_types(self) -> List[str]:
        """Get list of terrain types from dataset directory structure."""
        return sorted([d for d in os.listdir(self.dataset_dir) 
                      if os.path.isdir(os.path.join(self.dataset_dir, d))])

    def _create_mask_lookup_table(self):
        """Create a lookup table for terrain masks."""
        num_classes = self.num_terrains + 1
        masks = np.zeros((self.num_terrains, num_classes), dtype=np.float32)
        for i in range(self.num_terrains):
            masks[i, i + 1 ] = 1.0  # +1 to account for SAR domain at index 0
        return tf.convert_to_tensor(masks, dtype=tf.float32)

    @tf.function
    def _create_mask_vector(self, terrain_index):
        """Create one-hot encoded mask vector using lookup table."""
        masks = self._create_mask_lookup_table()
        return tf.gather(masks, terrain_index)
    
    @tf.function
    def _create_sar_mask(self):
        """Create SAR domain mask vector [1,0,...,0]."""
        mask = tf.zeros(self.num_terrains + 1, dtype=tf.float32)
        return tf.tensor_scatter_nd_update(mask, [[0]], [1.0])

    @tf.function
    def _expand_mask(self, mask):
        """Efficiently expand mask to image dimensions."""
        return tf.tile(
            tf.reshape(mask, [1, 1, -1]),
            [self.image_size, self.image_size, 1]
        )

    @tf.function
    def _process_image(self, file_path):
        """Process a single image."""
        img = tf.io.read_file(file_path)
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        img = tf.image.resize(img, [self.image_size, self.image_size])
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        return img
        
    def _create_dataset(
        self,
        sar_paths: List[str],
        optical_paths: List[str],
        sar_masks: np.ndarray,
        terrain_masks: np.ndarray,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """Create tf.data.Dataset from image paths and masks."""
         # Convert paths to absolute paths and normalize them
        sar_paths = [os.path.abspath(p) for p in sar_paths]
        optical_paths = [os.path.abspath(p) for p in optical_paths]
    
    # Reshape masks to [batch_size, H, W, C]
        sar_masks = np.array([mask.reshape(self.image_size, self.image_size, -1) for mask in sar_masks])
        terrain_masks = np.array([mask.reshape(self.image_size, self.image_size, -1) for mask in terrain_masks])
    
    # Create separate datasets for SAR and optical
        sar_ds = tf.data.Dataset.from_tensor_slices((sar_paths, sar_masks))
        optical_ds = tf.data.Dataset.from_tensor_slices((optical_paths, terrain_masks))

        # Add error handling to the mapping function
        def safe_parse_image(file_path, mask):
            try:
                return self._parse_image(file_path), mask
            except tf.errors.OpError as e:
                logger.error(f"Error loading image {file_path}: {str(e)}")
                # Return a placeholder image of correct shape
                return (tf.zeros([self.image_size, self.image_size, 3], dtype=tf.float32), 
                       mask)

        # Map paths to actual images with error handling
        sar_ds = sar_ds.map(
            safe_parse_image,
            num_parallel_calls=tf.data.AUTOTUNE
        ).filter(
            lambda x, y: tf.reduce_all(tf.math.is_finite(x))  # Filter out any invalid images
        )
        
        optical_ds = optical_ds.map(
            safe_parse_image,
            num_parallel_calls=tf.data.AUTOTUNE
        ).filter(
            lambda x, y: tf.reduce_all(tf.math.is_finite(x))  # Filter out any invalid images
        )

        if shuffle:
            sar_ds = sar_ds.shuffle(1000)
            optical_ds = optical_ds.shuffle(1000)

        # Zip the datasets together
        dataset = tf.data.Dataset.zip((sar_ds, optical_ds))
        
        # Add error checking before batching
        dataset = dataset.filter(
            lambda x, y: tf.reduce_all([
                tf.reduce_all(tf.math.is_finite(x[0])),  # Check SAR image
                tf.reduce_all(tf.math.is_finite(y[0]))   # Check optical image
            ])
        )
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset


    def load_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load and prepare unpaired training and validation datasets."""
        start_time = time.time()
        print("Starting to load unpaired SAR-optical dataset")

        sar_files = []
        optical_files = []
        terrain_indices = [] 

        # Collect file paths and corresponding terrain types
        for terrain_idx, terrain in enumerate(self.terrain_types):
            terrain_dir = os.path.join(self.dataset_dir, terrain)
            
            # Collect SAR files
            sar_path = os.path.join(terrain_dir, 's1')
            sar_files.extend([os.path.join(sar_path, f) for f in os.listdir(sar_path) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            # Collect optical files
            optical_path = os.path.join(terrain_dir, 's2')
            curr_optical_files = [os.path.join(optical_path, f) for f in os.listdir(optical_path) 
                                if f.endswith(('.png', '.jpg', '.jpeg'))]
            optical_files.extend(curr_optical_files)
            terrain_indices.extend([terrain_idx] * len(curr_optical_files))


        # Split into training and validation
        split_idx = int(len(sar_files) * (1 - self.validation_split))
        
        # Create SAR datasets
        train_sar_ds = tf.data.Dataset.from_tensor_slices(sar_files[:split_idx])
        val_sar_ds = tf.data.Dataset.from_tensor_slices(sar_files[split_idx:])
        
        train_optical_ds = tf.data.Dataset.from_tensor_slices({
            'path': optical_files[:split_idx],
            'terrain_idx': terrain_indices[:split_idx]
        })
        val_optical_ds = tf.data.Dataset.from_tensor_slices({
            'path': optical_files[split_idx:],
            'terrain_idx': terrain_indices[split_idx:]
        })

        def process_sar(file_path):
            image = self._process_image(file_path)
            mask = self._expand_mask(self._create_sar_mask())
            return image, mask  # Return actual data instead of None

        def process_optical(features):
            image = self._process_image(features['path'])
            mask = self._expand_mask(self._create_mask_vector(features['terrain_idx']))
            return image, mask  # Return actual data instead of None

        # Map processing functions
        train_sar_ds = train_sar_ds.map(process_sar, num_parallel_calls=tf.data.AUTOTUNE)
        val_sar_ds = val_sar_ds.map(process_sar, num_parallel_calls=tf.data.AUTOTUNE)
        train_optical_ds = train_optical_ds.map(process_optical, num_parallel_calls=tf.data.AUTOTUNE)
        val_optical_ds = val_optical_ds.map(process_optical, num_parallel_calls=tf.data.AUTOTUNE)

        # Cache, shuffle, batch, and prefetch
        def prepare_dataset(sar_ds, optical_ds, shuffle=True):
            if shuffle:
                sar_ds = sar_ds.shuffle(1000)
                optical_ds = optical_ds.shuffle(1000)
            
            sar_ds = sar_ds.batch(self.batch_size)
            optical_ds = optical_ds.batch(self.batch_size)
            
            # Zip datasets
            dataset = tf.data.Dataset.zip((sar_ds, optical_ds))
            return dataset.prefetch(tf.data.AUTOTUNE)

        train_ds = prepare_dataset(train_sar_ds, train_optical_ds, shuffle=True)
        val_ds = prepare_dataset(val_sar_ds, val_optical_ds, shuffle=False)

        print(f"Dataset loading completed in {time.time() - start_time:.2f} seconds")
        return train_ds, val_ds

    @tf.function
    def _preprocess_image(self, img: tf.Tensor) -> tf.Tensor:
        """Preprocess image according to paper specifications."""
        # Ensure the image has correct shape and type
        img = tf.ensure_shape(img, (None, None, 3))
        
        # Resize to consistent size
        img = tf.image.resize(img, [self.image_size, self.image_size])
        
        # Normalize to [-1, 1] range as commonly used in GANs
        img = tf.cast(img, tf.float32)
        img = (img / 127.5) - 1
        
        return img

    def _parse_image(self, file_path: str) -> tf.Tensor:
        """Load and parse image from file path."""
        # Convert file path tensor to string
        file_path = tf.convert_to_tensor(file_path)
        file_path = tf.strings.regex_replace(file_path, r'\\', '/')  # Handle Windows paths
        
        # Read and decode image
        img = tf.io.read_file(file_path)
        
        # Handle different image formats
        img = tf.image.decode_image(img, channels=3, expand_animations=False)
        
        # Ensure the image is always float32 and has 3 channels
        img = tf.cast(img, tf.float32)
        
        return self._preprocess_image(img)

    def _augment_image(self, image):
        """Apply random jittering and mirroring as specified in paper"""
        # Random jittering
        bigger_image = tf.image.resize(image, [286, 286])  # 286 = 256 + 30 as per paper
        cropped_image = tf.image.random_crop(bigger_image, [256, 256, 3])
        
        # Random mirroring
        if tf.random.uniform([]) > 0.5:
            cropped_image = tf.image.flip_left_right(cropped_image)
            
        return cropped_image

class MetricsLogger:
     def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.train_metrics = {
            'gen_loss': [],
            'disc_loss': [],
            'cycle_loss': [],
            'cls_loss': []
        }
        self.val_metrics = {
            'gen_loss': [],
            'disc_loss': [],
            'cycle_loss': [],
            'cls_loss': []
        }
    
     def update_train(self, metrics):
        for key, value in metrics.items():
            if key in self.train_metrics:
                self.train_metrics[key].append(float(value))
    
     def update_val(self, metrics):
        for key, value in metrics.items():
            if key in self.val_metrics:
                self.val_metrics[key].append(float(value))
    
     def log_epoch(self, epoch):
        metrics_str = " - ".join([
            f"{key}: {self.train_metrics[key][-1]:.4f}"
            for key in self.train_metrics
            if self.train_metrics[key]
        ])
        print(f"Epoch {epoch + 1} - {metrics_str}")

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def save_images(self, epoch, images, titles):
        try:
            plt.figure(figsize=(15, 5))
            for i, (img, title) in enumerate(zip(images, titles)):
                plt.subplot(1, len(images), i + 1)
                # Convert from [-1, 1] to [0, 1]
                img_numpy = img.numpy() if isinstance(img, tf.Tensor) else img
                img_display = (img_numpy + 1) * 0.5
                img_display = np.clip(img_display, 0, 1)  # Ensure values are in valid range
                plt.imshow(img_display)
                plt.title(title)
                plt.axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(self.save_dir, f'epoch_{epoch}.png')
            plt.savefig(save_path)
            plt.close()
            print(f"Saved visualization to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization: {str(e)}")

class InstanceNormalization(tf.keras.layers.Layer):
    """Custom Instance Normalization layer as used in the paper"""
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )
        self.offset = self.add_weight(
            name='offset',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )

    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        return self.scale * (x - mean) / tf.sqrt(variance + self.epsilon) + self.offset

class MultiDomainDiscriminator(tf.keras.Model):
    """Updated Discriminator architecture matching paper specifications"""
    def __init__(self, num_classes):
        super().__init__()
        
        # Common layers with dropout as specified in paper
        self.conv_blocks = [
            self._build_conv_block(64, first_block=True),
            self._build_conv_block(128),
            self._build_conv_block(256),
            self._build_conv_block(512)
        ]
        
        # Adversarial branch
        self.adv_conv = tf.keras.layers.Conv2D(1, 4, 1, padding='same')
        
        # Classification branch
        self.cls_conv = tf.keras.layers.Conv2D(num_classes, 4, 1, padding='same')
        self.global_avg = tf.keras.layers.GlobalAveragePooling2D()

    def _build_conv_block(self, filters, first_block=False):
        """Build conv block with optional normalization and dropout"""
        block = []
        block.append(tf.keras.layers.Conv2D(
            filters, 4, 2, padding='same',
            kernel_initializer='glorot_normal'
        ))
        if not first_block:
            block.append(LayerNormalization())
        block.append(tf.keras.layers.LeakyReLU(0.2))
        block.append(tf.keras.layers.Dropout(0.5))  # As mentioned in paper
        return block

    def call(self, x):
        for block in self.conv_blocks:
            for layer in block:
                x = layer(x)
        
        # Adversarial output
        adv_out = self.adv_conv(x)
        
        # Classification output
        cls_out = self.cls_conv(x)
        cls_out = self.global_avg(cls_out)
        
        return adv_out, cls_out

class Generator(tf.keras.Model):
    """Generator architecture matching paper specifications"""
    def __init__(self, image_size):
        super().__init__()
        
        # Initial convolution layers with instance normalization as per paper
        self.image_conv = tf.keras.Sequential([
            tf.keras.layers.Lambda(
                lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
            ),
            tf.keras.layers.Conv2D(32, 7, padding='valid', use_bias=False),
            InstanceNormalization(),
            tf.keras.layers.ReLU()
        ])
        
        self.mask_conv = tf.keras.Sequential([
            tf.keras.layers.Lambda(
                lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
            ),
            tf.keras.layers.Conv2D(32, 7, padding='valid', use_bias=False),
            InstanceNormalization(),
            tf.keras.layers.ReLU()
        ])
        
        self.concat = tf.keras.layers.Concatenate(axis=-1)
        
        # Shared encoder with instance normalization (keeping dimensions consistent)
        self.encoder = tf.keras.Sequential([
            # First downsampling
            tf.keras.layers.Conv2D(64, 3, strides=2, padding='same', use_bias=False),
            InstanceNormalization(),
            tf.keras.layers.ReLU(),
            
            # Second downsampling
            tf.keras.layers.Conv2D(128, 3, strides=2, padding='same', use_bias=False),
            InstanceNormalization(),
            tf.keras.layers.ReLU(),
            
            # Third downsampling
            tf.keras.layers.Conv2D(256, 3, strides=2, padding='same', use_bias=False),
            InstanceNormalization(),
            tf.keras.layers.ReLU()
        ])
        
        # Residual blocks
        self.res_blocks = [ResidualBlock(256) for _ in range(9)]
        
        # Decoder with instance normalization
        self.decoder = tf.keras.Sequential([
            # First upsampling
            tf.keras.layers.Conv2DTranspose(128, 3, strides=2, padding='same', use_bias=False),
            InstanceNormalization(),
            tf.keras.layers.ReLU(),
            
            # Second upsampling
            tf.keras.layers.Conv2DTranspose(64, 3, strides=2, padding='same', use_bias=False),
            InstanceNormalization(),
            tf.keras.layers.ReLU(),
            
            # Third upsampling
            tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', use_bias=False),
            InstanceNormalization(),
            tf.keras.layers.ReLU(),
            
            # Output layer
            tf.keras.layers.Lambda(
                lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]], 'REFLECT')
            ),
            tf.keras.layers.Conv2D(3, 7, padding='valid', activation='tanh')
        ])

    def call(self, inputs):
        image, mask = inputs
        
        # Process image and mask through initial convolutions
        x1 = self.image_conv(image)
        x2 = self.mask_conv(mask)
        
        # Concatenate features along channel dimension
        x = self.concat([x1, x2])
        
        # Encode
        x = self.encoder(x)
        
        # Apply residual blocks
        for block in self.res_blocks:
            x = block(x)
        
        # Decode
        x = self.decoder(x)
        
        return x

class ResidualBlock(tf.keras.layers.Layer):
    """Residual block with reflect padding as per paper"""
    def __init__(self, filters):
        super().__init__()
        self.filters = filters
        
        # First conv block
        self.conv1 = tf.keras.Sequential([
            tf.keras.layers.Lambda(
                lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            ),
            tf.keras.layers.Conv2D(
                filters, 3, padding='valid'
            )
        ])
        self.in1 = InstanceNormalization()
        
        # Second conv block
        self.conv2 = tf.keras.Sequential([
            tf.keras.layers.Lambda(
                lambda x: tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], 'REFLECT')
            ),
            tf.keras.layers.Conv2D(
                filters, 3, padding='valid'
            )
        ])
        self.in2 = InstanceNormalization()
        
    def call(self, x, training=True):
        residual = x
        x = self.conv1(x)
        x = self.in1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.in2(x)
        return x + residual

class CycleGAN:
    def __init__(
        self,
        image_size: int,
        num_classes: int,
        lambda_cyc: float = 10.0,    # As per paper section II.C.4
        lambda_cls: float = 1.0,     # As per paper section II.C.4
        learning_rate: float = 0.0002 # As per paper section II.D
    ):
        self.G1 = Generator(image_size)  # SAR to Optical
        self.G2 = Generator(image_size)  # Optical to SAR
        self.D1 = MultiDomainDiscriminator(num_classes)  # For Optical
        self.D2 = MultiDomainDiscriminator(num_classes)  # For SAR
        self.lambda_cyc = lambda_cyc
        self.lambda_cls = lambda_cls
        
        # Adam optimizer with β1 = 0.5 as specified in paper section II.D
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)

        dummy_image = tf.zeros((1, image_size, image_size, 3))
        dummy_mask = tf.zeros((1, image_size, image_size, num_classes))

        self.G1([dummy_image, dummy_mask])
        self.G2([dummy_image, dummy_mask])

        self.D1(dummy_image)
        self.D2(dummy_image)

    def compile(self):
        """Initialize metrics for tracking losses as described in paper."""
        self.gen_loss_metric = tf.keras.metrics.Mean('gen_loss')
        self.disc_loss_metric = tf.keras.metrics.Mean('disc_loss')
        self.cycle_loss_metric = tf.keras.metrics.Mean('cycle_loss')
        self.cls_loss_metric = tf.keras.metrics.Mean('cls_loss')

    @tf.function(experimental_compile=True)
    def train_step(self, inputs):
        """Modified train step to work with distributed training."""
        (real_sar, sar_mask), (real_optical, optical_mask) = inputs
        
        with tf.GradientTape(persistent=True) as tape:
            # Forward passes
            fake_optical = self.G1([real_sar, sar_mask])
            fake_sar = self.G2([real_optical, optical_mask])
            cycled_sar = self.G2([fake_optical, optical_mask])
            cycled_optical = self.G1([fake_sar, sar_mask])

            # Discriminator outputs for real images
            disc_real_optical_adv, disc_real_optical_cls = self.D1(real_optical)
            disc_real_sar_adv, disc_real_sar_cls = self.D2(real_sar)

            # Discriminator outputs for fake images
            disc_fake_optical_adv, disc_fake_optical_cls = self.D1(fake_optical)
            disc_fake_sar_adv, disc_fake_sar_cls = self.D2(fake_sar)

            # Generator adversarial loss
            gen_loss_optical = tf.reduce_mean(tf.square(disc_fake_optical_adv - 1.0))
            gen_loss_sar = tf.reduce_mean(tf.square(disc_fake_sar_adv - 1.0))
            gen_loss = gen_loss_optical + gen_loss_sar

            # Cycle consistency loss
            cycle_loss = (
                tf.reduce_mean(tf.abs(real_sar - cycled_sar)) +
                tf.reduce_mean(tf.abs(real_optical - cycled_optical))
            ) * self.lambda_cyc

            # Classification losses for generators
            cls_loss_G1 = self._classification_loss(optical_mask, disc_fake_optical_cls)
            cls_loss_G2 = self._classification_loss(sar_mask, disc_fake_sar_cls)
            cls_loss_G = (cls_loss_G1 + cls_loss_G2) * self.lambda_cls

            # Total generator loss
            total_gen_loss = gen_loss + cycle_loss + cls_loss_G

            # Discriminator losses
            disc_loss_optical = tf.reduce_mean(
                tf.square(disc_real_optical_adv - 1.0) + 
                tf.square(disc_fake_optical_adv)
            )
            disc_loss_sar = tf.reduce_mean(
                tf.square(disc_real_sar_adv - 1.0) + 
                tf.square(disc_fake_sar_adv)
            )
            disc_adv_loss = (disc_loss_optical + disc_loss_sar) * 0.5

            # Classification losses for discriminators
            cls_loss_D1 = self._classification_loss(optical_mask, disc_real_optical_cls)
            cls_loss_D2 = self._classification_loss(sar_mask, disc_real_sar_cls)
            cls_loss_D = (cls_loss_D1 + cls_loss_D2) * self.lambda_cls

            # Total discriminator loss
            total_disc_loss = disc_adv_loss + cls_loss_D

        # Calculate gradients
        gen_grads = tape.gradient(total_gen_loss, 
            self.G1.trainable_variables + self.G2.trainable_variables)
        disc_grads = tape.gradient(total_disc_loss,
            self.D1.trainable_variables + self.D2.trainable_variables)

        # Apply gradients
        self.gen_optimizer.apply_gradients(
            zip(gen_grads, self.G1.trainable_variables + self.G2.trainable_variables))
        self.disc_optimizer.apply_gradients(
            zip(disc_grads, self.D1.trainable_variables + self.D2.trainable_variables))

        return {
            'gen_loss': total_gen_loss,
            'disc_loss': total_disc_loss,
            'cycle_loss': cycle_loss,
            'cls_loss': cls_loss_G + cls_loss_D,
            'generated_images': (fake_optical, fake_sar, cycled_optical, cycled_sar)
        }

    def _adversarial_loss(
        self,
        pred: tf.Tensor,
        target_is_real: bool
    ) -> tf.Tensor:
        """Calculate adversarial loss as described in paper Section II.C.1"""
        if target_is_real:
            target = tf.ones_like(pred)
        else:
            target = tf.zeros_like(pred)
            
        return tf.reduce_mean(tf.keras.losses.binary_crossentropy(
            target, pred, from_logits=True
        ))

    def _classification_loss(self, labels: tf.Tensor, pred_cls: tf.Tensor) -> tf.Tensor:
        """Calculate classification loss.
        
        Args:
            labels: Tensor of shape (batch_size, H, W, num_classes)
            pred_cls: Tensor of shape (batch_size, num_classes)
        Returns:
            Classification loss value
        """
        # Average the labels over spatial dimensions to match discriminator output shape
        labels_reduced = tf.reduce_mean(labels, axis=[1, 2])  # Average over H,W dimensions
        
        # Both tensors now have shape (batch_size, num_classes)
        return tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                labels_reduced,  # Shape: (batch_size, num_classes) 
                pred_cls,       # Shape: (batch_size, num_classes)
                from_logits=True
            )
        )

class Trainer:
    """Updated Trainer with paper's learning rate schedule"""
    def __init__(
        self,
        model: CycleGAN,
        train_ds: tf.data.Dataset,
        
val_ds: tf.data.Dataset,
        output_dir: str,
        strategy: tf.distribute.Strategy,
        epochs: int = 200,
        lambda_cyc: float = 10.0,
        lambda_cls: float = 1.0,
        initial_epoch: int = 0
    ):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.output_dir = output_dir
        self.epochs = epochs
        self.lambda_cyc = lambda_cyc
        self.lambda_cls = lambda_cls
        self.initial_epoch = initial_epoch
        self.strategy = strategy
        with strategy.scope():
        # Initialize metrics
         self.gen_loss_metric = tf.keras.metrics.Mean()
         self.disc_loss_metric = tf.keras.metrics.Mean()
         self.cycle_loss_metric = tf.keras.metrics.Mean()
         self.cls_loss_metric = tf.keras.metrics.Mean()
         self.domain_cls_loss_metric = tf.keras.metrics.Mean()
         self.psnr_metric = tf.keras.metrics.Mean()
         self.ssim_metric = tf.keras.metrics.Mean()

        os.makedirs(output_dir, exist_ok=True)

        # Learning rate schedule as per paper
        self.initial_learning_rate = 0.0002
        self.decay_steps = 100  # Number of epochs after which to start decay

        # Add step counters for printing
        self.train_steps_per_epoch = 0
        for _ in self.train_ds:
            self.train_steps_per_epoch += 1

    @tf.function(experimental_compile=True)  # Add XLA compilation
    def distributed_train_step(self, inputs):
        """Distributed training step with XLA compilation for faster GPU execution"""
        per_replica_losses = self.strategy.run(self.model.train_step, args=(inputs,))
        
        # Reduce and return losses
        reduced_losses = {}
        for key, value in per_replica_losses.items():
            if key != 'generated_images':
                reduced_losses[key] = self.strategy.reduce(
                    tf.distribute.ReduceOp.MEAN, value, axis=None
                )
            else:
                reduced_losses[key] = value
        
        return reduced_losses

    def train(self):
        """Enhanced training loop with detailed metrics printing"""
        # Enable mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        
        print("\nStarting training with GPU strategy:", self.strategy)
        print(f"Steps per epoch: {self.train_steps_per_epoch}")

        # Ensure datasets are prefetched for better GPU utilization
        train_dist = self.strategy.experimental_distribute_dataset(
            self.train_ds.prefetch(tf.data.AUTOTUNE)
        )
        val_dist = self.strategy.experimental_distribute_dataset(
            self.val_ds.prefetch(tf.data.AUTOTUNE)
        )

        for epoch in range(self.initial_epoch, self.epochs):
            start_time = time.time()
            
            # Initialize metric accumulators for this epoch
            epoch_metrics = {
                'gen_loss': 0.0,
                'disc_loss': 0.0,
                'cycle_loss': 0.0,
                'cls_loss': 0.0
            }
            
             # Reset metrics at start of epoch
            self.psnr_metric.reset_state()
            self.ssim_metric.reset_state()

            # Training loop with progress printing
            for step, inputs in enumerate(train_dist):
                try:
                    metrics = self.distributed_train_step(inputs)
                    
                    # Accumulate metrics
                    for key in epoch_metrics.keys():
                        if key in metrics:
                            epoch_metrics[key] += metrics[key].numpy()
                    
                    # Log progress every 10 steps
                    if step % 10 == 0:
                        metrics_str = " - ".join([
                            f"{k}: {v.numpy():.4f}" if isinstance(v, tf.Tensor) 
                            else f"{k}: {v:.4f}"
                            for k, v in metrics.items() 
                            if k != 'generated_images'
                        ])
                        print(f"\rEpoch {epoch+1}/{self.epochs} - Step {step}/{self.train_steps_per_epoch} - {metrics_str}", end='')
                
                except Exception as e:
                    print(f"\nError in training step {step}: {str(e)}")
                    continue

            # Calculate epoch averages
            for key in epoch_metrics:
                epoch_metrics[key] /= self.train_steps_per_epoch
            
            # Get PSNR and SSIM from metrics
            avg_psnr = self.psnr_metric.result().numpy()
            avg_ssim = self.ssim_metric.result().numpy()

            # End of epoch
            epoch_time = time.time() - start_time
            
            # Print epoch summary
            print(f"\n{'='*80}")
            print(f"Epoch {epoch+1}/{self.epochs} Summary - Time: {epoch_time:.2f}s")
            print(f"Generator Loss: {epoch_metrics['gen_loss']:.4f}")
            print(f"Discriminator Loss: {epoch_metrics['disc_loss']:.4f}")
            print(f"Cycle Consistency Loss: {epoch_metrics['cycle_loss']:.4f}")
            print(f"Classification Loss: {epoch_metrics['cls_loss']:.4f}")
            print(f"PSNR: {avg_psnr:.4f}")
            print(f"SSIM: {avg_ssim:.4f}")
            print(f"{'='*80}\n")

            # Save samples and checkpoint
            if (epoch + 1) % 5 == 0:
                self._save_samples(epoch)
                self._save_checkpoint(epoch)
                
             # Reset metrics at epoch end
            self.gen_loss_metric.reset_state()
            self.disc_loss_metric.reset_state()
            self.cycle_loss_metric.reset_state()
            self.cls_loss_metric.reset_state()
            self.domain_cls_loss_metric.reset_state()

    def _classification_loss(self, labels: tf.Tensor, pred_cls: tf.Tensor) -> tf.Tensor:
      """Calculate classification loss."""
      # Reduce mask to 2D by averaging over spatial dimensions
     
        
        # Average the mask over spatial dimensions (H,W) to match discriminator output shape
        # Input labels shape: (batch_size, H, W, num_classes)
        # Input pred_cls shape: (batch_size, num_classes)
      labels_reduced = tf.reduce_mean(labels, axis=[1, 2])  # Average over H,W dimensions
        
      return tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                labels_reduced, pred_cls,
                from_logits=True
            )
        )

    def _calculate_image_metrics(self, real_img, generated_img):
        """Calculate PSNR and SSIM metrics between real and generated images."""
        # Convert from [-1,1] to [0,1] range
        real_img = (real_img + 1.0) / 2.0
        generated_img = (generated_img + 1.0) / 2.0
        
        # Calculate PSNR
        psnr = tf.image.psnr(real_img, generated_img, max_val=1.0)
        
        # Calculate SSIM
        ssim = tf.image.ssim(real_img, generated_img, max_val=1.0)
        
        return tf.reduce_mean(psnr), tf.reduce_mean(ssim)

    @tf.function(experimental_compile=True)
    def distributed_train_step(self, inputs):
        """Distributed training step using strategy.run()"""
        def _replica_step(inputs):
            (sar_img, sar_mask), (opt_img, opt_mask) = inputs

            with tf.GradientTape(persistent=True) as tape:
                # Forward passes
                fake_optical = self.model.G1([sar_img, sar_mask])
                fake_sar = self.model.G2([opt_img, opt_mask])
                cycled_optical = self.model.G1([fake_sar, sar_mask])
                cycled_sar = self.model.G2([fake_optical, opt_mask])

                # Discriminator outputs for real/fake images
                disc_real_optical = self.model.D1(opt_img)
                disc_fake_optical = self.model.D1(fake_optical)
                disc_real_sar = self.model.D2(sar_img)
                disc_fake_sar = self.model.D2(fake_sar)

                # Calculate per-replica losses
                gen_loss = self._compute_generator_loss(disc_fake_optical, disc_fake_sar)
                disc_loss = self._compute_discriminator_loss(
                    disc_real_optical, disc_fake_optical, 
                    disc_real_sar, disc_fake_sar
                )
                cycle_loss = self._compute_cycle_loss(
                    sar_img, opt_img, cycled_sar, cycled_optical
                )
                cls_loss = self._compute_classification_loss(
                    sar_mask, opt_mask, disc_fake_optical[1], disc_fake_sar[1]
                )

                # Scale losses by global batch size
                gen_loss = gen_loss / self.strategy.num_replicas_in_sync
                disc_loss = disc_loss / self.strategy.num_replicas_in_sync
                cycle_loss = cycle_loss / self.strategy.num_replicas_in_sync
                cls_loss = cls_loss / self.strategy.num_replicas_in_sync

                total_gen_loss = gen_loss + self.lambda_cyc * cycle_loss + self.lambda_cls * cls_loss
                total_disc_loss = disc_loss

            # Apply gradients
            gen_grads = tape.gradient(total_gen_loss, 
                self.model.G1.trainable_variables + self.model.G2.trainable_variables)
            disc_grads = tape.gradient(total_disc_loss,
                self.model.D1.trainable_variables + self.model.D2.trainable_variables)

            self.model.gen_optimizer.apply_gradients(
                zip(gen_grads, self.model.G1.trainable_variables + self.model.G2.trainable_variables))
            self.model.disc_optimizer.apply_gradients(
                zip(disc_grads, self.model.D1.trainable_variables + self.model.D2.trainable_variables))

            # Update metrics
            self.gen_loss_metric.update_state(gen_loss)
            self.disc_loss_metric.update_state(disc_loss)
            self.cycle_loss_metric.update_state(cycle_loss)
            self.cls_loss_metric.update_state(cls_loss)

            # Calculate image quality metrics
            psnr = tf.reduce_mean(tf.image.psnr(opt_img, fake_optical, max_val=2.0))
            ssim = tf.reduce_mean(tf.image.ssim(opt_img, fake_optical, max_val=2.0))

            return {
                'gen_loss': gen_loss,
                'disc_loss': disc_loss,
                'cycle_loss': cycle_loss,
                'cls_loss': cls_loss,
                'psnr': psnr,
                'ssim': ssim,
                'generated_images': (fake_optical, fake_sar, cycled_optical, cycled_sar)
            }

        # Run the replica step using strategy
        per_replica_metrics = self.strategy.run(_replica_step, args=(inputs,))

        # Reduce metrics across replicas
        reduced_metrics = {
            key: self.strategy.reduce(
                tf.distribute.ReduceOp.MEAN, 
                per_replica_metrics[key], 
                axis=None
            ) if key != 'generated_images' else per_replica_metrics[key]
            for key in per_replica_metrics
        }

        return reduced_metrics

    def train(self):
        """Execute training loop with proper distribution strategy."""
        # Remove the mapping operations on distributed datasets
        # Just use the datasets directly after distribution
        train_dist = self.strategy.experimental_distribute_dataset(self.train_ds)
        val_dist = self.strategy.experimental_distribute_dataset(self.val_ds)

        for epoch in range(self.initial_epoch, self.epochs):
            start_time = time.time()
            self._update_learning_rate(epoch)
            
             # Reset metrics at start of epoch
            self.psnr_metric.reset_state()
            self.ssim_metric.reset_state()
            
            # Initialize metric accumulators for this epoch
            epoch_metrics = {
                'gen_loss': 0.0,
                'disc_loss': 0.0,
                'cycle_loss': 0.0,
                'cls_loss': 0.0
            }
             
            # Training loop
            for step, inputs in enumerate(train_dist):
                try:
                    metrics = self.distributed_train_step(inputs)
                    
                     # Accumulate metrics
                    for key in epoch_metrics.keys():
                        if key in metrics:
                            epoch_metrics[key] += metrics[key].numpy()
                    
                    # Log progress every 100 steps
                    if step % 100 == 0:
                        metrics_str = " - ".join([
                            f"{k}: {v.numpy():.4f}" if isinstance(v, tf.Tensor) 
                            else f"{k}: {v:.4f}"
                            for k, v in metrics.items() 
                            if k != 'generated_images'
                        ])
                        print(f"Epoch {epoch+1}, Step {step}: {metrics_str}")
                
                except Exception as e:
                    logger.error(f"Error during training step {step}: {str(e)}")
                    continue

            # Calculate epoch averages
            for key in epoch_metrics:
                epoch_metrics[key] /= self.train_steps_per_epoch

            # Get PSNR and SSIM from metrics
            avg_psnr = self.psnr_metric.result().numpy()
            avg_ssim = self.ssim_metric.result().numpy()

            # End of epoch
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

            # Print the summary for the metrics at the end of each epoch
            print(f"Epoch {epoch+1} Summary:")
            print(f"  Generator Loss: {epoch_metrics['gen_loss']:.4f}")
            print(f"  Discriminator Loss: {epoch_metrics['disc_loss']:.4f}")
            print(f"  Cycle Consistency Loss: {epoch_metrics['cycle_loss']:.4f}")
            print(f"  Classification Loss: {epoch_metrics['cls_loss']:.4f}")
            print(f"  PSNR: {avg_psnr:.4f}")
            print(f"  SSIM: {avg_ssim:.4f}")

            # Save samples and checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                try:
                    self._save_samples(epoch)
                    self._save_checkpoint(epoch)
                except Exception as e:
                    logger.error(f"Error saving epoch {epoch+1} artifacts: {str(e)}")

            # Reset metrics at epoch end
            self.gen_loss_metric.reset_state()
            self.disc_loss_metric.reset_state()
            self.cycle_loss_metric.reset_state()
            self.cls_loss_metric.reset_state()
            self.domain_cls_loss_metric.reset_state()

    def _compute_losses(self, *args):
        """Helper to compute all losses"""
        (sar_img, opt_img, 
         fake_optical, fake_sar,
         cycled_optical, cycled_sar,
         disc_real_optical_adv, disc_fake_optical_adv,
         disc_real_sar_adv, disc_fake_sar_adv,
         disc_real_optical_cls, disc_fake_optical_cls,
         disc_real_sar_cls, disc_fake_sar_cls,
         sar_mask, opt_mask) = args

        # Generator adversarial losses
        gen_loss_optical = tf.reduce_mean(tf.square(disc_fake_optical_adv - 1))
        gen_loss_sar = tf.reduce_mean(tf.square(disc_fake_sar_adv - 1))

        # Cycle consistency losses
        cycle_loss = (
            tf.reduce_mean(tf.abs(sar_img - cycled_sar)) +
            tf.reduce_mean(tf.abs(opt_img - cycled_optical))
        ) * self.lambda_cyc

        # Classification losses
        cls_loss_G1 = self._classification_loss(opt_mask, disc_fake_optical_cls)
        cls_loss_G2 = self._classification_loss(sar_mask, disc_fake_sar_cls)

        # Discriminator losses
        disc_loss_optical = tf.reduce_mean(
            tf.square(disc_real_optical_adv - 1) + tf.square(disc_fake_optical_adv)
        )
        disc_loss_sar = tf.reduce_mean(
            tf.square(disc_real_sar_adv - 1) + tf.square(disc_fake_sar_adv)
        )

        cls_loss_D1 = self._classification_loss(opt_mask, disc_real_optical_cls)
        cls_loss_D2 = self._classification_loss(sar_mask, disc_real_sar_cls)

        # Total losses
        gen_total_loss = (
            gen_loss_optical + gen_loss_sar +
            cycle_loss +
            (cls_loss_G1 + cls_loss_G2) * self.lambda_cls
        )
        
        disc_total_loss = (
            (disc_loss_optical + disc_loss_sar +
             (cls_loss_D1 + cls_loss_D2) * self.lambda_cls) * 0.5
        )

        return gen_total_loss, disc_total_loss

    def _domain_classification_loss(self, disc_output, target_domain):
        # Remove this method since we now have _classification_loss
        return self._classification_loss(target_domain, disc_output)

    def _save_samples(self, epoch):
        """Save sample images during training"""
        sample_batch = next(iter(self.val_ds))
        (sar_img, sar_mask), (opt_img, opt_mask) = sample_batch
        metrics = self.model.train_step(sar_img, opt_img, sar_mask, opt_mask)
        
        if 'generated_images' in metrics:
            fake_optical, fake_sar, cycled_optical, cycled_sar = metrics['generated_images']
            
            plt.figure(figsize=(15, 10))
            titles = ['Real Optical', 'Fake SAR', 'Cycled Optical',
                     'Real SAR', 'Fake Optical', 'Cycled SAR']
            images = [opt_img[0], fake_sar[0], cycled_optical[0],
                     sar_img[0], fake_optical[0], cycled_sar[0]]
            
            for i, (img, title) in enumerate(zip(images, titles)):
                plt.subplot(2, 3, i + 1)
                plt.imshow(img * 0.5 + 0.5)
                plt.title(title)
                plt.axis('off')
            
            plt.savefig(os.path.join(self.output_dir, f'samples_epoch_{epoch+1}.png'))
            plt.close()

    def _save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(self.output_dir, f'checkpoint_epoch_{epoch + 1}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save(checkpoint_dir)

    def _update_learning_rate(self, epoch):
        """Implement paper's learning rate decay schedule"""
        if epoch > self.decay_steps:
            decay_factor = 1.0 - max(0, epoch - self.decay_steps) / float(self.epochs - self.decay_steps)
            new_lr = self.initial_learning_rate * decay_factor
            self.model.gen_optimizer.learning_rate.assign(new_lr)
            self.model.disc_optimizer.learning_rate.assign(new_lr)
            print(f"Learning rate adjusted to {new_lr}")

    def _compute_generator_loss(self, disc_fake_optical, disc_fake_sar):
        """Compute generator adversarial loss."""
        # Get adversarial outputs
        disc_fake_optical_adv, _ = disc_fake_optical
        disc_fake_sar_adv, _ = disc_fake_sar
        
        # Calculate losses (want discriminator to predict real (1) for generated images)
        gen_loss_optical = tf.reduce_mean(tf.square(disc_fake_optical_adv - 1))
        gen_loss_sar = tf.reduce_mean(tf.square(disc_fake_sar_adv - 1))
        
        return gen_loss_optical + gen_loss_sar

    def _compute_discriminator_loss(self, disc_real_optical, disc_fake_optical, 
                                  disc_real_sar, disc_fake_sar):
        """Compute discriminator adversarial loss."""
        # Get adversarial outputs
        disc_real_optical_adv, _ = disc_real_optical
        disc_fake_optical_adv, _ = disc_fake_optical
        disc_real_sar_adv, _ = disc_real_sar
        disc_fake_sar_adv, _ = disc_fake_sar
        
        # Calculate losses
        disc_loss_optical = tf.reduce_mean(
            tf.square(disc_real_optical_adv - 1) + tf.square(disc_fake_optical_adv)
        )
        disc_loss_sar = tf.reduce_mean(
            tf.square(disc_real_sar_adv - 1) + tf.square(disc_fake_sar_adv)
        )
        
        return (disc_loss_optical + disc_loss_sar) * 0.5

    def _compute_cycle_loss(self, real_sar, real_optical, cycled_sar, cycled_optical):
        """Compute cycle consistency loss."""
        return (
            tf.reduce_mean(tf.abs(real_sar - cycled_sar)) +
            tf.reduce_mean(tf.abs(real_optical - cycled_optical))
        ) * self.lambda_cyc

    def _compute_classification_loss(self, sar_mask, optical_mask, 
                                   fake_optical_cls, fake_sar_cls):
        """Compute classification loss for generated images."""
        cls_loss_optical = self._classification_loss(optical_mask, fake_optical_cls)
        cls_loss_sar = self._classification_loss(sar_mask, fake_sar_cls)
        return (cls_loss_optical + cls_loss_sar) * self.lambda_cls

    def _compute_total_generator_loss(self, gen_loss, cycle_loss, cls_loss):
        """Compute total generator loss with weighted components."""
        return gen_loss + cycle_loss + cls_loss

    def _compute_total_discriminator_loss(self, disc_loss, cls_loss):
        """Compute total discriminator loss with weighted components."""
        return disc_loss + cls_loss * self.lambda_cls

    def _compute_metrics(self, real_img, generated_img):
        """Compute PSNR and SSIM metrics."""
        # Convert from [-1,1] to [0,1] range
        real_img = (real_img + 1.0) / 2.0
        generated_img = (generated_img + 1.0) / 2.0
        
        # Calculate metrics
        psnr = tf.reduce_mean(tf.image.psnr(real_img, generated_img, max_val=1.0))
        ssim = tf.reduce_mean(tf.image.ssim(real_img, generated_img, max_val=1.0))
        
        return psnr, ssim

def configure_gpu():
     physical_devices = tf.config.list_physical_devices('GPU')
     if physical_devices:
        try:
            # Enable memory growth on all GPUs
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            
            # Visible devices - use both GPUs
            tf.config.set_visible_devices(physical_devices, 'GPU')
            print(f"Configured {len(physical_devices)} GPUs")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")

def main():
    configure_gpu()
    tf.config.set_soft_device_placement(False)
    # Configure logging first
    logging_file = f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logging_file)
        ]
    )

    # Use MirroredStrategy to utilize multiple GPUs
    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        # Update batch size based on strategy
        GLOBAL_BATCH_SIZE = 32 * strategy.num_replicas_in_sync

        # Configuration parameters
        config = {
            'image_size': 256,
            'batch_size': GLOBAL_BATCH_SIZE,
            'epochs': 1,
            'dataset_dir': '/kaggle/input/sentinel12-image-pairs-segregated-by-terrain/v_2',
            'output_dir': '/kaggle/working',
            'learning_rate': 0.0002,
            'lambda_cyc': 10.0,
            'lambda_cls': 1.0,
            'validation_split': 0.1,
        }

        print(f"Training config: {config}")

        # Initialize components within strategy scope
        with strategy.scope():
            try:
                # Initialize data loader
                data_loader = DataLoader(
                    dataset_dir=config['dataset_dir'],
                    image_size=config['image_size'],
                    batch_size=config['batch_size']
                )
                
                # Load datasets
                train_ds, val_ds = data_loader.load_data()
                
                # Apply options before distribution
                options = tf.data.Options()
                options.experimental_distribute.auto_shard_policy = \
                    tf.data.experimental.AutoShardPolicy.DATA
                
                train_ds = train_ds.with_options(options)
                val_ds = val_ds.with_options(options)

                # Initialize model
                model = CycleGAN(
                    image_size=config['image_size'],
                    num_classes=data_loader.num_terrains + 1,
                    lambda_cyc=config['lambda_cyc'],
                    lambda_cls=config['lambda_cls'],
                    learning_rate=config['learning_rate']
                )

                # Initialize with dummy data of correct shapes
                dummy_image = tf.zeros((1, config['image_size'], config['image_size'], 3))
                dummy_mask = tf.zeros((1, config['image_size'], config['image_size'], 
                                     data_loader.num_terrains + 1))

                # Build models explicitly
                model.G1.build([(None, config['image_size'], config['image_size'], 3),
                               (None, config['image_size'], config['image_size'], 
                                data_loader.num_terrains + 1)])
                model.G2.build([(None, config['image_size'], config['image_size'], 3),
                               (None, config['image_size'], config['image_size'], 
                                data_loader.num_terrains + 1)])
                model.D1.build((None, config['image_size'], config['image_size'], 3))
                model.D2.build((None, config['image_size'], config['image_size'], 3))

                # Initialize trainer
                trainer = Trainer(
                    model=model,
                    train_ds=train_ds,
                    val_ds=val_ds,
                    output_dir=config['output_dir'],
                    strategy=strategy,
                    epochs=config['epochs']
                )

                # Calculate and log total parameters
                total_params = sum([
                    np.sum([tf.size(w).numpy() for w in model.G1.trainable_weights]),
                    np.sum([tf.size(w).numpy() for w in model.G2.trainable_weights]),
                    np.sum([tf.size(w).numpy() for w in model.D1.trainable_weights]),
                    np.sum([tf.size(w).numpy() for w in model.D2.trainable_weights])
                ])
                
                # Log model summary
                print("Model architectures initialized successfully")
                print(f"Total trainable parameters: {total_params:,}")

                # Start training
                trainer.train()

            except Exception as e:
                print(f"Training failed: {str(e)}")
                raise

if __name__ == "__main__":
    main()