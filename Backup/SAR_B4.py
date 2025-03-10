import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import time
from typing import Tuple, List, Dict, Any
import json
import tensorflow_addons as tfa
from keras.applications.inception_v3 import preprocess_input
from scipy import linalg

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
            if physical_devices:
                logger.info(f"Found {len(physical_devices)} GPU(s)")
                for device in physical_devices:
                    tf.config.experimental.set_memory_growth(device, True)
                    logger.info(f"Enabled memory growth for GPU: {device}")
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
        batch_size: int = 1,  # As mentioned in paper Section II.D
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
        logger.info(f"Found terrain types: {self.terrain_types}")

    def _get_terrain_types(self) -> List[str]:
        """Get list of terrain types from dataset directory structure."""
        return sorted([d for d in os.listdir(self.dataset_dir) 
                      if os.path.isdir(os.path.join(self.dataset_dir, d))])

    def _create_mask_lookup_table(self):
        """Create a lookup table for terrain masks."""
        num_classes = self.num_terrains + 1
        masks = np.zeros((self.num_terrains, num_classes), dtype=np.float32)
        for i in range(self.num_terrains):
            masks[i, i + 1] = 1.0  # +1 to account for SAR domain at index 0
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
        logger.info("Starting to load unpaired SAR-optical dataset")

        sar_files = []
        optical_files = []
        terrain_indices = [] 

        # Collect file paths and corresponding terrain types
        for terrain_idx, terrain in enumerate(self.terrain_types):
            terrain_dir = os.path.join(self.dataset_dir, terrain)
            
            # Collect SAR files
            sar_path = os.path.join(terrain_dir, 'SAR')
            sar_files.extend([os.path.join(sar_path, f) for f in os.listdir(sar_path) 
                            if f.endswith(('.png', '.jpg', '.jpeg'))])
            
            # Collect optical files
            optical_path = os.path.join(terrain_dir, 'Color')
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
            return image, mask

        def process_optical(features):
            image = self._process_image(features['path'])
            mask = self._expand_mask(self._create_mask_vector(features['terrain_idx']))
            return image, mask

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

        logger.info(f"Dataset loading completed in {time.time() - start_time:.2f} seconds")
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
        try:
            img = tf.cond(
                tf.strings.regex_full_match(file_path, r'.*\.png$'),
                lambda: tf.image.decode_png(img, channels=3),
                lambda: tf.image.decode_jpeg(img, channels=3)
            )
        except:
            # Fallback decode method
            img = tf.image.decode_image(img, channels=3)
            img.set_shape([None, None, 3])
        
        # Ensure the image is always float32 and has 3 channels
        img = tf.cast(img, tf.float32)
        
        return self._preprocess_image(img)

    
    
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
        logger.info(f"Epoch {epoch + 1} - {metrics_str}")

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
            logger.info(f"Saved visualization to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save visualization: {str(e)}")

class MultiDomainDiscriminator(tf.keras.Model):
    """Discriminator architecture as shown in Fig. 3 of paper"""
    def __init__(self, num_classes):
        super().__init__()
        
        # Common layers
        self.conv1 = tf.keras.layers.Conv2D(64, 4, 2, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(128, 4, 2, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(256, 4, 2, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(512, 4, 2, padding='same')
        
        # Adversarial branch
        self.adv_conv = tf.keras.layers.Conv2D(1, 4, 1, padding='same')
        
        # Classification branch
        self.cls_conv = tf.keras.layers.Conv2D(num_classes, 4, 1, padding='same')
        self.global_avg = tf.keras.layers.GlobalAveragePooling2D()
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        x = tf.nn.leaky_relu(self.conv1(x))
        x = tf.nn.leaky_relu(self.conv2(x))
        x = tf.nn.leaky_relu(self.conv3(x))
        x = tf.nn.leaky_relu(self.conv4(x))
        
        # Adversarial output
        adv_out = self.adv_conv(x)
        
        # Classification output
        cls_out = self.cls_conv(x)
        cls_out = self.global_avg(cls_out)
        cls_out = self.softmax(cls_out)
        
        return adv_out, cls_out

class Generator(tf.keras.Model):
    """Generator architecture as shown in Fig. 2 of paper"""
    def __init__(self, image_size):
        super().__init__()
        self.initialized = False
        self.image_conv = None
        self.mask_conv = None
        
        # Rest of the architecture remains the same
        self.shared_conv = tf.keras.layers.Conv2D(64, 3, 1, padding='same')
        self.down_conv1 = tf.keras.layers.Conv2D(128, 3, 2, padding='same')
        self.down_conv2 = tf.keras.layers.Conv2D(256, 3, 2, padding='same')
        self.res_blocks = [ResidualBlock(256) for _ in range(9)]
        self.up_conv1 = tf.keras.layers.Conv2DTranspose(128, 3, 2, padding='same')
        self.up_conv2 = tf.keras.layers.Conv2DTranspose(64, 3, 2, padding='same')
        self.output_conv = tf.keras.layers.Conv2D(3, 7, 1, padding='same', activation='tanh')
        self.concat = tf.keras.layers.Concatenate(axis=-1)

    def build(self, input_shapes):
        """Build model with dynamic input shapes"""
        if not self.built:
            image_shape, mask_shape = input_shapes
            
            # Create conv layers with proper input channel dimensions
            self.image_conv = tf.keras.layers.Conv2D(
                32, 7, 1, padding='same',
                input_shape=(None, None, image_shape[-1])
            )
            self.mask_conv = tf.keras.layers.Conv2D(
                32, 7, 1, padding='same',
                input_shape=(None, None, mask_shape[-1])
            )
            
            # Call parent's build method
            super().build(input_shapes)

    def call(self, inputs):
        image, mask = inputs
        
        # Ensure 4D tensors
        if len(image.shape) == 3:
            image = tf.expand_dims(image, 0)
        if len(mask.shape) == 3:
            mask = tf.expand_dims(mask, 0)
        
        # Process image and mask separately
        image_features = self.image_conv(image)
        mask_features = self.mask_conv(mask)
        
        # Concatenate features
        x = self.concat([image_features, mask_features])
        
        # Rest of the forward pass
        x = self.shared_conv(x)
        x = tf.nn.relu(self.down_conv1(x))
        x = tf.nn.relu(self.down_conv2(x))
        
        for block in self.res_blocks:
            x = block(x)
        
        x = tf.nn.relu(self.up_conv1(x))
        x = tf.nn.relu(self.up_conv2(x))
        
        return self.output_conv(x)

class ResidualBlock(tf.keras.layers.Layer):
    """Residual block used in Generator"""
    def __init__(self, filters):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters, 3, 1, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(filters, 3, 1, padding='same')
        
    def call(self, x):
        residual = x
        x = tf.nn.relu(self.conv1(x))
        x = self.conv2(x)
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

    @tf.function
    def train_step(
        self,
        real_sar: tf.Tensor,
        real_optical: tf.Tensor,
        sar_mask: tf.Tensor,
        optical_mask: tf.Tensor
    ) -> dict:
        """Execute one training step following paper's Section II.C"""
        with tf.GradientTape(persistent=True) as tape:
            # Forward cycle (SAR → Optical → SAR)
            fake_optical = self.G1([real_sar, sar_mask])  # Ensure inputs are properly processed
            cycled_sar = self.G2([fake_optical, optical_mask])  # Ensure inputs are properly processed

            # Backward cycle (Optical → SAR → Optical)
            fake_sar = self.G2([real_optical, optical_mask])
            cycled_optical = self.G1([fake_sar, sar_mask])
            
            # Discriminator outputs for real and fake images
            disc_real_optical_adv, disc_real_optical_cls = self.D1(real_optical)
            disc_fake_optical_adv, disc_fake_optical_cls = self.D1(fake_optical)
            disc_real_sar_adv, disc_real_sar_cls = self.D2(real_sar)
            disc_fake_sar_adv, disc_fake_sar_cls = self.D2(fake_sar)
            
            # Generator adversarial losses (Eq. 2 in paper)
            gen_loss_G1 = self._adversarial_loss(disc_fake_optical_adv, True)
            gen_loss_G2 = self._adversarial_loss(disc_fake_sar_adv, True)
            
            # Classification losses (Eq. 4 and 5 in paper)
            cls_loss_G1 = self._classification_loss(optical_mask, disc_fake_optical_cls)
            cls_loss_G2 = self._classification_loss(sar_mask, disc_fake_sar_cls)
            
            # Cycle consistency loss (Eq. 3 in paper)
            cycle_loss = (
                tf.reduce_mean(tf.abs(real_sar - cycled_sar)) +
                tf.reduce_mean(tf.abs(real_optical - cycled_optical))
            ) * self.lambda_cyc
            
            # Total generator loss (Eq. 7 in paper)
            total_gen_loss = (
                gen_loss_G1 + gen_loss_G2 +
                cycle_loss +
                (cls_loss_G1 + cls_loss_G2) * self.lambda_cls
            )
            
            # Discriminator losses (Eq. 6 in paper)
            disc_loss_real_optical = self._adversarial_loss(disc_real_optical_adv, True)
            disc_loss_fake_optical = self._adversarial_loss(disc_fake_optical_adv, False)
            disc_loss_real_sar = self._adversarial_loss(disc_real_sar_adv, True)
            disc_loss_fake_sar = self._adversarial_loss(disc_fake_sar_adv, False)
            
            cls_loss_D1 = self._classification_loss(optical_mask, disc_real_optical_cls)
            cls_loss_D2 = self._classification_loss(sar_mask, disc_real_sar_cls)
            
            total_disc_loss = (
                disc_loss_real_optical + disc_loss_fake_optical +
                disc_loss_real_sar + disc_loss_fake_sar +
                (cls_loss_D1 + cls_loss_D2) * self.lambda_cls
            ) * 0.5

        # Calculate gradients
        gen_gradients = tape.gradient(
            total_gen_loss,
            self.G1.trainable_variables + self.G2.trainable_variables
        )
        disc_gradients = tape.gradient(
            total_disc_loss,
            self.D1.trainable_variables + self.D2.trainable_variables
        )
        
        # Apply gradients
        self.gen_optimizer.apply_gradients(zip(
            gen_gradients,
            self.G1.trainable_variables + self.G2.trainable_variables
        ))
        self.disc_optimizer.apply_gradients(zip(
            disc_gradients,
            self.D1.trainable_variables + self.D2.trainable_variables
        ))
        
        del tape
        
        # Update metrics
        self.gen_loss_metric.update_state(total_gen_loss)
        self.disc_loss_metric.update_state(total_disc_loss)
        self.cycle_loss_metric.update_state(cycle_loss)
        self.cls_loss_metric.update_state(cls_loss_G1 + cls_loss_G2)
        
        return {
            'gen_loss': total_gen_loss,
            'disc_loss': total_disc_loss,
            'cycle_loss': cycle_loss,
            'cls_loss': cls_loss_G1 + cls_loss_G2,
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

    def _classification_loss(
        self,
        labels: tf.Tensor,
        pred_cls: tf.Tensor
    ) -> tf.Tensor:
        """Calculate classification loss as described in paper Section II.C.3"""
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
            labels, pred_cls
        ))

class MultiDomainDiscriminator(tf.keras.Model):
    """Discriminator architecture as shown in Fig. 3 of paper"""
    def __init__(self, num_classes):
        super().__init__()
        
        # Common layers
        self.conv1 = tf.keras.layers.Conv2D(64, 4, 2, padding='same')
        self.conv2 = tf.keras.layers.Conv2D(128, 4, 2, padding='same')
        self.conv3 = tf.keras.layers.Conv2D(256, 4, 2, padding='same')
        self.conv4 = tf.keras.layers.Conv2D(512, 4, 2, padding='same')
        
        # Adversarial branch
        self.adv_conv = tf.keras.layers.Conv2D(1, 4, 1, padding='same')
        
        # Classification branch
        self.cls_conv = tf.keras.layers.Conv2D(num_classes, 4, 1, padding='same')
        self.global_avg = tf.keras.layers.GlobalAveragePooling2D()
        self.softmax = tf.keras.layers.Softmax()

    def call(self, x):
        x = tf.nn.leaky_relu(self.conv1(x))
        x = tf.nn.leaky_relu(self.conv2(x))
        x = tf.nn.leaky_relu(self.conv3(x))
        x = tf.nn.leaky_relu(self.conv4(x))
        
        # Adversarial output
        adv_out = self.adv_conv(x)
        
        # Classification output
        cls_out = self.cls_conv(x)
        cls_out = self.global_avg(cls_out)
        cls_out = self.softmax(cls_out)
        
        return adv_out, cls_out
    

class Trainer:
    def __init__(
        self,
        model: CycleGAN,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        output_dir: str,
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

        # Initialize metrics
        self.gen_loss_metric = tf.keras.metrics.Mean()
        self.disc_loss_metric = tf.keras.metrics.Mean()
        self.cycle_loss_metric = tf.keras.metrics.Mean()
        self.cls_loss_metric = tf.keras.metrics.Mean()
        self.domain_cls_loss_metric = tf.keras.metrics.Mean()

        os.makedirs(output_dir, exist_ok=True)

    @tf.function
    def _train_step(self, real_sar, real_optical, sar_mask, optical_mask):
        """Single training step implementing MC-GAN methodology"""
        with tf.GradientTape(persistent=True) as tape:
            # Generate fake images

            fake_optical = self.model.G1([real_sar, optical_mask])  # SAR to Optical
            fake_sar = self.model.G2([real_optical, sar_mask])      # Optical to SAR
            
            # Cycle consistency
            cycled_sar = self.model.G2([fake_optical, sar_mask])    # Optical back to SAR
            cycled_optical = self.model.G1([fake_sar, optical_mask]) # SAR back to Optical

            # Discriminator outputs for real/fake classification
            disc_real_optical_adv, disc_real_optical_cls = self.model.D1(real_optical)
            disc_fake_optical_adv, disc_fake_optical_cls = self.model.D1(fake_optical)
            disc_real_sar_adv, disc_real_sar_cls = self.model.D2(real_sar)
            disc_fake_sar_adv, disc_fake_sar_cls = self.model.D2(fake_sar)

            # Calculate losses according to paper
            # Adversarial losses
            gen_loss_optical = tf.reduce_mean(tf.square(disc_fake_optical_adv - 1))
            gen_loss_sar = tf.reduce_mean(tf.square(disc_fake_sar_adv - 1))
            disc_loss_optical = tf.reduce_mean(tf.square(disc_real_optical_adv - 1) + tf.square(disc_fake_optical_adv))
            disc_loss_sar = tf.reduce_mean(tf.square(disc_real_sar_adv - 1) + tf.square(disc_fake_sar_adv))

            # Cycle consistency losses
            cycle_loss_sar = tf.reduce_mean(tf.abs(real_sar - cycled_sar))
            cycle_loss_optical = tf.reduce_mean(tf.abs(real_optical - cycled_optical))
            cycle_loss = cycle_loss_sar + cycle_loss_optical

            # Domain classification losses for real images
            real_cls_loss_optical = self._domain_classification_loss(disc_real_optical_cls, optical_mask)
            real_cls_loss_sar = self._domain_classification_loss(disc_real_sar_cls, sar_mask)
            
            # Domain classification losses for fake images
            fake_cls_loss_optical = self._domain_classification_loss(disc_fake_optical_cls, optical_mask)
            fake_cls_loss_sar = self._domain_classification_loss(disc_fake_sar_cls, sar_mask)

            # Total losses
            gen_total_loss = (gen_loss_optical + gen_loss_sar + 
                            self.lambda_cyc * cycle_loss +
                            self.lambda_cls * (fake_cls_loss_optical + fake_cls_loss_sar))
                            
            disc_total_loss = (disc_loss_optical + disc_loss_sar + 
                             self.lambda_cls * (real_cls_loss_optical + real_cls_loss_sar))

        # Update generators
        gen_gradients = tape.gradient(gen_total_loss, 
                                    self.model.G1.trainable_variables + 
                                    self.model.G2.trainable_variables)
        self.model.gen_optimizer.apply_gradients(
            zip(gen_gradients, 
                self.model.G1.trainable_variables + 
                self.model.G2.trainable_variables))

        # Update discriminators
        disc_gradients = tape.gradient(disc_total_loss,
                                     self.model.D1.trainable_variables + 
                                     self.model.D2.trainable_variables)
        self.model.disc_optimizer.apply_gradients(
            zip(disc_gradients,
                self.model.D1.trainable_variables + 
                self.model.D2.trainable_variables))

        # Update metrics
        self.gen_loss_metric.update_state(gen_total_loss)
        self.disc_loss_metric.update_state(disc_total_loss)
        self.cycle_loss_metric.update_state(cycle_loss)
        self.cls_loss_metric.update_state(fake_cls_loss_optical + fake_cls_loss_sar)
        self.domain_cls_loss_metric.update_state(real_cls_loss_optical + real_cls_loss_sar)

        return {
            'generated_images': [fake_optical, fake_sar, cycled_optical, cycled_sar],
            'gen_loss': gen_total_loss,
            'disc_loss': disc_total_loss,
            'cycle_loss': cycle_loss
        }

    def _domain_classification_loss(self, disc_output, target_domain):
        target_domain = tf.reduce_max(target_domain, axis=[1, 2])
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
            target_domain, disc_output
        ))

    def train(self):
        """Execute training loop following MC-GAN methodology"""
        for epoch in range(self.initial_epoch, self.epochs):
            start_time = time.time()
            
            # Training loop
            for (sar_img, sar_mask), (opt_img, opt_mask) in self.train_ds:
                metrics = self._train_step(sar_img, opt_img, sar_mask, opt_mask)
                
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} - "
                f"Gen Loss: {self.gen_loss_metric.result():.4f}, "
                f"Disc Loss: {self.disc_loss_metric.result():.4f}, "
                f"Cycle Loss: {self.cycle_loss_metric.result():.4f}, "
                f"Cls Loss: {self.cls_loss_metric.result():.4f}, "
                f"Domain Cls Loss: {self.domain_cls_loss_metric.result():.4f}"
            )

            # Save sample images periodically
            if (epoch + 1) % 5 == 0:
                self._save_samples(epoch)

            # Reset metrics
            self.gen_loss_metric.reset_states()
            self.disc_loss_metric.reset_states()
            self.cycle_loss_metric.reset_states()
            self.cls_loss_metric.reset_states()
            self.domain_cls_loss_metric.reset_states()

            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)

    def _save_samples(self, epoch):
        """Save sample images during training"""
        sample_batch = next(iter(self.val_ds))
        (sar_img, sar_mask), (opt_img, opt_mask) = sample_batch
        metrics = self._train_step(sar_img, opt_img, sar_mask, opt_mask)
        
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

def main():
    """
    Main function to train the MC-GAN model for SAR image colorization.
    Aligns with the methodologies and configurations from the referenced paper.
    """
    # Configuration parameters based on the paper
    config = {
        'image_size': 256,  # Image size for resizing as per Section II.D
        'batch_size': 1,    # Batch size as per Section II.D
        'epochs': 200,      # Number of epochs for training
        'dataset_dir': './Dataset',  # Path to the dataset directory
        'output_dir': './output_updatedSAR',   # Directory to save outputs and checkpoints
        'learning_rate': 0.0002,    # Learning rate as specified in Section II.D
        'lambda_cyc': 10.0,         # Weight for cycle-consistency loss (Section II.C.4)
        'lambda_cls': 1.0,          # Weight for multidomain classification loss (Section II.C.4)
        'validation_split': 0.1,    # 10% for validation (Section II.D)
    }

    # Save configuration to output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Configure GPU settings
    GPUConfig.configure()

    # Initialize the data loader
    data_loader = DataLoader(
        dataset_dir='./Dataset',
        image_size=256,
        batch_size=1,
    )
    train_ds, val_ds = data_loader.load_data()

    # Initialize the MC-GAN model
    model = CycleGAN(
        image_size=config['image_size'],
        num_classes=data_loader.num_terrains + 1,  # Include SAR domain
        lambda_cyc=config['lambda_cyc'],
        lambda_cls=config['lambda_cls'],
        learning_rate=config['learning_rate']
    )

    # Log total trainable parameters
    total_params = sum(
        np.prod(v.get_shape().as_list())
        for v in model.G1.trainable_variables + model.G2.trainable_variables +
                  model.D1.trainable_variables + model.D2.trainable_variables
    )
    logger.info(f"Total trainable parameters: {total_params:,}")

    # Compile the model
    model.compile()

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        train_ds=train_ds,
        val_ds=val_ds,
        output_dir=config['output_dir'],
        epochs=config['epochs']
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()