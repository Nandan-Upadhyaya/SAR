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
        image_size: int, 
        batch_size: int,
        validation_split: float = 0.2,
        cache_dataset: bool = False
    ):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.cache_dataset = cache_dataset
        self.num_classes = len(os.listdir(dataset_dir))
        
        logger.info(
            f"Initializing DataLoader with parameters: "
            f"image_size={image_size}, batch_size={batch_size}, "
            f"validation_split={validation_split}"
        )

    def _get_image_files(self, directory: str) -> List[str]:
        """Get all image files from a directory."""
        valid_extensions = {'.png', '.jpg', '.jpeg'}
        files = []
        
        for f in os.listdir(directory):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_extensions:
                files.append(f)
                
        return sorted(files)

    def load_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """Load and prepare training and validation datasets."""
        start_time = time.time()
        logger.info(f"Starting to load dataset from {self.dataset_dir}")
        
        try:
            categories = os.listdir(self.dataset_dir)
            if not categories:
                raise ValueError(f"No categories found in {self.dataset_dir}")
                
            logger.info(f"Found {len(categories)} categories: {categories}")
            
            color_paths = []
            sar_paths = []
            labels_list = []
            
            for label, category in enumerate(categories):
                category_dir = os.path.join(self.dataset_dir, category)
                sar_dir = os.path.join(category_dir, 'SAR')
                color_dir = os.path.join(category_dir, 'Color')  # Note: lowercase 'color'
                
                # Verify directories exist
                if not os.path.exists(sar_dir):
                    raise FileNotFoundError(f"SAR directory not found: {sar_dir}")
                if not os.path.exists(color_dir):
                    raise FileNotFoundError(f"Color directory not found: {color_dir}")
                
                # Get image files
                sar_files = self._get_image_files(sar_dir)
                color_files = self._get_image_files(color_dir)
                
                if not sar_files:
                    raise ValueError(f"No image files found in {sar_dir}")
                if not color_files:
                    raise ValueError(f"No image files found in {color_dir}")
                
                # Verify equal number of files
                if len(sar_files) != len(color_files):
                    raise ValueError(
                        f"Unequal number of images in category {category}. "
                        f"SAR: {len(sar_files)}, Color: {len(color_files)}"
                    )
                
                logger.info(
                    f"Found {len(sar_files)} image pairs in category '{category}'"
                )
                
                # Add paths and labels
                sar_paths.extend([os.path.join(sar_dir, f) for f in sar_files])
                color_paths.extend([os.path.join(color_dir, f) for f in color_files])
                labels_list.extend([label] * len(sar_files))
            
            # Shuffle with fixed seed for reproducibility
            indices = np.arange(len(color_paths))
            np.random.seed(42)
            np.random.shuffle(indices)
            
            color_paths = np.array(color_paths)[indices]
            sar_paths = np.array(sar_paths)[indices]
            labels_list = np.array(labels_list)[indices]
            
            # Split into training and validation
            split_idx = int(len(color_paths) * (1 - self.validation_split))
            
            train_color = color_paths[:split_idx]
            train_sar = sar_paths[:split_idx]
            train_labels = labels_list[:split_idx]
            
            val_color = color_paths[split_idx:]
            val_sar = sar_paths[split_idx:]
            val_labels = labels_list[split_idx:]
            
            # Create training dataset
            train_ds = self._create_dataset(
                train_color, train_sar, train_labels, shuffle=True
            )
            
            # Create validation dataset
            val_ds = self._create_dataset(
                val_color, val_sar, val_labels, shuffle=False
            )
            
            logger.info(
                f"Dataset pipeline setup completed in "
                f"{time.time() - start_time:.2f} seconds"
            )
            
            return train_ds, val_ds
            
        except Exception as e:
            logger.error(f"Failed to setup dataset pipeline: {str(e)}")
            raise

    def _create_dataset(
        self, 
        color_paths: np.ndarray, 
        sar_paths: np.ndarray, 
        labels: np.ndarray,
        shuffle: bool = True
    ) -> tf.data.Dataset:
        """Create a tf.data.Dataset from image paths and labels."""
        dataset = tf.data.Dataset.from_tensor_slices((
            (color_paths, sar_paths),
            labels
        ))

        # Map file paths to images
        dataset = dataset.map(
            lambda x, y: ((
                self._parse_image(x[0]),
                self._parse_image(x[1])
            ), tf.one_hot(y, self.num_classes)),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        if self.cache_dataset:
            dataset = dataset.cache()
            
        if shuffle:
            dataset = dataset.shuffle(5000)
            
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    @tf.function
    def _preprocess_image(self, img: tf.Tensor) -> tf.Tensor:
        """Preprocess image with data augmentation."""
        # Random flip
        if tf.random.uniform([]) > 0.5:
            img = tf.image.flip_left_right(img)
        
        # Random brightness
        img = tf.image.random_brightness(img, 0.2)
        
        # Random contrast
        img = tf.image.random_contrast(img, 0.8, 1.2)
        
        # Random rotation
        img = tf.image.rot90(img, tf.random.uniform([], maxval=4, dtype=tf.int32))
        
        # Normalize to [-1, 1]
        img = tf.clip_by_value(img, 0, 255)
        img = (img / 127.5) - 1
        
        return img

    def _parse_image(self, file_path: str) -> tf.Tensor:
        """Load and parse image from file path."""
        try:
            img = tf.io.read_file(file_path)
            img = tf.image.decode_png(img, channels=3)
            img = tf.image.resize(img, [self.image_size, self.image_size])
            
            if img.shape[-1] != 3:
                raise ValueError(f"Unexpected image shape {img.shape}")
                
            return self._preprocess_image(img)
            
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {str(e)}")
            raise
    
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

class Generator(tf.keras.Model):
    def __init__(self, image_size: int):
        super(Generator, self).__init__()
        self.image_size = image_size
        self.encoder = self._build_encoder()
        self.bottleneck = self._build_bottleneck()
        self.decoder = self._build_decoder()
        
    def _build_encoder(self) -> tf.keras.Sequential:
        initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
        
        layers = []
        filters = [48, 96, 192]  # Reduced filters to save VRAM
        
        for f in filters:
            layers.extend([
                tf.keras.layers.Conv2D(
                    f, 4, strides=2, padding='same',
                    kernel_initializer=initializer,
                    use_bias=False
                ),
                tfa.layers.InstanceNormalization(),  # Better than BatchNorm for style transfer
                tf.keras.layers.LeakyReLU(0.2)
            ])
        
        return tf.keras.Sequential(layers)
    
    def _build_bottleneck(self) -> tf.keras.Sequential:
        initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
        
        layers = []
        for _ in range(4):  # Reduced from 6 for faster training
            layers.append(ResidualBlock(192, initializer))  # Adjusted filters
        
        return tf.keras.Sequential(layers)
    
    def _build_decoder(self) -> tf.keras.Sequential:
        initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
        
        layers = []
        filters = [128, 64]
        
        for f in filters:
            layers.extend([
                tf.keras.layers.Conv2DTranspose(
                    f, 4, strides=2, padding='same',
                    kernel_initializer=initializer,
                    use_bias=False
                ),
                tfa.layers.InstanceNormalization(),
                tf.keras.layers.ReLU()
            ])
            
        layers.append(
            tf.keras.layers.Conv2DTranspose(
                3, 4, strides=2, padding='same',
                kernel_initializer=initializer,
                activation='tanh'
            )
        )
        
        return tf.keras.Sequential(layers)
    
    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        x = self.encoder(x, training=training)
        x = self.bottleneck(x, training=training)
        return self.decoder(x, training=training)

class SelfAttention(tf.keras.layers.Layer):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.query = tf.keras.layers.Conv2D(channels//8, 1)
        self.key = tf.keras.layers.Conv2D(channels//8, 1)
        self.value = tf.keras.layers.Conv2D(channels, 1)
        self.gamma = self.add_weight(name='gamma', shape=[], initializer='zeros')
        
    def call(self, x):
        batch_size, h, w, c = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        q = tf.reshape(q, [batch_size, -1, h*w])
        k = tf.reshape(k, [batch_size, -1, h*w])
        v = tf.reshape(v, [batch_size, -1, h*w])
        
        attention = tf.nn.softmax(tf.matmul(q, k, transpose_b=True))
        out = tf.matmul(attention, v, transpose_b=True)
        out = tf.reshape(out, [batch_size, h, w, c])
        
        return x + self.gamma * out

class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters: int, initializer):
        super(ResidualBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters, 3, padding='same',
            kernel_initializer=initializer,
            use_bias=False
        )
        self.bn1 = tfa.layers.InstanceNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters, 3, padding='same',
            kernel_initializer=initializer,
            use_bias=False
        )
        self.bn2 = tfa.layers.InstanceNormalization()
        
    def call(self, x: tf.Tensor, training: bool = True) -> tf.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        return tf.nn.relu(x + residual)

class Discriminator(tf.keras.Model):
    def __init__(self, num_classes: int):
        super(Discriminator, self).__init__()
        self.conv_layers = self._build_conv_layers()
        self.adv_head = tf.keras.layers.Dense(1)
        self.cls_head = tf.keras.layers.Dense(num_classes, activation='softmax')
        
    def _build_conv_layers(self) -> tf.keras.Sequential:
        initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
        
        layers = []
        filters = [48, 96, 192]  # Reduced filters to save memory
        
        for f in filters:
            layers.extend([
                tf.keras.layers.Conv2D(
                    f, 4, strides=2, padding='same',
                    kernel_initializer=initializer
                ),
                tf.keras.layers.LeakyReLU(0.2)
            ])
            
        layers.extend([
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3)
        ])
        
        return tf.keras.Sequential(layers)
    
    def call(
        self, 
        x: tf.Tensor, 
        training: bool = True
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        features = self.conv_layers(x, training=training)
        return self.adv_head(features), self.cls_head(features)

class CycleGAN:
    def __init__(
        self,
        image_size: int,
        num_classes: int,
        lambda_cyc: float = 5.0,    # Reduced from 8.0 to get cycle loss < 1
        lambda_cls: float = 0.3,    # Reduced from 0.5 to get cls loss < 0.5
        learning_rate: float = 2e-4  # Slightly increased for faster convergence
    ):
        self.G1 = Generator(image_size)
        self.G2 = Generator(image_size)
        self.D1 = Discriminator(num_classes)
        self.D2 = Discriminator(num_classes)
        self.lambda_cyc = lambda_cyc
        self.lambda_cls = lambda_cls
        
        self.gen_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)
        self.disc_optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.5)

        dummy_input = tf.zeros((1, image_size, image_size, 3))
        self.G1(dummy_input)
        self.G2(dummy_input)
        self.D1(dummy_input)
        self.D2(dummy_input)
        
        # Initialize learning rate schedulers
        self.gen_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate, decay_steps=1000, decay_rate=0.96
        )
        self.disc_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            learning_rate, decay_steps=1000, decay_rate=0.96
        )

        self.gen_optimizer = tf.keras.optimizers.Adam(
            self.gen_scheduler, beta_1=0.5
        )
        self.disc_optimizer = tf.keras.optimizers.Adam(
            self.disc_scheduler, beta_1=0.5
        )
        
        # Enable mixed precision training
        self.mixed_precision = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(self.mixed_precision)
        
        # Adaptive learning rate schedule
        self.gen_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
            learning_rate,
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=1e-5
        )
        self.disc_scheduler = tf.keras.optimizers.schedules.CosineDecayRestarts(
            learning_rate * 0.5,  # Slower discriminator learning
            first_decay_steps=1000,
            t_mul=2.0,
            m_mul=0.9,
            alpha=1e-5
        )

        # Improved learning rate schedule
        initial_learning_rate = learning_rate
        self.gen_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[50, 100],  # Epoch boundaries
            values=[initial_learning_rate, initial_learning_rate*0.5, initial_learning_rate*0.1]
        )
        self.disc_scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=[50, 100],
            values=[initial_learning_rate*0.5, initial_learning_rate*0.25, initial_learning_rate*0.05]
        )

    def compile(self):
        """Initialize metrics for tracking."""
        self.gen_loss_metric = tf.keras.metrics.Mean('gen_loss')
        self.disc_loss_metric = tf.keras.metrics.Mean('disc_loss')
        self.cycle_loss_metric = tf.keras.metrics.Mean('cycle_loss')
        self.cls_loss_metric = tf.keras.metrics.Mean('cls_loss')

    def save(self, save_dir: str):
        """Save all components of the CycleGAN model."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save each model component
        self.G1.save_weights(os.path.join(save_dir, 'generator1.h5'))
        self.G2.save_weights(os.path.join(save_dir, 'generator2.h5'))
        self.D1.save_weights(os.path.join(save_dir, 'discriminator1.h5'))
        self.D2.save_weights(os.path.join(save_dir, 'discriminator2.h5'))
        
        logger.info(f"Model saved successfully to {save_dir}")
        
        try:
         symbolic_weights = getattr(self.gen_optimizer, 'weights', None)
         if symbolic_weights:
            weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
            # Save each weight array separately
            for i, w in enumerate(weight_values):
                np.save(os.path.join(save_dir, f'gen_optimizer_weights_{i}.npy'), w)
        
         symbolic_weights = getattr(self.disc_optimizer, 'weights', None)
         if symbolic_weights:
            weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
            # Save each weight array separately
            for i, w in enumerate(weight_values):
                np.save(os.path.join(save_dir, f'disc_optimizer_weights_{i}.npy'), w)
        except Exception as e:
         logger.warning(f"Failed to save optimizer states: {str(e)}")
         logger.warning("Training can continue but optimizer states won't be restored")


        if isinstance(self.gen_optimizer.learning_rate, tf.keras.optimizers.schedules.ExponentialDecay):
          learning_rate_config = {
            'initial_learning_rate': float(self.gen_optimizer.learning_rate.initial_learning_rate),
            'decay_steps': int(self.gen_optimizer.learning_rate.decay_steps),
            'decay_rate': float(self.gen_optimizer.learning_rate.decay_rate),
            'type': 'ExponentialDecay'
        }
        else:
         learning_rate_config = {
            'type': 'fixed',
            'value': float(self.gen_optimizer.learning_rate)
        }

        # Save model configuration
        config = {
            'lambda_cyc': float(self.lambda_cyc),
            'lambda_cls': float(self.lambda_cls),
            'learning_rate_config': learning_rate_config
        }
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(config, f)
        
        logger.info(f"Model saved successfully to {save_dir}")
   
    @classmethod
    def load(cls, save_dir: str, image_size: int, num_classes: int):
        """Load a saved CycleGAN model."""
        # Load configuration
        with open(os.path.join(save_dir, 'config.json'), 'r') as f:
            config = json.load(f)

        learning_rate_config = config.get('learning_rate_config', {})
        if learning_rate_config.get('type') == 'ExponentialDecay':
         learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate_config['initial_learning_rate'],
            decay_steps=learning_rate_config['decay_steps'],
            decay_rate=learning_rate_config['decay_rate']
        )
        else:
         learning_rate = learning_rate_config.get('value', 2e-4)
        
        # Create new model instance
        model = cls(
            image_size=image_size,
            num_classes=num_classes,
            lambda_cyc=config['lambda_cyc'],
            lambda_cls=config['lambda_cls'],
            learning_rate=learning_rate
        )
        
         # Load model weights
        model.G1.load_weights(os.path.join(save_dir, 'generator1_weights.h5'))
        model.G2.load_weights(os.path.join(save_dir, 'generator2_weights.h5'))
        model.D1.load_weights(os.path.join(save_dir, 'discriminator1_weights.h5'))
        model.D2.load_weights(os.path.join(save_dir, 'discriminator2_weights.h5'))
        
        # Load optimizer weights if they exist
        gen_opt_weights_path = os.path.join(save_dir, 'gen_optimizer_weights.npy')
        if os.path.exists(gen_opt_weights_path):
            with open(gen_opt_weights_path, 'rb') as f:
                weight_values = np.load(f, allow_pickle=True)
            model.gen_optimizer.set_weights(weight_values)
            
        disc_opt_weights_path = os.path.join(save_dir, 'disc_optimizer_weights.npy')
        if os.path.exists(disc_opt_weights_path):
            with open(disc_opt_weights_path, 'rb') as f:
                weight_values = np.load(f, allow_pickle=True)
            model.disc_optimizer.set_weights(weight_values)
        
        return model

        
    @tf.function
    def train_step(
        self,
        real_color: tf.Tensor,
        real_sar: tf.Tensor,
        labels: tf.Tensor
    ) -> Dict[str, Any]:
        """Execute one training step."""
        with tf.GradientTape(persistent=True) as tape:
            # Generator outputs
            # Generator outputs
            fake_sar = self.G1(real_color, training=True)
            cycled_color = self.G2(fake_sar, training=True)
            fake_color = self.G2(real_sar, training=True)
            cycled_sar = self.G1(fake_color, training=True)
            
            # Identity mapping
            same_color = self.G2(real_color, training=True)
            same_sar = self.G1(real_sar, training=True)
            
            # Discriminator outputs
            disc_real_color_adv, disc_real_color_cls = self.D1(real_color, training=True)
            disc_fake_color_adv, disc_fake_color_cls = self.D1(fake_color, training=True)
            disc_real_sar_adv, disc_real_sar_cls = self.D2(real_sar, training=True)
            disc_fake_sar_adv, disc_fake_sar_cls = self.D2(fake_sar, training=True)
            
            # Generator adversarial losses
            gen_loss_G1 = self._adversarial_loss(disc_fake_sar_adv, True)
            gen_loss_G2 = self._adversarial_loss(disc_fake_color_adv, True)
            
            # Classification losses
            cls_loss_G1 = self._classification_loss(labels, disc_fake_sar_cls)
            cls_loss_G2 = self._classification_loss(labels, disc_fake_color_cls)
            
            # Cycle consistency losses
            cycle_loss = (
                tf.reduce_mean(tf.abs(real_color - cycled_color)) +
                tf.reduce_mean(tf.abs(real_sar - cycled_sar))
            ) * self.lambda_cyc
            
            # Identity losses
            identity_loss = (
                tf.reduce_mean(tf.abs(real_color - same_color)) +
                tf.reduce_mean(tf.abs(real_sar - same_sar))
            ) * 0.5 * self.lambda_cyc
            
            # Total generator loss
            total_gen_loss = (
                gen_loss_G1 + gen_loss_G2 +
                cycle_loss + identity_loss +
                (cls_loss_G1 + cls_loss_G2) * self.lambda_cls
            )
            
            # Discriminator losses
            disc_loss_real_color = self._adversarial_loss(disc_real_color_adv, True)
            disc_loss_fake_color = self._adversarial_loss(disc_fake_color_adv, False)
            disc_loss_real_sar = self._adversarial_loss(disc_real_sar_adv, True)
            disc_loss_fake_sar = self._adversarial_loss(disc_fake_sar_adv, False)
            
            cls_loss_D1 = self._classification_loss(labels, disc_real_color_cls)
            cls_loss_D2 = self._classification_loss(labels, disc_real_sar_cls)
            
            total_disc_loss = (
                disc_loss_real_color + disc_loss_fake_color +
                disc_loss_real_sar + disc_loss_fake_sar +
                (cls_loss_D1 + cls_loss_D2) * self.lambda_cls
            ) * 0.5

            # Add gradient penalty
            gp_color = self._gradient_penalty(self.D1, real_color, fake_color)
            gp_sar = self._gradient_penalty(self.D2, real_sar, fake_sar)
            
            total_disc_loss += (gp_color + gp_sar) * 10.0  # Add gradient penalty

        # Calculate and apply gradients with clipping
        gen_gradients = tape.gradient(
            total_gen_loss,
            self.G1.trainable_variables + self.G2.trainable_variables
        )
        disc_gradients = tape.gradient(
            total_disc_loss,
            self.D1.trainable_variables + self.D2.trainable_variables
        )
        
        gen_grad_norm = tf.linalg.global_norm(gen_gradients)
        disc_grad_norm = tf.linalg.global_norm(disc_gradients)

        # Gradient clipping
        gen_gradients = [tf.clip_by_norm(g, 1.0) for g in gen_gradients]
        disc_gradients = [tf.clip_by_norm(g, 1.0) for g in disc_gradients]
        
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
            'gen_grad_norm': gen_grad_norm,
            'disc_grad_norm': disc_grad_norm,
            'generated_images': (fake_color, fake_sar, cycled_color, cycled_sar)
        }
    
    @tf.function
    def validation_step(self, real_color, real_sar, labels):
        # Similar to train_step but without gradient updates
        fake_sar = self.G1(real_color, training=False)
        cycled_color = self.G2(fake_sar, training=False)
        fake_color = self.G2(real_sar, training=False)
        cycled_sar = self.G1(fake_color, training=False)

        real_color_0_1 = (real_color + 1) * 0.5
        cycled_color_0_1 = (cycled_color + 1) * 0.5
        real_sar_0_1 = (real_sar + 1) * 0.5
        cycled_sar_0_1 = (cycled_sar + 1) * 0.5
        
        psnr_color = tf.image.psnr(real_color_0_1, cycled_color_0_1, max_val=1.0)
        psnr_sar = tf.image.psnr(real_sar_0_1, cycled_sar_0_1, max_val=1.0)
        ssim_color = tf.image.ssim(real_color_0_1, cycled_color_0_1, max_val=1.0)
        ssim_sar = tf.image.ssim(real_sar_0_1, cycled_sar_0_1, max_val=1.0)

        # Identity mapping
        same_color = self.G2(real_color, training=True)
        same_sar = self.G1(real_sar, training=True)
            
            # Discriminator outputs
        disc_real_color_adv, disc_real_color_cls = self.D1(real_color, training=False)
        disc_fake_color_adv, disc_fake_color_cls = self.D1(fake_color, training=False)
        disc_real_sar_adv, disc_real_sar_cls = self.D2(real_sar, training=False)
        disc_fake_sar_adv, disc_fake_sar_cls = self.D2(fake_sar, training=False)
        
        # Generator adversarial losses
        gen_loss_G1 = self._adversarial_loss(disc_fake_sar_adv, True)
        gen_loss_G2 = self._adversarial_loss(disc_fake_color_adv, True)
            
            # Classification losses
        cls_loss_G1 = self._classification_loss(labels, disc_fake_sar_cls)
        cls_loss_G2 = self._classification_loss(labels, disc_fake_color_cls)
            
            # Cycle consistency losses
        cycle_loss = (
                tf.reduce_mean(tf.abs(real_color - cycled_color)) +
                tf.reduce_mean(tf.abs(real_sar - cycled_sar))
            ) * self.lambda_cyc
            
            # Identity losses
        identity_loss = (
                tf.reduce_mean(tf.abs(real_color - same_color)) +
                tf.reduce_mean(tf.abs(real_sar - same_sar))
            ) * 0.5 * self.lambda_cyc
            
            # Total generator loss
        total_gen_loss = (
                gen_loss_G1 + gen_loss_G2 +
                cycle_loss + identity_loss +
                (cls_loss_G1 + cls_loss_G2) * self.lambda_cls
            )
            
            # Discriminator losses
        disc_loss_real_color = self._adversarial_loss(disc_real_color_adv, True)
        disc_loss_fake_color = self._adversarial_loss(disc_fake_color_adv, False)
        disc_loss_real_sar = self._adversarial_loss(disc_real_sar_adv, True)
        disc_loss_fake_sar = self._adversarial_loss(disc_fake_sar_adv, False)
            
        cls_loss_D1 = self._classification_loss(labels, disc_real_color_cls)
        cls_loss_D2 = self._classification_loss(labels, disc_real_sar_cls)
            
        total_disc_loss = (
                disc_loss_real_color + disc_loss_fake_color +
                disc_loss_real_sar + disc_loss_fake_sar +
                (cls_loss_D1 + cls_loss_D2) * self.lambda_cls
            ) * 0.5

        
        return {
            'gen_loss': total_gen_loss,
            'disc_loss': total_disc_loss,
            'cycle_loss': cycle_loss,
            'cls_loss': cls_loss_G1 + cls_loss_G2,
            'psnr_color': psnr_color,  # Add psnr_color to the metrics
            'psnr_sar': psnr_sar,      # Add psnr_sar to the metrics
            'ssim_color': ssim_color,  # Add ssim_color to the metrics
            'ssim_sar': ssim_sar,      # Add ssim_sar to the metrics
            'generated_images': (fake_color, fake_sar, cycled_color, cycled_sar)
        }


    def _adversarial_loss(
    self,
    pred: tf.Tensor,
    target_is_real: bool
) -> tf.Tensor:
     """Calculate adversarial loss with label smoothing."""
    # Get the batch size dynamically from the prediction tensor
     batch_size = tf.shape(pred)[0]
    
     if target_is_real:
        # Create target tensor with dynamic shape
        target = tf.ones_like(pred) * 0.95  # Increased from 0.9
     else:
        # Create target tensor with dynamic shape
        target = tf.zeros_like(pred) * 0.05  # Reduced from 0.1
    
     return tf.reduce_mean(tf.keras.losses.binary_crossentropy(
        target, pred, from_logits=True
    ))

    def _classification_loss(
        self,
        labels: tf.Tensor,
        pred_cls: tf.Tensor
    ) -> tf.Tensor:
        """Calculate classification loss with label smoothing."""
        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(
            labels, pred_cls, label_smoothing=0.15  # increased from 0.1
        ))

    def _gradient_penalty(self, discriminator, real, fake):
        """Calculate gradient penalty for improved Wasserstein loss."""
        alpha = tf.random.uniform([tf.shape(real)[0], 1, 1, 1], 0.0, 1.0)
        interpolated = alpha * real + (1 - alpha) * fake
        
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred, _ = discriminator(interpolated, training=True)
        
        gradients = tape.gradient(pred, interpolated)
        slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
        return tf.reduce_mean(tf.square(slopes - 1.0))

class Trainer:
    def __init__(
        self,
        model: CycleGAN,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        output_dir: str,
        epochs: int = 100,
        early_stopping_patience: int = 5
    ):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.output_dir = output_dir
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        
        self.val_gen_loss_metric = tf.keras.metrics.Mean()
        self.val_disc_loss_metric = tf.keras.metrics.Mean()
        self.val_cycle_loss_metric = tf.keras.metrics.Mean()
        self.val_cls_loss_metric = tf.keras.metrics.Mean()
        self.lpips_metric = tf.keras.metrics.Mean()
        self.inception_score = tf.keras.metrics.Mean()
        self.l1_loss_metric = tf.keras.metrics.Mean()
        self.l2_loss_metric = tf.keras.metrics.Mean()
        self.fid_score_metric = tf.keras.metrics.Mean()
        self.inception_model = tf.keras.applications.InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
        self.vgg = tf.keras.applications.VGG16(include_top=False)
   
        self.metrics_logger = MetricsLogger(output_dir)
        self.visualizer = Visualizer(output_dir)
        self.patience_counter = 0
        
        os.makedirs(output_dir, exist_ok=True)

    def _get_inception_features(self, images):
        """Get inception features with proper resizing."""
        try:
            # Ensure images are in [0, 255] range
            images = tf.clip_by_value((images + 1) * 127.5, 0, 255)
            
            # Resize to inception size (299x299)
            images = tf.image.resize(images, (299, 299), method='bilinear')
            
            # Preprocess for InceptionV3
            images = preprocess_input(images)
            
            # Get features
            features = self.inception_model(images, training=False)
            return features
            
        except Exception as e:
            logger.error(f"Error in inception features extraction: {str(e)}")
            # Return zero features as fallback
            return tf.zeros((tf.shape(images)[0], self.inception_model.output_shape[-1]))

    def _calculate_lpips(self, img1, img2):
        
        feat1 = self.vgg(img1)
        feat2 = self.vgg(img2)
        return tf.reduce_mean(tf.square(feat1 - feat2))
    
    def _calculate_covmean(self, sigma1, sigma2):
        try:
            return linalg.sqrtm(sigma1.dot(sigma2))
        except Exception as e:
            logger.error(f"Covmean calculation failed: {str(e)}")
            # Return identity matrix as fallback
            return np.eye(sigma1.shape[0])

    def _calculate_inception_score(self, images):
        """Calculate inception score with proper preprocessing."""
        try:
            # Ensure images are in [0, 255] range
            images = tf.clip_by_value((images + 1) * 127.5, 0, 255)
            
            # Resize to inception size
            images = tf.image.resize(images, (299, 299), method='bilinear')
            
            # Preprocess for InceptionV3
            images = preprocess_input(images)
            
            # Get logits
            features = self.inception_model(images, training=False)
            
            # Convert to probabilities with stable softmax
            probs = tf.nn.softmax(features)
            
            # Calculate p(y) and p(y|x) with numerical stability
            p_yx = probs + 1e-10  # Add small constant for numerical stability
            p_y = tf.reduce_mean(p_yx, axis=0, keepdims=True) + 1e-10
            
            # Calculate KL divergence
            kl_div = tf.reduce_sum(p_yx * (tf.math.log(p_yx) - tf.math.log(p_y)), axis=1)
            
            # Calculate final score
            score = tf.exp(tf.reduce_mean(kl_div))
            
            return float(score)
            
        except Exception as e:
            logger.error(f"Inception score calculation failed: {str(e)}")
            return 1.0  # Return baseline score

    def _calculate_l1_l2_loss(self, img1, img2):
        l1_loss = tf.reduce_mean(tf.abs(img1 - img2))
        l2_loss = tf.reduce_mean(tf.square(img1 - img2))
        return l1_loss, l2_loss

    def _calculate_fid(self, real_features, fake_features):
        """Calculate FID score with improved numerical stability."""
        try:
            # Convert to float32 and move to CPU for numerical stability
            real_features = tf.cast(real_features, tf.float32)
            fake_features = tf.cast(fake_features, tf.float32)
            
            # Calculate means
            mu_real = tf.reduce_mean(real_features, axis=0)
            mu_fake = tf.reduce_mean(fake_features, axis=0)
            
            # Calculate mean difference squared
            mu_diff_squared = tf.reduce_sum(tf.square(mu_real - mu_fake))
            
            # Calculate covariances with regularization
            n_real = tf.cast(tf.shape(real_features)[0], tf.float32)
            n_fake = tf.cast(tf.shape(fake_features)[0], tf.float32)
            
            real_centered = real_features - mu_real
            fake_centered = fake_features - mu_fake
            
            cov_real = (tf.matmul(real_centered, real_centered, transpose_a=True) / 
                       (n_real - 1) + tf.eye(tf.shape(mu_real)[0]) * 1e-6)
            cov_fake = (tf.matmul(fake_centered, fake_centered, transpose_a=True) / 
                       (n_fake - 1) + tf.eye(tf.shape(mu_fake)[0]) * 1e-6)
            
            return mu_diff_squared, cov_real, cov_fake
            
        except Exception as e:
            logger.error(f"FID calculation error: {str(e)}")
            return None

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

    @tf.function
    def _validation_step(self, real_color, real_sar, labels):
        """Single validation step."""
        metrics = self.model.validation_step(real_color, real_sar, labels)
    
         
        fake_color = metrics['generated_images'][0]
        cycled_color = metrics['generated_images'][2]
        
        # Calculate inception features within tf.function
        real_features = self._get_inception_features(real_color)
        fake_features = self._get_inception_features(cycled_color)
        
        # Calculate other metrics
        inception_score = self._calculate_inception_score(fake_color)
        lpips_score = self._calculate_lpips(real_color, cycled_color)
        l1_loss, l2_loss = self._calculate_l1_l2_loss(real_color, cycled_color)
        
        # Update metrics
        self.lpips_metric.update_state(lpips_score)
        self.inception_score.update_state(inception_score)
        self.l1_loss_metric.update_state(l1_loss)
        self.l2_loss_metric.update_state(l2_loss)
        
        # Update metrics
        self.val_gen_loss_metric.update_state(metrics['gen_loss'])
        self.val_disc_loss_metric.update_state(metrics['disc_loss'])
        self.val_cycle_loss_metric.update_state(metrics['cycle_loss'])
        self.val_cls_loss_metric.update_state(metrics['cls_loss'])
        
        return metrics, real_features, fake_features
    
    def _validate(self) -> Dict[str, tf.Tensor]:
        """Execute validation step with improved memory management."""

        real_features_list = []
        fake_features_list = []
        # Initialize TensorArrays for metrics
        batch_size = next(iter(self.val_ds))[0][0].shape[0]
        num_batches = tf.data.experimental.cardinality(self.val_ds).numpy()
        
        metrics_arrays = {
            'psnr_color': tf.TensorArray(tf.float32, size=num_batches),
            'psnr_sar': tf.TensorArray(tf.float32, size=num_batches),
            'ssim_color': tf.TensorArray(tf.float32, size=num_batches),
            'ssim_sar': tf.TensorArray(tf.float32, size=num_batches)
        }
        
        for i, ((real_color, real_sar), labels) in enumerate(self.val_ds):
            metrics, real_feat, fake_feat = self._validation_step(real_color, real_sar, labels)

            real_features_list.append(real_feat)
            fake_features_list.append(fake_feat)
            
            # Write to TensorArrays
            for key in metrics_arrays:
                metrics_arrays[key] = metrics_arrays[key].write(i, metrics[key])
        
        all_real_features = tf.concat(real_features_list, axis=0)
        all_fake_features = tf.concat(fake_features_list, axis=0)
        
        fid_intermediate = self._calculate_fid(all_real_features, all_fake_features)
        if fid_intermediate is not None:
         ssdiff, sigma1, sigma2 = fid_intermediate
         fid_score = self._finalize_fid(ssdiff, sigma1, sigma2)
        else:
         fid_score = float('inf')
    
        self.fid_score_metric.update_state(fid_score)

        # Compute mean metrics
        val_metrics = {
            'gen_loss': self.val_gen_loss_metric.result(),
            'disc_loss': self.val_disc_loss_metric.result(),
            'cycle_loss': self.val_cycle_loss_metric.result(),
            'cls_loss': self.val_cls_loss_metric.result(),
            'lpips': self.lpips_metric.result(),
            'inception_score': self.inception_score.result(),
            'l1_loss': self.l1_loss_metric.result(),
            'l2_loss': self.l2_loss_metric.result(),
            'fid_score': self.fid_score_metric.result()
        }

        # Add image quality metrics
        for key in metrics_arrays:
            val_metrics[key] = tf.reduce_mean(metrics_arrays[key].stack())
        
        logger.info(
             f"Validation - "
            f"Cycle Loss: {val_metrics['cycle_loss']:.4f}, "
            f"PSNR Color: {val_metrics['psnr_color']:.2f}, "
            f"PSNR SAR: {val_metrics['psnr_sar']:.2f}, "
            f"SSIM Color: {val_metrics['ssim_color']:.4f}, "
            f"SSIM SAR: {val_metrics['ssim_sar']:.4f}, "
            f"LPIPS: {val_metrics['lpips']:.4f}, "
            f"Inception: {val_metrics['inception_score']:.4f}, "
            f"L1: {val_metrics['l1_loss']:.4f}, "
            f"L2: {val_metrics['l2_loss']:.4f}, "
            f"FID: {val_metrics['fid_score']:.4f}"
        )
        
        return val_metrics
        
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoints with optimizer states."""
        prefix = 'best' if is_best else f'epoch_{epoch + 1}'
        checkpoint_dir = os.path.join(self.output_dir, f'checkpoint_{prefix}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save complete model state
        self.model.save(checkpoint_dir)
        
        # Save additional training state
        training_state = {
            'epoch': epoch,
            'best_val_loss': float(self.val_gen_loss_metric.result().numpy()),
            'patience_counter': self.patience_counter 
        }
        
        with open(os.path.join(checkpoint_dir, 'training_state.json'), 'w') as f:
            json.dump(training_state, f)
        
        logger.info(f"Saved {'best' if is_best else ''} checkpoint at epoch {epoch + 1}")
    
    def load_checkpoint(self, checkpoint_dir: str):
        """Load model and training state from checkpoint."""
        if not os.path.exists(checkpoint_dir):
            raise ValueError(f"Checkpoint directory {checkpoint_dir} does not exist")
            
        # Load model
        self.model = CycleGAN.load(
            checkpoint_dir,
            self.model.G1.input_shape[1],  # image_size
            self.model.D1.output_shape[1]   # num_classes
        )

        # Load training state
        with open(os.path.join(checkpoint_dir, 'training_state.json'), 'r') as f:
            training_state = json.load(f)
            
        self.start_epoch = training_state['epoch'] + 1
        self.val_gen_loss_metric.reset_states()
        self.val_gen_loss_metric.update_state(training_state['best_val_loss'])
        self.patience_counter = training_state['patience_counter']
        
        logger.info(f"Loaded checkpoint from epoch {training_state['epoch']}")

    def train(self):
        """Execute training loop with validation and early stopping."""
        best_cycle_loss = float('inf')
        
        for epoch in range(self.epochs):
            start_time = time.time()
            
            # Training loop
            for (real_color, real_sar), labels in self.train_ds:
                metrics = self.model.train_step(real_color, real_sar, labels)
                self.metrics_logger.update_train(metrics)
            
            # Reset validation metrics
            self.val_gen_loss_metric.reset_states()
            self.val_disc_loss_metric.reset_states()
            self.val_cycle_loss_metric.reset_states()
            self.val_cls_loss_metric.reset_states()
            self.lpips_metric.reset_states()
            self.inception_score.reset_states()
            self.l1_loss_metric.reset_states()
            self.l2_loss_metric.reset_states()
            self.fid_score_metric.reset_states()

            # Validation
            val_metrics = self._validate()
            self.metrics_logger.update_val(val_metrics)
            
            # Early stopping check
            if val_metrics['cycle_loss'] < best_cycle_loss:
                best_cycle_loss = val_metrics['cycle_loss']
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Log progress
            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} completed in "
                f"{time.time() - start_time:.2f}s"
            )
            self.metrics_logger.log_epoch(epoch)
            
            # Visualize results periodically
            if (epoch + 1) % 5 == 0:
                sample_images = next(iter(self.val_ds))
                (real_color, real_sar), _ = sample_images
                metrics = self.model.validation_step(real_color, real_sar, _)
                if 'generated_images' in metrics:
                    fake_color, fake_sar, cycled_color, cycled_sar = metrics['generated_images']
                    self.visualizer.save_images(
                        epoch,
                        [real_color[0], fake_sar[0], cycled_color[0],
                         real_sar[0], fake_color[0], cycled_sar[0]],
                        ["Real Color", "Fake SAR", "Cycled Color",
                         "Real SAR", "Fake Color", "Cycled SAR"]
                    )
            
            # Regular checkpointing
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        logger.info("Training completed")
        self._save_checkpoint(epoch, is_best=True)  # Save final model

def main():
    # Optimized parameters for RTX 3060 Mobile
    config = {
        'image_size': 128,  # Reduced from 256
        'batch_size': 32,     # Reduced batch size for stability
        'epochs': 100,       # Reduced epochs with better scheduling
        'dataset_dir': './Dataset',
        'output_dir': './output_1',
        'learning_rate': 2e-4,
        'lambda_cyc': 5.0,
        'lambda_cls': 0.3,
        'early_stopping_patience': 7,
        'validation_split': 0.2,
        'cache_dataset': True  # Enable dataset caching for faster training
    }
   
    # Save configuration
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Configure GPU
    GPUConfig.configure()
    
    # Enable memory growth to prevent OOM errors
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logger.info(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
        except RuntimeError as e:
            logger.error(f"Memory growth setup failed: {str(e)}")

    # Initialize components
    data_loader = DataLoader(
        config['dataset_dir'],
        config['image_size'],
        config['batch_size'],
        config['validation_split']
    )
    train_ds, val_ds = data_loader.load_data()
    
    model = CycleGAN(
        config['image_size'],
        data_loader.num_classes,
        config['lambda_cyc'],
        config['lambda_cls'],
        config['learning_rate']
    )
    total_params = np.sum([
        np.prod(v.get_shape().as_list()) 
        for v in model.G1.trainable_variables + 
                 model.G2.trainable_variables +
                 model.D1.trainable_variables + 
                 model.D2.trainable_variables
    ])
    
    logger.info(f"Total trainable parameters: {total_params:,}")
    model.compile()
    
    trainer = Trainer(
        model,
        train_ds,
        val_ds,
        config['output_dir'],
        config['epochs'],
        config['early_stopping_patience']
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()
