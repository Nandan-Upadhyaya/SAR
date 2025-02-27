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
        
        symbolic_weights = getattr(self.gen_optimizer, 'weights', None)
        if symbolic_weights:
            weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
            with open(os.path.join(save_dir, 'gen_optimizer_weights.npy'), 'wb') as f:
                np.save(f, weight_values)
        
        
        symbolic_weights = getattr(self.disc_optimizer, 'weights', None)
        if symbolic_weights:
            weight_values = tf.keras.backend.batch_get_value(symbolic_weights)
            with open(os.path.join(save_dir, 'disc_optimizer_weights.npy'), 'wb') as f:
                np.save(f, weight_values)

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
        early_stopping_patience: int = 5,
        resume_from: str = None  # Add parameter to specify checkpoint to resume from
    ):
        self.model = model
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.output_dir = output_dir
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.start_epoch = 0  # Track starting epoch for resuming
        
        self.val_gen_loss_metric = tf.keras.metrics.Mean()
        self.val_disc_loss_metric = tf.keras.metrics.Mean()
        self.val_cycle_loss_metric = tf.keras.metrics.Mean()
        self.val_cls_loss_metric = tf.keras.metrics.Mean()

        self.metrics_logger = MetricsLogger(output_dir)
        self.visualizer = Visualizer(output_dir)
        self.patience_counter = 0
        
        os.makedirs(output_dir, exist_ok=True)

        if resume_from:
            self.load_checkpoint(resume_from)
            logger.info(f"Resuming training from epoch {self.start_epoch}")
    
    def train(self):
        """Execute training loop with validation and early stopping."""
        best_cycle_loss = float('inf')
        patience_counter = 0
        last_epoch = self.start_epoch - 1  # Track the last completed epoch
        
        # Track best metrics
        best_metrics = {
            'epoch': 0,
            'gen_loss': float('inf'),
            'disc_loss': float('inf'),
            'cycle_loss': float('inf'),
            'cls_loss': float('inf'),
            'psnr_color': 0,
            'psnr_sar': 0,
            'ssim_color': 0,
            'ssim_sar': 0
        }
        
        # Track final metrics for averaging
        final_metrics = {
            'gen_loss': [],
            'disc_loss': [],
            'cycle_loss': [],
            'cls_loss': [],
            'psnr_color': [],
            'psnr_sar': [],
            'ssim_color': [],
            'ssim_sar': []
        }
        
        # Check if we've already completed all epochs
        if self.start_epoch >= self.epochs:
            logger.info(f"Training already completed ({self.start_epoch} >= {self.epochs} epochs)")
            return
            
        for epoch in range(self.start_epoch, self.epochs):
            start_time = time.time()
            last_epoch = epoch  # Update last completed epoch
            
            # Training
            for (real_color, real_sar), labels in self.train_ds:
                metrics = self.model.train_step(real_color, real_sar, labels)
                self.metrics_logger.update_train(metrics)
            
            self.val_gen_loss_metric.reset_states()
            self.val_disc_loss_metric.reset_states()
            self.val_cycle_loss_metric.reset_states()
            self.val_cls_loss_metric.reset_states()

            # Validation
            val_metrics = self._validate()
            self.metrics_logger.update_val(val_metrics)
            
            # Update final metrics
            for key in final_metrics:
                if key in val_metrics:
                    final_metrics[key].append(val_metrics[key])
            
            # Track best metrics
           
            if val_metrics['cycle_loss'] < best_cycle_loss:
                best_cycle_loss = val_metrics['cycle_loss']
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Rest of the existing training loop code...
            self.metrics_logger.log_epoch(epoch)
                
                # Visualize results
            if (epoch + 1) % 5 == 0:
                    sample_images = next(iter(self.val_ds))
                    (real_color, real_sar), _ = sample_images
                    metrics = self.model.validation_step(real_color, real_sar, _)
                    if 'generated_images' in metrics:
                     fake_color, fake_sar, cycled_color, cycled_sar = (
                        metrics['generated_images']
                    )
                     self.visualizer.save_images(
                        epoch,
                        [
                            real_color[0],
                            fake_sar[0],
                            cycled_color[0],
                            real_sar[0],
                            fake_color[0],
                            cycled_sar[0]
                        ],
                        [
                            "Real Color",
                            "Fake SAR",
                            "Cycled Color",
                            "Real SAR",
                            "Fake Color",
                            "Cycled SAR"
                        ]
                    )
                
                # Save checkpoints
            if (epoch + 1) % 10 == 0:
                    self._save_checkpoint(epoch)
                
                # Early stopping
                
                
            if self.patience_counter >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {epoch + 1} epochs"
                    )
                    break
                
        if last_epoch >= 0:  # Only log if at least one epoch was completed
            logger.info(
                f"Epoch {last_epoch + 1}/{self.epochs} completed in "
                f"{time.time() - start_time:.2f}s"
            )
        
        logger.info("\n" + "="*50)
        logger.info("TRAINING COMPLETED - FINAL METRICS REPORT")
        logger.info("="*50)
        
        # Average metrics over last N epochs (e.g., last 5)
        n_final_epochs = min(5, len(next(iter(final_metrics.values()))))
        avg_final_metrics = {
            key: np.mean(values[-n_final_epochs:]) 
            for key, values in final_metrics.items()
        }
        
        logger.info("\nBest Model Metrics (Epoch {}):".format(best_metrics['epoch']))
        logger.info("-"*30)
        for key, value in best_metrics.items():
            if key != 'epoch':
                logger.info(f"{key}: {value:.4f}")
        
        logger.info(f"\nFinal Average Metrics (Last {n_final_epochs} epochs):")
        logger.info("-"*30)
        for key, value in avg_final_metrics.items():
            logger.info(f"{key}: {value:.4f}")
        
        logger.info("="*50)
        
        # Save final metrics to JSON
        final_metrics_path = os.path.join(self.output_dir, 'final_metrics.json')
        metrics_to_save = {
            'best_metrics': {k: float(v) for k, v in best_metrics.items()},
            'final_average_metrics': {k: float(v) for k, v in avg_final_metrics.items()}
        }
        with open(final_metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        
        logger.info(f"\nFinal metrics saved to: {final_metrics_path}")

    @tf.function
    def _validation_step(self, real_color, real_sar, labels):
        """Single validation step."""
        metrics = self.model.validation_step(real_color, real_sar, labels)
        
        # Update metrics
        self.val_gen_loss_metric.update_state(metrics['gen_loss'])
        self.val_disc_loss_metric.update_state(metrics['disc_loss'])
        self.val_cycle_loss_metric.update_state(metrics['cycle_loss'])
        self.val_cls_loss_metric.update_state(metrics['cls_loss'])
        
        return metrics
    
    def _validate(self) -> Dict[str, tf.Tensor]:
        """Execute validation step with improved memory management."""
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
            metrics = self._validation_step(real_color, real_sar, labels)
            
            # Write to TensorArrays
            for key in metrics_arrays:
                metrics_arrays[key] = metrics_arrays[key].write(i, metrics[key])
        
        # Compute mean metrics
        val_metrics = {
            'gen_loss': self.val_gen_loss_metric.result(),
            'disc_loss': self.val_disc_loss_metric.result(),
            'cycle_loss': self.val_cycle_loss_metric.result(),
            'cls_loss': self.val_cls_loss_metric.result(),
        }
        
        # Add image quality metrics
        for key in metrics_arrays:
            val_metrics[key] = tf.reduce_mean(metrics_arrays[key].stack())
        
        logger.info(
            f"Validation - Cycle Loss: {val_metrics['cycle_loss']:.4f}, "
            f"PSNR Color: {val_metrics['psnr_color']:.2f}, "
            f"PSNR SAR: {val_metrics['psnr_sar']:.2f}, "
            f"SSIM Color: {val_metrics['ssim_color']:.4f}, "
            f"SSIM SAR: {val_metrics['ssim_sar']:.4f}"
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
            
        # Load configuration first
        with open(os.path.join(checkpoint_dir, 'config.json'), 'r') as f:
            config = json.load(f)
            
        # Load training state
        with open(os.path.join(checkpoint_dir, 'training_state.json'), 'r') as f:
            training_state = json.load(f)
        
        # Load model weights after initializing model
        self.model.G1.load_weights(os.path.join(checkpoint_dir, 'generator1.h5'))
        self.model.G2.load_weights(os.path.join(checkpoint_dir, 'generator2.h5'))
        self.model.D1.load_weights(os.path.join(checkpoint_dir, 'discriminator1.h5'))
        self.model.D2.load_weights(os.path.join(checkpoint_dir, 'discriminator2.h5'))
            
        self.start_epoch = training_state['epoch'] + 1
        self.val_gen_loss_metric.reset_states()
        self.val_gen_loss_metric.update_state(training_state['best_val_loss'])
        self.patience_counter = training_state['patience_counter']
        
        logger.info(f"Loaded checkpoint from epoch {training_state['epoch']}")

def main():
    # Optimized parameters for RTX 3060 Mobile
    config = {
        'image_size': 128,  # Reduced from 256
        'batch_size': 8,     # Reduced batch size for stability
        'epochs': 50,       # Reduced epochs with better scheduling
        'epochs': 20,       # Reduced epochs with better scheduling
        'dataset_dir': './Dataset',
        'output_dir': './output',
        'learning_rate': 2e-4,
        'lambda_cyc': 5.0,
        'lambda_cls': 0.5,
        'early_stopping_patience': 10,
        'validation_split': 0.2,
        'cache_dataset': True,  # Enable dataset caching for faster training
        'resume_from': './output/checkpoint_epoch_10'  # Specify checkpoint to resume from, or None
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
    
    # Initialize model first with a dummy input to build shapes
    model = CycleGAN(
        config['image_size'],
        data_loader.num_classes,
        config['lambda_cyc'],
        config['lambda_cls'],
        config['learning_rate']
    )
    
    # Build model shapes with dummy input
    dummy_input = tf.zeros((1, config['image_size'], config['image_size'], 3))
    _ = model.G1(dummy_input)
    _ = model.G2(dummy_input)
    _ = model.D1(dummy_input)
    _ = model.D2(dummy_input)
    
    model.compile()
    
    trainer = Trainer(
        model,
        train_ds,
        val_ds,
        config['output_dir'],
        config['epochs'],
        config['early_stopping_patience'],
        config['resume_from']
    )
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()

