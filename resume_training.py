import tensorflow as tf
from SAR_B3 import CycleGAN, DataLoader, GPUConfig, Trainer
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'resume_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def resume_training(checkpoint_dir: str):
    """Resume training from a saved checkpoint."""
    try:
        # Configure GPU and set memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"Found {len(gpus)} GPU(s)")
            
            # Force TensorFlow to use GPU
            strategy = tf.distribute.OneDeviceStrategy("/GPU:0")
            with strategy.scope():
                # Load configuration from the original training
                config_path = os.path.join(os.path.dirname(checkpoint_dir), 'config.json')
                if not os.path.exists(config_path):
                    raise FileNotFoundError(f"Config file not found at {config_path}")
                    
                with open(config_path, 'r') as f:
                    import json
                    config = json.load(f)
                
                # First create the model and initialize it
                model = CycleGAN(
                    config['image_size'],
                    len(os.listdir(config['dataset_dir'])),  # num_classes
                    config['lambda_cyc'],
                    config['lambda_cls'],
                    config['learning_rate']
                )
                
                # Build the model first with dummy inputs
                dummy_input = tf.zeros((1, config['image_size'], config['image_size'], 3))
                _ = model.G1(dummy_input)
                _ = model.G2(dummy_input)
                _ = model.D1(dummy_input)
                _ = model.D2(dummy_input)
                
                # Load the weights after model is built
                model.G1.load_weights(os.path.join(checkpoint_dir, 'generator1.h5'))
                model.G2.load_weights(os.path.join(checkpoint_dir, 'generator2.h5'))
                model.D1.load_weights(os.path.join(checkpoint_dir, 'discriminator1.h5'))
                model.D2.load_weights(os.path.join(checkpoint_dir, 'discriminator2.h5'))
                
                # Compile model
                model.compile()
                
                # Now initialize data loader with strategy
                data_loader = DataLoader(
                    config['dataset_dir'],
                    config['image_size'],
                    config['batch_size'],
                    config['validation_split']
                )
                
                train_ds, val_ds = data_loader.load_data()
                
                # Create new output directory for resumed training
                new_output_dir = f"./resumed_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(new_output_dir, exist_ok=True)
                
                # Create trainer instance with initialized model
                trainer = Trainer(
                    model,
                    train_ds,
                    val_ds,
                    new_output_dir,
                    config['epochs'],
                    config['early_stopping_patience']
                )
                
                # Load training state
                if os.path.exists(os.path.join(checkpoint_dir, 'training_state.json')):
                    with open(os.path.join(checkpoint_dir, 'training_state.json'), 'r') as f:
                        training_state = json.load(f)
                        trainer.patience_counter = training_state.get('patience_counter', 0)
                        trainer.val_gen_loss_metric.reset_states()
                        trainer.val_gen_loss_metric.update_state(training_state.get('best_val_loss', float('inf')))
                
                # Resume training
                logger.info("Resuming training...")
                trainer.train()
        else:
            raise RuntimeError("No GPU available for training")
            
    except Exception as e:
        logger.error(f"Failed to resume training: {str(e)}")
        raise

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Resume CycleGAN training from checkpoint')
    parser.add_argument('--checkpoint_dir', 
                       type=str,
                       default='./output/checkpoint_best',
                       help='Path to checkpoint directory')
    
    args = parser.parse_args()
    
    # Check if checkpoint directory exists
    if not os.path.exists(args.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")
        
    logger.info(f"Starting resume process from checkpoint: {args.checkpoint_dir}")
    resume_training(args.checkpoint_dir)
