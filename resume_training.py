import os
import json
import numpy as np
import tensorflow as tf
from SAR_B3 import GPUConfig, logger,DataLoader, CycleGAN, Trainer

def resume_training(checkpoint_epoch=80, total_epochs=200):
    # Load original configuration
    with open('./output/config.json', 'r') as f:
        config = json.load(f)
    
    # Create new output directory for resumed training
    config['output_dir'] = './output_resumed'
    
    # Adjust epochs for remaining training
    config['epochs'] = total_epochs  # Set the total desired epochs
    remaining_epochs = total_epochs - checkpoint_epoch  

    # Save new configuration
    os.makedirs(config['output_dir'], exist_ok=True)
    with open(os.path.join(config['output_dir'], 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Configure GPU
    GPUConfig.configure()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            logger.info(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
        except RuntimeError as e:
            logger.error(f"Memory growth setup failed: {str(e)}")

    # Initialize DataLoader with same configuration
    data_loader = DataLoader(
        config['dataset_dir'],
        config['image_size'],
        config['batch_size'],
        config['validation_split']
    )
    train_ds, val_ds = data_loader.load_data()
    
    # Load checkpoint configuration
    checkpoint_path = os.path.join('./output', f'checkpoint_epoch_{checkpoint_epoch}')
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint at epoch {checkpoint_epoch} not found in {checkpoint_path}")
    
    with open(os.path.join(checkpoint_path, 'config.json'), 'r') as f:
        checkpoint_config = json.load(f)
    
    # Initialize model with the original learning rate value
    learning_rate = checkpoint_config.get('learning_rate_config', {}).get('value', 2e-4)
    
    logger.info(f"Loading checkpoint from epoch {checkpoint_epoch}")
    model = CycleGAN(
        image_size=config['image_size'],
        num_classes=data_loader.num_classes,
        lambda_cyc=checkpoint_config.get('lambda_cyc', 5.0),
        lambda_cls=checkpoint_config.get('lambda_cls', 0.5),
        learning_rate=learning_rate  # Pass the scalar learning rate
    )
    
    model.compile()

    # Load weights
    model.G1.load_weights(os.path.join(checkpoint_path, 'generator1.h5'))
    model.G2.load_weights(os.path.join(checkpoint_path, 'generator2.h5'))
    model.D1.load_weights(os.path.join(checkpoint_path, 'discriminator1.h5'))
    model.D2.load_weights(os.path.join(checkpoint_path, 'discriminator2.h5'))
    
    # Log model parameters
    total_params = np.sum([
        np.prod(v.get_shape().as_list()) 
        for v in model.G1.trainable_variables + 
                 model.G2.trainable_variables +
                 model.D1.trainable_variables + 
                 model.D2.trainable_variables
    ])
    logger.info(f"Total trainable parameters: {total_params:,}")
    
    # Initialize trainer with remaining epochs
    trainer = Trainer(
        model,
        train_ds,
        val_ds,
        config['output_dir'],
        config['epochs'],
        config['early_stopping_patience'],
        initial_epoch=checkpoint_epoch
    )
    
    # Resume training
    logger.info(f"Resuming training from epoch {checkpoint_epoch} for {remaining_epochs} epochs")
    trainer.train()

if __name__ == "__main__":
    resume_training(checkpoint_epoch=80, total_epochs=200)