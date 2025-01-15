import keras_tuner as kt
from SAR_B3 import CycleGAN, DataLoader, Trainer, GPUConfig, logger
import tensorflow as tf
import os

class CycleGANHyperModel(kt.HyperModel):
    """HyperModel for CycleGAN to define the model architecture and hyperparameters."""
    
    def __init__(self, image_size: int, num_classes: int):
        super().__init__()
        self.image_size = image_size
        self.num_classes = num_classes
    
    def build(self, hp):
        """Build CycleGAN model with hyperparameters."""
        # Define hyperparameters
        lambda_cls = hp.Float("lambda_cls", min_value=0.1, max_value=1.0, step=0.1)
        lambda_cyc = hp.Float("lambda_cyc", min_value=1.0, max_value=10.0, step=1.0)
        learning_rate = hp.Choice("learning_rate", values=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 1e-6, 2e-6, 1e-7])
        
        # Create model
        model = CycleGAN(
            image_size=self.image_size,
            num_classes=self.num_classes,
            lambda_cyc=lambda_cyc,
            lambda_cls=lambda_cls,
            learning_rate=learning_rate
        )
        model.compile()
        
        # Create a wrapper Keras model that represents the CycleGAN
        class KerasWrapper(tf.keras.Model):
            def __init__(self, cycle_gan):
                super().__init__()
                self.cycle_gan = cycle_gan
                
            def call(self, inputs):
                # Dummy forward pass - not actually used for training
                return self.cycle_gan.G1(inputs)
        
        return KerasWrapper(model)

class CycleGANTuner(kt.Tuner):
    """Custom tuner for CycleGAN to handle the complex training process."""
    
    def run_trial(self, trial, *args, **kwargs):
        """Execute a trial with given hyperparameters."""
        hp = trial.hyperparameters
        
        # Get the wrapped model and extract CycleGAN
        wrapped_model = self.hypermodel.build(hp)
        model = wrapped_model.cycle_gan
        
        # Create trainer for this trial
        output_dir = os.path.join(
            self.project_dir,
            f"trial_{trial.trial_id}_cls_{hp.get('lambda_cls')}_cyc_{hp.get('lambda_cyc')}_lr_{hp.get('learning_rate')}".replace('.', '_')
        )
        
        trainer = Trainer(
            model=model,
            train_ds=self.train_ds,
            val_ds=self.val_ds,
            output_dir=output_dir,
            epochs=self.epochs,
            early_stopping_patience=5
        )
        
        # Train the model
        trainer.train()
        
        # Calculate weighted validation metrics
        metrics = {
            'cycle_loss': float(trainer.val_cycle_loss_metric.result()),
            'gen_loss': float(trainer.val_gen_loss_metric.result()),
            'disc_loss': float(trainer.val_disc_loss_metric.result()),
            'cls_loss': float(trainer.val_cls_loss_metric.result()),
            'lpips': float(trainer.lpips_metric.result()),
            'fid_score': float(trainer.fid_score_metric.result()),
            'l1_loss': float(trainer.l1_loss_metric.result()),
            'l2_loss': float(trainer.l2_loss_metric.result()),
            'inception_score': float(trainer.inception_score.result())
        }
        
        # Calculate weighted score
        weighted_score = (
            metrics['cycle_loss'] * 0.3 +
            metrics['gen_loss'] * 0.15 +
            metrics['disc_loss'] * 0.15 +
            metrics['cls_loss'] * 0.1 +
            metrics['lpips'] * 0.1 +
            metrics['fid_score'] * 0.1 +
            metrics['l1_loss'] * 0.05 +
            metrics['l2_loss'] * 0.05 +
            (10.0 - metrics['inception_score']) * 0.05  # Convert inception score to loss
        )

        metrics['weighted_score'] = weighted_score
        
        # Log all metrics
        self.oracle.update_trial(trial.trial_id, metrics)
        
        # Return the weighted score for optimization
        return weighted_score

def tuner_pipeline():
    """Execute the hyperparameter tuning pipeline."""
    # GPU configuration
    GPUConfig.configure()
    
    # Data loading
    dataset_dir = "./Dataset"
    image_size = 128
    num_classes = 4
    epochs = 5
    
    data_loader = DataLoader(
        dataset_dir=dataset_dir,
        image_size=image_size,
        batch_size=16,
        validation_split=0.2
    )
    train_ds, val_ds = data_loader.load_data()
    
    # Create hypermodel
    hypermodel = CycleGANHyperModel(image_size=image_size, num_classes=num_classes)
    
    # Create custom tuner
    tuner = CycleGANTuner(
        hypermodel=hypermodel,
        oracle=kt.oracles.BayesianOptimization(
            objective=kt.Objective('weighted_score', direction='min'),
            max_trials=30
        ),
        directory='./tuner_dir',
        project_name='cycle_gan_tuning',
        overwrite=True
    )
    
    # Add required attributes for training
    tuner.train_ds = train_ds
    tuner.val_ds = val_ds
    tuner.epochs = epochs
    
    # Start the search
    tuner.search()
    
    # Get best hyperparameters
    best_hp = tuner.get_best_hyperparameters(1)[0]
    
    # Log best hyperparameters
    logger.info("\nBest Hyperparameters:")
    logger.info(f"Lambda_cls: {best_hp.get('lambda_cls')}")
    logger.info(f"Lambda_cyc: {best_hp.get('lambda_cyc')}")
    logger.info(f"Learning rate: {best_hp.get('learning_rate')}")
    
    # Create and save best model
    best_model = CycleGAN(
        image_size=image_size,
        num_classes=num_classes,
        lambda_cyc=best_hp.get('lambda_cyc'),
        lambda_cls=best_hp.get('lambda_cls'),
        learning_rate=best_hp.get('learning_rate')
    )
    best_model.compile()
    
    # Save the best model
    best_model_dir = os.path.join('tuner_results', 'best_model')
    os.makedirs(best_model_dir, exist_ok=True)
    best_model.save(best_model_dir)
    logger.info(f"\nBest model saved to: {best_model_dir}")

if __name__ == "__main__":
    tuner_pipeline()