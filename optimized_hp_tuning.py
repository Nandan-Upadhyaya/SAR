import keras_tuner as kt
import tensorflow as tf
from SAR_B3 import CycleGAN, DataLoader, Trainer, GPUConfig, logger

# Build model function for Keras Tuner
def build_model(hp):
    """Build and return a CycleGAN model with tunable hyperparameters."""
    lambda_cls = hp.Float("lambda_cls", min_value=0.1, max_value=1.0, step=0.1)
    lambda_cyc = hp.Float("lambda_cyc", min_value=1.0, max_value=10.0, step=1.0)
    learning_rate = hp.Choice("learning_rate", values=[1e-5, 2e-5, 5e-5, 1e-4, 2e-4])

    model = CycleGAN(
        image_size=128,  # Adjusted for RTX 4060 constraints
        num_classes=4,  # Example value; replace with actual number of classes
        lambda_cyc=lambda_cyc,
        lambda_cls=lambda_cls,
        learning_rate=learning_rate
    )

    model.compile()
    return model

class HyperparameterTrainer:
    def __init__(self, train_ds, val_ds, epochs=20):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.epochs = epochs

    def fit(self, hp):
        # Build the model with the hyperparameters
        model = build_model(hp)

        # Create output directory for this trial
        lambda_cls = hp['lambda_cls']  # Access hyperparameter directly
        lambda_cyc = hp['lambda_cyc']
        learning_rate = hp['learning_rate']
        logger.info(f"Starting trial with lambda_cls={lambda_cls}, lambda_cyc={lambda_cyc}, learning_rate={learning_rate}")
        output_dir = f"./tuner_results/lambda_cls_{lambda_cls}_lambda_cyc_{lambda_cyc}_lr_{learning_rate}".replace('.', '_')
        num_classes = 4 
        trainer = Trainer(
            model,
            self.train_ds,
            self.val_ds,
            output_dir=output_dir,
            epochs=self.epochs,
            early_stopping_patience=5
        )

        trainer.train()

        # Return weighted loss for Keras Tuner to optimize
        weighted_loss = trainer._calculate_weighted_loss(trainer.best_metrics)
        return weighted_loss

def tuner_pipeline():
    # GPU configuration
    GPUConfig.configure()

    # Data loading
    dataset_dir = "./Dataset"  # Replace with actual dataset directory
    data_loader = DataLoader(
        dataset_dir=dataset_dir,
        image_size=128,
        batch_size=16,  # Adjusted for RTX 4060
        validation_split=0.2
    )
    train_ds, val_ds = data_loader.load_data()

    # Define hyperparameter tuner
    tuner = kt.BayesianOptimization(
        HyperparameterTrainer(train_ds, val_ds).fit,
        objective=kt.Objective("val_loss", direction="min"),
        max_trials=20,
        directory="./tuner_dir",
        project_name="cycle_gan_tuning"
    )

    # Perform hyperparameter search
    tuner.search()

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Best Hyperparameters:")
    print(f"Lambda_cls: {best_hps.get('lambda_cls')}")
    print(f"Lambda_cyc: {best_hps.get('lambda_cyc')}")
    print(f"Learning rate: {best_hps.get('learning_rate')}")

if __name__ == "__main__":
    tuner_pipeline()
