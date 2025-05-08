import os
import torch
from torch.utils.tensorboard import SummaryWriter
import time
from datetime import datetime
from tqdm import tqdm

# Import our modules
from model import get_model_instance, train_one_epoch, evaluate, visualize_predictions
from data_loader import get_maskrcnn_dataloader

class Config:
     def __init__(self):
        # Paths
        self.data_dir = 'carton_data'
        self.output_dir = 'checkpoints'
        
        # ******    Change before each run  *************

        self.run_no = 6

        # ***********************************************

        # Training hyperparameters
        self.num_epochs = 12
        self.batch_size = 4
        self.learning_rate = 0.005
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.lr_step_size = 10
        self.lr_gamma = 0.2
        
        # Data loading
        self.num_workers = 4
        self.max_samples = 840 # Total dataset is of 8400 images
        self.train_ratio = 0.8
        self.seed = 42
        
        # Device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Logging
        self.log_dir = '.logs'
        self.log_images = True
        self.log_freq = 5  # Log every N epochs


def main():
    # Initialize config
    config = Config()
    
    
    def log_hparam(_config):
        # hparam_dict = {key: value for key, value in vars(_config).items() if not key.startswith('__')}
        hparam_dict = {key: value for key, value in _config.__dict__.items() if not key.startswith('__')}
        
        # Save the hyperparameters to a text file
        file_path = os.path.join(_config.log_dir, f"hyperparameters_run_{_config.run_no}.txt")
        with open(file_path, "w") as file:
            for key, value in hparam_dict.items():
                file.write(f"{key}: {value}\n")
    
        
    # Create directories
    run = config.run_no
    os.makedirs(os.path.join(config.output_dir, f'run_{run}'), exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    log_hparam(config)

    # Initialize tensorboard writer
    current_time = datetime.now().strftime('%b%d_%H-%M')
    
    tb_log_dir = os.path.join(config.log_dir, "tensorboard",f'run_{run}')
    writer = SummaryWriter(log_dir=tb_log_dir)
    
    print(f"Using device: {config.device}")
    
    # Load data
    print("Loading data...")
    train_dataloader, val_dataloader = get_maskrcnn_dataloader(
        data_dir=config.data_dir,
        max_samples=config.max_samples,
        train_ratio=config.train_ratio,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed
    )
    print(f"Training samples: {len(train_dataloader.dataset)}")
    print(f"Validation samples: {len(val_dataloader.dataset)}")
    
    # Initialize model (assuming 2 classes: background and carton)
    num_classes = 2  # Background + Carton
    model = get_model_instance(num_classes)
    model.to(config.device)
    
    # Initialize optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.lr_step_size,
        gamma=config.lr_gamma
    )

    # Training loop
    print(f"Starting training for {config.num_epochs} epochs...")
    best_loss = float('inf')
    start_time = time.time()
    hparam_dict = log_hparam(config)
    
    for epoch in range(config.num_epochs):
        # Train for one epoch
        train_metrics = train_one_epoch(
            model, optimizer, train_dataloader, config.device, epoch, writer
        )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_dataloader, config.device)
        torch.cuda.empty_cache()
        
        # Log validation metrics
        for k, v in val_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
        
        # Log learning rate
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print metrics
        print(f"Epoch {epoch} - Train loss: {train_metrics['loss']:.6f}, Val loss: {val_metrics['loss']:.6f}")
        
        # Visualize predictions if configured
        if config.log_images and epoch % config.log_freq == 0:
            visualize_predictions(model, val_dataloader, config.device)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict(),
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss'],
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, os.path.join(config.output_dir, f'run_{run}','latest_checkpoint.pth'))
        
        # Save best model
        if val_metrics['loss'] < best_loss:
            best_loss = val_metrics['loss']
            torch.save(checkpoint, os.path.join(config.output_dir, f'run_{run}','best_model.pth'))
            print(f"New best model saved with validation loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()