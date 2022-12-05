import os
import torch
import argparse
import torchvision
from tqdm.auto import tqdm
from omegaconf import OmegaConf
from hydra.utils import instantiate
import torch.nn.functional as F
from collections import namedtuple
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from utils import fix_random_seeds, set_logger
from losses import get_loss, MSSIM


logger = set_logger()


def get_params():
    parser = argparse.ArgumentParser(description='Generic runner for models')
    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()
    return args


def train(cfg):
    fix_random_seeds(cfg.exp_params['seed'])
    logger.info("Training with config:\n{}".format(cfg))
    
    train_dataset = instantiate(cfg.data_params, split="train")
    val_dataset = instantiate(cfg.data_params, split="val")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=cfg.data_params['batch_size'],
        num_workers=cfg.data_params['workers'],
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=int(cfg.data_params['batch_size']*2),
        num_workers=cfg.data_params['workers'],
    )
    logger.info("Building data done with train {} images loaded and val {} images loaded.".format(len(train_dataset), len(val_dataset)))
    
    # Model to GPU
    criterion = get_loss(cfg.exp_params['criterion_name'], delta=cfg.exp_params['criterion_delta'])
    model = instantiate(cfg.model_params, recon_loss=criterion)
    model = model.to(cfg.exp_params['device'])
    
    logger.info(f"Building model {cfg.model_params['name']} done")
    
    if "Pix2Pix" in cfg.model_params['name']:
        optimizer = instantiate(cfg.optimizer, params=model.netD.parameters())
        optimizer_G = instantiate(cfg.optimizer_g, params=model.netG.parameters())
    else:
        optimizer = instantiate(cfg.optimizer, params=model.parameters())
        optimizer_G = None
    logger.info("Building optimizer and criterion done.")
    
    # Tensorboard
    name_log = f"{cfg.exp_params['seed']}_{cfg.model_params['name']}_{cfg.data_params['name']}_{cfg.exp_params['criterion_name']}_{cfg.optimizer['_target_']}"
    writer = SummaryWriter(
        log_dir=cfg.logging_params['log_dir'] + "/" + name_log
    )
    mssim = MSSIM()
    
    if os.path.exists(cfg.logging_params['log_dir'] + "/" + name_log):
        logger.info("Log directory already exists, so we are taking the last checkpoint.")
        checkpoint = torch.load(cfg.logging_params['log_dir'] + "/" + name_log + "/last_checkpoint.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if "Pix2Pix" in cfg.model_params['name']:
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        best = checkpoint['mae']
    else:
        os.makedirs(os.path.join(cfg.logging_params['checkpoint_dir'], name_log), exist_ok=False)
        best = None
    
    logger.info("Start training")
    best = None
    for epoch in range(cfg.exp_params['epochs']):
        logger.info(f"Epoch {epoch + 1}/{cfg.exp_params['epochs']}")
        
        model.train()
        scaler = GradScaler()
        
        avg_losses = {}
        for batch in tqdm(train_loader, desc="Training", leave=False):
            optimizer.zero_grad(set_to_none=True)
            batch = batch.to(cfg.exp_params['device'])
            with autocast():  # mixed precision
                out = model(batch)
            losses = model.loss_function(*out)
            loss = losses["loss"]
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            for key, value in losses.items():
                if f"train/{key}" not in avg_losses:
                    avg_losses[f"train/{key}"] = [value.item()]
                else:
                    avg_losses[f"train/{key}"].append(value.item())
            
            if optimizer_G:
                optimizer_G.zero_grad(set_to_none=True)
                loss = model.loss_function_G(*out)
                loss = losses["loss_G"]
                loss.backward()
                optimizer_G.step()
                
                for key, value in losses.items():
                    if f"train/{key}" not in avg_losses:
                        avg_losses[f"train/{key}"] = [value.item()]
                    else:
                        avg_losses[f"train/{key}"].append(value.item())
        
        model.eval()
        with torch.inference_mode():
            logged = False
            for batch in tqdm(val_loader, desc="Evaluating", leave=False):
                batch = batch.to(cfg.exp_params['device'])
                out = model(batch)
                recons = out[0]
                input = out[1]
                
                losses = model.loss_function(*out)
                losses['mse'] = F.mse_loss(recons, input)
                losses['psnr'] = 10 * torch.log10(1 / losses['mse'])
                losses['mssim'] = torch.mean(torch.as_tensor(([mssim(rec, inp) for rec, inp in zip(recons, input)])))
                losses['mae'] = F.l1_loss(recons, input)
                
                for key, value in losses.items():
                    if f"val/{key}" not in avg_losses:
                        avg_losses[f"val/{key}"] = [value.item()]
                    else:
                        avg_losses[f"val/{key}"].append(value.item())
                
                if not logged and epoch % cfg.exp_params['log_images_interval'] == 0:
                    if input.shape[0] > 8:
                        input = input[:8]
                        recons = recons[:8]
                    log_img = torch.concat([input, recons], dim=0).cpu()
                    log_img = val_dataset.post_process(log_img)
                    grid = torchvision.utils.make_grid(log_img, nrow=8)
                    writer.add_image("val/imgs", grid, epoch)
                    logged = True
        
        # logging
        log_message = "METRICS: "
        for key, value in avg_losses.items():
            avg_losses[key] = torch.mean(torch.as_tensor(value))
            log_message += f"{key}: {avg_losses[key]}, "
            writer.add_scalar(key, avg_losses[key], epoch)
        logger.info(log_message)    
        
        # Save our checkpoint loc
        if best is None or best > avg_losses['val/mae']:
            best = avg_losses['val/mae']
            checkpoint_path = os.path.join(cfg.logging_params['checkpoint_dir'], name_log, f"best_checkpoint.pth.tar")
            logger.info(f"best model so far, saving checkpoint to {checkpoint_path} with mae {best}")
            torch.save({ 
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict() if optimizer_G else None,
                'loss': avg_losses['train/loss'],
                'mae': avg_losses['val/mae'],
            }, checkpoint_path)
        if epoch % cfg.exp_params['checkpoint_freq'] == 0:
            checkpoint_path = os.path.join(cfg.logging_params['checkpoint_dir'], name_log, f"{epoch}_checkpoint.pth.tar")
            checkpoint_last = os.path.join(cfg.logging_params['checkpoint_dir'], name_log, "last_checkpoint.pth.tar")
            torch.save({ 
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict() if optimizer_G else None,
                'loss': avg_losses['train/loss'],
                'mae': avg_losses['val/mae'],
            }, checkpoint_path)
            torch.save({ 
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict() if optimizer_G else None,
                'loss': avg_losses['train/loss'],
                'mae': avg_losses['val/mae'],
            }, checkpoint_last)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        elif epoch+1 == cfg.exp_params['epochs']:
            checkpoint_path = os.path.join(cfg.logging_params['checkpoint_dir'], name_log, "last_checkpoint.pth.tar")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict() if optimizer_G else None,
                'loss': avg_losses['train/loss'],
                'mae': avg_losses['val/mae'],
            }, checkpoint_last)
            logger.info(f"Checkpoint last saved to {checkpoint_last}")

    logger.info("Training done.")
    

if __name__ == "__main__":
    args = get_params()
    cfg_yaml = OmegaConf.load(args.filename)
    cfg_dict = OmegaConf.to_container(cfg_yaml, resolve=True)
    MyTuple = namedtuple('MyTuple', cfg_dict)
    cfg = MyTuple(**cfg_dict)
    train(cfg)
