from pathlib import Path
from copy import deepcopy
import gc
from collections import defaultdict

from tqdm import tqdm
import wandb

import torch

from LSTM.modules.pipeline_config.utils import validate_model


def train(model, dataloaders, criterion, class_weights, optimizer, metrics, lr, scheduler=None, 
          num_epochs=5, start_epoch=-1, prev_metrics=dict(), device=torch.device('cpu'),
          folder_for_checkpoints=Path('.'), dataset_name='IMDB & Goodreads', run=None):
    for key, vals in prev_metrics.items():
        for val in vals:
            wandb.log({key: val[1]}, step=val[0])
            
    if len(prev_metrics) > 0:
        history = deepcopy(prev_metrics)
        curr_step = prev_metrics['train_loss'][-1][0] + 1
    else:
        history = defaultdict(list)
        curr_step = 1
    
    update_log_iteration = len(dataloaders['train']) // 2
    train_dataloader_len = len(dataloaders['train'])
    
    model.train()
    for epoch in range(start_epoch + 1, start_epoch + 1 + num_epochs):
        running_loss = 0.0
        running_score = defaultdict(float)

        print('-' * 20)
        print(f'Epoch: {epoch}/{start_epoch + num_epochs}')
        print('-' * 20)
        print('Train: ')

        for batch_idx, (X, y_true) in enumerate(tqdm(dataloaders['train'])):
            X = X.to(device)
            y_true = y_true.to(device, dtype=torch.float32)
            
            y_pred = model(X).squeeze()
            
            # Weighted BCELoss.  
            target_weights = class_weights[y_true.data.view(-1).long()].view_as(y_true)
            loss = criterion(y_pred, y_true)
            loss *= target_weights
            loss = torch.mean(loss)
            
            optimizer.zero_grad()
            
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            running_loss += loss.item()
            
            if batch_idx % (update_log_iteration + 1) == update_log_iteration or \
                batch_idx == train_dataloader_len - 1:
                val_result = validate_model(
                    model, dataloaders['val'], criterion, class_weights, 
                    metrics, device, return_train=True
                )
                
                wandb.log({
                    'val_loss': val_result.loss,
                    'train_loss': running_loss / (batch_idx + 1)
                }, step=curr_step)

                for metric, score in val_result.scores.items():
                    wandb.log({'val_' + metric: score}, step=curr_step)
                    history['val_' + metric].append((curr_step, score))
                
                history['train_loss'].append((curr_step, running_loss / (batch_idx + 1)))
                history['val_loss'].append((curr_step, val_result.loss))

                curr_step += 1

        print('-'*20 + '\n')

        state = {
            'epoch': epoch,
            'batch_size_training': dataloaders['train'].batch_size,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'whole_history': history, 
            'lr': lr, 
            'dataset': dataset_name
        }
        
        curr_checkpoint_path = folder_for_checkpoints / f'checkpoint_epoch_{epoch%5 + 1}.pt'
        torch.save(state, curr_checkpoint_path)
        
        model_art = wandb.Artifact('checkpoints', type='model')
        model_art.add_file(curr_checkpoint_path)
        run.log_artifact(model_art)
