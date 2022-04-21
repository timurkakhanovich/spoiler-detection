from pathlib import Path
import argparse
import textwrap

import torch
from torch.optim import AdamW

from TextCNN.modules.model.model import TextCNN
from TextCNN.modules.pipeline_config.utils import BPEMB_EN, set_device


class ModelConfig:
    '''
    Gets arguments from command line to initialize model.
    STRICTLY REQUIRED: 
    * for chosen checkpoint: CHECKPOINT, NUM_EPOCHS, DEVICE;
    * if checkpoint is not chosen: LR, BATCH_SIZE, NUM_EPOCHS, DEVICE
    '''

    def __init__(self, checkpoint_path=Path('.')):
        self.checkpoint_path = checkpoint_path
        self.model = TextCNN(BPEMB_EN, hid_size=128)
    
    def load_from_checkpoint(self, check_epoch, batch_size, 
                            num_epochs, device):
        full_check_path = self.checkpoint_path / f'checkpoint_epoch_{check_epoch}.pt'
        checkpoint = torch.load(full_check_path, map_location=device)
        
        epoch = checkpoint['epoch']
        self.model = self.model.to(device)
        self.model.device = device
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        lr = checkpoint['lr']
        history = checkpoint['whole_history']
        
        optimizer = AdamW(self.model.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if checkpoint['scheduler_state_dict']:
            scheduler = StepLR(optimizer)
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return {
            'model': self.model, 
            'optimizer': optimizer, 
            'scheduler': scheduler if checkpoint['scheduler_state_dict'] else None, 
            'history': history, 
            'batch_size': batch_size, 
            'start_epoch': epoch, 
            'lr': lr, 
            'num_epochs': num_epochs, 
            'device': device, 
            'dataset': checkpoint['dataset']
        }
    
    
    def get_arguments(self):
        parser = argparse.ArgumentParser(
            prog='PROG',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description=textwrap.dedent('''
            Gets arguments from command line to initialize model.
            STRICTLY REQUIRED: 
            * for chosen checkpoint: CHECKPOINT, NUM_EPOCHS, DEVICE;
            * if checkpoint is not chosen: LR, BATCH_SIZE, NUM_EPOCHS, DEVICE
            ''')
        )
        parser.add_argument('--checkpoint', type=int, help='Loading from checkpoint')
        parser.add_argument('--batch_size', type=int, help='Batch size')
        parser.add_argument('--device', type=str, help='Device type')
        parser.add_argument('--lr', type=float, help='Learning rate')
        parser.add_argument('--num_epochs', type=int, help='Num epochs for training')

        args = parser.parse_args()
        
        device = set_device(args.device)
        
        if args.checkpoint:
            return self.load_from_checkpoint(check_epoch=args.checkpoint, 
                                             batch_size=args.batch_size,
                                             num_epochs=args.num_epochs, 
                                             device=device)
        else:
            self.model = self.model.to(device)
            
            optimizer = AdamW(self.model.parameters(), lr=args.lr)
            scheduler = None
                
            return {
                'model': self.model, 
                'optimizer': optimizer, 
                'scheduler': scheduler, 
                'history': dict(), 
                'batch_size': args.batch_size, 
                'start_epoch': -1, 
                'lr': args.lr, 
                'num_epochs': args.num_epochs, 
                'device': device, 
                'dataset': 'IMDB & Goodreads'
            }
