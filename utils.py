"""Seed, di chuyển batch và lưu/đọc checkpoint tương thích notebook."""
import os
import random
from pathlib import Path
import torch


def set_seed(seed=23022006):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def move_batch(batch, device):
    return dict(words=batch['words'], tags=batch['tags'].to(device),
                mask=batch['mask'].to(device), lengths=batch['lengths'])



def save_model(model, target_dir, model_name='best_modular.pt', **metadata):
    path = Path(target_dir) / model_name
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model_state_dict=model.state_dict(), **metadata), path)
    return path


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(Path(path), map_location=device, weights_only=True)
    required = {'model_state_dict', 'model_config', 'feature_config', 'labels'}
    if not isinstance(checkpoint, dict) or not required <= checkpoint.keys():
        raise ValueError(f'Checkpoint thiếu metadata: {required}')
    return checkpoint
