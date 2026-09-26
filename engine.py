"""Vòng train/evaluate với strict IOB2 entity precision, recall và F1."""
import json
from pathlib import Path
import torch
from utils import move_batch, save_model

def train_epoch(model, loader, optimizer, device, epoch=None, total_epochs=None):
    from tqdm.auto import tqdm
    model.train()
    total_loss = 0.0
    
    # Tiêu đề thanh chạy batch (ví dụ: Epoch 01/40)
    desc = f"Epoch {epoch:02d}/{total_epochs}" if epoch and total_epochs else "Training"
    
    # Bọc loader vào tqdm: leave=False giúp thanh chạy tự thu gọn/ẩn khi xong 1 epoch để đỡ rối màn hình
    batch_pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    
    for raw in batch_pbar:
        batch = move_batch(raw, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(batch)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        optimizer.step()
        
        batch_loss = loss.item()
        total_loss += batch_loss
        
        # Cập nhật loss của batch hiện tại theo thời gian thực
        batch_pbar.set_postfix({"batch_loss": f"{batch_loss:.4f}"})
        
    return total_loss / len(loader) if len(loader) > 0 else 0.0
@torch.no_grad()
def evaluate_model(model, loader, labels, device):
    from seqeval.metrics import precision_score, recall_score, f1_score
    from seqeval.scheme import IOB2
    model.eval()
    gold, pred = [], []
    total_loss = 0.0

    for raw in loader:
        batch = move_batch(raw, device)
        
        # Tính validation loss
        batch_loss = model.loss(batch)
        total_loss += batch_loss.item()

        for row, path in enumerate(model.decode(batch)):
            gold.append([labels[i] for i in batch['tags'][row, :len(path)].tolist()])
            pred.append([labels[i] for i in path])
            
    kwargs = dict(mode='strict', scheme=IOB2, zero_division=0)
    avg_loss = total_loss / len(loader) if len(loader) > 0 else 0.0

    return dict(
        loss=float(avg_loss),
        precision=float(precision_score(gold, pred, **kwargs)),
        recall=float(recall_score(gold, pred, **kwargs)),
        f1=float(f1_score(gold, pred, **kwargs))
    )


# Tên hàm tương ứng với hướng dẫn Going Modular.
train_step = train_epoch
test_step = evaluate_model


def train(model, train_loader, dev_loader, optimizer, labels, device,
          model_config, feature_config, split_manifest, output_dir,
          checkpoint_path, epochs=10, patience=3, seed=23022006):
    """Chọn checkpoint theo dev F1 và dừng sớm sau patience epoch không tăng."""
    if epochs < 1 or patience < 1:
        raise ValueError('epochs và patience phải >= 1')
    best_f1, stale_epochs, history = -1.0, 0, []
    checkpoint_path = Path(checkpoint_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, epochs)
        train_eval = evaluate_model(model, train_loader, labels, device)
        dev_eval = evaluate_model(model, dev_loader, labels, device)
        row = dict(epoch=epoch, optimization_loss=train_loss,
                   **{f'train_{k}': v for k, v in train_eval.items()},
                   **{f'dev_{k}': v for k, v in dev_eval.items()})
        history.append(row)
        print(f"Epoch {epoch}/{epochs} | train loss={train_loss:.4f} | "
              f"dev loss={dev_eval['loss']:.4f} | dev F1={dev_eval['f1']:.4f}")
        if dev_eval['f1'] > best_f1:
            best_f1, stale_epochs = dev_eval['f1'], 0
            save_model(model, checkpoint_path.parent, checkpoint_path.name,
                       model_config=model_config, feature_config=feature_config,
                       labels=labels, seed=seed, epoch=epoch, dev_f1=best_f1,
                       split_manifest=split_manifest)
        else:
            stale_epochs += 1
        (Path(output_dir) / 'history.json').write_text(json.dumps(history, indent=2), encoding='utf-8')
        if stale_epochs >= patience:
            print(f'Early stopping tại epoch {epoch}')
            break
    return history
