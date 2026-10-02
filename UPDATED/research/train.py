from pathlib import Path
import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .config import Config
from .data import load_splits
from .metrics import probabilities, classification_metrics, fit_temperature
from .models import MatchedViT
from .runtime import seed_all, json_write, snapshot, event, save_checkpoint, restore_checkpoint, torch_save_atomic, environment


@torch.no_grad()
def predict(model, dataset, batch_size):
    model.eval()
    logits, targets = [], []
    start = time.perf_counter()
    for x,y in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        logits.append(model(x).numpy())
        targets.append(y.numpy())
    z, y = np.concatenate(logits), np.concatenate(targets)
    assert len(y) == len(dataset), 'Incomplete split evaluation'
    return z, y, time.perf_counter()-start


def train(c, output='runs', data_dir='data', validation_only=False, stop_after=None):
    seed_all(c.seed, c.threads)
    folder = Path(output)/c.experiment_id
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'test_metrics.json').exists():
        return folder
    if (folder/'config.json').exists():
        assert json.loads((folder/'config.json').read_text()) == c.dict()
    else:
        json_write(folder/'config.json', c.dict())
        snapshot(folder)
    datasets, channels, classes, data_meta = load_splits(c, data_dir)
    json_write(folder/'data.json', data_meta)
    model = MatchedViT(c, channels, classes)
    json_write(folder/'resources.json', model.resources())
    generator = torch.Generator().manual_seed(c.seed+100000)
    loader = DataLoader(datasets['train'], batch_size=c.batch_size, shuffle=True, generator=generator)
    counts = torch.bincount(datasets['train'].tensors[1], minlength=classes).float()
    weight = counts.sum()/(classes*counts.clamp_min(1)) if c.class_weight else None
    criterion = torch.nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=c.lr, weight_decay=c.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=c.epochs, eta_min=1e-6)
    start_epoch, best, elapsed = 0, float('inf'), 0.
    if (folder/'last.pt').exists():
        state = restore_checkpoint(folder/'last.pt', model, optimizer, scheduler, generator)
        start_epoch,best,elapsed = state['epoch'],state['best'],state['elapsed']
        event(folder, kind='resume', epoch=start_epoch, source_sha256=environment()['source_sha256'])
    for epoch in range(start_epoch+1,c.epochs+1):
        start = time.perf_counter()
        model.train()
        loss_sum, correct = 0.,0
        for x,y in loader:
            if c.augment:
                # Explicit, reproducible small translations and horizontal flips. No test augmentation.
                flip = torch.rand(len(x)) < .5
                x = x.clone()
                x[flip] = x[flip].flip(-1)
                shift = torch.randint(-1,2,(2,)).tolist()
                x = torch.roll(x, shifts=shift, dims=(-2,-1))
            optimizer.zero_grad(set_to_none=True)
            z = model(x)
            loss = criterion(z,y)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach())*len(y)
            correct += int((z.argmax(1)==y).sum())
        val_z,val_y,_ = predict(model,datasets['val'],c.batch_size)
        val_nll = float(torch.nn.functional.cross_entropy(torch.from_numpy(val_z),torch.from_numpy(val_y)))
        scheduler.step()
        elapsed += time.perf_counter()-start
        if val_nll < best:
            best = val_nll
            torch_save_atomic({'model': model.state_dict(), 'epoch': epoch, 'val_nll': best}, folder/'best.pt')
        save_checkpoint(folder/'last.pt', model, optimizer, scheduler, generator, epoch,best,elapsed)
        event(folder, kind='epoch', epoch=epoch, train_loss=loss_sum/len(datasets['train']),
              train_accuracy=correct/len(datasets['train']), val_nll=val_nll,
              val_accuracy=float((val_z.argmax(1)==val_y).mean()), cumulative_training_seconds=elapsed)
        if stop_after is not None and epoch >= stop_after:
            return folder
    best_state = torch.load(folder/'best.pt', weights_only=True)
    model.load_state_dict(best_state['model'])
    val_z,val_y,_ = predict(model,datasets['val'],c.batch_size)
    temperature = fit_temperature(val_z,val_y)
    np.savez_compressed(folder/'validation_predictions.npz', logits=val_z, labels=val_y,
                        probabilities=probabilities(val_z,temperature), sample_ids=np.arange(len(val_y)))
    json_write(folder/'calibration.json', {'temperature':temperature,'fit_split':'validation','best_epoch':best_state['epoch']})
    json_write(folder/'validation_metrics.json', classification_metrics(val_y,probabilities(val_z,temperature)))
    if not validation_only:
        z,y,seconds = predict(model,datasets['test'],c.batch_size)
        p = probabilities(z,temperature)
        np.savez_compressed(folder/'test_predictions.npz', logits=z,labels=y,probabilities=p,
                            sample_ids=np.arange(len(y)), uncalibrated_probabilities=probabilities(z))
        m = classification_metrics(y,p)
        m.update({'training_seconds':elapsed,'inference_seconds':seconds,'execution_mode':'classical_analytical',
                  'shots':0,'circuit_calls':0,'temperature':temperature,'best_epoch':best_state['epoch'],
                  'uncalibrated':classification_metrics(y, probabilities(z))})
        json_write(folder/'test_metrics.json',m)
        event(folder,kind='complete_test',n=len(y),accuracy=m['accuracy'],auroc=m['auroc'])
    print(c.experiment_id, 'validation' if validation_only else f"test accuracy={m['accuracy']:.4f}", flush=True)
    return folder


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--output',default='runs')
    p.add_argument('--data-dir',default='data')
    p.add_argument('--validation-only',action='store_true')
    a=p.parse_args()
    train(Config.read(a.config),a.output,a.data_dir,a.validation_only)


if __name__=='__main__':
    main()
