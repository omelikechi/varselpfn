# Train or fine-tune
'''
Trains the network on datasets simulated from the prior in prior.py, or fine-tunes CD-PFN or PC-PFN on a new prior:

    python -m varselpfn.train --config cd_pfn --run-dir my_run                 # train CD-PFN's prior from scratch
    python -m varselpfn.train --config my_prior.json --init cd_pfn --run-dir my_run   # fine-tune CD-PFN

--config is a bundled config name ('cd_pfn', 'pc_pfn') or a path to a JSON file with the same blocks: prior (see
prior.py), model, and train. The prior's "family" can also be a path to a Python file, relative to the config, that
defines sample_n, sample_p, sample_dataset, and sample_batch as in prior.py; datasets are then simulated from it.
--init starts from existing weights with a fresh optimizer: a bundled name or a path to a
.safetensors file or a checkpoint. Use the fine-tuned weights with CDPFN(weights='my_run/best.pt').

Freshly simulated datasets are streamed from worker processes. Each batch draws n and p from the configured ranges;
the batch size is chosen so each batch has about cells_per_batch cells. After warmup, the learning rate follows cosine
decay or warmup-stable-decay (saving predecay.pt at the end of the stable phase, from which a longer run can resume
with --resume after copying it to last.pt), driven by step or wall-time progress, whichever is further along.
Validation BCE on a fixed set of datasets of varied size selects the checkpoint; for a mixture prior, the validation
set has equal numbers of datasets from each family and BCE is also logged per family; with val_match_training (train
config), checkpoints are selected on BCE averaged per dataset, then over datasets, with families weighted by
component_weights, rather than on BCE pooled over covariates. Stops at max_steps or max_train_minutes.
head_weights (train config) weight the conditional dependence, parent, and child losses: [1, 0, 0] for CD-PFN and
[0, 1, 1] for PC-PFN; validation BCE (and checkpoint selection) uses the same weights.
Writes log.jsonl, best.pt, last.pt, prevalence.json, and train_summary.json to the run directory.
'''

import argparse
from functools import cache
import importlib.util
import json
import math
import os
from pathlib import Path
import pickle
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset

from varselpfn import prior as general_prior
from varselpfn.api import load_weights
from varselpfn.model import Network, count_parameters
from varselpfn.prior import sample_dataset_scm, sample_dataset_xfirst

head_names = ['cd', 'parent', 'child']
config_dir = Path(__file__).resolve().parent / 'configs'

#----------------------------------------------------------------
# Data stream
#----------------------------------------------------------------
class PriorStream(IterableDataset):
    '''Endless batches; each worker has its own random stream derived from (seed, start step, worker id).'''
    def __init__(self, prior, entropy, cells_per_batch, max_batch_size):
        self.prior = prior
        self.entropy = entropy
        self.cells_per_batch = cells_per_batch
        self.max_batch_size = max_batch_size

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        if worker is not None:
            # one BLAS thread per worker: workers run in parallel, and multithreaded numpy in each oversubscribes the CPUs
            threadpool_limits(1)
        rng = np.random.default_rng(np.random.SeedSequence(self.entropy + [0 if worker is None else worker.id]))
        while True:
            sampler = prior_module(self.prior)
            n = sampler.sample_n(rng, self.prior)
            p = sampler.sample_p(rng, self.prior)
            batch_size = int(np.clip(self.cells_per_batch // (n * (p + 1)), 1, self.max_batch_size))
            X, y, labels, failures, _ = sampler.sample_batch(rng, self.prior, batch_size, n=n, p=p)
            yield torch.from_numpy(X), torch.from_numpy(y), torch.from_numpy(labels), failures

#----------------------------------------------------------------
# Helpers
#----------------------------------------------------------------
@cache
def load_module(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def prior_module(prior):
    '''Sampler for the prior's family: the general prior (prior.py) or a user-supplied Python file.'''
    family = prior.get('family', 'general')
    return load_module(family) if family.endswith('.py') else general_prior

def load_config(config):
    '''A bundled config name or a path to a JSON file; a prior family given as a Python file is resolved relative to it.'''
    path = config_dir / f'{config}.json' if (config_dir / f'{config}.json').exists() else Path(config).expanduser()
    with open(path) as f:
        config = json.load(f)
    family = config['prior'].get('family', 'general')
    if family.endswith('.py'):
        config['prior']['family'] = str((path.parent / family).resolve())
    return config

def get_device(name='auto'):
    if name == 'auto':
        name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    return torch.device(name)

def peak_memory_mb(device):
    out = {}
    if device.type == 'cuda':
        out['cuda_peak_allocated_mb'] = torch.cuda.max_memory_allocated() / 2**20
    return out

def head_bce(logits, labels):
    return F.binary_cross_entropy_with_logits(logits.float(), labels, reduction='none').mean(dim=(0, 1))

@torch.no_grad()
def validation_bce(model, val, device, autocast, weights, family_weights=None):
    '''
    Mean BCE per head over all covariates of all validation datasets (datasets have different sizes), and its weighted
    mean over heads ('mean'). val maps a family name to its datasets; with several families, also returns each family's
    weighted mean BCE. If family_weights is given (matching the training objective, where each batch has one size),
    BCE is instead averaged within each dataset, then over datasets, and the per-head values and 'mean' weight the
    families by family_weights.
    '''
    model.eval()
    total = torch.zeros(3)
    count = 0
    by_family = {}
    family_means = {}
    family_heads = {}
    for family, datasets in val.items():
        family_total = torch.zeros(3)
        family_count = 0
        dataset_total = torch.zeros(3)
        for X, y, labels in datasets:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                logits = model(torch.as_tensor(X, device=device)[None], torch.as_tensor(y, device=device)[None])[0]
            bce = F.binary_cross_entropy_with_logits(logits.float().cpu(), torch.as_tensor(labels), reduction='none')
            family_total += bce.sum(dim=0)
            family_count += len(labels)
            dataset_total += bce.mean(dim=0)
        total += family_total
        count += family_count
        per_head_family = dataset_total / len(datasets) if family_weights is not None else family_total / family_count
        family_heads[family] = per_head_family.numpy()
        family_means[family] = float(np.average(family_heads[family], weights=weights))
        by_family[f'mean_{family}'] = family_means[family]
    if family_weights is not None:
        per_head = np.average([family_heads[family] for family in val], axis=0, weights=[family_weights.get(family, 1.0) for family in val])
    else:
        per_head = (total / count).numpy()
    mean = float(np.average(per_head, weights=weights))
    return dict(zip(head_names, per_head.tolist())) | {'mean': mean} | (by_family if len(val) > 1 else {})

def save_checkpoint(path, model, optimizer, step, extra, config):
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step, 'extra': extra,
        'config': config}, path)

def progress(step, minutes, tc):
    '''Fraction of the budget used after warmup: the larger of step progress and wall-time progress.'''
    return min(1.0, max((step - tc['warmup_steps']) / max(1, tc['max_steps'] - tc['warmup_steps']), minutes / tc['max_train_minutes']))

def learning_rate(step, minutes, tc):
    '''
    Linear warmup, then either cosine decay over the whole budget (schedule 'cosine') or warmup-stable-decay
    (schedule 'wsd'): constant until the last decay_fraction of the budget, then linear decay to min_lr_ratio * lr.
    '''
    if step < tc['warmup_steps']:
        return tc['lr'] * (step + 1) / tc['warmup_steps']
    t = progress(step, minutes, tc)
    if tc.get('schedule', 'cosine') == 'wsd':
        start = 1 - tc['decay_fraction']
        return tc['lr'] * (1 - (1 - tc['min_lr_ratio']) * max(0.0, t - start) / tc['decay_fraction'])
    return tc['lr'] * (tc['min_lr_ratio'] + (1 - tc['min_lr_ratio']) * 0.5 * (1 + math.cos(math.pi * t)))

#----------------------------------------------------------------
# Main
#----------------------------------------------------------------
if __name__ == '__main__':
    print()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help="bundled config name ('cd_pfn', 'pc_pfn') or path to a JSON config")
    parser.add_argument('--run-dir', required=True, help='directory for checkpoints and logs')
    parser.add_argument('--init', default=None, help="weights to start from (fresh optimizer): 'cd_pfn', 'pc_pfn', or a path")
    parser.add_argument('--device', default='auto')
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--max-minutes', type=float, default=None, help="training time limit, overriding the config's max_train_minutes")
    parser.add_argument('--workers', type=int, default=None, help='data worker processes (default: the config value, at most the number of CPUs minus one)')
    parser.add_argument('--resume', action='store_true', help='continue from last.pt in the run directory')
    args = parser.parse_args()
    # expandable segments reduce CUDA memory fragmentation, which can otherwise exhaust memory on the largest batches
    os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

    config = load_config(args.config)
    config['run_dir'] = str(Path(args.run_dir).expanduser())
    tc = config['train']
    if args.max_steps is not None:
        tc['max_steps'] = args.max_steps
    if args.max_minutes is not None:
        tc['max_train_minutes'] = args.max_minutes
    tc['workers'] = args.workers if args.workers is not None else min(tc['workers'], max(os.cpu_count() - 1, 0))
    prior = config['prior']
    head_weights = np.array(tc.get('head_weights', [1.0, 1.0, 1.0]), dtype=float)
    if 'child_keep_range' in prior.get('xfirst', {}) and head_weights[0] != 0:
        # with child_keep_range the X-first conditional dependence label is a subset of the conditional dependence set
        raise ValueError('xfirst child_keep_range requires head_weights[0] = 0 (conditional dependence labels are not exact)')
    run_dir = Path(config['run_dir'])
    run_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)
    autocast = device.type == 'cuda'
    print(f'device {device}; bf16 autocast {autocast}; run dir {run_dir}')
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    #----------------------------------------------------------------
    # Prevalence (training-only labels) and validation set
    #----------------------------------------------------------------
    rng_prev = np.random.default_rng(np.random.SeedSequence([config['seed'], 1]))
    prev_labels = [prior_module(prior).sample_dataset(rng_prev, prior)[2] for _ in range(tc['prevalence_datasets'])]
    prevalence = np.concatenate(prev_labels).mean(axis=0)
    with open(run_dir / 'prevalence.json', 'w') as f:
        json.dump({'datasets': tc['prevalence_datasets'], 'prevalence': dict(zip(head_names, prevalence.tolist()))}, f, indent=2)
    print('training-label prevalence (pooled over covariates):', dict(zip(head_names, prevalence.round(4).tolist())))

    rng_val = np.random.default_rng(np.random.SeedSequence([config['seed'], 2]))
    weights = prior.get('component_weights', {'scm': 1.0})
    if len(weights) == 1:
        val = {'all': [prior_module(prior).sample_dataset(rng_val, prior)[:3] for _ in range(tc['val_datasets'])]}
    else:
        samplers = {'scm': sample_dataset_scm, 'xfirst': sample_dataset_xfirst}
        def val_dataset(family):
            n, p = general_prior.sample_n(rng_val, prior), general_prior.sample_p(rng_val, prior)
            return samplers[family](rng_val, prior, n, p)[:3]
        val = {family: [val_dataset(family) for _ in range(tc['val_datasets'] // len(weights))] for family in weights}
    val_family_weights = weights if tc.get('val_match_training', False) else None
    with open(run_dir / 'val.pkl', 'wb') as f:
        pickle.dump(val, f)

    #----------------------------------------------------------------
    # Model and optimizer
    #----------------------------------------------------------------
    torch.manual_seed(config['seed'])
    if device.type == 'cuda' and torch.cuda.get_device_properties(device).total_memory < 12 * 2**30:
        # smaller GPUs: recompute block activations in the backward pass so the largest datasets fit
        config['model']['grad_checkpoint'] = True
        print('GPU memory < 12 GB: gradient checkpointing on')
    model = Network(**config['model'])
    model.set_output_bias(prevalence)
    if args.init is not None and not (args.resume and (run_dir / 'last.pt').exists()):
        model.load_state_dict(load_weights(args.init)[1])
        print(f'initialized from {args.init}')
    model.to(device)
    print('parameters:', count_parameters(model))
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc['lr'], weight_decay=tc['weight_decay'])
    step = 0
    best = {'val_mean_bce': math.inf, 'step': None}
    prior_minutes = 0.0
    if args.resume and (run_dir / 'last.pt').exists():
        checkpoint = torch.load(run_dir / 'last.pt', map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step = checkpoint['step']
        best = checkpoint['extra']['best']
        prior_minutes = checkpoint['extra']['train_minutes']
        print(f'resumed at step {step}')

    # the data stream is re-seeded from the start step, so a resumed run continues with fresh datasets
    stream = PriorStream(prior, [config['seed'], 3, step], tc['cells_per_batch'], tc['max_batch_size'])
    loader = DataLoader(stream, batch_size=None, num_workers=tc['workers'], persistent_workers=tc['workers'] > 0,
        prefetch_factor=4 if tc['workers'] > 0 else None)

    #----------------------------------------------------------------
    # Training loop
    #----------------------------------------------------------------
    log_file = open(run_dir / 'log.jsonl', 'a')
    if step == 0:
        # validation before any update: the baseline for fine-tuning
        record = {'step': 0, 'val_bce': validation_bce(model, val, device, autocast, head_weights, val_family_weights)}
        log_file.write(json.dumps(record) + '\n')
        print(f"step 0  val {record['val_bce']}", flush=True)
    t_start = time.time()
    wait_seconds = 0.0
    window_loss, window_cells, window_batch = [], [], []
    failures = 0
    oom_skips = 0
    skipped_shapes = []
    stop_reason = 'max_steps'
    t_wait = time.time()
    for X, y, labels, batch_failures in loader:
        wait_seconds += time.time() - t_wait
        minutes = prior_minutes + (time.time() - t_start) / 60
        if step >= tc['max_steps']:
            break
        if minutes >= tc['max_train_minutes']:
            stop_reason = 'time_limit'
            break
        failures += int(batch_failures)
        X, y, labels = X.to(device, non_blocking=True), y.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        if tc.get('schedule', 'cosine') == 'wsd' and step >= tc['warmup_steps'] and progress(step, minutes, tc) >= 1 - tc['decay_fraction'] \
                and not (run_dir / 'predecay.pt').exists():
            # end of the stable phase: to train longer, resume from this checkpoint with a larger max_train_minutes
            save_checkpoint(run_dir / 'predecay.pt', model, optimizer, step, {'best': best, 'train_minutes': minutes}, config)
            print(f'step {step}: saved predecay.pt at {minutes:.1f} min', flush=True)
        for group in optimizer.param_groups:
            group['lr'] = learning_rate(step, minutes, tc)
        model.train()
        # the largest batches recompute block activations in the backward pass instead of storing them
        model.grad_checkpoint = config['model']['grad_checkpoint'] or X.shape[0] * X.shape[1] * (X.shape[2] + 1) > tc.get('checkpoint_cells', float('inf'))
        out_of_memory = False
        try:
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=autocast):
                logits = model(X, y)
            per_head = head_bce(logits, labels)
            loss = (per_head * torch.as_tensor(head_weights, device=per_head.device, dtype=per_head.dtype)).sum() / head_weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
        except torch.OutOfMemoryError:
            out_of_memory = True
        if out_of_memory:
            # skip the batch rather than lose the run (memory is freed outside the except block, where the traceback
            # no longer holds the tensors); repeated failures still stop training
            optimizer.zero_grad(set_to_none=True)
            logits = per_head = loss = None
            torch.cuda.empty_cache()
            oom_skips += 1
            skipped_shapes.append(tuple(X.shape))
            print(f'step {step}: out of memory on a batch of shape {tuple(X.shape)}; skipped ({oom_skips} so far)', flush=True)
            if oom_skips > tc.get('max_oom_skips', 20):
                # keep the progress, then stop (the job script requeues and resumes from last.pt)
                save_checkpoint(run_dir / 'last.pt', model, optimizer, step, {'best': best, 'train_minutes': minutes}, config)
                raise RuntimeError(f'more than {tc.get("max_oom_skips", 20)} batches ran out of memory: {skipped_shapes}')
            t_wait = time.time()
            continue
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc['grad_clip'])
        if torch.isfinite(grad_norm):
            optimizer.step()
        step += 1
        window_loss.append(per_head.detach().cpu().numpy())
        window_cells.append(X.shape[0] * X.shape[1] * (X.shape[2] + 1))
        window_batch.append(X.shape[0])

        if step % tc['log_every'] == 0 or step % tc['val_every'] == 0:
            minutes = prior_minutes + (time.time() - t_start) / 60
            record = {'step': step, 'train_bce': dict(zip(head_names, np.mean(window_loss, axis=0).tolist())),
                'mean_cells': float(np.mean(window_cells)), 'mean_batch_size': float(np.mean(window_batch)),
                'grad_norm': float(grad_norm), 'lr': learning_rate(step, minutes, tc), 'elapsed_minutes': minutes,
                'data_wait_seconds': wait_seconds, 'generator_failures': failures, 'oom_skips': oom_skips}
            window_loss, window_cells, window_batch = [], [], []
            if step % tc['val_every'] == 0:
                record['val_bce'] = validation_bce(model, val, device, autocast, head_weights, val_family_weights)
                record['memory'] = peak_memory_mb(device)
                extra = {'best': best, 'train_minutes': minutes}
                if record['val_bce']['mean'] < best['val_mean_bce']:
                    best = {'val_mean_bce': record['val_bce']['mean'], 'step': step, 'val_bce': record['val_bce']}
                    extra['best'] = best
                    save_checkpoint(run_dir / 'best.pt', model, optimizer, step, extra, config)
                save_checkpoint(run_dir / 'last.pt', model, optimizer, step, extra, config)
                print(f"step {step}  train {np.round(list(record['train_bce'].values()), 4)}  val {np.round(list(record['val_bce'].values()), 4)}  "
                    f"{minutes:.1f} min  data wait {wait_seconds:.0f}s", flush=True)
            log_file.write(json.dumps(record) + '\n')
            log_file.flush()
        t_wait = time.time()
    log_file.close()

    minutes = prior_minutes + (time.time() - t_start) / 60
    final_val = validation_bce(model, val, device, autocast, head_weights, val_family_weights)
    extra = {'best': best, 'train_minutes': minutes}
    if final_val['mean'] < best['val_mean_bce']:
        best = {'val_mean_bce': final_val['mean'], 'step': step, 'val_bce': final_val}
        extra['best'] = best
        save_checkpoint(run_dir / 'best.pt', model, optimizer, step, extra, config)
    save_checkpoint(run_dir / 'last.pt', model, optimizer, step, extra, config)
    summary = {'device': str(device), 'parameters': count_parameters(model), 'steps': step, 'stop_reason': stop_reason,
        'train_minutes': minutes, 'data_wait_seconds': wait_seconds, 'generator_failures': failures, 'oom_skips': oom_skips, 'skipped_shapes (batch, n, p)': skipped_shapes, 'final_val_bce': final_val,
        'best': best, 'memory': peak_memory_mb(device)}
    with open(run_dir / 'train_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
