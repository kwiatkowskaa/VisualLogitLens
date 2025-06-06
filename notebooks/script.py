import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_root)

from metrics.CKA import linear_CKA
from metrics.CKNNA import AlignmentMetrics

def extract_layer_outputs(model, x):
    '''
    Extracts intermediate layer outputs from a ViT model during a forward pass.
    '''
    activations = []

    def hook_fn(module, input, output):
        activations.append(output)

    hooks = [blk.register_forward_hook(hook_fn) for blk in model.blocks]

    with torch.no_grad():
        _ = model(x)

    for hook in hooks:
        hook.remove()

    return activations


def compute_logits_from_layers(model, layer_outputs):
    '''
    Computes classification logits from intermediate layer representations using the model's head.
    '''
    cls_tokens = [output[:, 0] for output in layer_outputs]  # CLS token
    norm = model.norm
    head = model.head
    logits = [head(norm(cls)) for cls in cls_tokens]
    return logits


def collect_logits_from_batches(model, dataloader, max_batches=None):
    '''
    Collects logits from all transformer layers for multiple batches of data.
    '''
    all_logits = []

    for batch_idx, (x_batch, _) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x_batch = x_batch.to(next(model.parameters()).device)
        layer_outputs = extract_layer_outputs(model, x_batch)
        logits = compute_logits_from_layers(model, layer_outputs)
        all_logits.append(logits)

    return all_logits


def compute_cka_matrix(logits):
    L = len(logits)
    matrix = torch.zeros(L, L)
    logits = [F.normalize(logit.detach(), dim=-1) for logit in logits]

    for i in range(L):
        for j in range(i, L):
            val = linear_CKA(logits[i], logits[j])
            matrix[i, j] = val
            matrix[j, i] = val

    return matrix.numpy()


def compute_cosine_similarity_matrix(logits):
    """
    Fully vectorized computation of cosine similarity matrix between L layer outputs.
    Uses F.cosine_similarity and removes nested loops.
    """
    x = torch.stack([F.normalize(t.detach(), dim=-1) for t in logits])

    x1 = x[:, None, :, :] 
    x2 = x[None, :, :, :]

    cos = F.cosine_similarity(x1, x2, dim=-1)

    sim_matrix = cos.mean(dim=-1)

    return sim_matrix.cpu().numpy()


def compute_cknna_matrix(logits, k=10):
    L = len(logits)
    matrix = np.zeros((L, L))
    
    for i in range(L):
        for j in range(i, L):
            sim = AlignmentMetrics.cknna(logits[i], logits[j], topk=k)
            matrix[i, j] = sim
            matrix[j, i] = sim
    
    return matrix


@torch.no_grad()
def forward_with_skipping(model, x, skip_layers=[]):
    x = model.patch_embed(x)
    cls_token = model.cls_token.expand(x.shape[0], -1, -1)
    x = torch.cat((cls_token, x), dim=1)
    x = x + model.pos_embed
    x = model.pos_drop(x)

    for i, blk in enumerate(model.blocks):
        if i in skip_layers:
            continue
        x = blk(x)

    x = model.norm(x)
    cls_token_out = x[:, 0]
    return cls_token_out


@torch.no_grad()
def collect_outputs_with_skipping(model, dataloader, skip_layers=[], max_batches=None, device='cpu'):
    outputs = []
    all_labels = []
    for i, (images, labels) in enumerate(tqdm(dataloader, desc="Collecting outputs")):
        if max_batches and i >= max_batches:
            break
        images = images.to(device)
        cls_token_out = forward_with_skipping(model, images, skip_layers)
        outputs.append(cls_token_out.cpu())
        all_labels.extend(labels.numpy())

    return torch.cat(outputs).numpy(), np.array(all_labels)


def evaluate_full_model(model, dataloader, linear_classifier, max_layer=12, max_batches=None, device='cpu'):
    results = []

    print("\n[INFO] Evaluating full model...\n")
    
    skipped = []
    X_val, y_val = collect_outputs_with_skipping(model, dataloader, skip_layers=skipped, max_batches=max_batches)

    inputs = torch.tensor(X_val, dtype=torch.float32).to(device)
    with torch.no_grad():
        outputs = linear_classifier(inputs)
        y_pred = outputs.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_val, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

    print(f"→ Accuracy: {acc:.4f}")

    results.append({
        'skipped_layers': skipped,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1
    })

    print(results)


def evaluate_skipped_layers(model, dataloader, linear_classifier, max_layer=12, max_batches=None, device='cpu'):
    results = []

    print("\n[INFO] Evaluating individual skipped layers...\n")
    for i in range(max_layer):
        skipped = [i]
        print(f"\n[INFO] Skipping layer: {skipped}")
        X_val, y_val = collect_outputs_with_skipping(model, dataloader, skip_layers=skipped, max_batches=max_batches)

        inputs = torch.tensor(X_val, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = linear_classifier(inputs)
            y_pred = outputs.argmax(dim=1).cpu().numpy()

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_val, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

        print(f"→ Accuracy: {acc:.4f}")

        results.append({
            'skipped_layers': skipped,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        })

    print("\n[INFO] Evaluating pairs of consecutive skipped layers...\n")
    for i in range(max_layer - 1):
        skipped = [i, i + 1]
        print(f"\n[INFO] Skipping layers: {skipped}")
        X_val, y_val = collect_outputs_with_skipping(model, dataloader, skip_layers=skipped, max_batches=max_batches)

        inputs = torch.tensor(X_val, dtype=torch.float32).to(device)
        with torch.no_grad():
            outputs = linear_classifier(inputs)
            y_pred = outputs.argmax(dim=1).cpu().numpy()

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_val, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

        results.append({
            'skipped_layers': skipped,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1
        })

    return results