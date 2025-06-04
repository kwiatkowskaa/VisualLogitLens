import os
import argparse
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
from torch import nn
from torchvision import transforms

# Utility classes for metrics and checkpointing
class MetricLogger:
    def __init__(self, delimiter="  "):
        self.meters = {}
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = SmoothedValue()
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def add_meter(self, name, meter=None):
        if meter is None:
            meter = SmoothedValue()
        self.meters[name] = meter

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if header is not None:
            print(header)
        for obj in tqdm(iterable):
            yield obj
            if i % print_freq == 0:
                print(str(self))
            i += 1

class SmoothedValue:
    def __init__(self, window_size=20, fmt='{value:.4f}'):
        self.deque = []
        self.window_size = window_size
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value):
        self.deque.append(value)
        if len(self.deque) > self.window_size:
            self.deque.pop(0)
        self.count += 1
        self.total += value

    @property
    def global_avg(self):
        return self.total / self.count if self.count > 0 else 0

    def __str__(self):
        return self.fmt.format(value=self.global_avg)

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class SelectedImageNet(torch.utils.data.Dataset):
    def __init__(self, root_dirs, selected_classes, transform=None):
        """
        Args:
            root_dirs (list or str): List of base directories for image classes
                                     (e.g., ['imagenet100/train.X1', 'imagenet100/train.X2'])
                                     or a single directory (e.g., 'imagenet100/val.X').
            selected_classes (list): List of class folder names to include.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dirs = [root_dirs] if isinstance(root_dirs, str) else root_dirs
        self.selected_classes = selected_classes
        self.transform = transform
        # Create a mapping from class name (e.g., 'n01632777') to an integer index (0, 1, 2...)
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(list(self.selected_classes)))}
        self.samples = self._find_samples()

    def _find_samples(self):
        samples = []
        for root_dir in self.root_dirs:
            if not os.path.isdir(root_dir):
                print(f"Warning: Directory not found {root_dir}, skipping.")
                continue
            for class_name in os.listdir(root_dir):
                if class_name in self.selected_classes:
                    class_path = os.path.join(root_dir, class_name)
                    if os.path.isdir(class_path):
                        for img_name in os.listdir(class_path):
                            img_path = os.path.join(class_path, img_name)
                            # Check if it's a file and has a common image extension
                            if os.path.isfile(img_path) and img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                                item = (img_path, self.class_to_idx[class_name])
                                samples.append(item)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        try:
            image = Image.open(img_path).convert('RGB') # Ensure 3 channels
        except FileNotFoundError:
            print(f"Warning: File not found {img_path}, skipping or returning None.")
            # Handle this case as per your requirement, e.g., return None or raise error
            # For DataLoader with default collate_fn, returning None might cause issues.
            # A robust way is to filter out bad samples during _find_samples or handle it in collate_fn.
            # For now, let's try to get the next valid sample if this happens, though it's not ideal.
            if len(self.samples) > 1:
                return self.__getitem__((idx + 1) % len(self.samples)) # Risky, could lead to infinite loop if all fail
            else:
                raise # Or return a placeholder if your training loop can handle it
        except Exception as e:
            print(f"Warning: Could not load image {img_path} due to {e}. Skipping.")
            if len(self.samples) > 1:
                return self.__getitem__((idx + 1) % len(self.samples)) # Risky
            else:
                raise

        if self.transform:
            image = self.transform(image)

        return image, label

# Selected ImageNet classes for the task
SELECTED_CLASSES = ['n01632777', 'n01984695', 'n01728572', 'n01514859',
                    'n01614925', 'n01582220', 'n01806143', 'n01819313']

def train_linear(args):
    # Determine device
    if args.device.startswith('cuda') and torch.cuda.is_available():
        device = torch.device('cuda')
    elif args.device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load model
    model = torch.hub.load('facebookresearch/dino:main', args.arch, pretrained=True)
    embed_dim = model.embed_dim

    # Move model to device
    model.to(device)
    model.eval()    

    # Create linear classifier
    linear_classifier = LinearClassifier(dim=embed_dim, num_labels=len(SELECTED_CLASSES))
    linear_classifier = linear_classifier.to(device)
    
    # Load train and val data
    train_dirs = [os.path.join(args.data_path, f"train.X{i}") for i in range(1, 5)]
    val_path = os.path.join(args.data_path, "val.X")
    
    # Load data
    train_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    
    train_subset = SelectedImageNet(train_dirs, SELECTED_CLASSES, train_transform)
    val_subset = SelectedImageNet(val_path, SELECTED_CLASSES, eval_transform)
    
    print(f"Train subset size: {len(train_subset)}")
    print(f"Val subset size: {len(val_subset)}")

    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Set optimizer
    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.lr * args.batch_size / 256., # linear scaling rule
        momentum=0.9,
        weight_decay=0,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    best_acc = 0.0
    for epoch in range(args.epochs):
        train_stats = train(model, linear_classifier, optimizer, train_loader, epoch, device, args)
        scheduler.step()
        print(f"Train stats: {train_stats}")

        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, linear_classifier, device, args)
            print(f"Accuracy at epoch {epoch} of the network on the {len(val_subset)} test images: {test_stats['acc1']:.1f}%")
            if test_stats["acc1"] > best_acc:
                torch.save({'model': linear_classifier.state_dict(), 'class_to_idx': train_subset.class_to_idx}, os.path.join(args.output_dir, "best_model.pth"))
            best_acc = max(best_acc, test_stats["acc1"])
            print(f'Max accuracy so far: {best_acc:.2f}%')
            
        save_dict = {
            "epoch": epoch + 1,
            "state_dict": linear_classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_acc": best_acc,
        }
        torch.save(save_dict, os.path.join(args.output_dir, f"checkpoint_{epoch}.pth"))

    print("Training of the supervised linear classifier on frozen features completed.\n"
          "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


def train(model, linear_classifier, optimizer, loader, epoch, device, args):
    linear_classifier.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.3f}'))
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    for (inp, target) in metric_logger.log_every(loader, 20, header):
        # move to device
        inp = inp.to(device)
        target = target.to(device)

        # forward
        with torch.no_grad():
            output = model(inp) 
        output = linear_classifier(output)

        # compute cross entropy loss
        loss = nn.CrossEntropyLoss()(output, target)

        # compute the gradients
        optimizer.zero_grad()
        loss.backward()

        # step
        optimizer.step()

        # log
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifier, device, args):
    linear_classifier.eval()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('acc1', SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('loss', SmoothedValue(window_size=1, fmt='{value:.3f}'))
    header = 'Test:'
    for inp, target in metric_logger.log_every(val_loader, 20, header):
        # move to device
        inp = inp.to(device)
        target = target.to(device)

        # forward
        output = model(inp)
        output = linear_classifier(output)
        loss = nn.CrossEntropyLoss()(output, target)

        acc1, = accuracy(output, target, topk=(1,))

        metric_logger.update(loss=loss.item())
        metric_logger.update(acc1=acc1.item())

    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--arch', default='dino_vits16', type=str, help='Architecture')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size', default=128, type=int, help='Per-GPU batch-size (now effectively total batch size)')
    parser.add_argument('--device', default='cpu', type=str, help="Device to use (e.g., 'cuda', 'mps', 'cpu')")
    parser.add_argument('--data_path', default='imagenet100', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="dino_out", help='Path to save logs and checkpoints')
    args = parser.parse_args()

    # Create output dir if it doesn't exist
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    train_linear(args)