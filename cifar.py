import os
import torch
import torch.nn.functional as F
from homura import optim, lr_scheduler, callbacks, reporters
from homura.trainers import SupervisedTrainer as Trainer
from homura.vision.data.loaders import vision_loaders

from senet.baseline import resnet20
from senet.baseline_ASP import resnet20_ASP
from senet.baseline_ASP import resnet32_ASP
from senet.se_resnet import se_resnet20
from senet.se_resnet import se_resnet18
from senet.se_resnet import se_resnet50



def main():
    train_loader, test_loader = vision_loaders(args.data_name, args.batch_size, download=True, num_workers=1)

    if args.baseline:
        model = resnet20()
    else:
        # model = se_resnet18(num_classes=1000)
        # model = se_resnet50(num_classes=100)
        model = resnet20_ASP(num_classes=10)
        # model = resnet32_ASP(num_classes=10)

    optimizer = optim.SGD(lr=1e-1, momentum=0.9, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(80, 0.1)
    tqdm_rep = reporters.TQDMReporter(range(args.epochs))
    _callbacks = [tqdm_rep, callbacks.AccuracyCallback()]
    with Trainer(model, optimizer, F.cross_entropy, scheduler=scheduler, callbacks=_callbacks) as trainer:
        for i in tqdm_rep:
            trainer.train(train_loader)
            trainer.test(test_loader)


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=250)
    p.add_argument("--data_name", type=str, default='cifar10')
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--reduction", type=int, default=16)
    p.add_argument("--baseline", action="store_true")
    args = p.parse_args()
    main()
