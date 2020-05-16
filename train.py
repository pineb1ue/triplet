import argparse
import os
from os.path import join

import torch
import torch.optim as optim

from data import get_transforms
from models import get_embedding_net


def main(args):
    if args.metric in ("t", "triplet"):
        from data import TripletDataset
        from engine import TripletTrainer
        from losses import TripletLoss
        from models import TripletNet

        Dataset = TripletDataset
        Trainer = TripletTrainer
        Loss = TripletLoss
        Net = TripletNet

    elif args.metric in ("q", "quadruplet"):
        from data import QuadrupletDataset
        from engine import QuadrupletTrainer
        from losses import QuadrupletLoss
        from models import QuadrupletNet

        Dataset = QuadrupletDataset
        Trainer = QuadrupletTrainer
        Loss = QuadrupletLoss
        Net = QuadrupletNet

    elif args.metric in ("a", "arcface"):
        from data import Dataset
        from engine import ArcfaceTrainer
        from losses import FocalLoss
        from models import Arcface

        Trainer = ArcfaceTrainer
        Loss = FocalLoss
        Net = Arcface

    else:
        raise ValueError

    train_dataset = Dataset(
        root_dirs=args.train_dirs, transforms=get_transforms("train")
    )
    test_dataset = Dataset(
        root_dirs=args.test_dirs, transforms=get_transforms("test")
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    embedding_net = get_embedding_net(
        model=args.embedding_net,
        pretrained=args.pretrained,
        fine_tuning=args.fine_tuning,
        weights_path=args.weights_path,
    )
    if args.metric != "arcface":
        model = Net(embedding_net)
    else:
        model = Net(embedding_net, 1000, train_dataset.num_classes)

    criterion = Loss(*args.margin) if args.margin is not None else Loss()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate)
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.99
    )

    trainer = Trainer(
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=args.num_epochs,
        device=device,
        interval=args.interval,
        output_dir=args.output_dir,
        project_name=join("runs", args.output_dir.split("/")[-1]),
    )
    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-train", "--train-dirs", nargs="+")
    parser.add_argument("-test", "--test-dirs", nargs="+")
    parser.add_argument("-o", "--output-dir")
    parser.add_argument("-m", "--metric", default="triplet")

    parser.add_argument("-en", "--embedding-net")
    parser.add_argument("-pt", "--pretrained", action="store_true")
    parser.add_argument("-ft", "--fine-tuning", action="store_true")
    parser.add_argument("-w", "--weights-path")
    parser.add_argument("-mg", "--margin", nargs="+", type=float)

    parser.add_argument("-bs", "--batch-size", type=int, default=128)
    parser.add_argument("-nw", "--num-workers", type=int, default=8)
    parser.add_argument("-lr", "--learning-rate", type=float, default=0.01)
    parser.add_argument("-ne", "--num-epochs", type=int, default=100)
    parser.add_argument("-it", "--interval", type=int, default=10)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    main(args)
