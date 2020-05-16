import copy
from os.path import join

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class _Trainer:
    def __init__(
        self,
        train_loader,
        test_loader,
        model,
        criterion,
        optimizer,
        lr_scheduler,
        num_epochs,
        device,
        interval,
        output_dir,
        project_name,
    ):
        self._train_loader = train_loader
        self._test_loader = test_loader
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        self._num_epochs = num_epochs
        self._device = device
        self._interval = interval
        self._output_dir = output_dir

        self._model = self._model.to(self._device)
        self._writer = SummaryWriter(project_name)

        self._best_model = None
        self._best_epoch = -1
        self._best_metric = 10e9  # loss value

    def fit(self):
        for epoch in tqdm(range(self._num_epochs)):
            self._train_epoch(epoch)
            self._test_epoch(epoch)

            self._writer.add_scalar(
                "lr", self._get_lr(self._optimizer), epoch + 1
            )

            if (epoch + 1) % self._interval == 0:
                self._save_model(epoch)

        print("Best epoch was {}.".format(self._best_epoch + 1))
        self._save_model("best_model")

    def _train_epoch(self, epoch):
        pass

    def _test_epoch(self, epoch):
        pass

    def _save_model(self, keyword):
        if isinstance(keyword, (int, float)):
            filename = "triplet_{:>04d}.pth".format(keyword + 1)
        else:
            filename = "triplet_{}.pth".format(keyword)

        save_path = join(self._output_dir, filename)
        torch.save(self._model.embedding_net.state_dict(), save_path)

    @staticmethod
    def _get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]


class TripletTrainer(_Trainer):
    def _train_epoch(self, epoch):
        self._model.train()
        epoch_loss = 0.0

        for images, _ in self._train_loader:
            images = tuple(image.to(self._device) for image in images)

            self._optimizer.zero_grad()

            outputs = self._model(*images)
            loss = self._criterion(*outputs)
            epoch_loss += loss.item()
            loss.backward()
            self._optimizer.step()

        self._lr_scheduler.step()

        self._writer.add_scalar(
            "train/loss",
            epoch_loss / len(self._train_loader.dataset),
            epoch + 1,
        )

    @torch.no_grad()
    def _test_epoch(self, epoch):
        self._model.eval()
        epoch_loss = 0.0

        for images, _ in self._test_loader:
            images = tuple(image.to(self._device) for image in images)

            outputs = self._model(*images)
            loss = self._criterion(*outputs)
            epoch_loss += loss.item()

        self._writer.add_scalar(
            "test/loss", epoch_loss / len(self._test_loader.dataset), epoch + 1
        )

        if epoch_loss < self._best_metric:
            self._best_model = copy.deepcopy(self._model).cpu()
            self._best_epoch = epoch
            self._best_metric = epoch_loss


QuadrupletTrainer = TripletTrainer


class ArcfaceTrainer(_Trainer):
    def _train_epoch(self, epoch):
        self._model.train()
        epoch_loss = 0.0

        for images, labels in self._train_loader:
            images = images.to(self._device)
            labels = labels.to(self._device)

            self._optimizer.zero_grad()

            outputs = self._model(images, labels)
            loss = self._criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            self._optimizer.step()

        self._lr_scheduler.step()

        self._writer.add_scalar(
            "train/loss",
            epoch_loss / len(self._train_loader.dataset),
            epoch + 1,
        )

    @torch.no_grad()
    def _test_epoch(self, epoch):
        self._model.eval()
        epoch_loss = 0.0

        for images, labels in self._test_loader:
            images = images.to(self._device)
            labels = labels.to(self._device)

            outputs = self._model(images, labels)
            loss = self._criterion(outputs, labels)
            epoch_loss += loss.item()

        self._writer.add_scalar(
            "test/loss", epoch_loss / len(self._test_loader.dataset), epoch + 1
        )
