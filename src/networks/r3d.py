import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score

# modified from https://github.com/jfzhang95/pytorch-video-recognition/blob/master/network/R3D_model.py
# paper : https://arxiv.org/abs/1711.11248


class Conv(nn.Module):

    def __init__(
        self, in_channels, out_channels, kernel_size, stride=(1, 1), padding=(0, 0)
    ):
        super(Conv, self).__init__()

        self.st_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.st_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, downsample=False):
        super(ResBlock, self).__init__()

        self.downsample = downsample

        padding = tuple(i // 2 for i in kernel_size)

        if self.downsample:
            self.downsampleconv = Conv(
                in_channels, out_channels, kernel_size=(1, 1), stride=(2, 2)
            )
            self.downsamplebn = nn.BatchNorm2d(out_channels)
            self.conv1 = Conv(
                in_channels, out_channels, kernel_size, padding=padding, stride=(2, 2)
            )
        else:
            self.conv1 = Conv(in_channels, out_channels, kernel_size, padding=padding)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = Conv(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.outrelu = nn.ReLU()

    def forward(self, x):
        res = self.relu1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))

        if self.downsample:
            x = self.downsamplebn(self.downsampleconv(x))

        res = self.outrelu(x + res)

        return res


class ResLayer(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        layer_size,
        block_type=ResBlock,
        downsample=False,
    ):
        super(ResLayer, self).__init__()

        self.block1 = block_type(in_channels, out_channels, kernel_size, downsample)

        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            self.blocks += [block_type(out_channels, out_channels, kernel_size)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)

        return x


class R3D(L.LightningModule):

    def __init__(self, num_classes, layer_sizes, block_type=ResBlock):
        super().__init__()

        self.example_input_array = torch.Tensor(16, 8, 32, 32)

        self.conv1 = Conv(8, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv2 = ResLayer(
            64, 64, kernel_size=(3, 3), layer_size=layer_sizes[0], block_type=block_type
        )
        self.conv3 = ResLayer(
            64,
            128,
            kernel_size=(3, 3),
            layer_size=layer_sizes[1],
            block_type=block_type,
            downsample=True,
        )
        self.conv4 = ResLayer(
            128,
            256,
            kernel_size=(3, 3),
            layer_size=layer_sizes[2],
            block_type=block_type,
            downsample=True,
        )
        self.conv5 = ResLayer(
            256,
            512,
            kernel_size=(3, 3),
            layer_size=layer_sizes[3],
            block_type=block_type,
            downsample=True,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(512, num_classes)

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.precision = Precision(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self.recall = Recall(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self.f1 = F1Score(task="multiclass", average="macro", num_classes=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        x = self.pool(x)
        x = self.flatten(x)
        logits = self.linear(x)

        return logits

    def loss_fn(self, out, target):
        loss = nn.CrossEntropyLoss()(out, target)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0003 / 255)
        return optimizer

    def training_step(self, batch, batch_idx):
        video, label = batch
        logits = self(video)
        loss = self.loss_fn(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(pred, label)

        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        video, label = batch
        logits = self(video)
        loss = self.loss_fn(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(pred, label)

        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        video, label = batch
        logits = self(video)
        loss = self.loss_fn(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(pred, label)
        precision = self.precision(logits, label)
        recall = self.recall(logits, label)
        f1 = self.f1(logits, label)

        self.log("test_acc", acc, on_step=False, on_epoch=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("precision", precision, on_step=False, on_epoch=True)
        self.log("recall", recall, on_step=False, on_epoch=True)
        self.log("f1", f1, on_step=False, on_epoch=True)

        return loss


class R3DSmall(L.LightningModule):
    def __init__(self, num_classes, layer_sizes, block_type=ResBlock):
        super().__init__()

        self.example_input_array = torch.Tensor(16, 8, 32, 32)

        self.conv1 = Conv(8, 16, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))
        self.conv2 = ResLayer(
            16, 16, kernel_size=(3, 3), layer_size=layer_sizes[0], block_type=ResBlock
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.linear = nn.Linear(16, num_classes)  # num classes

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

        self.precision = Precision(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self.recall = Recall(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self.f1 = F1Score(task="multiclass", average="macro", num_classes=num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.flatten(x)
        logits = self.linear(x)

        return logits

    def loss_fn(self, out, target):
        loss = nn.CrossEntropyLoss()(out, target)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.003 / 255)
        return optimizer

    def training_step(self, batch, batch_idx):
        video, label = batch
        logits = self(video)
        loss = self.loss_fn(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(pred, label)

        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        video, label = batch
        logits = self(video)
        loss = self.loss_fn(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(pred, label)

        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        video, label = batch
        logits = self(video)
        loss = self.loss_fn(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = self.test_accuracy(pred, label)
        precision = self.precision(logits, label)
        recall = self.recall(logits, label)
        f1 = self.f1(logits, label)

        self.log("test_acc", acc, on_step=False, on_epoch=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("precision", precision, on_step=False, on_epoch=True)
        self.log("recall", recall, on_step=False, on_epoch=True)
        self.log("f1", f1, on_step=False, on_epoch=True)

        return loss


if __name__ == "__main__":

    net = R3D((2, 2, 2, 2))  # for R3D with 18 layers
