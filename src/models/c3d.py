import torch
import torch.nn as nn
import lightning as L
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score


# modified from https://github.com/jfzhang95/pytorch-video-recognition/blob/master/network/C3D_model.py
class C3D(L.LightningModule):
    """
    The C3D network.

    conv1   in:10*3*64*64    out:10*64*64*64
    pool1   in:10*64*32*32   out:10*64*32*32

    conv2   in:10*64*32*32   out:10*128*32*32
    pool2   in:10*128*32*32  out:5*128*16*16

    conv3a  in:5*128*16*16   out:5*256*16*16
    conv3b  in:5*256*16*16   out:5*256*16*16
    pool3   in:5*256*16*16   out:2*256*8*8

    conv4a  in:2*256*8*8     out:2*512*8*8
    conv4b  in:2*512*8*8     out:2*512*8*8
    pool4   in:2*512*8*8     out:1*512*4*4

    conv5a  in:1*512*4*4     out:1*512*4*4
    conv5b  in:1*512*4*4     out:1*512*4*4
    pool5   in:1*512*4*4     out:1*512*3*3

    fc6     in:512*3*3       out:256*3*3
    fc7     in:256*3*3       out:256*3*3
    fc8     in:256*3*3       out:num_classes
    """

    def __init__(self, img_res=32, num_frames=8, num_classes=10):
        super().__init__()

        self.example_input_array = torch.Tensor(16, num_frames, img_res, img_res)

        self.conv1 = nn.Conv2d(num_frames, 64, kernel_size=(3, 3), padding=(1, 1))
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.pool2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3a = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        self.conv3b = nn.Conv2d(256, 256, kernel_size=(3, 3), padding=(1, 1))
        self.pool3 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv4a = nn.Conv2d(256, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv4b = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.pool4 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv5a = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.conv5b = nn.Conv2d(512, 512, kernel_size=(3, 3), padding=(1, 1))
        self.pool5 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(2048, 1024)
        self.fc7 = nn.Linear(1024, 1024)
        self.fc8 = nn.Linear(1024, num_classes)

        self.flatten = nn.Flatten(start_dim=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclasS", num_classes=num_classes)

        self.precision = Precision(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self.recall = Recall(
            task="multiclass", average="macro", num_classes=num_classes
        )
        self.f1 = F1Score(task="multiclass", average="macro", num_classes=num_classes)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.pool1(out)

        out = self.relu(self.conv2(x))
        out = self.pool2(out)

        out = self.relu(self.conv3a(out))
        out = self.relu(self.conv3b(out))
        out = self.pool3(out)

        out = self.relu(self.conv4a(out))
        out = self.relu(self.conv4b(out))
        out = self.pool4(out)

        out = self.relu(self.conv5a(out))
        out = self.relu(self.conv5b(out))
        out = self.pool5(out)

        out = self.flatten(out)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc7(out)
        out = self.relu(out)
        out = self.dropout(out)

        logits = self.fc8(out)

        return logits

    def loss_fn(self, out, target):
        loss = nn.CrossEntropyLoss()(out, target)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.003 / 255)

    def training_step(self, batch, batch_idx):
        video, label = batch
        logits = self(video)
        loss = self.loss_fn(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = self.train_accuracy(pred, label)

        # record metrics
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        video, label = batch
        logits = self(video)
        loss = self.loss_fn(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(pred, label)

        # record metrics
        self.log("val_acc", acc, on_step=False, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        video, label = batch
        logits = self(video)
        loss = self.loss_fn(logits, label)
        pred = torch.argmax(logits, dim=1)
        acc = self.val_accuracy(pred, label)
        precision = self.precision(logits, label)
        recall = self.recall(logits, label)
        f1 = self.f1(logits, label)

        # record metrics
        self.log("test_acc", acc, on_step=False, on_epoch=True)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.log("precision", precision, on_step=False, on_epoch=True)
        self.log("recall", recall, on_step=False, on_epoch=True)
        self.log("f1", f1, on_step=False, on_epoch=True)

        return loss


class C3DSmall(L.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        self.example_input_array = torch.Tensor(16, 8, 32, 32)

        self.conv1 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=(1, 1))
        self.pool1 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.fc8 = nn.Linear(4096, num_classes)

        self.flatten = nn.Flatten(start_dim=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

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
        out = self.relu(self.conv1(x))
        out = self.pool1(out)

        out = self.flatten(out)
        out = self.dropout(out)

        logits = self.fc8(out)

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
    pass
