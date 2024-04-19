import torch
import torch.nn as nn
import lightning as L

class C3D(L.LightningModule):
    def __init__(self, img_res=64, num_frames=10, num_classes=10):
        super().__init__()

        self.example_input_array = torch.Tensor(1, 1, 20, 64, 64)

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_accuracy = Accuracy(task="multiclasS", num_classes=num_classes)

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

        out = out.view(-1, 8192)
        out = self.fc6(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc7(out)
        out = self.relu(out)
        out = self.dropout(out)

        logits = self.fc8(out)

        return logits

    def loss_fn(self, out, target):
        return nn.CrossEntropyLoss()(out, target)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        video, label = batch
        video = video.permute(0, 2, 1, 3, 4)
        # label = label.view(-1) what does this do???
        out = self(video)
        prob = nn.Softmax()(out)
        pred = torch.argmax(prob, dim=1)
        acc = self.train_accuracy(pred, label)
        loss = self.loss_fn(out, label)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        video, label = batch
        video = video.permute(0, 2, 1, 3, 4)
        # label = label.view(-1) what does this do???
        out = self(video)
        prob = nn.Softmax()(out)
        pred = torch.argmax(prob, dim=1)
        acc = self.train_accuracy(pred, label)
        loss = self.loss_fn(out, label)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        video, label = batch
        video = video.permute(0, 2, 1, 3, 4)
        # label = label.view(-1) what does this do???
        out = self(video)
        prob = nn.Softmax()(out)
        pred = torch.argmax(prob, dim=1)
        acc = self.train_accuracy(pred, label)
        loss = self.loss_fn(out, label)
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    


if __name__ == "__main__":
    model = C3D()
    dataset = ""
    trainer = L.Trainer(max_epochs=1)
    trainer.fit(model=model, datamodule=dataset)
    




class C3D(nn.Module):
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

    def __init__(self, img_res=64, num_frames=10, num_classes):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool1 = nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool2 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool3 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool4 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.pool5 = nn.AvgPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc6 = nn.Linear(8192, 4096)
        self.fc7 = nn.Linear(4096, 4096)
        self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

        self.__init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.relu(x)
        x = self.conv3b(x)
        x = self.relu(x)
        x = self.pool3(x)

        x = self.conv4a(x)
        x = self.relu(x)
        x = self.conv4b(x)
        x = self.relu(x)
        x = self.pool4(x)

        x = self.conv5a(x)
        x = self.relu(x)
        x = self.conv5b(x)
        x = self.relu(x)
        x = self.pool5(x)

        x = x.view(-1, 8192)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc7(x)
        x = self.relu(x)
        x = self.dropout(x)

        logits = self.fc8(x)

        return logits

