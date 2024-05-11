import os
import urllib.request
from urllib.error import HTTPError
import io
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import pytorch_lightning as pl
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import torchvision
from IPython.display import set_matplotlib_formats
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, EarlyStopping
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm.notebook import tqdm
from torch.utils.data import DataLoader, Dataset
from helper import *
from models.unet.model import *
from models.smaatunet.model import *
from models.segformer.model import *
from models.vp_suite.models import *
import PIL.Image
import math

# Path to the folder where the pretrained models are saved
CHECKPOINT_PATH = os.environ.get("PATH_CHECKPOINT", "saved_models")

# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

gpu_id = 3
device = torch.device("cuda:%s"%gpu_id) if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

colev = ["#99DBEA", "#52A5D1", "#3753AD", "#80C505", "#52C10D",
         "#35972A", "#FAE33B", "#EAB81E", "#F78C2C", "#E2331F",
         "#992B27", "#471713", "#BC5CC2", "#975CC0"]
cmap = colors.ListedColormap(colev)
levels = np.arange(0, 71, 5)
norm = colors.BoundaryNorm(levels, cmap.N)

# Transformations applied on each image => only make them a tensor
transform = transforms.Compose([transforms.ToTensor(), transforms.CenterCrop(224)])#, transforms.Normalize((0.5,), (0.5,))])

# Loading the training dataset. We need to split it into a training and validation part
trainset = MetData(split='train', transform=transform)
validset = MetData(split='valid', transform=transform)
testset = MetData(split='test', transform=transform)

train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=32)
valid_loader = DataLoader(validset, batch_size=1, shuffle=False, num_workers=32)
test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=32)

model_kwargs = {
    "action_size": 0,
    "img_shape": (1, 224, 224),
    "temporal_dim": 3,
    "action_conditional": False,
    "tensor_value_range": [0.0, 1.0]
}
out_len = 20
model_name = PhyDNet # STPhy # EF_TrajGRU # EF_ConvLSTM # UNet3D #

class Autoencoder(pl.LightningModule):
    def __init__(
        self,
        model_class: object = model_name,
    ):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        # self.net = model_class(num_inchannels=5, num_classes=20, backbone='MobileNetV3-S')
        self.net = model_class(device, **model_kwargs).to(device)

    def forward(self, x):
        """The forward function takes in an image and returns the reconstructed image."""
        z = self.net(x, out_len)
        return z

    def _get_reconstruction_loss(self, batch):
        """Given a batch of images, this function returns the reconstruction loss (MSE in our case)"""
        x, y = batch[:,:5].unsqueeze(2), batch[:,5:].unsqueeze(2)  # We do not need the labels
        y_hat, _ = self.forward(x)
        loss = F.l1_loss(y, y_hat, reduction="none")
        loss = loss.mean(dim=[0,1,2,3,4])#.sum(dim=[1, 2, 3])
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.2, patience=20, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    def training_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)

class GenerateCallback(pl.Callback):
    def __init__(self, input_imgs, every_n_epochs=1):
        super().__init__()
        self.input_imgs = input_imgs  # Images to reconstruct during training
        # Only save those images every N epochs (otherwise tensorboard gets quite large)
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Reconstruct images
            target_imgs = self.input_imgs[:,5:]
            input_imgs = self.input_imgs[:,:5].unsqueeze(2).to(pl_module.device)
            with torch.no_grad():
                pl_module.eval()
                reconst_imgs, _ = pl_module(input_imgs)
                pl_module.train()

            # Plot and add to tensorboard
            for i in range(len(input_imgs)):
                tar = target_imgs[i:i+1, ::5]
                rec = reconst_imgs[i:i+1, ::5].squeeze(2)
                target_grid = custom_grid(tar.permute(1,0,2,3), nrow=4, normalize=True, value_range=(0, 70))
                reconst_grid = custom_grid(rec.permute(1,0,2,3), nrow=4, normalize=True, value_range=(0, 70))

                grid = torch.cat([target_grid.to(pl_module.device), reconst_grid], dim=1)

                plot_buf = gen_plot(grid.cpu().squeeze(0))
                image = PIL.Image.open(plot_buf)
                grid = transforms.ToTensor()(image)
                trainer.logger.experiment.add_image("Example-%s, Targets vs. Reconstructions"%i, grid, global_step=trainer.global_step)

def get_train_images(num):
    return torch.stack([trainset[i] for i in range(num)], dim=0)

def train_cifar():
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, str(model_name).split('.')[-1][:-2]),
        gpus=[gpu_id] if str(device).startswith("cuda") else 0,
        max_epochs=500,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, monitor='val_loss', save_top_k=3),
            LearningRateMonitor("epoch"),
            GenerateCallback(get_train_images(5), every_n_epochs=1),
            EarlyStopping(monitor='val_loss', mode='min', patience=30),
        ],
        log_every_n_steps=1,
    )
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    model = Autoencoder()
    trainer.fit(model, train_loader, valid_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=valid_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}
    return model, result


def gen_plot(grid):
    # plt.figure()
    plt.imshow(grid, vmin=0, vmax=70, cmap=cmap)
    plt.axis('off')
    # plt.colorbar()
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    return buf

def custom_grid(tensor, value_range, nrow: int = 2, padding: int = 2, normalize: bool = False, pad_value: float = 1.0,):

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        tensor = tensor*(value_range[1]-value_range[0])+value_range[0]
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid

def visualize_reconstructions(model):
    for i in range(5):
        input_imgs = trainset[i].unsqueeze(0)
        target_imgs = input_imgs[:, 5:]
        input_imgs = input_imgs[:, :5].unsqueeze(2)
        # Reconstruct images
        model.eval()
        with torch.no_grad():
            reconst_imgs = model(input_imgs.to(model.device))
        reconst_imgs = reconst_imgs.cpu().squeeze(2)

        # Plotting
        target_grid = custom_grid(target_imgs.permute(1, 0, 2, 3), nrow=20, normalize=True, value_range=(0, 70))
        reconst_grid = custom_grid(reconst_imgs.permute(1, 0, 2, 3), nrow=20, normalize=True, value_range=(0, 70))
        grid = torch.cat([target_grid.to(model.device), reconst_grid], dim=1)
        grid = grid.cpu().squeeze(0)
        plt.figure()
        plt.title("Targets vs. Reconstructed")
        plt.imshow(grid, vmax=70, vmin=0, cmap=cmap)
        plt.axis("off")
        plt.show()


# Plot and add to tensorboard
# imgs = torch.stack([target_imgs, reconst_imgs], dim=1).flatten(0, 1)




def test_cifar():
    model = Autoencoder()
    model = model.load_from_checkpoint(
        # './saved_models/unet/lightning_logs/version_1/checkpoints/epoch=199-step=200.ckpt')
        './saved_models/unet/lightning_logs/version_2/checkpoints/epoch=133-step=670.ckpt')
    visualize_reconstructions(model)


if __name__ == '__main__':
    model, result = train_cifar()
    print(result)
    # test_cifar()

