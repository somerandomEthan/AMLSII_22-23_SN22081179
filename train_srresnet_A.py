import time
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader, random_split
from skimage.metrics import peak_signal_noise_ratio
from matplotlib import pyplot as plt
from src.models import SRResNet, Generator, Discriminator, TruncatedVGG19
from A.datasets import SRDataset 
from src.utils import *
from pathlib import Path


# Data parameters
BASEDIR = Path.cwd() # get the parent directory of the current file
DATA_FOLDER = BASEDIR / 'Datasets'
DESIRED_SIZE = 96 # size of target HR images
SCALING_FACTOR = 2  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor
TRAIN_SPLIT = 0.8 # 20% of the dataset will be used for validation

# Model parameters
large_kernel_size = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
n_channels = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
n_blocks = 16  # number of residual blocks


# Learning parameters
checkpoint = None  # path to model checkpoint, None if none
BATCH_SIZE = 16  # batch size
start_epoch = 0  # start at this epoch
iterations = 1e4  # number of training iterations
WORKERS = 4  # number of workers for loading data in the DataLoader
print_freq = 500  # print training status once every __ batches
lr = 1e-4  # learning rate
grad_clip = None  # clip if gradients are exploding
train_loss_list = []
Traning_PSNR_list = []
Validation_PSNR_list = []
valid_loss_list = []



# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True



def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint


    # Initialize model or load checkpoint
    if checkpoint is None:
        model = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                         n_channels=n_channels, n_blocks=n_blocks, scaling_factor=SCALING_FACTOR)
        # Initialize the optimizer
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # Custom dataloaders
    whole_dataset = SRDataset(
    DATA_FOLDER,
    split="train",
    process="crop",
    desired_size=DESIRED_SIZE,
    scaling_factor=SCALING_FACTOR,
    lr_img_type="[0, 1]",
    hr_img_type="[-1, 1]",
)

    # Calculate the sizes of train and validation sets
    train_size = int(TRAIN_SPLIT * len(whole_dataset))
    val_size = len(whole_dataset) - train_size

    # Divide the dataset into train and validation sets
    train_dataset, val_dataset = random_split(whole_dataset, [train_size, val_size])


    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    valid_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

    # Total number of epochs to train for
    epochs = int(iterations // len(train_loader) + 1)
    print(f'The number of epochs is {epochs}')
    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train_loss, Traning_PSNR = train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)
        train_loss_list.append(train_loss)
        Traning_PSNR_list.append(Traning_PSNR)
        if epoch % 10 == 0:
            # validation every 10 epochs
            valid_loss, Validation_PSNR = valid(valid_loader=valid_loader,
              model=model,
              criterion=criterion,
              epoch=epoch)
            valid_loss_list.append(valid_loss)
            Validation_PSNR_list.append(Validation_PSNR)

            # Save checkpoint
            torch.save({'epoch': epoch,  
                    'model': model,
                    'optimizer': optimizer},
                   'checkpoint_srresnet_A.pth.tar')
    plt.figure(1)
    plt.plot([i for i in range(len(train_loss_list))], train_loss_list)
    plt.ylabel("Triaining loss")
    plt.xlabel('Number of epochs')
    plt.title("Triaining loss per epoch for SRResNet in Task A")
    plt.savefig('Triaining loss per epoch for SRResNet in Task A.png')

    plt.figure(2)
    plt.plot([i for i in range(len(Traning_PSNR_list))], Traning_PSNR_list)
    plt.ylabel("Triaining PSNR")
    plt.xlabel('Number of epochs')
    plt.title("Triaining PSNR per epoch for SRResNet in Task A")
    plt.savefig('Triaining PSNR per epoch for SRResNet in Task A.png')

    plt.figure(3)
    plt.plot([i*10 for i in range(len(valid_loss_list))], valid_loss_list)
    plt.ylabel("Validation loss")
    plt.xlabel('Number of epochs')
    plt.title("Validation loss per epoch for SRResNet in Task A")
    plt.savefig("Validation loss per epoch for SRResNet in Task A.png")

    plt.figure(4)
    plt.plot([i*10 for i in range(len(Validation_PSNR_list))], Validation_PSNR_list)
    plt.ylabel("Validation PSNR")
    plt.xlabel('Number of epochs')
    plt.title("Validation PSNR per epoch for SRResNet in Task A")
    plt.savefig("Validation PSNR per epoch for SRResNet in Task A.png")





def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: content loss function (Mean Squared-Error loss)
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    train_losses = AverageMeter()  # training loss

    # Keep track of the PSNRs and the SSIMs across batches
    Traning_PSNRs = AverageMeter()

    start = time.time()

    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), in [-1, 1]

        # Forward prop.
        sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

        # Loss
        loss = criterion(sr_imgs, hr_imgs)  # scalar
        

        # Calculate PSNR 
        sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
        hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
        psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().detach().numpy(), sr_imgs_y.cpu().detach().numpy(),data_range=255.)
        Traning_PSNRs.update(psnr, lr_imgs.size(0))

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        # Keep track of loss
        train_losses.update(loss.item(), lr_imgs.size(0))

        # Keep track of batch time
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]----'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  'Training Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=train_losses))
            print('Training PSNR - {psnrs.avg:.3f}'.format(psnrs=Traning_PSNRs))
            
    del lr_imgs, hr_imgs, sr_imgs  # free some memory since their histories may be stored
    return train_losses.avg, Traning_PSNRs.avg

def valid(valid_loader, model, criterion, epoch):
    """
    One epoch's training.

    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: content loss function (Mean Squared-Error loss)
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.eval()  # evaluation mode

    valid_losses = AverageMeter()  # loss
    # Keep track of the PSNRs and the SSIMs across batches
    Validation_PSNRs = AverageMeter()

    # start = time.time()

    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(valid_loader):
        # data_time.update(time.time() - start)

        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), in [-1, 1]

        # Forward prop.
        sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

        # Loss
        loss = criterion(sr_imgs, hr_imgs)  # scalar

        # Calculate PSNR 
        sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
        hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
        psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().detach().numpy(), sr_imgs_y.cpu().detach().numpy(),data_range=255.)
        Validation_PSNRs.update(psnr, lr_imgs.size(0))


        # # Backward prop.
        # optimizer.zero_grad()
        # loss.backward()

        # # Clip gradients, if necessary
        # if grad_clip is not None:
        #     clip_gradient(optimizer, grad_clip)

        # # Update model
        # optimizer.step()

        # Keep track of loss
        valid_losses.update(loss.item(), lr_imgs.size(0))

        # # Keep track of batch time
        # batch_time.update(time.time() - start)

        # Reset start time
        # start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]----'
                  'Validation Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(valid_loader),
                                                                    loss=valid_losses))
            print('Validation PSNR - {psnrs.avg:.3f}'.format(psnrs=Validation_PSNRs))
    del lr_imgs, hr_imgs, sr_imgs  # free some memory since their histories may be stored
    return valid_losses.avg, Validation_PSNRs.avg


if __name__ == '__main__':
    main()
