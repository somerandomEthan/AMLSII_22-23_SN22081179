from src.utils import *
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from B.datasets import SRDataset
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
BASEDIR = Path.cwd() # get the parent directory of the current file
DATA_FOLDER = BASEDIR / 'Datasets'
# test_data_names = ["Set5", "Set14", "BSDS100"]
DESIRED_SIZE = None # size of target HR images
SCALING_FACTOR = 2  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor


# Model checkpoints
srresnet_checkpoint = "./checkpoint_srresnet_B.pth.tar"

# Load model, either the SRResNet or the SRGAN
srresnet = torch.load(srresnet_checkpoint)['model'].to(device)
srresnet.eval()
model = srresnet

# Evaluate


 # Custom dataloader
test_dataset = SRDataset(
    DATA_FOLDER,
    split="test",
    process="crop",
    desired_size=DESIRED_SIZE,
    scaling_factor=SCALING_FACTOR,
    lr_img_type="[0, 1]",
    hr_img_type="[-1, 1]",
)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4,
                                              pin_memory=True)

    # Keep track of the PSNRs and the SSIMs across batches
PSNRs = AverageMeter()
SSIMs = AverageMeter()

    # Prohibit gradient computation explicitly because I had some problems with memory
with torch.no_grad():
    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(test_loader):
        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (1), 3, w / 4, h / 4), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (1), 3, w, h), in [-1, 1]

        # Forward prop.
        sr_imgs = model(lr_imgs)  # (1, 3, w, h), in [-1, 1]

        # Calculate PSNR and SSIM
        sr_imgs_y = convert_image(sr_imgs, source='[-1, 1]', target='y-channel').squeeze(
            0)  # (w, h), in y-channel
        hr_imgs_y = convert_image(hr_imgs, source='[-1, 1]', target='y-channel').squeeze(0)  # (w, h), in y-channel
        psnr = peak_signal_noise_ratio(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                        data_range=255.)
        ssim = structural_similarity(hr_imgs_y.cpu().numpy(), sr_imgs_y.cpu().numpy(),
                                         data_range=255.)
        PSNRs.update(psnr, lr_imgs.size(0))
        SSIMs.update(ssim, lr_imgs.size(0))

# Print average PSNR and SSIM
print('PSNR - {psnrs.avg:.3f}'.format(psnrs=PSNRs))
print('SSIM - {ssims.avg:.3f}'.format(ssims=SSIMs))

