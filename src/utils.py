from PIL import Image, ImageOps


import torchvision.transforms.functional as FT

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Some constants
rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(device)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)



def convert_image(img, source, target):
    """
    Convert an image from a source format to a target format.

    :param img: image
    :param source: source format, one of 'pil' (PIL image), '[0, 1]' or '[-1, 1]' (pixel value ranges)
    :param target: target format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                   'imagenet-norm' (pixel values standardized by imagenet mean and std.),
                   'y-channel' (luminance channel Y in the YCbCr color format, used to calculate PSNR and SSIM)
    :return: converted image
    """
    assert source in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % source
    assert target in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
                      'y-channel'}, "Cannot convert to target format %s!" % target

    # Convert from source to [0, 1]
    if source == 'pil':
        img = FT.to_tensor(img)

    elif source == '[0, 1]':
        pass  # already in [0, 1]

    elif source == '[-1, 1]':
        img = (img + 1.) / 2.

    # Convert from [0, 1] to target
    if target == 'pil':
        img = FT.to_pil_image(img)

    elif target == '[0, 255]':
        img = 255. * img

    elif target == '[0, 1]':
        pass  # already in [0, 1]

    elif target == '[-1, 1]':
        img = 2. * img - 1.

    elif target == 'imagenet-norm':
        if img.ndimension() == 3:
            img = (img - imagenet_mean) / imagenet_std
        elif img.ndimension() == 4:
            img = (img - imagenet_mean_cuda) / imagenet_std_cuda

    elif target == 'y-channel':
        # Based on definitions at https://github.com/xinntao/BasicSR/wiki/Color-conversion-in-SR
        # torch.dot() does not work the same way as numpy.dot()
        # So, use torch.matmul() to find the dot product between the last dimension of an 4-D tensor and a 1-D tensor
        img = torch.matmul(255. * img.permute(0, 2, 3, 1)[:, 4:-4, 4:-4, :], rgb_weights) / 255. + 16.

    return img

def image_padding(img, padding_size):
    """
    Pad an image with zeros.
    :param img: image
    :param padding_size: padding size
    :return: padded image
    """
    width, _ = img.size
    if width < int(padding_size+1/2):
        # Pad the low resolution(lr) image to half the size of the high resolution(hr) image(square) using zero padding

        img= ImageOps.pad(img, size=(int(padding_size/2), int(padding_size/2)),centering=(0.5, 0.5))

    else :
        # Pad the high resolution(hr) image to the desired size(square)
        img= ImageOps.pad(img, size=(int(padding_size), int(padding_size)),centering=(0.5, 0.5))

    return img

def image_crop(img, crop_size):
    """
    Crop an image to the desired size.

    :param img: image
    :param crop_size: crop size
    :return: cropped image
    """

   
    if img.width < int(1022):
        # Crop the low resolution(lr) image to half the size of the high resolution(hr) image(square)
        left = (img.width - int(crop_size/2))/2
        top = (img.height - int(crop_size/2))/2
        right = (img.width + int(crop_size/2))/2
        bottom = (img.height + int(crop_size/2))/2 

    else :
        # Crop the high resolution(hr) image to the desired size(square)
        left = (img.width - crop_size)/2
        top = (img.height - crop_size)/2
        right = (img.width + crop_size)/2
        bottom = (img.height + crop_size)/2 
    
    img= img.crop((left, top, right, bottom))


    return img

def image_flip(img):
    """
    Flip an image to horizontal if the image is oriented vertically. Keep the horizontal image as it is.

    :param img: image
    :return: flipped image
    """

    width, height = img.size
    aspect_ratio = width / height

    if aspect_ratio < 1:
    # The image is vertical, so rotate it
        img = img.transpose(method=Image.Transpose.ROTATE_90)

    return img

class ImageTransforms(object):
    """
    Image transformation pipeline.
    """

    def __init__(self, process, desired_size, lr_img_type, hr_img_type):
        """
        :param process: process type, one of 'crop' or 'padding'
        :param desired_size: desired image size
        :param source_type: source image format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges)
        :param target_type: target image format, one of 'pil' (PIL image), '[0, 255]', '[0, 1]', '[-1, 1]' (pixel value ranges),
                            'imagenet-norm' (pixel values standardized by imagenet mean and std.), 'y-channel' (luminance   channel Y in the YCbCr color format, used to calculate PSNR and SSIM)                                                                                
        """
        self.process = process.lower()
        self.desired_size = desired_size
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type


        assert self.process in {'crop', 'padding'}
        # assert self.source_type in {'pil', '[0, 1]', '[-1, 1]'}, "Cannot convert from source format %s!" % self.source_type
        # assert self.target_type in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm',
        #               'y-channel'}, "Cannot convert to target format %s!" % self.target_type

    def __call__(self, img):
        """
        :param img: a PIL source image from which the HR image will be cropped, and then downsampled to create the LR image
        :return: LR and HR images in the specified format
        """
        if self.desired_size is None:
            # Convert the LR and HR image to the required type
            if img.width < int(1022):
                img = convert_image(img, source='pil', target=self.lr_img_type)
            else :
                img = convert_image(img, source='pil', target=self.hr_img_type)
            return img
        else:
            _width = img.width
            if self.process == 'crop':
                img = image_crop(img, crop_size=self.desired_size)
            elif self.process == 'padding':
                img = image_flip(img)
                img = image_padding(img, padding_size=self.desired_size)

            # Convert the LR and HR image to the required type
            if _width < int(1022):
                img = convert_image(img, source='pil', target=self.lr_img_type)
            else :
                img = convert_image(img, source='pil', target=self.hr_img_type)

            return img


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        # self.list = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        # self.list.append(val)


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def save_checkpoint(state, filename):
    """
    Save model checkpoint.

    :param state: checkpoint contents
    """

    torch.save(state, filename)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))
