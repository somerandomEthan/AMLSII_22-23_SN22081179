from torch.utils.data import Dataset
import os
from PIL import Image
from src.utils import ImageTransforms


class SRDataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """

    def __init__(self, data_folder, split, width, scaling_factor):
        """
        :param data_folder: # pass the data folder path object into the class
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of target HR images
        :param scaling_factor: the input LR images will be downsampled from the target HR images by this factor; the scaling done in the super-resolution
        # :param lr_img_type: the format for the LR image supplied to the model; see convert_image() in utils.py for available formats
        # :param hr_img_type: the format for the HR image supplied to the model; see convert_image() in utils.py for available formats
        # :param test_data_name: if this is the 'test' split, which test dataset? (for example, "Set14")
        """

        self.data_folder = data_folder
        self.split = split.lower()
        self.width = int(width)
        self.scaling_factor = int(scaling_factor)


        assert self.split in {'train', 'test'}


        # Read list of image-paths
        hr_images_list = []
        if self.split == 'train':
            hd = data_folder / 'DIV2K_train_HR'
            for i in os.listdir(hd):
                img_path = hd / str(i)
                hr_images_list.append(img_path)
            self.images = hr_images_list
        else:
            hd = data_folder / 'DIV2K_valid_HR'
            for i in os.listdir(hd):
                img_path = hd / str(i)
                hr_images_list.append(img_path)
            self.images = hr_images_list

                
             

        # Select the correct set of transforms
        self.transform = ImageTransforms(split=self.split,
                                         width=self.width, 
                                         scaling_factor=self.scaling_factor)

    def __getitem__(self, i):
        """
        This method is required to be defined for use in the PyTorch DataLoader.
        :param i: index to retrieve
        :return: the 'i'th pair LR and HR images to be fed into the model
        """
        # Read image
        img_hr_dir = self.images[i]
        print(img_hr_dir)
        index = img_hr_dir.stem
        if self.split == 'train':
            img_lr_dir = self.data_folder /  f"DIV2K_train_LR_bicubic_X{self.scaling_factor}" / "DIV2K_train_LR_bicubic" / f"X{self.scaling_factor}"/ f'{index}x{self.scaling_factor}.png'
        else:
            img_lr_dir = self.data_folder /  f"DIV2K_valid_LR_bicubic_X{self.scaling_factor}" / "DIV2K_valid_LR_bicubic" / f"X{self.scaling_factor}"/ f'{index}x{self.scaling_factor}.png'
        print(img_lr_dir)


        img_lr = Image.open(img_lr_dir)
        lr_img = img_lr.convert('RGB')
        img_hr = Image.open(img_hr_dir)
        hr_img = img_hr.convert('RGB')
        lr_img= self.transform(lr_img)
        hr_img = self.transform(hr_img)

        return lr_img, hr_img

    def __len__(self):
        """
        This method is required to be defined for use in the PyTorch DataLoader.

        :return: size of this data (in number of images)
        """
        return len(self.images)