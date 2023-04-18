from torch.utils.data import Dataset
import os
from PIL import Image
from src.utils import ImageTransforms


class SRDataset(Dataset):
    """
    A PyTorch Dataset to be used by a PyTorch DataLoader.
    """

    def __init__(self, data_folder, split, process, desired_size, scaling_factor, lr_img_type, hr_img_type):
        """
        :param data_folder: # pass the data folder path object into the class
        :param split: one of 'train' or 'test'
        :param crop_size: crop size of target HR images
        :param scaling_factor: the input LR images will be downsampled from the target HR images by this factor; the scaling done in the super-resolution
        # :param lr_img_type: the format for the LR image supplied to the model; see convert_image() in utils.py for available formats
        # :param hr_img_type: the format for the HR image supplied to the model; see convert_image() in utils.py for available formats
        """

        self.data_folder = data_folder
        self.split = split.lower()
        self.process = process.lower()
        self.desired_size = desired_size
        self.scaling_factor = int(scaling_factor)
        self.lr_img_type = lr_img_type
        self.hr_img_type = hr_img_type


        assert self.split in {'train', 'test'}
        assert lr_img_type in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm','y-channel'}
        assert hr_img_type in {'pil', '[0, 255]', '[0, 1]', '[-1, 1]', 'imagenet-norm','y-channel'}


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
        self.transform = ImageTransforms(process=self.process,
                                         desired_size=self.desired_size, 
                                         lr_img_type=self.lr_img_type,
                                         hr_img_type=self.hr_img_type
                                         )

    def __getitem__(self, i):
        """
        This method is required to be defined for use in the PyTorch DataLoader.
        :param i: index to retrieve
        :return: the 'i'th pair LR and HR images to be fed into the model
        """
        # Read image       
        img_hr_dir = self.images[i]
        index = img_hr_dir.stem
        if self.split == 'train':
            img_lr_dir = self.data_folder /  f"DIV2K_train_LR_unknown" / f"X{self.scaling_factor}"/ f'{index}x{self.scaling_factor}.png'
        else:
            img_lr_dir = self.data_folder /  f"DIV2K_valid_LR_unknown" / f"X{self.scaling_factor}"/ f'{index}x{self.scaling_factor}.png'



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
    

    # class SRCNNDataset(Dataset):
        
    #     def __init__(self, image_dir, zoom_factor):
    #         super(SRCNNDataset, self).__init__()
    #         hr_images_list = []
    #     if self.split == 'train':
    #         hd = data_folder / 'DIV2K_train_HR'
    #         for i in os.listdir(hd):
    #             img_path = hd / str(i)
    #             hr_images_list.append(img_path)
    #         self.images = hr_images_list
    #     else:
    #         hd = data_folder / 'DIV2K_valid_HR'
    #         for i in os.listdir(hd):
    #             img_path = hd / str(i)
    #             hr_images_list.append(img_path)
    #         self.images = hr_images_list
    #         self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

    #         crop_size = CROP_SIZE - (CROP_SIZE % zoom_factor) # Valid crop size
    #         self.input_transform = transforms.Compose([transforms.CenterCrop(crop_size), # cropping the image
    #                                   transforms.Resize(crop_size//zoom_factor),  # subsampling the image (half size)
    #                                   transforms.Resize(crop_size, interpolation=Image.BICUBIC),  # bicubic upsampling to get back the original size 
    #                                   transforms.ToTensor()])
    #         self.target_transform = transforms.Compose([transforms.CenterCrop(crop_size), # since it's the target, we keep its original quality
    #                                    transforms.ToTensor()])

    #     def __getitem__(self, index):
    #         input = load_img(self.image_filenames[index])
    #         target = input.copy()
        
    #         # input = input.filter(ImageFilter.GaussianBlur(1)) 
    #         input = self.input_transform(input)
    #         target = self.target_transform(target)

    #         return input, target

    #     def __len__(self):
    #         return len(self.image_filenames)