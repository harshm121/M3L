import torch
import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image
from utils.img_utils import pad_image_to_shape

class BaseDataset(data.Dataset):

    def __init__(self, dataset_settings, mode, unsupervised):
        self._mode = mode
        self.unsupervised = unsupervised
        self._rgb_path = dataset_settings['rgb_root']
        self._depth_path = dataset_settings['depth_root']
        self._gt_path = dataset_settings['gt_root']
        self._train_source = dataset_settings['train_source']
        self._eval_source = dataset_settings['eval_source']
        self.modalities = dataset_settings['modalities']
        # self._file_length = dataset_settings['max_samples'] 
        self._required_length = dataset_settings['required_length'] 
        self._file_names = self._get_file_names(mode)
        self.model_input_shape = (dataset_settings['image_height'], dataset_settings['image_width'])
        
    def __len__(self):
        if self._required_length is not None:
            return self._required_length
        return len(self._file_names) # when model == "val"

    def _get_file_names(self, mode):
        assert mode in ['train', 'val']
        source = self._train_source
        if mode == "val":
            source = self._eval_source

        file_names = []
        with open(source) as f:
            files = f.readlines()

        for item in files:
            names = self._process_item_names(item)
            file_names.append(names)
        
        if mode == "val":
            return file_names
        elif self._required_length <= len(file_names):
            return file_names[:self._required_length]
        else:
            return self._construct_new_file_names(file_names, self._required_length)

    def _construct_new_file_names(self, file_names, length):
        assert isinstance(length, int)
        files_len = len(file_names)

        new_file_names = file_names * (length // files_len) #length % files_len items remaining

        rand_indices = torch.randperm(files_len).tolist()
        new_indices = rand_indices[:length % files_len]

        new_file_names += [file_names[i] for i in new_indices]

        return new_file_names

    def _process_item_names(self, item):
        item = item.strip()
        item = item.split('\t')
        num_modalities = len(self.modalities)
        num_items = len(item)
        names = {}
        if not self.unsupervised:
            assert num_modalities + 1 == num_items, f"Number of modalities and number of items in file name don't match, len(modalities) = {num_modalities} and len(item) = {num_items}" + item[0]
            for i, modality in enumerate(self.modalities):
                names[modality] = item[i]
            names['gt'] = item[-1]
        else:
            assert num_modalities == num_items, f"Number of modalities and number of items in file name don't match, len(modalities) = {num_modalities} and len(item) = {num_items}"
            for i, modality in enumerate(self.modalities):
                names[modality] = item[i]
            names['gt'] = None

        return names

    def _open_rgb(self, rgb_path, dtype = None):
        bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR) #cv2 reads in BGR format, HxWxC
        rgb = np.array(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), dtype=dtype) #Pretrained PyTorch model accepts image in RGB
        return rgb

    def _open_depth(self, depth_path, dtype = None): #returns in HxWx3 with the same image in all channels
        img_arr = np.array(Image.open(depth_path))
        if len(img_arr.shape) == 2:  # grayscale
            img_arr = np.array(np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0), dtype = dtype)
        img_arr = (img_arr - img_arr.min()) * 255.0 / (img_arr.max() - img_arr.min())
        return img_arr
    
    def _open_depth_tf_nyu(self, depth_path, dtype = None): #returns in HxWx3 with the same image in all channels
        img_arr = np.array(Image.open(depth_path))
        if len(img_arr.shape) == 2:  # grayscale
            img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
        return img_arr

    def _open_gt(self, gt_path, dtype = None):
        return np.array(cv2.imread(gt_path,  cv2.IMREAD_GRAYSCALE), dtype=dtype)

    def slide_over_image(self, img, crop_size, stride_rate):
        H, W, C = img.shape
        long_size = H if H > W else W
        output = []
        if long_size <= min(crop_size[0], crop_size[1]):
            raise Exception("Crop size is greater than the image size itself. Not handeled right now")
        
        else:
            stride_0 = int(np.ceil(crop_size[0] * stride_rate))
            stride_1 = int(np.ceil(crop_size[1] * stride_rate))
            r_grid = int(np.ceil((H - crop_size[0]) / stride_0)) + 1
            c_grid = int(np.ceil((W - crop_size[1]) / stride_1)) + 1

            for grid_yidx in range(r_grid):
                for grid_xidx in range(c_grid):
                    s_x = grid_xidx * stride_1
                    s_y = grid_yidx * stride_0
                    e_x = min(s_x + crop_size[1], W)
                    e_y = min(s_y + crop_size[0], H)
                    s_x = e_x - crop_size[1]
                    s_y = e_y - crop_size[0]
                    img_sub = img[s_y:e_y, s_x: e_x, :]
                    img_sub, margin = pad_image_to_shape(img_sub, crop_size, cv2.BORDER_CONSTANT, value=0)
                    output.append((img_sub, np.array([s_y, e_y, s_x, e_x]), margin))
        
        return output
