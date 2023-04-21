import torch
import numpy as np

from datasets.base_dataset import BaseDataset


class CityScapesSunRGBD(BaseDataset):

    def __init__(self, dataset_settings, mode, unsupervised, preprocess, sliding = False, stride_rate = None):
        super(CityScapesSunRGBD, self).__init__(dataset_settings, mode, unsupervised)
        self.preprocess = preprocess
        self.sliding = sliding
        self.stride_rate = stride_rate
        if self.sliding and self._mode == 'train':
            print("Ensure correct preprocessing is being done!")

    def __getitem__(self, index):
        # if self._file_length is not None:
        #     names = self._construct_new_file_names(self._file_length)[index]
        # else:
        #     names = self._file_names[index]
        names = self._file_names[index]
        rgb_path = self._rgb_path+names['rgb'] 
        depth_path = self._rgb_path+names['depth'] 
        if not self.unsupervised:
            gt_path = self._gt_path+names['gt']
        item_name = names['rgb'].split("/")[-1].split(".")[0]

        rgb = self._open_rgb(rgb_path)
        depth = self._open_depth(depth_path)
        gt = None
        if not self.unsupervised:
            gt = self._open_gt(gt_path)

        if not self.sliding:
            if self.preprocess is not None:
                rgb, depth, gt = self.preprocess(rgb, depth, gt)

            if self._mode in ['train', 'val']:
                rgb = torch.from_numpy(np.ascontiguousarray(rgb)).float()
                depth = torch.from_numpy(np.ascontiguousarray(depth)).float()
                if gt is not None:
                    gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
            else:
                raise Exception(f"{self._mode} not supported in CityScapesSunRGB")
                
            # output_dict = dict(rgb=rgb, fn=str(item_name),
            #                    n=len(self._file_names))
            output_dict = dict(data=[rgb, depth], name = item_name)
            if gt is not None:
                output_dict['gt'] = gt
            return output_dict
    
        else:
            sliding_ouptut = self.slide_over_image(rgb, self.model_input_shape, self.stride_rate)
            output_dict = {}
            if self._mode in ['train', 'val']:
                if gt is not None:
                    gt = torch.from_numpy(np.ascontiguousarray(gt)).long()
                output_dict['gt'] = gt
                output_dict['sliding_output'] = []
                for img_sub, pos, margin in sliding_ouptut:
                    if self.preprocess is not None:
                        img_sub, _ = self.preprocess(img_sub, None)
                    img_sub = torch.from_numpy(np.ascontiguousarray(img_sub)).float()
                    output_dict['sliding_output'].append(([img_sub], pos, margin))
            return output_dict
