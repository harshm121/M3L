from utils.img_utils import normalizedepth, random_crop_pad_to_shape, random_mirror, random_scale, normalize, resizedepth, resizergb, tfnyu_normalizedepth

class RGBTrainPre(object):
    def __init__(self, pytorch_mean, pytorch_std, dataset_settings):
        self.pytorch_mean = pytorch_mean
        self.pytorch_std = pytorch_std
        self.train_scale_array = dataset_settings['train_scale_array']
        self.crop_size = (dataset_settings['image_height'], dataset_settings['image_width'])

    def __call__(self, rgb, gt):
        transformed_dict = random_mirror({"rgb":rgb, "gt":gt})
        if self.train_scale_array is not None:
            transformed_dict, _ = random_scale(transformed_dict, self.train_scale_array, (rgb.shape[0], rgb.shape[1]))

        transformed_dict, _ = random_crop_pad_to_shape(transformed_dict, transformed_dict['rgb'].shape[:2], self.crop_size) #Makes gt HxWx1
        rgb = transformed_dict['rgb']
        gt = transformed_dict['gt']
        rgb = normalize(rgb, self.pytorch_mean, self.pytorch_std)

        rgb = rgb.transpose(2, 0, 1) #Brings the channel dimension in the top. Final output = CxHxW
        return rgb, gt 


class RGBValPre(object):
    def __init__(self, pytorch_mean, pytorch_std, dataset_settings):
        self.pytorch_mean = pytorch_mean
        self.pytorch_std = pytorch_std
        self.model_input_shape = (dataset_settings['image_height'], dataset_settings['image_width'])

    def __call__(self, rgb, gt):
        rgb = resizergb(rgb, self.model_input_shape)
        rgb = normalize(rgb, self.pytorch_mean, self.pytorch_std)
        rgb = rgb.transpose(2, 0, 1) #Brings the channel dimension in the top. Final output = CxHxW
        return rgb, gt


class RGBDTrainPre(object):
    def __init__(self, pytorch_mean, pytorch_std, dataset_settings):
        self.pytorch_mean = pytorch_mean
        self.pytorch_std = pytorch_std
        self.train_scale_array = dataset_settings['train_scale_array']
        self.crop_size = (dataset_settings['image_height'], dataset_settings['image_width'])

    def __call__(self, rgb, depth, gt):
        transformed_dict = random_mirror({"rgb":rgb, "depth": depth, "gt":gt})
        if self.train_scale_array is not None:
            transformed_dict, _ = random_scale(transformed_dict, self.train_scale_array, (rgb.shape[0], rgb.shape[1]))

        transformed_dict, _ = random_crop_pad_to_shape(transformed_dict, transformed_dict['rgb'].shape[:2], self.crop_size) #Makes gt HxWx1
        rgb = transformed_dict['rgb']
        depth = transformed_dict['depth']
        gt = transformed_dict['gt']
        rgb = normalize(rgb, self.pytorch_mean, self.pytorch_std)
        depth = normalizedepth(depth)
        rgb = rgb.transpose(2, 0, 1) #Brings the channel dimension in the top. Final output = CxHxW
        depth = depth.transpose(2, 0, 1) #Brings the channel dimension in the top. Final output = CxHxW
        return rgb, depth, gt 


class RGBDValPre(object):
    def __init__(self, pytorch_mean, pytorch_std, dataset_settings):
        self.pytorch_mean = pytorch_mean
        self.pytorch_std = pytorch_std
        self.model_input_shape = (dataset_settings['image_height'], dataset_settings['image_width'])

    def __call__(self, rgb, depth, gt):
        rgb = resizergb(rgb, self.model_input_shape)
        depth = resizedepth(depth, self.model_input_shape)
        rgb = normalize(rgb, self.pytorch_mean, self.pytorch_std)
        depth = normalizedepth(depth)
        rgb = rgb.transpose(2, 0, 1) #Brings the channel dimension in the top. Final output = CxHxW
        depth = depth.transpose(2, 0, 1) #Brings the channel dimension in the top. Final output = CxHxW
        return rgb, depth, gt

class DepthTrainPre(object):
    def __init__(self, dataset_settings):
        self.train_scale_array = dataset_settings['train_scale_array']
        self.crop_size = (dataset_settings['image_height'], dataset_settings['image_width'])

    def __call__(self, depth, gt):
        transformed_dict = random_mirror({"depth": depth, "gt":gt})
        if self.train_scale_array is not None:
            transformed_dict, _ = random_scale(transformed_dict, self.train_scale_array, (depth.shape[0], depth.shape[1]))

        transformed_dict, _ = random_crop_pad_to_shape(transformed_dict, transformed_dict['depth'].shape[:2], self.crop_size) #Makes gt HxWx1
        depth = transformed_dict['depth']
        gt = transformed_dict['gt']
        depth = normalizedepth(depth)
        depth = depth.transpose(2, 0, 1) #Brings the channel dimension in the top. Final output = CxHxW
        return depth, gt 


class DepthValPre(object):
    def __init__(self, dataset_settings):
        self.model_input_shape = (dataset_settings['image_height'], dataset_settings['image_width'])

    def __call__(self, depth, gt):
        depth = resizedepth(depth, self.model_input_shape)
        depth = normalizedepth(depth)
        depth = depth.transpose(2, 0, 1) #Brings the channel dimension in the top. Final output = CxHxW
        return depth, gt