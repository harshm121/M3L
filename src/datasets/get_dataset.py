import torch
from torch.utils.data import DataLoader
from datasets.citysundepth import CityScapesSunDepth
from datasets.citysunrgb import CityScapesSunRGB
from datasets.citysunrgbd import CityScapesSunRGBD
from datasets.preprocessors import DepthTrainPre, DepthValPre, RGBDTrainPre, RGBDValPre, RGBTrainPre, RGBValPre
from datasets.tfnyu import TFNYU
from utils.constants import Constants as C

def get_dataset(args):
    datasetClass = None
    if args.data == "city" or args.data == "sunrgbd" or args.data == 'stanford_indoor':
        if len(args.modalities) == 1 and args.modalities[0] == 'rgb':
            datasetClass = CityScapesSunRGB
        elif len(args.modalities) == 1 and args.modalities[0] == 'depth':
            datasetClass = CityScapesSunDepth
        elif len(args.modalities) == 2 and args.modalities[0] == 'rgb' and args.modalities[1] == 'depth':
            datasetClass = CityScapesSunRGBD
        else:
            raise Exception(f"{args.modalities} not configured in get_dataset function.")
    else:
        raise Exception(f"{args.data} not configured in get_dataset function.")
    return datasetClass

def get_preprocessors(args, dataset_settings, mode):
    if len(args.modalities) == 1 and args.modalities[0] == 'rgb':
        if mode == 'train':
            return RGBTrainPre(C.pytorch_mean, C.pytorch_std, dataset_settings)
        elif mode == 'val':
            return RGBValPre(C.pytorch_mean, C.pytorch_std, dataset_settings)
        else:
            return Exception("%s mode not defined" % mode)
    elif len(args.modalities) == 1 and args.modalities[0] == 'depth':
        if mode == 'train':
            return DepthTrainPre(dataset_settings)
        elif mode == 'val':
            return DepthValPre(dataset_settings)
        else:
            return Exception("%s mode not defined" % mode)
    elif len(args.modalities) == 2 and args.modalities[0] == 'rgb' and args.modalities[1] == 'depth':
        if mode == 'train':
            return RGBDTrainPre(C.pytorch_mean, C.pytorch_std, dataset_settings)
        elif mode == 'val':
            return RGBDValPre(C.pytorch_mean, C.pytorch_std, dataset_settings)
        else:
            return Exception("%s mode not defined" % mode)
    else:
        raise Exception("%s not configured for preprocessing" % args.modalities)

def get_train_loader(datasetClass, args, train_source, unsupervised = False):
    dataset_settings = {'rgb_root': args.rgb_root,
                        'gt_root': args.gt_root,
                        'depth_root': args.depth_root,
                        'train_source': train_source,
                        'eval_source': args.eval_source, 
                        'required_length': args.total_train_imgs, #Every dataloader will have  Total Train Images / batch size iterations to be consistent
                        'train_scale_array': args.train_scale_array,
                        'image_height': args.image_height,
                        'image_width': args.image_width,
                        'modalities': args.modalities}

    preprocessing = get_preprocessors(args, dataset_settings, "train")
    train_dataset = datasetClass(dataset_settings, "train", unsupervised, preprocessing)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas = args.world_size, rank = args.rank)
    if unsupervised and "unsup_batch_size" in args:
        batch_size = args.unsup_batch_size
    else:
        batch_size = args.batch_size
    train_loader = DataLoader(train_dataset, 
                                batch_size = batch_size // args.world_size,
                                num_workers = args.num_workers, 
                                drop_last = True, 
                                shuffle = False,
                                sampler = train_sampler)
    return train_loader
    
def get_val_loader(datasetClass, args):
    dataset_settings = {'rgb_root': args.rgb_root,
                        'gt_root': args.gt_root,
                        'depth_root': args.depth_root,
                        'train_source': None,
                        'eval_source': args.eval_source, 
                        'required_length': None,
                        'max_samples': None,
                        'train_scale_array': args.train_scale_array,
                        'image_height': args.image_height,
                        'image_width': args.image_width,
                        'modalities': args.modalities}
    if args.data == 'sunrgbd':
        eval_sources = []
        for shape in ['427_561', '441_591', '530_730', '531_681']:
            eval_sources.append(dataset_settings['eval_source'].split('.')[0] + '_' +  shape + '.txt')
    else:
        eval_sources = [args.eval_source]

    preprocessing = get_preprocessors(args, dataset_settings, "val")
    if args.sliding_eval:
        collate_fn = _sliding_collate_fn
    else:
        collate_fn = None

    val_loaders = []
    for eval_source in eval_sources:
        dataset_settings['eval_source'] = eval_source
        val_dataset = datasetClass(dataset_settings, "val", False, preprocessing, args.sliding_eval, args.stride_rate)
        if args.rank is not None: #DDP Evaluation
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas = args.world_size, rank = args.rank)
            batch_size = args.val_batch_size // args.world_size
        else: #DP Evaluation
            val_sampler = None
            batch_size = args.val_batch_size

        val_loader = DataLoader(val_dataset,
                            batch_size = batch_size,
                            num_workers = 4, 
                            drop_last = False, 
                            shuffle = False, 
                            collate_fn = collate_fn, 
                            sampler = val_sampler)
        val_loaders.append(val_loader)
    return val_loaders
    

def _sliding_collate_fn(batch):
        gt = torch.stack([b['gt'] for b in batch])
        sliding_output = []
        num_modalities = len(batch[0]['sliding_output'][0][0])
        for i in range(len(batch[0]['sliding_output'])): #i iterates over positions
            imgs = [torch.stack([b['sliding_output'][i][0][m] for b in batch]) for m in range(num_modalities)]
            pos = batch[0]['sliding_output'][i][1]
            pos_compare = [(b['sliding_output'][i][1] == pos).all() for b in batch]
            assert all(pos_compare), f"Position not same for all points in the batch: {pos_compare}, {[b['sliding_output'][i][1] for b in batch]}"
            margin = batch[0]['sliding_output'][i][2]
            margin_compare = [(b['sliding_output'][i][2] == margin).all() for b in batch]
            assert all(margin_compare), f"Margin not same for all points in the batch: {margin_compare}, {[b['sliding_output'][i][2] for b in batch]}"
            sliding_output.append((imgs, pos, margin))
        return {"gt": gt, "sliding_output": sliding_output}