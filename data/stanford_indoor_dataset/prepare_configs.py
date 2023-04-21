from functools import partial
import os

def get_rgbd_semantic_list(oglist, path_prefix, with_gt = True):
    final_list = []
    for line in oglist:
        area, name = line.strip().split(' ')
        rgb_path = os.path.join(path_prefix, area, 'image', name) + '.png'
        depth_path = os.path.join(path_prefix, area, 'depth', name) + '.png'
        if with_gt:
            semantic_path = os.path.join(path_prefix, area, 'label', name) + '.png'
            final_line = "\t".join([rgb_path, depth_path, semantic_path]) + '\n'
        else:
            final_line = "\t".join([rgb_path, depth_path]) + '\n'
        final_list.append(final_line)
    return final_list

def get_rgb_semantic_list(oglist, path_prefix, with_gt = True):
    final_list = []
    for line in oglist:
        area, name = line.strip().split(' ')
        rgb_path = os.path.join(path_prefix, area, 'image', name) + '.png'
        if with_gt:
            semantic_path = os.path.join(path_prefix, area, 'label', name) + '.png'
            final_line = "\t".join([rgb_path, semantic_path]) + '\n'
        else:
            final_line = "\t".join([rgb_path]) + '\n'
        final_list.append(final_line)
    return final_list

def get_depth_semantic_list(oglist, path_prefix, with_gt = True):
    final_list = []
    for line in oglist:
        area, name = line.strip().split(' ')
        depth_path = os.path.join(path_prefix, area, 'depth', name) + '.png'
        if with_gt:
            semantic_path = os.path.join(path_prefix, area, 'label', name) + '.png'
            final_line = "\t".join([depth_path, semantic_path]) + '\n'
        else:
            final_line = "\t".join([depth_path]) + '\n'
        final_list.append(final_line)
    return final_list

os.makedirs('./configs', exist_ok=True)
os.makedirs('./configs/config_rgbd/', exist_ok=True)
os.makedirs('./configs/config_rgbd/train_config', exist_ok=True)
os.makedirs('./configs/config_rgb/', exist_ok=True)
os.makedirs('./configs/config_rgb/train_config', exist_ok=True)
os.makedirs('./configs/config_depth/', exist_ok=True)
os.makedirs('./configs/config_depth/train_config', exist_ok=True)


def create_files(get_spemantic_func, input_file, output_path):
    lines = open(input_file).readlines()
    converted_lines = get_spemantic_func(lines)
    with open(output_path, 'w') as f:
        f.writelines(converted_lines)


test = open('./semi_supervised_configs/test.txt').readlines()
test_rgbd = get_rgbd_semantic_list(test, path_prefix = '/data/', with_gt = True)
with open('./configs/config_rgbd/test.txt', 'w') as f:
    f.writelines(test_rgbd)

create_files(partial(get_rgbd_semantic_list, path_prefix = '/data/', with_gt = True), './semi_supervised_configs/train.txt', './configs/config_rgbd/train.txt')
create_files(partial(get_rgbd_semantic_list, path_prefix = '/data/', with_gt = True), './semi_supervised_configs/val.txt', './configs/config_rgbd/val.txt')
create_files(partial(get_rgbd_semantic_list, path_prefix = '/data/', with_gt = True), './semi_supervised_configs/test.txt', './configs/config_rgbd/test.txt')

create_files(partial(get_rgb_semantic_list, path_prefix = '/data/', with_gt = True), './semi_supervised_configs/train.txt', './configs/config_rgb/train.txt')
create_files(partial(get_rgb_semantic_list, path_prefix = '/data/', with_gt = True), './semi_supervised_configs/val.txt', './configs/config_rgb/val.txt')
create_files(partial(get_rgb_semantic_list, path_prefix = '/data/', with_gt = True), './semi_supervised_configs/test.txt', './configs/config_rgb/test.txt')

create_files(partial(get_depth_semantic_list, path_prefix = '/data/', with_gt = True), './semi_supervised_configs/train.txt', './configs/config_depth/train.txt')
create_files(partial(get_depth_semantic_list, path_prefix = '/data/', with_gt = True), './semi_supervised_configs/val.txt', './configs/config_depth/val.txt')
create_files(partial(get_depth_semantic_list, path_prefix = '/data/', with_gt = True), './semi_supervised_configs/test.txt', './configs/config_depth/test.txt')


ratios = os.listdir('./semi_supervised_configs/train_config/')

for ratio in ratios:
    rgbd_func_labeled = partial(get_rgbd_semantic_list, path_prefix = '/data/', with_gt = True)
    rgbd_func_unlabeled = partial(get_rgbd_semantic_list, path_prefix = '/data/', with_gt = False)
    create_files(rgbd_func_labeled, './semi_supervised_configs/train_config/{}/train_labeled.txt'.format(ratio), './configs/config_rgbd/train_config/train_labeled_{}'.format(ratio))
    create_files(rgbd_func_unlabeled, './semi_supervised_configs/train_config/{}/train_unlabeled.txt'.format(ratio), './configs/config_rgbd/train_config/train_unlabeled_{}'.format(ratio))

    rgb_func_labeled = partial(get_rgb_semantic_list, path_prefix = '/data/', with_gt = True)
    rgb_func_unlabeled = partial(get_rgb_semantic_list, path_prefix = '/data/', with_gt = False)
    create_files(rgb_func_labeled, './semi_supervised_configs/train_config/{}/train_labeled.txt'.format(ratio), './configs/config_rgb/train_config/train_labeled_{}'.format(ratio))
    create_files(rgb_func_unlabeled, './semi_supervised_configs/train_config/{}/train_unlabeled.txt'.format(ratio), './configs/config_rgb/train_config/train_unlabeled_{}'.format(ratio))

    depth_func_labeled = partial(get_depth_semantic_list, path_prefix = '/data/', with_gt = True)
    depth_func_unlabeled = partial(get_depth_semantic_list, path_prefix = '/data/', with_gt = False)
    create_files(depth_func_labeled, './semi_supervised_configs/train_config/{}/train_labeled.txt'.format(ratio), './configs/config_depth/train_config/train_labeled_{}'.format(ratio))
    create_files(depth_func_unlabeled, './semi_supervised_configs/train_config/{}/train_unlabeled.txt'.format(ratio), './configs/config_depth/train_config/train_unlabeled_{}'.format(ratio))
    
    