import sys
import os
import numpy as np

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from utils.lr_policy import WarmUpPolyLR
from utils.seg_opr.metrics import compute_metrics, hist_info
from semi_supervised.base_framework import AbstractFramework
from datasets.get_dataset import get_train_loader, get_val_loader
        
class CPSNetwork(nn.Module):
    
    def __init__(self, segModel1, segModel2):
        super(CPSNetwork, self).__init__()
        self.branch1 = segModel1
        self.branch2 = segModel2
        
    def forward(self, data, step1 = True, step2 = False, **kwargs):
        if not self.training:
            return self.branch1(data, **kwargs)
        
        ret = []
        if step1:
            ret.append(self.branch1(data, **kwargs))
        if step2:
            ret.append(self.branch2(data, **kwargs))
        return ret

    def get_params(self, branch):
        if branch == 1:
            return self.branch1.get_params()
        elif branch == 2:
            return self.branch2.get_params()
        else:
            raise Exception(f"Branch: {branch} not defined.")


class CPSFramework(AbstractFramework):
    
    def __init__(self, seg_model, sup_criterion, unsup_criterion, datasetClass, args):
        super(CPSFramework, self).__init__(seg_model, args)
        self.sup_criterion = sup_criterion
        self.unsup_criterion = unsup_criterion
        self.base_lr = args.base_lr
        self.n_sup_epochs = args.n_sup_epochs
        assert self.n_sup_epochs == 0, "n_sup_epochs should be 0 for CPS"
        self.n_unsup_epochs = args.n_unsup_epochs
        self.sup_niters_per_epoch = args.total_train_imgs // args.batch_size
        self.sup_total_iterations = self.n_sup_epochs * self.sup_niters_per_epoch
        self.unsup_niters_per_epoch = args.total_train_imgs // args.batch_size
        self.niters_per_epoch = args.total_train_imgs // args.batch_size
        self.unsup_total_iterations = self.n_unsup_epochs * self.unsup_niters_per_epoch
        self.total_iterations = self.sup_total_iterations + self.unsup_total_iterations
        self.lr_policy = [WarmUpPolyLR(base_lr, args.lr_power, self.total_iterations, self.unsup_niters_per_epoch * args.lr_warm_up_epoch) for base_lr in self.base_lr]
        self.freeze_bn = args.freeze_bn
        
        if self.rank == None:
            self.seg_model = nn.DataParallel(seg_model).to(self.device)
        else:
            self.seg_model = self.seg_model.to(self.rank)
            self.seg_model = DDP(seg_model, device_ids=[self.rank], output_device = self.rank)
        self.optimizers_l = self._get_optimizer(args.optim, args.weight_decay, args.momentum, branch = 1)
        self.optimizers_r = self._get_optimizer(args.optim, args.weight_decay, args.momentum, branch = 2)
        self.bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        if self.rank is not None:
            self.sup_train_loader = get_train_loader(datasetClass, args, args.sup_train_source, False)
            self.unsup_train_loader = get_train_loader(datasetClass, args, args.unsup_train_source, True)
        self.cps_weight = args.cps_weight
        self.num_classes = args.num_classes
        self.val_loader = get_val_loader(datasetClass, args)
        if not type(self.val_loader) == list:
            self.val_loader = [self.val_loader]
        self.val_size = np.sum([len(l) for l in self.val_loader])
        self.val_output_size = (args.data_image_height, args.data_image_width)

    def _get_optimizer(self, optim, weight_decays, momentum, branch):
        param_list = self.seg_model.module.get_params(branch = branch)
        optimizers = []
        if optim == 'sgd':
            for i, params in enumerate(param_list):
                optimizers.append(torch.optim.SGD(params,
                                    lr=self.base_lr[i],
                                    momentum=momentum,
                                    weight_decay=weight_decays[i]))
        elif optim == "adamw":
            for i, params in enumerate(param_list):
                optimizers.append(torch.optim.AdamW(params,
                                    lr=self.base_lr[i],
                                    weight_decay=weight_decays[i]))
        else:
            raise Exception(f"{optim} not configured")
        return optimizers                 

    def train(self, wandblogger, verbose = "epoch"):
        self.seg_model.train()
        if self.freeze_bn:
            for module in self.seg_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        for epoch in range(self.n_unsup_epochs):
            self.sup_train_loader.sampler.set_epoch(epoch)
            self.unsup_train_loader.sampler.set_epoch(epoch)
            pbar = tqdm(range(self.unsup_niters_per_epoch), file=sys.stdout, bar_format=self.bar_format)

            sup_dataloader = iter(self.sup_train_loader)
            unsup_dataloader = iter(self.unsup_train_loader)
            
            sum_loss_sup_l = 0
            sum_loss_sup_r = 0
            sum_loss_cps = 0
            sum_total_loss = 0
            all_results = []
            for idx in pbar:
                for optimizer in self.optimizers_l:
                    optimizer.zero_grad()
                for optimizer in self.optimizers_r:
                    optimizer.zero_grad()

                #NOTE: Only one forward pass through the model can be done because mmcv.ConvModule used in SegFormer has an inplace batch norm operation which raises this error:
                # "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
                minibatch = sup_dataloader.next()
                unsup_minibatch = unsup_dataloader.next()
                data = minibatch['data']
                gt = minibatch['gt']
                gt = gt.to(self.device)
                unsup_data = unsup_minibatch['data']
                data = [d.to(self.device) for d in data]
                unsup_data = [ud.to(self.device) for ud in unsup_data]
                all_data = [torch.cat([d_s, d_u], dim = 0) for d_s, d_u in zip(data, unsup_data)]
                out1, out2 = self.seg_model(all_data, step1 = True, step2 = True, get_sup_loss = True, gt = gt, criterion = self.sup_criterion)
                pred_l, loss_sup_l = out1
                pred_r, loss_sup_r = out2
                
                if type(pred_l) == list:
                    pred_l = pred_l[-1]
                if type(pred_r) == list:
                    pred_r = pred_r[-1]
                _, max_l = torch.max(pred_l, dim=1)
                _, max_r = torch.max(pred_r, dim=1)
                max_l = max_l.long()
                max_r = max_r.long()

                cps_loss = self.unsup_criterion(pred_l, max_r) + self.unsup_criterion(pred_r, max_l)
                dist.all_reduce(cps_loss, dist.ReduceOp.SUM)
                cps_loss = cps_loss / self.world_size
                
                dist.all_reduce(loss_sup_l, dist.ReduceOp.SUM)
                loss_sup_l = loss_sup_l / self.world_size

                dist.all_reduce(loss_sup_r, dist.ReduceOp.SUM)
                loss_sup_r = loss_sup_r / self.world_size

                current_idx = epoch * self.niters_per_epoch + idx
                lrs = [lr_policy.get_lr(current_idx) for lr_policy in self.lr_policy]
                for lr, optimizer in zip(lrs, self.optimizers_l):
                    for i in range(len(optimizer.param_groups)):
                        optimizer.param_groups[i]['lr'] = lr

                for lr, optimizer in zip(lrs, self.optimizers_r):
                    for i in range(len(optimizer.param_groups)):
                        optimizer.param_groups[i]['lr'] = lr
                
                loss = loss_sup_l + loss_sup_r + self.cps_weight * cps_loss
                
                loss.backward()
                for optimizer in self.optimizers_l:
                    optimizer.step()
                for optimizer in self.optimizers_r:
                    optimizer.step()
                
                sum_loss_sup_l += torch.tensor([loss_sup_l.item()], device = self.device)
                sum_loss_sup_r += torch.tensor([loss_sup_r.item()], device = self.device)
                sum_loss_cps += torch.tensor([cps_loss.item()], device = self.device)
                sum_total_loss += torch.tensor([loss.item()], device = self.device)


                print_str = 'Epoch{}/{}'.format(epoch, self.n_unsup_epochs) \
                        + ' Iter{}/{}:'.format(idx + 1, self.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup_l=%.2f' % loss_sup_l.item() \
                        + ' loss_sup_r=%.2f' % loss_sup_r.item() \
                        + ' loss_cps=%.4f' % cps_loss.item() \
                        + ' total_loss=%.4f' % loss.item()

                pbar.set_description(print_str, refresh = False)
            
                if wandblogger and verbose == 'iter':
                    if current_idx % self.snapshot_iter == 0:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.seg_model.module.state_dict(),
                            'optimizer_l_state_dict': [optimizer.state_dict() for optimizer in self.optimizers_l],
                            'optimizer_r_state_dict': [optimizer.state_dict() for optimizer in self.optimizers_r],
                            'current_idx': current_idx
                            }, os.path.join(self.snapshot_dir, "model_iter_" + str(current_idx) + ".pth"))
                    wandblogger.log({"Train/SupLoss_l": loss_sup_l.item(), 
                                    "Train/SupLoss_r": loss_sup_r.item(), 
                                    "Train/CPSLoss": cps_loss.item(), 
                                    "Train/TotalLoss": loss.item(), 
                                    # "Train/Mean_IU": mean_IU,
                                    # "Train/MeanPixelAcc": mean_pixel_acc
                                    }, step = current_idx)
            if wandblogger:
                if epoch % self.snapshot_epoch == 0:
                    torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.seg_model.module.state_dict(),
                            'optimizer_l_state_dict': [optimizer.state_dict() for optimizer in self.optimizers_l],
                            'optimizer_r_state_dict': [optimizer.state_dict() for optimizer in self.optimizers_r],
                            'current_idx': current_idx
                            }, os.path.join(self.snapshot_dir, "model_" + str(epoch) + ".pth"))
                    wandblogger.log({"Train/SupLoss_l": sum_loss_sup_l.item() / (len(pbar)), 
                                    "Train/SupLoss_r": sum_loss_sup_r.item() / (len(pbar)), 
                                    "Train/CPSLoss": sum_loss_cps.item() / (len(pbar)), 
                                    "Train/TotalLoss": sum_total_loss.item() / (len(pbar)),
                                    }, step = current_idx) 

    def evaluate_ddp_itr(self, wandblogger, itr, ddp = False):
        self.seg_model.eval()
        with torch.no_grad():
            sum_loss =[]
            all_results = []
            for mini_val_loader in self.val_loader:
                for minibatch in tqdm(mini_val_loader, desc="Rank:{}".format(self.rank)):
                    if not self.sliding_eval:
                        data = minibatch['data']
                        gt = minibatch['gt']
                        data = [d.to(self.device) for d in data]
                        gt = gt.to(self.device)
                        scores = self.seg_model(data)
                        if not type(scores) == list: #to make it consistent with models who return multiple predictions (like CEN or Token Exchange)
                            scores = [scores]
                        scores = [torch.nn.functional.interpolate(score, size = (gt.shape[1], gt.shape[2]), mode = 'bilinear', align_corners = True) for score in scores]
                        if self.eval_flip:
                            data = [d.flip(-1) for d in data]
                            scores_flip = self.seg_model(data)
                            if not type(scores_flip) == list: #to make it consistent with models who return multiple predictions (like CEN or Token Exchange)
                                scores_flip = [scores_flip]
                            scores_flip = [torch.nn.functional.interpolate(score_flip, size = (gt.shape[1], gt.shape[2]), mode = 'bilinear', align_corners = True) for score_flip in scores_flip]
                            scores = [(scores[i] + scores_flip[i].flip(-1))/2 for i in range(len(scores))]                  
                            
                    predictions = [score.argmax(1).cpu().numpy() for score in scores]
                    count = data[0].shape[0]
                    if len(sum_loss) == 0:
                        sum_loss = [0 for _ in range(len(predictions))]
                        all_results = [[] for _ in range(len(predictions))]

                    for i, (prediction, score) in enumerate(zip(predictions, scores)):
                        hist_tmp, labeled_tmp, correct_tmp = hist_info(self.num_classes,
                                                                    prediction,
                                                                    gt.cpu().numpy())
                        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp,
                                'correct': correct_tmp, "count": count}
                        loss = self.sup_criterion(score, gt)
                        sum_loss[i] += loss.item()
                        all_results[i].append(results_dict)
            
            wandbdict = {}
            for i, (all_result, sum_loss_i),  in enumerate(zip(all_results, sum_loss)):
                mean_IU, mean_pixel_acc, mean_acc = compute_metrics(all_result, self.num_classes)
                if ddp:
                    mean_IU = torch.tensor(mean_IU, device = self.device)
                    dist.all_reduce(mean_IU, dist.ReduceOp.SUM)
                    mean_IU = mean_IU / self.world_size

                    mean_pixel_acc = torch.tensor(mean_pixel_acc, device = self.device)
                    dist.all_reduce(mean_pixel_acc, dist.ReduceOp.SUM)
                    mean_pixel_acc = mean_pixel_acc / self.world_size
                    
                    mean_acc = torch.tensor(mean_acc, device = self.device)
                    dist.all_reduce(mean_acc, dist.ReduceOp.SUM)
                    mean_acc = mean_acc / self.world_size
                if wandblogger:
                    wandbdict.update({f"Val_Loss/Pred-{i}": sum_loss_i / self.val_size, 
                                f"Val_Mean_IU/Pred-{i}": mean_IU,
                                f"Val_Mean_Acc/Pred-{i}": mean_acc,
                                f"Val_MeanPixelAcc/Pred-{i}": mean_pixel_acc}
                                )    
                if self.rank == 0:
                    print(f"Iter: {itr}, Val Loss: {sum_loss_i / self.val_size}, Pred-{i} Val mean IoU: {mean_IU}, Val mean pix acc: {mean_pixel_acc}, Val mean acc: {mean_acc}")
            if wandblogger:
                wandblogger.log(wandbdict, step = itr)

    def evaluate(self, wandblogger, verbose, test_checkpoint = None, test_checkpoint_path = None):
        if test_checkpoint_path is not None:
            self.seg_model.module.load_state_dict(torch.load(path)['model_state_dict'])
            if self.rank is not None: #DDP Evaluation
                self.evaluate_ddp_itr(None, os.path.basename(test_checkpoint_path), ddp = True)    
            else:
                self.evaluate_ddp_itr(None, os.path.basename(test_checkpoint_path), ddp = False)    
        else:
            if verbose == 'epoch':
                rangePoints = range(0, self.n_unsup_epochs)
                if test_checkpoint is not None:
                    rangePoints = range(test_checkpoint, test_checkpoint + 1)
                for epoch in rangePoints:
                    print(epoch)
                    path = os.path.join(self.snapshot_dir, f"model_{epoch}.pth")
                    if os.path.exists(path):
                        self.seg_model.module.load_state_dict(torch.load(path)['model_state_dict'])
                        if self.rank is not None: #DDP Evaluation
                            self.evaluate_ddp_itr(wandblogger, epoch, ddp = True)    
                        else:
                            self.evaluate_ddp_itr(wandblogger, epoch, ddp = False)    
            else:
                rangePoints = range(0, self.total_iterations, self.snapshot_iter)
                if test_checkpoint is not None:
                    rangePoints = range(test_checkpoint, test_checkpoint + 1)
                for itr in rangePoints:
                    path = os.path.join(self.snapshot_dir, f"model_iter_{itr}.pth")
                    if os.path.exists(path):
                        print(itr)
                        self.seg_model.module.load_state_dict(torch.load(path)['model_state_dict'])
                        if self.rank is not None: #DDP Evaluation
                            self.evaluate_ddp_itr(wandblogger, itr, ddp = True)    
                        else:
                            self.evaluate_ddp_itr(wandblogger, itr, ddp=False)    
        return