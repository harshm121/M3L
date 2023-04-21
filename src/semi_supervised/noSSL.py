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


class NoSSL(AbstractFramework):
    
    def __init__(self, seg_model, sup_criterion, datasetClass, args):
        super(NoSSL, self).__init__(seg_model, args)
        self.sup_criterion = sup_criterion
        self.base_lr = args.base_lr
        self.nepochs = args.n_sup_epochs
        self.niters_per_epoch = args.total_train_imgs // args.batch_size
        self.total_iterations = self.nepochs * self.niters_per_epoch
        self.lr_policy = [WarmUpPolyLR(base_lr, args.lr_power, self.total_iterations, args.niters_per_epoch * args.lr_warm_up_epoch) for base_lr in self.base_lr]
        if "restart" in args and args.restart:
            self.restart = True 
            self.restart_epoch = args.restart_epoch
        else:
            self.restart = False
            self.restart_epoch = 0
        if self.rank == None:
            self.seg_model = nn.DataParallel(seg_model).to(self.device)
        else:
            self.seg_model = self.seg_model.to(self.rank)
            self.seg_model = DDP(seg_model, device_ids=[self.rank], output_device = self.rank)

        self.freeze_bn = args.freeze_bn
        self.optimizers = self._get_optimizer(args.optim, args.weight_decay, args.momentum)
        self.bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        if self.rank is not None:
            self.sup_train_loader = get_train_loader(datasetClass, args, args.sup_train_source, False)
        self.num_classes = args.num_classes
        self.val_loader = get_val_loader(datasetClass, args)
        if not type(self.val_loader) == list:
            self.val_loader = [self.val_loader]
        self.val_size = np.sum([len(l) for l in self.val_loader])
        self.val_output_size = (args.data_image_height, args.data_image_width)

    def _get_optimizer(self, optim, weight_decays, momentum = None):
        param_list = self.seg_model.module.get_params()
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
        if self.restart: 
            checkpoint = torch.load(os.path.join(self.snapshot_dir, "model_" + str(self.restart_epoch) + ".pth"))
            self.seg_model.module.load_state_dict(checkpoint['model_state_dict'])
            for i in range(len(self.optimizers)):
                self.optimizers[i].load_state_dict(checkpoint['optimizer_state_dict'][i])
            epoch = checkpoint['epoch']
            assert self.restart_epoch == epoch
        self.seg_model.train()
        if self.freeze_bn:
            for module in self.seg_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        for epoch in range(self.restart_epoch, self.nepochs):
            self.sup_train_loader.sampler.set_epoch(epoch)
            pbar = tqdm(range(self.niters_per_epoch), file=sys.stdout, bar_format=self.bar_format)
            sup_dataloader = iter(self.sup_train_loader)
            
            sum_loss_sup = 0
            sum_total_loss = 0
            all_results = []
            
            for idx in pbar:
                for optimizer in self.optimizers:
                    optimizer.zero_grad()

                minibatch = sup_dataloader.next()
                
                data = minibatch['data']
                gt = minibatch['gt']
                data = [d.to(self.device) for d in data]
                gt = gt.to(self.device)
                
                pred_sup, loss_sup = self.seg_model(data, get_sup_loss = True, gt = gt, criterion = self.sup_criterion)
                dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
                loss_sup = loss_sup / self.world_size
                
                
                current_idx = epoch * self.niters_per_epoch + idx
                lrs = [lr_policy.get_lr(current_idx) for lr_policy in self.lr_policy]
                for lr, optimizer in zip(lrs, self.optimizers):
                    for i in range(len(optimizer.param_groups)):
                        optimizer.param_groups[i]['lr'] = lr
                
                loss = loss_sup
                
                loss.backward()
                for optimizer in self.optimizers:
                    optimizer.step()
                
                sum_loss_sup += torch.tensor([loss_sup.item()], device = self.device)
                
                sum_total_loss += torch.tensor([loss.item()], device = self.device)


                print_str = 'Epoch{}/{}'.format(epoch, self.nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, self.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup=%.2f' % loss_sup.item() \
                        + ' total_loss=%.4f' % loss.item()

                pbar.set_description(print_str, refresh = False)

                if wandblogger and verbose == 'iter':
                    wandblogger.log({"Train/SupLoss": loss_sup.item(), 
                                    "Train/TotalLoss": loss.item(),
                                    }, step = current_idx)
                    if current_idx % self.snapshot_iter == 0:
                        torch.save({
                            'iter': current_idx,
                            'model_state_dict': self.seg_model.module.state_dict(),
                            'optimizer_state_dict': [optimizer.state_dict() for optimizer in self.optimizers],
                            'current_idx': current_idx
                            }, os.path.join(self.snapshot_dir, "model_iter_" + str(current_idx) + ".pth"))
            if wandblogger:
                if epoch % self.snapshot_epoch == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.seg_model.module.state_dict(),
                        'optimizer_state_dict': [optimizer.state_dict() for optimizer in self.optimizers],
                        'current_idx': current_idx
                        }, os.path.join(self.snapshot_dir, "model_" + str(epoch) + ".pth"))
                if verbose == 'epoch':
                    wandblogger.log({"Train/SupLoss": sum_loss_sup.item() / (len(pbar)), 
                                    "Train/TotalLoss": sum_total_loss.item() / (len(pbar)),
                                    }, step = epoch)
                print(f"Logging complete in {self.rank}")


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
                        # data[0] = torch.rand_like(data[0])
                        # data[1] = torch.zeros_like(data[1])
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
                    print(f"Iter: {itr}, Val Loss: {sum_loss_i / self.val_size}, Student-{i} Val mean IoU: {mean_IU}, Val mean pix acc: {mean_pixel_acc}, Val mean acc: {mean_acc}")
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
                rangePoints = range(0, self.nepochs)
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