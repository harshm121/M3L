import copy
import sys
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import SyncBatchNorm as BatchNorm2d
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from utils.init_utils import group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.seg_opr.metrics import compute_metrics, hist_info

from semi_supervised.base_framework import AbstractFramework
from datasets.get_dataset import get_train_loader, get_val_loader

class StudentTeacher(nn.Module):
    
    def __init__(self, student, teacher, ema_alpha):
        super(StudentTeacher, self).__init__()
        self.student = student
        self.teacher = teacher
        self.teacher = self._detach_teacher(self.teacher)
        self.ema_alpha = ema_alpha

    def forward(self, data, student = True, teacher = True, **kwargs):
        ret = []
        if student:
            ret.append(self.student(data, **kwargs))
        if teacher:
            ret.append(self.teacher(data, **kwargs))
        return ret
    
    def _detach_teacher(self, model):
        for param in model.parameters():
            param.detach_()
        return model

    def update_teacher_models(self, global_step):
        alpha = min(1 - 1 / (global_step + 1), self.ema_alpha)
        for ema_param, param in zip(self.teacher.parameters(), self.student.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)
        return
    
    def copy_student_to_teacher(self):
        for ema_param, param in zip(self.teacher.parameters(), self.student.parameters()):
            ema_param.data.mul_(0).add_(param.data)
        return 

    def get_params(self):
        student_params = self.student.get_params()
        teacher_params = self.teacher.get_params()
        return student_params
        


class MeanTeacher(AbstractFramework):
    
    def __init__(self, seg_model, sup_criterion, pseudo_sup_criterion, datasetClass, args):
        super(MeanTeacher, self).__init__(seg_model, args)
        self.sup_criterion = sup_criterion
        self.pseudo_sup_criterion = pseudo_sup_criterion
        self.base_lr = args.base_lr
        self.n_sup_epochs = args.n_sup_epochs
        self.n_unsup_epochs = args.n_unsup_epochs
        self.sup_niters_per_epoch = args.total_train_imgs // args.batch_size
        self.sup_total_iterations = self.n_sup_epochs * self.sup_niters_per_epoch
        self.unsup_niters_per_epoch = args.total_train_imgs // args.batch_size
        self.niters_per_epoch = args.total_train_imgs // args.batch_size
        self.unsup_total_iterations = self.n_unsup_epochs * self.unsup_niters_per_epoch
        self.total_iterations = self.sup_total_iterations + self.unsup_total_iterations
        self.lr_policy = [WarmUpPolyLR(base_lr, args.lr_power, self.total_iterations, self.unsup_niters_per_epoch * args.lr_warm_up_epoch) for base_lr in self.base_lr]
        self.freeze_bn = args.freeze_bn
        self.pseudo_loss_coeff = args.pseudo_loss_coeff
        if self.rank == None:
            self.seg_model = nn.DataParallel(seg_model).to(self.device)
        else:
            self.seg_model = self.seg_model.to(self.rank)
            self.seg_model = DDP(seg_model, device_ids=[self.rank], output_device = self.rank)

        self.optimizers = self._get_optimizer(args.optim, args.weight_decay, args.momentum)
        self.bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
        if self.rank is not None:
            self.sup_train_loader = get_train_loader(datasetClass, args, args.sup_train_source, False)
            self.unsup_train_loader = get_train_loader(datasetClass, args, args.unsup_train_source, True)
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
        self.seg_model.module.student.train()
        if self.freeze_bn:
            for module in self.seg_model.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()
        #Supervised loop
        for epoch in range(self.n_sup_epochs):
            self.sup_train_loader.sampler.set_epoch(epoch)
            pbar = tqdm(range(self.sup_niters_per_epoch), file=sys.stdout, bar_format=self.bar_format)
            sup_dataloader = iter(self.sup_train_loader)
            
            sum_loss_sup = 0
            sum_total_loss = 0
            for idx in pbar:
                for optimizer in self.optimizers:
                    optimizer.zero_grad()

                minibatch = sup_dataloader.next()
                
                data = minibatch['data']
                gt = minibatch['gt']
                data = [d.to(self.device) for d in data]
                gt = gt.to(self.device)
                _, loss_sup = self.seg_model(data, student = True, teacher = False, get_sup_loss = True, gt = gt, criterion = self.sup_criterion)[0]
                dist.all_reduce(loss_sup, dist.ReduceOp.SUM)
                loss_sup = loss_sup / self.world_size
                
                current_idx = epoch * self.sup_niters_per_epoch + idx
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

                print_str = 'Epoch{}/{}'.format(epoch, self.n_sup_epochs) \
                        + ' Iter{}/{}:'.format(idx + 1, self.sup_niters_per_epoch) \
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
        torch.save(self.seg_model.module.state_dict(), os.path.join(self.snapshot_dir, "model_sup_final" + str(self.n_sup_epochs) + ".pth"))
        
        #copy student parameters to teacher
        print("Copying Student to Teacher")
        self.seg_model.module.copy_student_to_teacher()

        #Unsupervised loop
        global_step = 0
        for epoch in range(self.n_sup_epochs, self.n_unsup_epochs+self.n_sup_epochs):
            self.sup_train_loader.sampler.set_epoch(epoch)
            self.unsup_train_loader.sampler.set_epoch(epoch)
            pbar = tqdm(range(self.unsup_niters_per_epoch), file=sys.stdout, bar_format=self.bar_format)
            
            sup_dataloader = iter(self.sup_train_loader)
            unsup_dataloader = iter(self.unsup_train_loader)
            sum_loss_sup = 0
            sum_total_loss = 0
            sum_loss_pseudo = 0
            sum_teacher_sup_loss = 0
            for idx in pbar:
                for optimizer in self.optimizers:
                    optimizer.zero_grad()
                
                #NOTE: Only one forward pass through the model can be done because mmcv.ConvModule used in SegFormer has an inplace batch norm operation which raises this error:
                # "RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation"
                minibatch = sup_dataloader.next()
                data = minibatch['data']
                gt = minibatch['gt']
                data = [d.to(self.device) for d in data]
                u_minibatch = unsup_dataloader.next()
                u_data = u_minibatch['data']
                u_data = [d.to(self.device) for d in u_data]
                # u_gt = torch.zeros(u_data[0].shape[0], *gt.shape[1:], dtype = torch.long) + 255 #Creating a fake gt for unsup data with all 255 which will be ignored
                all_data = [torch.cat([d_s, d_u], dim = 0) for d_s, d_u in zip(data, u_data)]
                # all_gt = torch.cat([gt, u_gt], dim = 0)
                gt = gt.to(self.device)
                out_student, out_teacher = self.seg_model(all_data, student = True, teacher = True, get_sup_loss = True, gt = gt, criterion = self.sup_criterion)
                sup_loss_student = out_student[1] #Unsup data would have incurred zero loss as all the labels were 255 (which is ignored in the sup_criterion)
                dist.all_reduce(sup_loss_student, dist.ReduceOp.SUM)
                sup_loss_student = sup_loss_student / self.world_size

                sup_loss_teacher = out_teacher[1] #Sup teacher loss, not used for backprop but just for reporting
                dist.all_reduce(sup_loss_teacher, dist.ReduceOp.SUM)
                sup_loss_teacher = sup_loss_teacher / self.world_size

                pred_teacher = out_teacher[0]
                pred_student = out_student[0]
                if type(pred_teacher) == list: #Some seg_models return a list of predictions
                    pred_teacher = pred_teacher[-1] #using the last element in the list as that is the ensemble prediction
                pred_teacher = nn.functional.softmax(pred_teacher, dim=1)
                pseudo_logits, pseudo_gt = torch.max(pred_teacher, dim=1)
                pesudo_loss_all = self.seg_model.module.student.get_sup_loss(pred_student, pseudo_gt, self.pseudo_sup_criterion)
                dist.all_reduce(pesudo_loss_all, dist.ReduceOp.SUM)
                pesudo_loss_all = pesudo_loss_all / self.world_size

                current_idx = epoch * self.niters_per_epoch + idx
                lrs = [lr_policy.get_lr(current_idx) for lr_policy in self.lr_policy]
                for lr, optimizer in zip(lrs, self.optimizers):
                    for i in range(len(optimizer.param_groups)):
                        optimizer.param_groups[i]['lr'] = lr
                
                loss = sup_loss_student + self.pseudo_loss_coeff * pesudo_loss_all
                loss.backward()

                for optimizer in self.optimizers:
                    optimizer.step()
                self.seg_model.module.update_teacher_models(global_step)
                global_step += 1
                sum_loss_sup += torch.tensor([sup_loss_student.item()], device = self.device)
                sum_loss_pseudo += torch.tensor([pesudo_loss_all.item()], device = self.device)
                sum_total_loss += torch.tensor([loss.item()], device = self.device)
                sum_teacher_sup_loss += torch.tensor([sup_loss_teacher.item()], device = self.device)

                print_str = 'Unsup Epoch{}/{}'.format(epoch, self.n_unsup_epochs + self.n_sup_epochs) \
                        + ' Iter{}/{}:'.format(idx + 1, self.niters_per_epoch) \
                        + ' lr=%.2e' % lr \
                        + ' loss_sup=%.2f' % sup_loss_student.item() \
                        + ' total_loss=%.4f' % loss.item()

                pbar.set_description(print_str, refresh = False)
                if wandblogger and verbose == 'iter':
                    wandblogger.log({"Train/SupLoss": sup_loss_student.item(), 
                                    "Train/PseudoSupLoss": pesudo_loss_all.item(), 
                                    "Train/TotalLoss": loss.item(), 
                                    "Train/TeacherSupLoss": sup_loss_teacher.item()}, step = current_idx)
                    if current_idx % self.snapshot_iter == 0:
                        torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.seg_model.module.state_dict(),
                        'optimizer_state_dict': [optimizer.state_dict() for optimizer in self.optimizers],
                        'current_idx': current_idx
                        }, os.path.join(self.snapshot_dir, "model_iter_" + str(current_idx) + ".pth"))
            if wandblogger:
                print(f"Logging in {self.rank}")
                if epoch % self.snapshot_epoch == 0:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.seg_model.module.state_dict(),
                        'optimizer_state_dict': [optimizer.state_dict() for optimizer in self.optimizers],
                        'current_idx': current_idx
                        }, os.path.join(self.snapshot_dir, "model_" + str(epoch) + ".pth"))
                if verbose == 'epoch':
                    wandblogger.log({"Train/SupLoss": sum_loss_sup.item() / (len(pbar)), 
                                    "Train/PseudoSupLoss": sum_loss_pseudo.item() / (len(pbar)), 
                                    "Train/TotalLoss": sum_total_loss.item() / (len(pbar)), 
                                    "Train/TeacherSupLoss": sum_teacher_sup_loss.item() / (len(pbar)),
                                    }, step = epoch)
                print(f"Logging complete in {self.rank}")
            
    def evaluate_ddp_itr(self, wandblogger, itr, ddp = False):
        self.seg_model.eval()
        with torch.no_grad():
            sum_loss_student = []
            all_results_student = []
            sum_loss_teacher = []
            for mini_val_loader in self.val_loader:
                for minibatch in tqdm(mini_val_loader):
                    if not self.sliding_eval:
                        data = minibatch['data']
                        gt = minibatch['gt']
                        data = [d.to(self.device) for d in data]
                        gt = gt.to(self.device)
                        student_scores, teacher_scores = self.seg_model(data = data, student = True, teacher = True)
                        if not type(student_scores) == list:
                            student_scores = [student_scores]

                        if not type(teacher_scores) == list:
                            teacher_scores = [teacher_scores]

                        
                        student_scores = [torch.nn.functional.interpolate(student_score, size = (gt.shape[1], gt.shape[2]), mode = 'bilinear', align_corners = True) for student_score in student_scores]
                        teacher_scores = [torch.nn.functional.interpolate(teacher_score, size = (gt.shape[1], gt.shape[2]), mode = 'bilinear', align_corners = True) for teacher_score in teacher_scores]                        
                        
                        if self.eval_flip:
                            data = [d.flip(-1) for d in data]
                            student_scores_flip, teacher_scores_flip = self.seg_model(data, student = True, teacher = True)
                            student_scores_flip = [torch.nn.functional.interpolate(student_score, size = (gt.shape[1], gt.shape[2]), mode = 'bilinear', align_corners = True) for student_score in student_scores_flip]
                            student_scores = [(student_scores[i] + student_scores_flip[i].flip(-1))/2 for i in range(len(student_scores))]                  
                            teacher_scores_flip = [torch.nn.functional.interpolate(teacher_score, size = (gt.shape[1], gt.shape[2]), mode = 'bilinear', align_corners = True) for teacher_score in teacher_scores_flip]                        
                            teacher_scores = [(teacher_scores[i] + teacher_scores_flip[i].flip(-1))/2 for i in range(len(teacher_scores))]                  
                        
                    else:
                        raise Exception("Sliding eval not configured for mean teacher")
                    
                    
                    student_predictions = [student_score.argmax(1).cpu().numpy() for student_score in student_scores]
                    teacher_predictions = [teacher_score.argmax(1).cpu().numpy() for teacher_score in teacher_scores]
                    
                    count = data[0].shape[0]
                    if len(sum_loss_student) == 0:
                        sum_loss_student = [0 for _ in range(len(student_predictions))]
                        all_results_student = [[] for _ in range(len(student_predictions))]
                    for i, (predictions, scores) in enumerate(zip(student_predictions, student_scores)):
                        hist_tmp, labeled_tmp, correct_tmp = hist_info(self.num_classes,
                                                                    predictions,
                                                                    gt.cpu().numpy())
                        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp,
                                'correct': correct_tmp, "count": count}
                        loss = self.sup_criterion(scores, gt)
                        sum_loss_student[i] += loss.item()
                        all_results_student[i].append(results_dict)

                    if len(sum_loss_teacher) == 0:
                        sum_loss_teacher = [0 for _ in range(len(teacher_predictions))]
                        all_results_teacher = [[] for _ in range(len(teacher_predictions))]
                    for i, (predictions, scores) in enumerate(zip(teacher_predictions, teacher_scores)):
                        hist_tmp, labeled_tmp, correct_tmp = hist_info(self.num_classes,
                                                                    predictions,
                                                                    gt.cpu().numpy())
                        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp,
                                'correct': correct_tmp, "count": count}
                        loss = self.sup_criterion(scores, gt)
                        sum_loss_teacher[i] += loss.item()
                        all_results_teacher[i].append(results_dict)
            
            wandbdict = {}
            for i, (all_results, sum_loss),  in enumerate(zip(all_results_student, sum_loss_student)):
                mean_IU, mean_pixel_acc, mean_acc = compute_metrics(all_results, self.num_classes)
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
                print(f"Epoch: {itr}, Val Loss: {sum_loss / self.val_size}, Student-{i} Val mean IoU: {mean_IU}, Val mean acc: {mean_acc}, Val mean pix acc: {mean_pixel_acc}")
                wandbdict.update({f"Val_Loss/Student-{i}": sum_loss / self.val_size, 
                            f"Val_Mean_IU/Student-{i}": mean_IU,
                            f"Val_Mean_Acc/Student-{i}": mean_acc,
                            f"Val_MeanPixelAcc/Student-{i}": mean_pixel_acc}
                            )    
            for i, (all_results, sum_loss),  in enumerate(zip(all_results_teacher, sum_loss_teacher)):
                mean_IU_t, mean_pixel_acc_t, mean_acc_t = compute_metrics(all_results, self.num_classes)
                if ddp:
                    mean_IU_t = torch.tensor(mean_IU_t, device = self.device)
                    dist.all_reduce(mean_IU_t, dist.ReduceOp.SUM)
                    mean_IU_t = mean_IU_t / self.world_size

                    mean_pixel_acc_t = torch.tensor(mean_pixel_acc_t, device = self.device)
                    dist.all_reduce(mean_pixel_acc_t, dist.ReduceOp.SUM)
                    mean_pixel_acc_t = mean_pixel_acc_t / self.world_size
                    
                    mean_acc_t = torch.tensor(mean_acc_t, device = self.device)
                    dist.all_reduce(mean_acc_t, dist.ReduceOp.SUM)
                    mean_acc_t = mean_acc_t / self.world_size
                print(f"Epoch: {itr}, Val Loss: {sum_loss / self.val_size}, Teacher-{i} Val mean IoU: {mean_IU_t}, Val mean acc: {mean_acc_t}, Val mean pix acc: {mean_pixel_acc_t}")
                wandbdict.update({f"Val_Loss/Teacher-{i}": sum_loss / self.val_size, 
                            f"Val_Mean_IU/Teacher-{i}": mean_IU_t,
                            f"Val_Mean_Acc/Teacher-{i}": mean_acc_t,
                            f"Val_MeanPixelAcc/Teacher-{i}": mean_pixel_acc_t}
                            )
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
            if verbose == "epoch":
                rangePoints = range(0, self.n_sup_epochs + self.n_unsup_epochs)
                if test_checkpoint is not None:
                    rangePoints = range(test_checkpoint, test_checkpoint + 1)
                for epoch in rangePoints:
                # for epoch in [21]:
                    print(epoch)
                    path = os.path.join(self.snapshot_dir, f"model_{epoch}.pth")
                    if os.path.exists(path):
                        self.seg_model.module.load_state_dict(torch.load(path)['model_state_dict'])
                        self.seg_model.module.mode = 'val'
                        if self.rank is not None: #DDP Evaluation
                            self.evaluate_ddp_itr(wandblogger, epoch, ddp = True)    
                        else:
                            self.evaluate_ddp_itr(wandblogger, epoch, ddp = False)
            else:
                rangePoints = range(0, self.total_iterations, self.snapshot_iter)
                if test_checkpoint is not None:
                    rangePoints = range(test_checkpoint, test_checkpoint + 1)
                for itr in rangePoints:
                # for itr in [7200]:
                    path = os.path.join(self.snapshot_dir, f"model_iter_{itr}.pth")
                    if os.path.exists(path):
                        print(itr)
                        self.seg_model.module.load_state_dict(torch.load(path)['model_state_dict'])
                        if self.rank is not None: #DDP Evaluation
                            self.evaluate_ddp_itr(wandblogger, itr, ddp=True)    
                        else:
                            self.evaluate_ddp_itr(wandblogger, itr, ddp=False)
        return