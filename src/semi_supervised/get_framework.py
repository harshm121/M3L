import torch.nn as nn
from datasets.get_dataset import get_dataset
from semi_supervised.meanteacher import MeanTeacher, StudentTeacher
from semi_supervised.meanteachermaskstudent import MeanTeacherMaskStudent, MaskStudentTeacher, MaskStudentMaskTeacher
from semi_supervised.noSSLModDrop import NoSSLModDrop
from semi_supervised.cps import CPSFramework, CPSNetwork
from semi_supervised.noSSL import NoSSL
from utils.seg_opr.loss_func import ProbOhemCrossEntropy2d
from models.get_model import get_model



def get_framework(args):
    datasetClass = get_dataset(args)
    if args.rank == None:
        pixel_num = int(args.image_height * args.image_width * 0.08) * args.val_batch_size
    else:
        pixel_num = int(args.image_height * args.image_width * 0.08) * args.batch_size // args.world_size
    if args.ssl_framework == "cps":
        branch1 = get_model(args)
        branch2 = get_model(args)
        ssl_model = CPSNetwork(branch1, branch2)
        sup_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=pixel_num, use_weight=False)
        cps_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
        framework = CPSFramework(ssl_model, sup_criterion, cps_criterion, datasetClass, args)
        return framework
    if args.ssl_framework == "nossl":
        seg_model = get_model(args)
        sup_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=pixel_num, use_weight=False)#nn.CrossEntropyLoss(reduction='mean', ignore_index=255)#
        framework = NoSSL(seg_model, sup_criterion, datasetClass, args)
        return framework
    if args.ssl_framework == "nosslmoddrop":
        seg_model = get_model(args)
        sup_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=pixel_num, use_weight=False)#nn.CrossEntropyLoss(reduction='mean', ignore_index=255)#
        framework = NoSSLModDrop(seg_model, sup_criterion, datasetClass, args)
        return framework
    if args.ssl_framework == "meanteacher":
        student = get_model(args)
        teacher = get_model(args)
        seg_model = StudentTeacher(student, teacher, args.ema_alpha)
        sup_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=pixel_num, use_weight=False)
        pseudo_sup_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
        framework = MeanTeacher(seg_model = seg_model, sup_criterion = sup_criterion, pseudo_sup_criterion = pseudo_sup_criterion, datasetClass = datasetClass, args = args)
        return framework
    if args.ssl_framework == "meanteachermaskedstudent":
        teacher = get_model(args)
        student = get_model(args)
        seg_model = MaskStudentTeacher(student, teacher, args.ema_alpha)
        sup_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=pixel_num, use_weight=False)
        pseudo_sup_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
        framework = MeanTeacherMaskStudent(seg_model = seg_model, sup_criterion = sup_criterion, pseudo_sup_criterion = pseudo_sup_criterion, datasetClass = datasetClass, args = args)
        return framework
    if args.ssl_framework == "meanteachermoddrop":
        teacher = get_model(args)
        student = get_model(args)
        seg_model = MaskStudentMaskTeacher(student, teacher, args.ema_alpha)
        sup_criterion = ProbOhemCrossEntropy2d(ignore_label=255, thresh=0.7, min_kept=pixel_num, use_weight=False)
        pseudo_sup_criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=255)
        framework = MeanTeacherMaskStudent(seg_model = seg_model, sup_criterion = sup_criterion, pseudo_sup_criterion = pseudo_sup_criterion, datasetClass = datasetClass, args = args)
        return framework
    else:
        raise Exception(f"{args.ssl_framework} not configured")

