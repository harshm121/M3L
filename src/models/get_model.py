from models.segmentation_models.cen import ChannelExchangingNetwork
from models.segmentation_models.deeplabv3p import DeepLabV3p_r101, DeepLabV3p_r18, DeepLabV3p_r50
from models.segmentation_models.linearfuse.segformer import WeTrLinearFusion
from models.segmentation_models.linearfusemaskedconsmixbatch.segformer import LinearFusionMaskedConsistencyMixBatch
from models.segmentation_models.refinenet import MyRefineNet
from models.segmentation_models.segformer.segformer import SegFormer
from models.segmentation_models.tokenfusion.segformer import WeTr
from models.segmentation_models.tokenfusionmaskedconsistencymixbatch.segformer import TokenFusionMaskedConsistencyMixBatch
from models.segmentation_models.unifiedrepresentation.segformer import UnifiedRepresentationNetwork
from models.segmentation_models.unifiedrepresentationmoddrop.segformer import UnifiedRepresentationNetworkModDrop

def get_model(args, **kwargs):
    if args.seg_model == "dlv3p":
        if args.base_model == "r18":
            return DeepLabV3p_r18(args.num_classes, args)
        elif args.base_model == "r50":
            return DeepLabV3p_r50(args.num_classes, args)
        elif args.base_model == "r101":
            return DeepLabV3p_r101(args.num_classes, args)
        else:
            raise Exception(f"{args.base_model} not configured")
    elif args.seg_model == 'refinenet':
        if args.base_model == 'r18':
            return MyRefineNet(num_layers = 18, num_classes = args.num_classes)
        if args.base_model == 'r50':
            return MyRefineNet(num_layers = 50, num_classes = args.num_classes)
        if args.base_model == 'r101':
            return MyRefineNet(num_layers = 101, num_classes = args.num_classes)
    elif args.seg_model == 'cen':
        if args.base_model == 'r18':
            return ChannelExchangingNetwork(num_layers = 18, num_classes = args.num_classes,  num_parallel = 2, l1_lambda = args.l1_lambda, bn_threshold = args.exchange_threshold)
        if args.base_model == 'r50':
            return ChannelExchangingNetwork(num_layers = 50, num_classes = args.num_classes,  num_parallel = 2, l1_lambda = args.l1_lambda, bn_threshold = args.exchange_threshold)
        if args.base_model == 'r101':
            return ChannelExchangingNetwork(num_layers = 101, num_classes = args.num_classes,  num_parallel = 2, l1_lambda = args.l1_lambda, bn_threshold = args.exchange_threshold)
    elif args.seg_model == 'segformer':
        return SegFormer(args.base_model, args, num_classes=  args.num_classes)
    elif args.seg_model == 'tokenfusion':
        return WeTr(args.base_model, args, l1_lambda = args.l1_lambda, num_classes = args.num_classes)
    elif args.seg_model == 'linearfusion':
        return WeTrLinearFusion(args.base_model, args, num_classes = args.num_classes)
    elif args.seg_model == 'linearfusionmaskedconsmixbatch':
        return LinearFusionMaskedConsistencyMixBatch(args.base_model, args, num_classes = args.num_classes)
    elif args.seg_model == 'tokenfusionmaskedconsmixbatch':
        return TokenFusionMaskedConsistencyMixBatch(args.base_model, args, l1_lambda = args.l1_lambda, num_classes = args.num_classes)
    elif args.seg_model == "unifiedrepresentationnetwork":
        return UnifiedRepresentationNetwork(args.base_model, args, num_classes = args.num_classes)
    elif args.seg_model == "unifiedrepresentationnetworkmoddrop":
        return UnifiedRepresentationNetworkModDrop(args.base_model, args, num_classes = args.num_classes)  
    else:
        raise Exception(f"{args.seg_model} not configured")