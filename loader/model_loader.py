import settings
import torch
import torchvision

from methods.protonet import ProtoNet
from backbone import Genotype
import backbone


def loadmodel(hook_fn):
    if settings.MODEL_FILE is None:
        model = torchvision.models.__dict__[settings.MODEL](pretrained=True)
    else:
        checkpoint = torch.load(settings.MODEL_FILE)
        if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
            if settings.MODEL == 'genotype':
                backbone.geneC = settings.CHANNELS #params.channels
                backbone.geneLayers = settings.LAYERS #params.layers
                backbone.geneName = settings.GENE_NAME #params.gene_name
                model = ProtoNet(Genotype, settings.N_WAY, settings.N_SHOT)
                state_dict = checkpoint['state']
                model.load_state_dict(state_dict)
            else:
                model = torchvision.models.__dict__[settings.MODEL](num_classes=settings.NUM_CLASSES)
                if settings.MODEL_PARALLEL:
                    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                        'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
                else:
                    state_dict = checkpoint
                model.load_state_dict(state_dict)
        else:
            model = checkpoint
    for name in settings.FEATURE_NAMES:
        if settings.MODEL == 'genotype':
            model._modules.get('feature')._modules.get('cells')._modules.get('%d'%(settings.LAYER-1)).register_forward_hook(hook_fn)
        else:
            model._modules.get(name).register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model
