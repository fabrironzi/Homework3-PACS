import torch
import torch.nn as nn
import copy
from torch.autograd import Function

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

''' 
Very easy template to start for developing your AlexNet with DANN 
Has not been tested, might contain incompatibilities with most recent versions of PyTorch (you should address this)
However, the logic is consistent
'''

__all__ = ['AlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class ReverseLayerF(Function):
    # Forwards identity
    # Sends backward reversed gradients
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=1, stride=1),
            nn.BatchNorm2d(64), nn.MaxPool2d(2), nn.ReLU(True),
            nn.Conv2d(64, 50, kernel_size=5, padding=1, stride=1),
            nn.BatchNorm2d(50), nn.MaxPool2d(2), nn.ReLU(True),
            nn.Dropout2d(),
        )
        self.num_cnn_features = 50 * 5 * 5
        self.class_classifier = nn.Sequential(
            nn.Linear(self.num_cnn_features, 100),
            nn.BatchNorm1d(100), nn.Dropout2d(), nn.ReLU(True),
            nn.Linear(100, 100), 
            nn.BatchNorm1d(100), nn.ReLU(True),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1),
        )
        self.domain_classifier = nn.Sequential(
            nn.Linear(self.num_cnn_features, 100),
            nn.BatchNorm1d(100), nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x, alpha=None):
        feature_extractor = self.feature_extractor(x)
        #feature_extractor = self.avgpool(x)
        
        # Flatten the features:
        feature_extractor = feature_extractor.view(-1, self.cnn_num_features)
        #feature_extractor = feature_extractor.view(feature_extractor.size(0), -1)
        #feature_extractor = torch.flatten(x, 1)
        # If we pass alpha, we can assume we are training the discriminator
        if alpha is not None:
            # gradient reversal layer (backward gradients will be reversed)
            reverse_feature = ReverseLayerF.apply(feature_extractor, alpha)
            discriminator_output = self.domain_classifier(reverse_feature)
            return discriminator_output
        # If we don't pass alpha, we assume we are training with supervision
        else:
            # do something else
            class_outputs = self.class_classifier(feature_extractor)
            return class_outputs
                
def Myalexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = DANN(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['alexnet'],
                                              progress=progress)
        state_dict.popitem("classifier.6.bias")
        state_dict.popitem("classifier.6.weight") 
        model.load_state_dict(state_dict, strict=False)
        model.domain_classifier[1].weight.data =  copy.deepcopy(model.class_classifier[1].weight.data)
        model.domain_classifier[1].bias.data = copy.deepcopy(model.class_classifier[1].bias.data)
        model.domain_classifier[4].weight.data = copy.deepcopy(model.class_classifier[4].weight.data)
        model.domain_classifier[4].bias.data = copy.deepcopy(model.class_classifier[4].bias.data)
        
    return model
