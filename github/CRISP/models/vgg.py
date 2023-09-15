import torch.nn as nn
import torch.utils.model_zoo as model_zoo


from utils.builder import get_builder

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(nn.Module):

    def __init__(self, builder, features, num_classes=1000, init_weights=False):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            builder.conv1x1_fc(512 * 7 * 7, 4096),
            # nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            builder.conv1x1_fc(4096, 4096),   
            #nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            builder.conv1x1_fc(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1).unsqueeze(dim=2).unsqueeze(dim=3)
        #print(x.shape)
        x = self.classifier(x)
        return x.flatten(1)
        # x = self.classifier2(x)
        # x = self.classifier3(x)
        # return x.flatten(1)

def make_layers(builder, cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = builder.conv3x3(in_channels, v, bias=True)
            #conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(get_builder(), make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11']))
    return model


def vgg11_bn(pretrained=False, **kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(get_builder(), make_layers(cfg['A'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))
    return model


def vgg13(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(get_builder(), make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13']))
    return model


def vgg13_bn(pretrained=False, **kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(get_builder(), make_layers(cfg['B'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13_bn']))
    return model


def vgg16(pretrained=True, **kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    builder = get_builder()
    model = VGG(builder, make_layers(builder, cfg['D']), **kwargs)
    # for name, param in model.named_parameters():
    #     print(name)
    
    if pretrained:
        ckpt = model_zoo.load_url(model_urls['vgg16'])
        ckpt_ld = model.state_dict()
        for param in ckpt.keys():
            if param in ckpt_ld.keys():
                ckpt_ld[param] = ckpt[param]
        ckpt_ld['classifier.0.weight'] = ckpt['classifier.0.weight'].view(
            ckpt['classifier.0.weight'].size(0), ckpt['classifier.0.weight'].size(1), 1, 1)
        ckpt_ld['classifier.3.weight'] = ckpt['classifier.3.weight'].view(
            ckpt['classifier.3.weight'].size(0), ckpt['classifier.3.weight'].size(1), 1, 1)
        ckpt_ld['classifier.6.weight'] = ckpt['classifier.6.weight'].view(
            ckpt['classifier.6.weight'].size(0), ckpt['classifier.6.weight'].size(1), 1, 1)

        model.load_state_dict(ckpt_ld, strict=False)
    return model


def vgg16_bn(pretrained=True, **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    builder = get_builder()
    model = VGG(builder, make_layers(builder, cfg['D'], batch_norm=True), **kwargs)
    # if pretrained:
    #     ckpt = model_zoo.load_url(model_urls['vgg16_bn'])
    #     for param in ckpt.keys():
    #         if 'classifier' in param:
    #             ckpt[param] =  ckpt[param].view(
    #         ckpt[param].size(0), ckpt[param].size(1), 1, 1)
    #     model.load_state_dict(ckpt)

    if pretrained:
        ckpt = model_zoo.load_url(model_urls['vgg16_bn'])
        ckpt_ld = model.state_dict()
        for param in ckpt.keys():
            if param in ckpt_ld.keys():
                ckpt_ld[param] = ckpt[param]
        ckpt_ld['classifier.0.weight'] = ckpt['classifier.0.weight'].view(
            ckpt['classifier.0.weight'].size(0), ckpt['classifier.0.weight'].size(1), 1, 1)
        ckpt_ld['classifier.3.weight'] = ckpt['classifier.3.weight'].view(
            ckpt['classifier.3.weight'].size(0), ckpt['classifier.3.weight'].size(1), 1, 1)
        ckpt_ld['classifier.6.weight'] = ckpt['classifier.6.weight'].view(
            ckpt['classifier.6.weight'].size(0), ckpt['classifier.6.weight'].size(1), 1, 1)
        model.load_state_dict(ckpt_ld, strict=False)

    return model


def vgg19(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(get_builder(),make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19']))
    return model


def vgg19_bn(pretrained=False, **kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(get_builder(),make_layers(cfg['E'], batch_norm=True), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19_bn']))
    return model