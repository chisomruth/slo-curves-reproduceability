import torch.nn as nn
import torchvision.models as tvm

NUM_CLASSES = {'cifar10': 10, 'cifar100': 100}


def _build_resnet(base_fn, num_classes: int) -> nn.Module:
    model = base_fn(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_densenet(base_fn, num_classes: int) -> nn.Module:
    model = base_fn(weights=None)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.features.pool0 = nn.Identity()
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def _build_vgg19(num_classes: int) -> nn.Module:
    model = tvm.vgg19_bn(weights=None)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(512, 512),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes),
    )
    return model


def _build_alexnet(num_classes: int) -> nn.Module:
    features = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(64, 192, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(192, 384, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(384, 256, kernel_size=3, padding=1),
        nn.ReLU(True),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(True),
    )
    classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(256, 2048),
        nn.ReLU(True),
        nn.Dropout(0.5),
        nn.Linear(2048, 2048),
        nn.ReLU(True),
        nn.Linear(2048, num_classes),
    )
    model = tvm.alexnet(weights=None)   
    model.features = features
    model.avgpool  = nn.AdaptiveAvgPool2d((1, 1))
    model.classifier = classifier
    return model


def _build_mobilenet_v3_small(num_classes: int) -> nn.Module:
    model = tvm.mobilenet_v3_small(weights=None)
    first_conv = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        first_conv.in_channels, first_conv.out_channels,
        kernel_size=3, stride=1, padding=1, bias=False,
    )
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


def get_model(architecture: str, dataset: str) -> nn.Module:
    num_classes = NUM_CLASSES[dataset]
    arch = architecture.lower()

    builders = {
        'resnet18':          lambda: _build_resnet(tvm.resnet18,   num_classes),
        'resnet101':         lambda: _build_resnet(tvm.resnet101,  num_classes),
        'vgg19':             lambda: _build_vgg19(num_classes),
        'alexnet':           lambda: _build_alexnet(num_classes),
        'mobilenet_v3_small':lambda: _build_mobilenet_v3_small(num_classes),
        'densenet121':       lambda: _build_densenet(tvm.densenet121, num_classes),
    }
    if arch not in builders:
        raise ValueError(f"Unknown architecture '{arch}'. "
                         f"Choose from {list(builders.keys())}")
    return builders[arch]()