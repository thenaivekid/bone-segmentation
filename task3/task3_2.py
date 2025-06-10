import torchvision.models as models
import torch
import torch.nn as nn
from collections import OrderedDict
import copy


def convert_conv2d_to_conv3d(conv2d):
    """Convert a 2D convolution layer to 3D with inflated weights"""
    conv3d = nn.Conv3d(
        in_channels=conv2d.in_channels,
        out_channels=conv2d.out_channels,
        kernel_size=(conv2d.kernel_size[0], conv2d.kernel_size[0], conv2d.kernel_size[1]),
        stride=(conv2d.stride[0], conv2d.stride[0], conv2d.stride[1]),
        padding=(conv2d.padding[0], conv2d.padding[0], conv2d.padding[1]),
        bias=conv2d.bias is not None
    )
    
    weight_2d = conv2d.weight.data  
    kernel_depth = conv2d.kernel_size[0]  # Use same as height/width
    weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, kernel_depth, 1, 1) / kernel_depth
    conv3d.weight.data = weight_3d
    
    if conv2d.bias is not None:
        conv3d.bias.data = conv2d.bias.data.clone()
    # print("expanded 2d conv")
    return conv3d

def convert_batchnorm2d_to_batchnorm3d(bn2d):
    """Convert 2D batch norm to 3D batch norm"""
    bn3d = nn.BatchNorm3d(bn2d.num_features)
    bn3d.weight.data = bn2d.weight.data.clone()
    bn3d.bias.data = bn2d.bias.data.clone()
    bn3d.running_mean.data = bn2d.running_mean.data.clone()
    bn3d.running_var.data = bn2d.running_var.data.clone()
    bn3d.eps = bn2d.eps
    bn3d.momentum = bn2d.momentum
    return bn3d

def convert_densenet_to_3d(model_2d):
    """Recursively convert DenseNet 2D model to 3D"""
    def convert_module(module):
        if isinstance(module, nn.Conv2d):
            return convert_conv2d_to_conv3d(module)
        elif isinstance(module, nn.BatchNorm2d):
            return convert_batchnorm2d_to_batchnorm3d(module)

        elif isinstance(module, nn.AvgPool2d):
            return nn.AvgPool3d(
                kernel_size=(module.kernel_size, module.kernel_size, module.kernel_size) if isinstance(module.kernel_size, int) else (module.kernel_size[0], module.kernel_size[0], module.kernel_size[1]),
                stride=(module.stride, module.stride, module.stride) if isinstance(module.stride, int) else (module.stride[0], module.stride[0], module.stride[1]),
                padding=(module.padding, module.padding, module.padding) if isinstance(module.padding, int) else (module.padding[0], module.padding[0], module.padding[1])
            )
        elif isinstance(module, nn.MaxPool2d):
            return nn.MaxPool3d(
                kernel_size=(module.kernel_size, module.kernel_size, module.kernel_size) if isinstance(module.kernel_size, int) else (module.kernel_size[0], module.kernel_size[0], module.kernel_size[1]),
                stride=(module.stride, module.stride, module.stride) if isinstance(module.stride, int) else (module.stride[0], module.stride[0], module.stride[1]),
                padding=(module.padding, module.padding, module.padding) if isinstance(module.padding, int) else (module.padding[0], module.padding[0], module.padding[1])
            )
        elif hasattr(module, 'children') and len(list(module.children())) > 0:
            converted_children = OrderedDict()
            for name, child in module.named_children():
                converted_children[name] = convert_module(child)
            
            new_module = copy.deepcopy(module)
            for name, child in converted_children.items():
                setattr(new_module, name, child)
            return new_module
        else:
            return copy.deepcopy(module)
    
    return convert_module(model_2d)

def get_loaded_3d_model():
    model_2d = models.densenet121(pretrained=True)
    model_3d = convert_densenet_to_3d(model_2d)

    # Modify the first conv layer to use single channel input
    first_conv_2d = model_2d.features.conv0
    first_conv_3d = nn.Conv3d(
        in_channels=1, 
        out_channels=first_conv_2d.out_channels,
        kernel_size=(first_conv_2d.kernel_size[0], first_conv_2d.kernel_size[0], first_conv_2d.kernel_size[1]),
        stride=(first_conv_2d.stride[0], first_conv_2d.stride[0], first_conv_2d.stride[1]),
        padding=(first_conv_2d.padding[0], first_conv_2d.padding[0], first_conv_2d.padding[1]),
        bias=first_conv_2d.bias is not None
    )

    # Average the RGB weights to create single channel weights
    rgb_weights = first_conv_2d.weight.data
    single_channel_weight = rgb_weights.mean(dim=1, keepdim=True)
    kernel_depth = first_conv_2d.kernel_size[0]
    inflated_weight = single_channel_weight.unsqueeze(2).repeat(1, 1, kernel_depth, 1, 1) / kernel_depth
    first_conv_3d.weight.data = inflated_weight

    if first_conv_2d.bias is not None:
        first_conv_3d.bias.data = first_conv_2d.bias.data.clone()

    model_3d.features.conv0 = first_conv_3d
    return model_3d

if __name__ == "__main__":
    x = torch.randn(1, 1, 216, 512, 512)  # (batch, channels, depth, height, width)
    model_3d = get_loaded_3d_model()
    with torch.no_grad():
        features_output = model_3d.features(x)
    print(f"Features output shape: {features_output.shape}")#[1, 1024, 6, 16, 16]

    print(f"\nModel successfully converted from 2D to 3D!")