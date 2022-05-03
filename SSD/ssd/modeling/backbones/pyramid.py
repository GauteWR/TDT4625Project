import torch
from typing import Tuple, List
import torchvision.models as models
import torchvision.ops.feature_pyramid_network as fpn


class PyramidModel(torch.nn.Module):

    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        # Implement RetinaNet

        self.resnet = models.resnet101(pretrained=True)

        self.additional_layers = torch.nn.ModuleList([
            torch.nn.Sequential( # 19 x 19 out
                torch.nn.Conv2d(output_channels[0], 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, output_channels[1], kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential( # 10x10 out
                torch.nn.Conv2d(output_channels[1], 2, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(2, output_channels[2], kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential( # 5 x 5 out
                torch.nn.Conv2d(output_channels[2], 4, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(4, output_channels[3], kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential( # 3 x 3 out
                torch.nn.Conv2d(output_channels[3], 8, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(8, output_channels[4], kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential( # 1 x 1 out
                torch.nn.Conv2d(output_channels[4], 16, kernel_size=2, padding=2),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, output_channels[5], kernel_size=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2,2)
            ),
        ])

    def forward(self, x):
        x = self.resnet.conv1(x)
        print(x.shape)
        x = self.resnet.bn1(x)
        print(x.shape)
        x = self.resnet.relu(x)
        print(x.shape)
        x = self.resnet.maxpool(x)
        print(x.shape)
        x1 = x
        for layer in self.resnet.layer1:
            x1 = layer(x1)
        
        x2 = x1
        for layer in self.resnet.layer2:
            x2 = layer(x2)
        x3 = x2
        for layer in self.resnet.layer3:
            x3 = layer(x3)
        x4 = x3
        for layer in self.resnet.layer4:
            x4 = layer(x4)
        print("ResNet finished")
        out_features = [x1, x2, x3, x4]
        for x in out_features:
            print(x.shape)
        x5 = x4
        print(x5.shape)
        for layer in self.additional_layers.children():
            print(layer)
            x5 = layer(x5)
            print(x.shape)
            out_features.append(x5)
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

