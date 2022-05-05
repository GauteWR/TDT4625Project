import torch
from typing import Tuple, List
import torchvision.models as models
import torchvision


class PyramidModel(torch.nn.Module):

    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        self.m = torchvision.ops.FeaturePyramidNetwork([1024, 512, 512, 128, 128], 1024)

        # Implement RetinaNet

        self.resnet = models.resnet18(pretrained=True)

        self.additional_layers = torch.nn.ModuleList([
            torch.nn.Sequential( 
                torch.nn.Conv2d(output_channels[0], 2048, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(2048, output_channels[1], kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential( 
                torch.nn.Conv2d(output_channels[1], 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, output_channels[2], kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(output_channels[2], 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, output_channels[3], kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential( 
                torch.nn.Conv2d(output_channels[3], 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[4], kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
            ),
        ])
    
    def forward(self, x):
        #print(x.shape)
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = x
        #print(x1.shape)
        for layer in self.resnet.layer1:
            x1 = layer(x1)
        x2 = x1
        #print(x2.shape)
        for layer in self.resnet.layer2:
            x2 = layer(x2)
        x3 = x2
        #print(x3.shape)
        for layer in self.resnet.layer3:
            x3 = layer(x3)
        x4 = x3
        #print(x4.shape)
        for layer in self.resnet.layer4:
            x4 = layer(x4)
        x5 = x4
        #print(x5.shape)
        out_features = [x4]
        for i, layer in enumerate(self.additional_layers.children()):
            x5 = layer(x5)
            # if i == len(self.additional_layers)-1:
            out_features.append(x5)
            # print(x5.shape)
        # print(len(out_features))
        # out_feat = dict(out_features)
        m_dict = {
            "tensor1": out_features[len(out_features)-1],
        }
        x6 = self.m.forward(m_dict)
        x6 = x6['tensor1'].to('cuda')
        out_features.append(x6)
        # print(x6['tensor1'].shape)
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            # print(feature.shape, "vs", expected_shape, " at idx:", idx)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

