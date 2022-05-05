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

        self.m = torchvision.ops.FeaturePyramidNetwork([64, 128, 256, 512, 512, 1024], 256)

        # Implement RetinaNet

        self.resnet = models.resnet18(pretrained=True)

        self.additional_layers = torch.nn.ModuleList([
            torch.nn.Sequential( 
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential( 
                torch.nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(1024, 1024 , kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU(),
            ),
        ])
        def init_weights(layer):
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.normal_(layer.weight, 0, 0.01)
                layer.bias.data.fill_(0)
                
        self.additional_layers.apply(init_weights)

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = x
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
        out_features = [x1, x2, x3, x4]
        #print("Done with resnet")
        for i, layer in enumerate(self.additional_layers.children()):
            x5 = layer(x5)
            out_features.append(x5)
            #print(x5.shape)

        
        """for layer in self.m.children():
            print(layer)"""

        m_dict = {
            "tensor1": out_features[0],
            "tensor2": out_features[1],
            "tensor3": out_features[2],
            "tensor4": out_features[3],
            "tensor5": out_features[4],
            "tensor6": out_features[5],
        }
        out_features = self.m.forward(m_dict)
        out_features = list(out_features.values())
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

