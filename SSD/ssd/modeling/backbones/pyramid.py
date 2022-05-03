import torch
from typing import Tuple, List


class PyramidModel(torch.nn.Module):

    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes

        print(image_channels)

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(image_channels, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2), # 64x512 out
            torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2,2), 
            # 1, 128, 32 x 256 out
        )

        self.additional_layers = torch.nn.ModuleList([
            torch.nn.Sequential( # 19 x 19 out
                torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[1], kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential( # 10x10 out
                torch.nn.Conv2d(output_channels[1], 256, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, output_channels[2], kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential( # 5 x 5 out
                torch.nn.Conv2d(output_channels[2], 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[3], kernel_size=3, padding=1, stride=2),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential( # 3 x 3 out
                torch.nn.Conv2d(output_channels[3], 128, kernel_size=3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[4], kernel_size=3, stride=2, padding=1),
                torch.nn.ReLU(),
            ),
            torch.nn.Sequential( # 1 x 1 out
                torch.nn.Conv2d(output_channels[4], 128, kernel_size=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, output_channels[5], kernel_size=2),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2,2)
            ),
        ])

    def forward(self, x):
        x = self.feature_extractor(x)
        out_features = [x]
        for layer in self.additional_layers.children():
            x = layer(x)
            out_features.append(x)
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

