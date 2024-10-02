import torch
from torch import nn


class ConvNet(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int, output_dim: int):
        super().__init__()

        self.backbone = nn.Sequential(
            self.__double_conv_block(input_channels, 16),
            self.__double_conv_block(16, 32, second_padding=2),
            self.__double_conv_block(32, 64),
        )
        self.head = nn.Sequential(
            nn.Linear(64 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.Dropout(0.5),
        )

    def __double_conv_block(
        self,
        in_channels: int,
        out_channels: int,
        first_padding: int = 1,
        second_padding: int = 1,
    ) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=first_padding,
            ),
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=second_padding,
            ),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = x.view(-1, 64 * 4 * 4)
        return self.head(x)
