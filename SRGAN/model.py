import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    # Conv -> BN -> PReLU
    def __init__(
        self,
        in_channels,
        out_channels,
        disc=False,
        use_activ=False,
        use_bn=True,
        **kwargs
    ) -> None:
        super().__init__()

        self.use_activ = use_activ
        self.use_bn = use_bn

        self.cnn = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            **kwargs,
            bias=not use_bn,
        )
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.activ = (
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            if disc
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return (
            self.activ(self.bn(self.cnn(x))) if self.use_activ else self.bn(self.cnn(x))
        )


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scaling_factor) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            in_channels * scaling_factor**2,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.ps = nn.PixelShuffle(scaling_factor)
        self.activ = nn.PReLU(num_parameters=in_channels)

    def forward(self, x):
        return self.activ(self.ps(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()
        self.b1 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.b2 = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            use_activ=False,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        return x + self.b2(self.b1(x))


class Generator(nn.Module):
    def __init__(
        self, in_channels=3, num_channels=64, num_blocks=16, scaling_factor=2
    ) -> None:
        super().__init__()
        self.initial_block = ConvBlock(
            in_channels=in_channels,
            out_channels=num_channels,
            use_bn=False,
            kernel_size=9,
            stride=1,
            padding=4,
        )
        self.residual_block = nn.Sequential(
            *[ResidualBlock(num_channels) for _ in range(num_blocks)]
        )
        self.conv_block = ConvBlock(
            num_channels,
            num_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_activ=False,
        )
        self.upsample_block = UpsampleBlock(
            in_channels=in_channels, scaling_factor=scaling_factor
        )
        self.final_block = nn.Conv2d(
            num_channels, in_channels, kernel_size=9, stride=1, padding=4
        )

    def forward(self, x):
        init = self.initial_block(x)
        x = self.residual_block(x)
        x = self.conv_block(x) + init
        x = self.upsample_block(x)
        return torch.tanh(self.final_block(x))


class Discriminator(nn.Module):
    def __init__(
        self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]
    ) -> None:
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    use_activ=True,
                    use_bn=idx != 0,
                    disc=True,
                )
            )
            in_channels = feature
        self.blocks = nn.Sequential(*blocks)
        self.classifier_block = nn.Sequential(
            nn.AdaptiveMaxPool2d(
                (6, 6),
            ),
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.classifier_block(self.blocks(x))
