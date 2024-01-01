import torch
import torch.nn as nn

class SeparableConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(SeparableConv3D, self).__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SeparableUNet3D(nn.Module):
    def __init__(self, class_num=1):
        super(SeparableUNet3D, self).__init__()

        self.conv1a = SeparableConv3D(3, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm3d(64)
        self.activation1a = nn.ReLU()
        self.conv1b = SeparableConv3D(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm3d(64)
        self.activation1b = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(2)

        self.conv2a = SeparableConv3D(64, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm3d(128)
        self.activation2a = nn.ReLU()
        self.conv2b = SeparableConv3D(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm3d(128)
        self.activation2b = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d(2)

        self.conv3a = SeparableConv3D(128, 256, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm3d(256)
        self.activation3a = nn.ReLU()
        self.conv3b = SeparableConv3D(256, 256, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm3d(256)
        self.activation3b = nn.ReLU()
        self.maxpool3 = nn.MaxPool3d(2)

        self.conv4a = SeparableConv3D(256, 512, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm3d(512)
        self.activation4a = nn.ReLU()
        self.conv4b = SeparableConv3D(512, 512, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm3d(512)
        self.activation4b = nn.ReLU()
        self.maxpool4 = nn.MaxPool3d(2)

        self.up_conv1 = nn.ConvTranspose3d(512, 512, kernel_size=2, stride=2)
        self.conv5a = SeparableConv3D(512 + 256, 256, kernel_size=3, padding=1)
        self.bn5a = nn.BatchNorm3d(256)
        self.activation5a = nn.ReLU()
        self.conv5b = SeparableConv3D(256, 256, kernel_size=3, padding=1)
        self.bn5b = nn.BatchNorm3d(256)
        self.activation5b = nn.ReLU()

        self.up_conv2 = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.conv6a = SeparableConv3D(256 + 128, 128, kernel_size=3, padding=1)
        self.bn6a = nn.BatchNorm3d(128)
        self.activation6a = nn.ReLU()
        self.conv6b = SeparableConv3D(128, 128, kernel_size=3, padding=1)
        self.bn6b = nn.BatchNorm3d(128)
        self.activation6b = nn.ReLU()

        self.up_conv3 = nn.ConvTranspose3d(128, 128, kernel_size=2, stride=2)
        self.conv7a = SeparableConv3D(128 + 64, 64, kernel_size=3, padding=1)
        self.bn7a = nn.BatchNorm3d(64)
        self.activation7a = nn.ReLU()
        self.conv7b = SeparableConv3D(64, 64, kernel_size=3, padding=1)
        self.bn7b = nn.BatchNorm3d(64)
        self.activation7b = nn.ReLU()

        self.finalconv = SeparableConv3D(64, class_num, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        self.skip_out64 = self.activation1b(self.bn1b(self.conv1b(self.activation1a(self.bn1a(self.conv1a(x))))))
        self.out64 = self.maxpool1(self.skip_out64)

        self.skip_out128 = self.activation2b(
            self.bn2b(self.conv2b(self.activation2a(self.bn2a(self.conv2a(self.out64))))))
        self.out128 = self.maxpool2(self.skip_out128)

        self.skip_out256 = self.activation3b(
            self.bn3b(self.conv3b(self.activation3a(self.bn3a(self.conv3a(self.out128))))))
        self.out256 = self.maxpool3(self.skip_out256)

        self.skip_out512 = self.activation4b(
            self.bn4b(self.conv4b(self.activation4a(self.bn4a(self.conv4a(self.out256))))))
        self.out512 = self.skip_out512

        self.out_up_conv1 = self.up_conv1(self.out512)
        self.concat1 = torch.cat((self.out_up_conv1, self.skip_out256), 1)
        self.out_up_256 = self.activation5b(
            self.bn5b(self.conv5b(self.activation5a(self.bn5a(self.conv5a(self.concat1))))))

        self.out_up_conv2 = self.up_conv2(self.out_up_256)
        self.concat2 = torch.cat((self.out_up_conv2, self.skip_out128), 1)
        self.out_up_128 = self.activation6b(
            self.bn6b(self.conv6b(self.activation6a(self.bn6a(self.conv6a(self.concat2))))))

        self.out_up_conv3 = self.up_conv3(self.out_up_128)
        self.concat3 = torch.cat((self.out_up_conv3, self.skip_out64), 1)
        self.out_up_64 = self.activation7b(
            self.bn7b(self.conv7b(self.activation7a(self.bn7a(self.conv7a(self.concat3))))))

        self.out = self.sigmoid(self.finalconv(self.out_up_64))

        return self.out


