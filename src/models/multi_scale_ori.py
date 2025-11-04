"""
Multi-Scale 1D ResNet for Time-Series Classification

This module implements a multi-scale residual neural network architecture for 1D time-series data.
The model processes input signals through three parallel paths using different kernel sizes (3x3, 5x5, 7x7)
to capture features at multiple temporal scales.

Adapted from: https://github.com/geekfeiw/Multi-Scale-1D-ResNet/blob/master/model/multi_scale_ori.py
"""

import torch.nn as nn
import torch


def conv3x3(in_planes, out_planes, stride=1):
    """
    3x3 convolution with padding.

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int, optional): Stride for convolution. Defaults to 1.

    Returns:
        nn.Conv1d: 1D convolutional layer
    """
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv5x5(in_planes, out_planes, stride=1):
    """
    5x5 convolution with padding.

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int, optional): Stride for convolution. Defaults to 1.

    Returns:
        nn.Conv1d: 1D convolutional layer
    """
    return nn.Conv1d(in_planes, out_planes, kernel_size=5, stride=stride, padding=1, bias=False)


def conv7x7(in_planes, out_planes, stride=1):
    """
    7x7 convolution with padding.

    Args:
        in_planes (int): Number of input channels
        out_planes (int): Number of output channels
        stride (int, optional): Stride for convolution. Defaults to 1.

    Returns:
        nn.Conv1d: 1D convolutional layer
    """
    return nn.Conv1d(in_planes, out_planes, kernel_size=7, stride=stride, padding=1, bias=False)


class BasicBlock3x3(nn.Module):
    """
    Basic residual block with 3x3 convolutions.

    This block consists of two 3x3 convolutional layers with batch normalization
    and ReLU activation, followed by a residual connection.

    Attributes:
        expansion (int): Expansion factor for the number of output channels
        conv1 (nn.Conv1d): First convolutional layer
        bn1 (nn.BatchNorm1d): Batch normalization after first convolution
        conv2 (nn.Conv1d): Second convolutional layer
        bn2 (nn.BatchNorm1d): Batch normalization after second convolution
        downsample (nn.Module or None): Downsampling layer for residual connection
        relu (nn.ReLU): ReLU activation function
    """

    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        """
        Initialize BasicBlock3x3.

        Args:
            inplanes3 (int): Number of input channels
            planes (int): Number of output channels
            stride (int, optional): Stride for convolutions. Defaults to 1.
            downsample (nn.Module, optional): Downsampling layer for residual. Defaults to None.
        """
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length)

        Returns:
            torch.Tensor: Output tensor after residual addition and activation
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock5x5(nn.Module):
    """
    Basic residual block with 5x5 convolutions.

    This block consists of two 5x5 convolutional layers with batch normalization
    and ReLU activation, followed by a residual connection. Includes dimension
    alignment for the residual connection.

    Attributes:
        expansion (int): Expansion factor for the number of output channels
        conv1 (nn.Conv1d): First convolutional layer
        bn1 (nn.BatchNorm1d): Batch normalization after first convolution
        conv2 (nn.Conv1d): Second convolutional layer
        bn2 (nn.BatchNorm1d): Batch normalization after second convolution
        downsample (nn.Module or None): Downsampling layer for residual connection
        relu (nn.ReLU): ReLU activation function
    """

    expansion = 1

    def __init__(self, inplanes5, planes, stride=1, downsample=None):
        """
        Initialize BasicBlock5x5.

        Args:
            inplanes5 (int): Number of input channels
            planes (int): Number of output channels
            stride (int, optional): Stride for convolutions. Defaults to 1.
            downsample (nn.Module, optional): Downsampling layer for residual. Defaults to None.
        """
        super(BasicBlock5x5, self).__init__()
        self.conv1 = conv5x5(inplanes5, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv5x5(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass through the residual block with dimension alignment.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length)

        Returns:
            torch.Tensor: Output tensor after residual addition and activation
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # Align dimensions by trimming residual to match output length
        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)

        return out1


class BasicBlock7x7(nn.Module):
    """
    Basic residual block with 7x7 convolutions.

    This block consists of two 7x7 convolutional layers with batch normalization
    and ReLU activation, followed by a residual connection. Includes dimension
    alignment for the residual connection.

    Attributes:
        expansion (int): Expansion factor for the number of output channels
        conv1 (nn.Conv1d): First convolutional layer
        bn1 (nn.BatchNorm1d): Batch normalization after first convolution
        conv2 (nn.Conv1d): Second convolutional layer
        bn2 (nn.BatchNorm1d): Batch normalization after second convolution
        downsample (nn.Module or None): Downsampling layer for residual connection
        relu (nn.ReLU): ReLU activation function
    """

    expansion = 1

    def __init__(self, inplanes7, planes, stride=1, downsample=None):
        """
        Initialize BasicBlock7x7.

        Args:
            inplanes7 (int): Number of input channels
            planes (int): Number of output channels
            stride (int, optional): Stride for convolutions. Defaults to 1.
            downsample (nn.Module, optional): Downsampling layer for residual. Defaults to None.
        """
        super(BasicBlock7x7, self).__init__()
        self.conv1 = conv7x7(inplanes7, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv7x7(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        """
        Forward pass through the residual block with dimension alignment.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, length)

        Returns:
            torch.Tensor: Output tensor after residual addition and activation
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        # Align dimensions by trimming residual to match output length
        d = residual.shape[2] - out.shape[2]
        out1 = residual[:, :, 0:-d] + out
        out1 = self.relu(out1)

        return out1


class MSResNet(nn.Module):
    """
    Multi-Scale Residual Network for 1D time-series data.

    This architecture processes input signals through three parallel ResNet paths with
    different kernel sizes (3x3, 5x5, 7x7) to capture features at multiple temporal scales.
    The outputs from all three paths are concatenated and passed through a fully connected
    layer for classification.

    Architecture:
        - Initial convolution (7x7) + batch norm + ReLU + max pooling
        - Three parallel paths:
            * Path 1: ResNet blocks with 3x3 kernels
            * Path 2: ResNet blocks with 5x5 kernels
            * Path 3: ResNet blocks with 7x7 kernels
        - Each path: 3 residual layers with channels [64, 128, 256]
        - Average pooling for each path
        - Concatenation of all paths
        - Dropout + fully connected layer

    Attributes:
        conv1 (nn.Conv1d): Initial convolution layer
        bn1 (nn.BatchNorm1d): Batch normalization after initial convolution
        maxpool (nn.MaxPool1d): Max pooling layer
        layer3x3_1-3 (nn.Sequential): ResNet layers for 3x3 path
        layer5x5_1-3 (nn.Sequential): ResNet layers for 5x5 path
        layer7x7_1-3 (nn.Sequential): ResNet layers for 7x7 path
        avgpool3, avgpool5, avgpool7 (nn.AvgPool1d): Average pooling for each path
        drop (nn.Dropout): Dropout layer
        fc (nn.Linear): Final fully connected layer
    """

    def __init__(self, input_channel=1, layers=[1, 1, 1], num_classes=10, dropout_rate=0.2):
        """
        Initialize Multi-Scale ResNet.

        Args:
            input_channel (int): Number of input channels. Defaults to 1.
            layers (list, optional): Number of residual blocks in each layer. Defaults to [1, 1, 1].
            num_classes (int, optional): Number of output classes. Defaults to 10.
            dropout_rate (float, optional): Dropout probability. Defaults to 0.2.
        """
        self.inplanes3 = 64
        self.inplanes5 = 64
        self.inplanes7 = 64

        super(MSResNet, self).__init__()

        # Initial convolution and pooling
        self.conv1 = nn.Conv1d(input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 3x3 kernel path
        self.layer3x3_1 = self._make_layer3(BasicBlock3x3, 64, layers[0], stride=2)
        self.layer3x3_2 = self._make_layer3(BasicBlock3x3, 128, layers[1], stride=2)
        self.layer3x3_3 = self._make_layer3(BasicBlock3x3, 256, layers[2], stride=2)
        self.avgpool3 = nn.AvgPool1d(kernel_size=16, stride=1, padding=0)

        # 5x5 kernel path
        self.layer5x5_1 = self._make_layer5(BasicBlock5x5, 64, layers[0], stride=2)
        self.layer5x5_2 = self._make_layer5(BasicBlock5x5, 128, layers[1], stride=2)
        self.layer5x5_3 = self._make_layer5(BasicBlock5x5, 256, layers[2], stride=2)
        self.avgpool5 = nn.AvgPool1d(kernel_size=10, stride=1, padding=0)

        # 7x7 kernel path
        self.layer7x7_1 = self._make_layer7(BasicBlock7x7, 64, layers[0], stride=2)
        self.layer7x7_2 = self._make_layer7(BasicBlock7x7, 128, layers[1], stride=2)
        self.layer7x7_3 = self._make_layer7(BasicBlock7x7, 256, layers[2], stride=2)
        self.avgpool7 = nn.AvgPool1d(kernel_size=5, stride=1, padding=0)

        # Final layers
        self.drop = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(256 * 3, num_classes)

    def _make_layer3(self, block, planes, blocks, stride=2):
        """
        Create a sequential layer of 3x3 residual blocks.

        Args:
            block (nn.Module): Residual block class to use
            planes (int): Number of output channels
            blocks (int): Number of residual blocks
            stride (int, optional): Stride for first block. Defaults to 2.

        Returns:
            nn.Sequential: Sequential container of residual blocks
        """
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes3,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def _make_layer5(self, block, planes, blocks, stride=2):
        """
        Create a sequential layer of 5x5 residual blocks.

        Args:
            block (nn.Module): Residual block class to use
            planes (int): Number of output channels
            blocks (int): Number of residual blocks
            stride (int, optional): Stride for first block. Defaults to 2.

        Returns:
            nn.Sequential: Sequential container of residual blocks
        """
        downsample = None
        if stride != 1 or self.inplanes5 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes5,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes5, planes, stride, downsample))
        self.inplanes5 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes5, planes))

        return nn.Sequential(*layers)

    def _make_layer7(self, block, planes, blocks, stride=2):
        """
        Create a sequential layer of 7x7 residual blocks.

        Args:
            block (nn.Module): Residual block class to use
            planes (int): Number of output channels
            blocks (int): Number of residual blocks
            stride (int, optional): Stride for first block. Defaults to 2.

        Returns:
            nn.Sequential: Sequential container of residual blocks
        """
        downsample = None
        if stride != 1 or self.inplanes7 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.inplanes7,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes7, planes, stride, downsample))
        self.inplanes7 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes7, planes))

        return nn.Sequential(*layers)

    def forward(self, x0):
        """
        Forward pass through the multi-scale network.

        Args:
            x0 (torch.Tensor): Input tensor of shape (batch_size, input_channel, length)

        Returns:
            torch.Tensor: Feature embeddings of shape (batch_size, 768)
                         Ready for classification or further processing

        Note:
            The final fully connected layer (self.fc) is not applied in this forward pass.
            The output is the concatenated features from all three scales after dropout.
        """
        # Initial processing
        x0 = self.conv1(x0)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)

        # 3x3 path
        x = self.layer3x3_1(x0)
        x = self.layer3x3_2(x)
        x = self.layer3x3_3(x)
        x = self.avgpool3(x)

        # 5x5 path
        y = self.layer5x5_1(x0)
        y = self.layer5x5_2(y)
        y = self.layer5x5_3(y)
        y = self.avgpool5(y)

        # 7x7 path
        z = self.layer7x7_1(x0)
        z = self.layer7x7_2(z)
        z = self.layer7x7_3(z)
        z = self.avgpool7(z)

        # Concatenate all paths
        out = torch.cat([x, y, z], dim=1)

        out = out[:, :, 0]
        out = self.drop(out)
        out = self.fc(out)

        return out
