import torch
import torch.nn as nn

class StudentModel(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 32, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 128, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

class StudentModel_Improved(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModel_Improved, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 16, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
    
class BasicModel(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 32, kernel_size= 3, stride= 1, padding= 1, bias = False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size= 3, stride= stride, padding= 1, bias = False)
        self.bn3x3 = nn.BatchNorm2d(out_channels) #??

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size= 1, stride=stride, padding = 0, bias= False)
        self.bn1x1 = nn.BatchNorm2d(out_channels)

        if stride == 1 and in_channels == out_channels:
            self.bn_indentity = nn.BatchNorm2d(out_channels)
        else:
            self.bn_indentity = None

        self.relu = nn.ReLU(inplace= True) #??

    def forward(self, x):
        out3x3 = self.bn3x3(self.conv3x3(x))
        out1x1 = self.bn1x1(self.conv1x1(x))

        if self.bn_indentity is not None:
            out_identity = self.bn_indentity(x)
            out = out3x3 + out1x1 + out_identity

        else:
            out = out3x3 + out1x1

        return self.relu(out)

class TeacherModel(nn.Module):
    def __init__(self, num_classes=10):
        super(TeacherModel, self).__init__()

        self.conv_layers = nn.Sequential(
            BasicBlock(in_channels=1, out_channels=8, stride=1),
            BasicBlock(in_channels=8, out_channels=16, stride=2),
            BasicBlock(in_channels=16, out_channels=16, stride=1),
            BasicBlock(in_channels=16, out_channels=16, stride=1),
            BasicBlock(in_channels=16, out_channels=32, stride=2),
            BasicBlock(in_channels=32, out_channels=32, stride=1),
            BasicBlock(in_channels=32, out_channels=32, stride=1),
            BasicBlock(in_channels=32, out_channels=320, stride=2)
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(320, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

class TFAM1DCNN_simplified(nn.Module):
    def __init__(self, num_classes=3, dropout_rate=0.5):
        super(TFAM1DCNN_simplified, self).__init__()

        # --- Khối 1 ---
        self.block1 = nn.Sequential(
            # Lớp 1: Convolution
            # in_channels=1, out_channels=16, kernel_size=64, stride=16, padding=24
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=64, stride=16, padding=24),
            # Lớp 2: BatchNorm
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            # Lớp 3: Dropout
            nn.Dropout(p=dropout_rate),
            # Lớp 4: Pooling
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Output shape: (batch, 16, 64)

        # --- Khối 2 ---
        self.block2 = nn.Sequential(
            # Lớp 5: Convolution
            # in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            # Lớp 6: BatchNorm
            nn.BatchNorm1d(num_features=32),
            nn.ReLU()
            # Lớp 7: TFAM (Bỏ qua)
        )
        # Output shape: (batch, 32, 64)

        # --- Khối 3 ---
        self.block3 = nn.Sequential(
            # Lớp 8: Convolution
            # in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            # Lớp 9: BatchNorm
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            # Lớp 10: Pooling (sửa stride=2)
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Output shape: (batch, 64, 32)

        # --- Khối 4 ---
        self.block4 = nn.Sequential(
            # Lớp 11: Convolution
            # in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # Lớp 12: BatchNorm
            nn.BatchNorm1d(num_features=64),
            nn.ReLU()
            # Lớp 13: TFAM (Bỏ qua)
        )
        # Output shape: (batch, 64, 32)

        # --- Khối Residual ---
        self.residual_block = nn.Sequential(
            # Lớp 14: Convolution
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # Lớp 15: BatchNorm
            nn.BatchNorm1d(num_features=64),
            nn.ReLU(),
            # Lớp 16: TFAM (Bỏ qua)
            # Lớp 17: Convolution
            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(num_features=64) # BN trước khi cộng
        )
        self.residual_activation = nn.ReLU()
        # Output shape: (batch, 64, 32)

        # --- Khối cuối ---
        self.final_block = nn.Sequential(
            # Lớp 19: Pooling
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Lớp 20: Global Average Pooling
            nn.AdaptiveAvgPool1d(1)
        )
        # Output shape: (batch, 64, 1)

        # --- Lớp phân loại ---
        # Lớp 21: Dense (Linear)
        self.classifier = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        # x shape: (batch, 1, 2048)
        x = self.block1(x)
        # x shape: (batch, 16, 64)
        x = self.block2(x)
        # x shape: (batch, 32, 64)
        x = self.block3(x)
        # x shape: (batch, 64, 32)
        x = self.block4(x)
        # x shape: (batch, 64, 32)

        # Lưu lại đầu vào cho kết nối tắt
        residual_input = x
        
        # Đi qua khối residual
        x_res = self.residual_block(x)
        
        # Lớp 18: Add
        x = x_res + residual_input
        x = self.residual_activation(x)
        # x shape: (batch, 64, 32)

        # Đi qua khối cuối
        x = self.final_block(x)
        # x shape: (batch, 64, 1)

        # Flatten trước khi đưa vào lớp Linear
        # x.view(x.size(0), -1) tương đương với flatten
        x = x.view(x.size(0), -1)
        # x shape: (batch, 64)

        # Lớp phân loại
        output = self.classifier(x)
        # output shape: (batch, num_classes)
        
        return output

class ChannelAttention(nn.Module):
    """Channel Attention Module (CAM) cho dữ liệu 1D"""
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        attention_weights = self.sigmoid(avg_out + max_out)
        return attention_weights

class SpatialAttention(nn.Module):
    """Spatial/Frequency Attention Module (FAM) cho dữ liệu 1D"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # Kernel size phải là số lẻ để padding hoạt động đúng
        assert kernel_size in (3, 5, 7), 'kernel size must be 3, 5, or 7'
        padding = (kernel_size - 1) // 2
        
        self.conv1d = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        attention_weights = self.sigmoid(self.conv1d(x_cat))
        return attention_weights

class TFAM(nn.Module):
    """Time-Frequency Attention Mechanism (TFAM) Module"""
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(TFAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        # Áp dụng Channel Attention
        ca_weights = self.ca(x)
        x_ca = x * ca_weights  # M_CAM
        
        # Kết nối tắt trong CAM theo công thức (3) của bài báo
        x_ca_res = x_ca + x    # Y_CAM
        
        # Áp dụng Spatial/Frequency Attention
        sa_weights = self.sa(x_ca_res)
        x_out = x_ca_res * sa_weights
        
        return x_out

class TFAM1DCNN(nn.Module):
    """
    Triển khai đầy đủ mô hình TFAM1DCNN từ bài báo "EdgeCog".
    Kiến trúc dựa trên Bảng I và mô tả module TFAM trong Mục III-B.
    """
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(TFAM1DCNN, self).__init__()

        # --- Khối 1: Lớp 1-4 ---
        self.block1 = nn.Sequential(
            nn.Conv1d(1, 16, 64, 16, 24), # Lớp 1
            nn.BatchNorm1d(16),           # Lớp 2
            nn.ReLU(),
            nn.Dropout(dropout_rate),     # Lớp 3
            nn.MaxPool1d(2, 2)            # Lớp 4
        )

        # --- Khối 2: Lớp 5-7 ---
        self.block2 = nn.Sequential(
            nn.Conv1d(16, 32, 3, 1, 1),   # Lớp 5
            nn.BatchNorm1d(32),           # Lớp 6
            nn.ReLU()
        )
        self.tfam1 = TFAM(in_channels=32) # Lớp 7

        # --- Khối 3: Lớp 8-10 ---
        self.block3 = nn.Sequential(
            nn.Conv1d(32, 64, 3, 1, 1),   # Lớp 8
            nn.BatchNorm1d(64),           # Lớp 9
            nn.ReLU(),
            nn.MaxPool1d(2, 2)            # Lớp 10 (sửa stride=2)
        )

        # --- Khối 4: Lớp 11-13 ---
        self.block4 = nn.Sequential(
            nn.Conv1d(64, 64, 3, 1, 1),   # Lớp 11
            nn.BatchNorm1d(64),           # Lớp 12
            nn.ReLU()
        )
        self.tfam2 = TFAM(in_channels=64) # Lớp 13

        # --- Khối Residual: Lớp 14-18 ---
        self.residual_block = nn.Sequential(
            nn.Conv1d(64, 64, 3, 1, 1),   # Lớp 14
            nn.BatchNorm1d(64),           # Lớp 15
            nn.ReLU()
        )
        self.tfam3 = TFAM(in_channels=64) # Lớp 16
        self.conv_after_tfam3 = nn.Sequential(
            nn.Conv1d(64, 64, 3, 1, 1),   # Lớp 17
            nn.BatchNorm1d(64)
        )
        self.residual_activation = nn.ReLU()

        # --- Khối cuối: Lớp 19-20 ---
        self.final_block = nn.Sequential(
            nn.MaxPool1d(2, 2),           # Lớp 19
            nn.AdaptiveAvgPool1d(1)       # Lớp 20
        )

        # --- Lớp phân loại: Lớp 21 ---
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        # Input: (N, 1, 2048)
        x = self.block1(x)          # -> (N, 16, 64)
        
        x = self.block2(x)          # -> (N, 32, 64)
        x = self.tfam1(x)           # Lớp 7
        
        x = self.block3(x)          # -> (N, 64, 32)
        
        identity = self.block4(x)   # -> (N, 64, 32)
        identity = self.tfam2(identity) # Lớp 13
        
        # Khối residual
        res_out = self.residual_block(identity) # Lớp 14, 15
        res_out = self.tfam3(res_out)           # Lớp 16
        res_out = self.conv_after_tfam3(res_out)# Lớp 17
        
        x = res_out + identity      # Lớp 18: Add
        x = self.residual_activation(x)
        
        x = self.final_block(x)     # -> (N, 64, 1)
        x = x.view(x.size(0), -1)   # -> (N, 64)
        output = self.classifier(x) # -> (N, num_classes)
        
        return output

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        reduced_channels = max(1, in_channels // reduction_ratio) # Đảm bảo không bằng 0
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv1d(in_channels, reduced_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv1d(reduced_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv1d = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1d(x_cat))

class TFAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(TFAM, self).__init__()
        self.ca = ChannelAttention(in_channels, reduction_ratio)
        self.sa = SpatialAttention(spatial_kernel_size)
    def forward(self, x):
        x_ca = x * self.ca(x)
        x_ca_res = x_ca + x
        x_out = x_ca_res * self.sa(x_ca_res)
        return x_out

class TFAM1DCNN_Light(nn.Module):
    def __init__(self, num_classes=10, dropout_rate=0.5):
        super(TFAM1DCNN_Light, self).__init__()

        # Giảm số kênh: 16->8, 32->16, 64->32
        # Giảm kernel lớp đầu: 64->32
        
        # --- Khối 1 ---
        self.block1 = nn.Sequential(
            # Kernel giảm từ 64 -> 32, channels giảm từ 16 -> 8
            nn.Conv1d(1, 8, 32, 16, 8), # Lớp 1
            nn.BatchNorm1d(8),          # Lớp 2
            nn.ReLU(),
            nn.Dropout(dropout_rate),   # Lớp 3
            nn.MaxPool1d(2, 2)          # Lớp 4
        )

        # --- Khối 2 ---
        self.block2 = nn.Sequential(
            nn.Conv1d(8, 16, 3, 1, 1),  # Lớp 5 (channels 16->16)
            nn.BatchNorm1d(16),         # Lớp 6
            nn.ReLU()
        )
        self.tfam1 = TFAM(in_channels=16) # Lớp 7

        # --- Khối 3 ---
        self.block3 = nn.Sequential(
            nn.Conv1d(16, 32, 3, 1, 1), # Lớp 8 (channels 32->32)
            nn.BatchNorm1d(32),         # Lớp 9
            nn.ReLU(),
            nn.MaxPool1d(2, 2)          # Lớp 10
        )

        # --- Khối 4 ---
        self.block4 = nn.Sequential(
            nn.Conv1d(32, 32, 3, 1, 1), # Lớp 11 (channels 64->32)
            nn.BatchNorm1d(32),         # Lớp 12
            nn.ReLU()
        )
        self.tfam2 = TFAM(in_channels=32) # Lớp 13

        # --- Khối Residual ---
        self.residual_block = nn.Sequential(
            nn.Conv1d(32, 32, 3, 1, 1), # Lớp 14
            nn.BatchNorm1d(32),         # Lớp 15
            nn.ReLU()
        )
        self.tfam3 = TFAM(in_channels=32) # Lớp 16
        self.conv_after_tfam3 = nn.Sequential(
            nn.Conv1d(32, 32, 3, 1, 1), # Lớp 17
            nn.BatchNorm1d(32)
        )
        self.residual_activation = nn.ReLU()

        # --- Khối cuối ---
        self.final_block = nn.Sequential(
            nn.MaxPool1d(2, 2),         # Lớp 19
            nn.AdaptiveAvgPool1d(1)     # Lớp 20
        )

        # --- Lớp phân loại ---
        self.classifier = nn.Linear(32, num_classes) # in_features giảm từ 64->32

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.tfam1(x)
        x = self.block3(x)
        identity = self.block4(x)
        identity = self.tfam2(identity)
        res_out = self.residual_block(identity)
        res_out = self.tfam3(res_out)
        res_out = self.conv_after_tfam3(res_out)
        x = res_out + identity
        x = self.residual_activation(x)
        x = self.final_block(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output
    
class Basic1DCNN(nn.Module):
    def __init__(self,
                 input_channels: int,
                 num_classes: int,
                 dropout_rate: float = 0.3):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=input_channels,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(output_size=1)   # Global-Max-Pooling
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                      # 128 × 1  → 128
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        x shape: (batch_size, input_channels, timesteps)
        """
        x = self.features(x)
        x = self.classifier(x)
        return x
