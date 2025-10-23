import torch
import torch.nn as nn
from ultralytics import YOLO

# ResCBAM Module
class ResCBAM(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ResCBAM, self).__init__()
        # Channel Attention
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        # Spatial Attention
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel Attention
        channel_att = self.channel_attention(x)
        x = x * channel_att
        # Spatial Attention
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_input = torch.cat([max_pool, avg_pool], dim=1)
        spatial_att = self.spatial_attention(spatial_input)
        return x * spatial_att

# Modified YOLOv11 Model with ResCBAM
class YOLOv11WithResCBAM(YOLO):
    def __init__(self, model="yolo11n.pt"):
        super(YOLOv11WithResCBAM, self).__init__(model=model)
        self.modify_backbone()

    def modify_backbone(self):
        # Modify YOLOv11's backbone to include ResCBAM
        backbone = self.model.model[0]  # Access the backbone
        if isinstance(backbone, nn.Sequential):
            for i, block in enumerate(backbone):
                # Apply ResCBAM to selected layers based on their structure
                if hasattr(block, 'cv1') and hasattr(block, 'cv2'):  # YOLO-specific layers
                    in_channels = block.cv2.c2  # cv2's output channels
                    rescbam = ResCBAM(in_channels)
                    backbone[i] = nn.Sequential(block, rescbam)

    def forward(self, x):
        return super().forward(x)

# Initialize the YOLOv11 model with ResCBAM
model_rescbam = YOLOv11WithResCBAM()

# Verify the backbone structure to ensure ResCBAM integration
print(model_rescbam.model.model[0])

# Train the model on Google Colab
model_rescbam.train(
    data="/content/yolov11/Drone_Airplane_Bird-5/data.yaml",  # Path to dataset
    epochs=100,                # Number of epochs
    imgsz=640, 
    batch=16,
    optimizer="Adam", 
    lr0=0.0005,
    conf=0.25, 
    iou=0.45,
    momentum=0.9
    augment=True,
    project='runs',            # Save location
    name='yolov11_rescbam'
)
