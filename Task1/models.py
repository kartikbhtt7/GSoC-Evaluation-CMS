import torch
import torch.nn as nn
from torchvision.models.resnet import ResNet, BasicBlock

class ResNet15(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet15, self).__init__()
        # ResNet18 backbone using [2, 2, 2, 2] blocks.
        self.resnet = ResNet(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)
        
        # Override the first conv layer to accept 2 channels (hit energy and time) 
        self.resnet.conv1 = nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # Remove the maxpool layer to preserve spatial resolution for 32x32 inputs.
        self.resnet.maxpool = nn.Identity()
        
        # Remove the last block (layer4) by replacing it with an identity mapping.
        self.resnet.layer4 = nn.Identity()
        
        # Add a new convolutional block after the initial conv block.
        self.new_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.new_bn   = nn.BatchNorm2d(64)
        self.new_relu = nn.ReLU(inplace=True)
        
        # Since we removed layer4, the output from the feature extractor is 256 channels (from layer3).
        # Therefore, update the fc layer accordingly (was originally expecting 512).
        self.resnet.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Initial conv block: conv1 + bn1 + relu.
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        
        # New conv block.
        x = self.new_conv(x)
        x = self.new_bn(x)
        x = self.new_relu(x)
        
        # Continue through the remaining layers.
        x = self.resnet.maxpool(x)  # Identity.
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)   # Identity.
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.resnet.fc(x)
        return x

def resnet15v2(num_classes=2):
    """Factory function to create a ResNet15 model instance."""
    return ResNet15(num_classes=num_classes)

if __name__ == '__main__':
    # Instantiate the model (for 2 classes: electrons and photons).
    model = resnet15v2(num_classes=2)
    print(model)
    
    # Test the model with a dummy input of shape [batch, channels, height, width] = [1, 2, 32, 32]
    dummy_input = torch.randn(1, 2, 32, 32)
    output = model(dummy_input)
    print("Output shape:", output.shape)
    
    # Save the model's state dictionary for later fine-tuning.
    save_path = "resnet15v2_aligned.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model state dictionary saved to '{save_path}'.")
