import torch
import torch.nn.functional as F


class CNNClassifier(torch.nn.Module):
    def __init__(self):
        
        super().__init__()
        self.convlayer1 = torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.convlayer2 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.convlayer3 = torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        self.bn3 = torch.nn.BatchNorm2d(128)

        self.linearlayer1 = torch.nn.Linear(in_features = 8*8*128, out_features = 128)
        self.linearbn1 = torch.nn.BatchNorm1d(128)
        self.linearlayer2 = torch.nn.Linear(in_features = 128, out_features = 6)       
        self.dropout_rate = 0.5

    def forward(self, x):
        """
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        applying input normalization inside the network
        """
        x = self.bn1(self.convlayer1(x))    
        x = F.relu(F.max_pool2d(x, 2))     
        x = self.bn2(self.convlayer2(x))     
        x = F.relu(F.max_pool2d(x, 2))    
        x = self.bn3(self.convlayer3(x))   
        x = F.relu(F.max_pool2d(x, 2))     

        x = x.view(-1, 8*8*128)  

        x = F.dropout(F.relu(self.linearbn1(self.linearlayer1(x))), 
        p=self.dropout_rate, training=self.training)    
        x = self.linearlayer2(x)                                    

        return F.log_softmax(x, dim=1)



class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Using up-convolutions, skip connections, residual connections, padding by kernel_size / 2
        
        
        """
        self.conv1 = torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.relu = torch.nn.ReLU(inplace=True)
        self.maxpool = torch.nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size = 3, stride = 2, padding = 1, bias = False)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.fc = torch.nn.Linear(64, 1)
        
        #self.downsample = torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 1, stride =2)
        self.downsample = torch.nn.Sequential(torch.nn.Conv2d(3, 5, 1),torch.nn.BatchNorm2d(5))
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x):
        """
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,6,H,W))
       
        """
        residual = x

        out = self.conv1(x) # 3 X 32 X 96/2 X 128/2
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual 
        #out = self.relu(out)
        
       
        return out 

model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
