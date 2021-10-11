import torch.nn 
import torch.nn.functional as F

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        
        super().__init__()
        self.convlayer1 = torch.nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, stride = 1, padding = 1)
        self.convlayer2 = torch.nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 1, padding = 1)
        self.convlayer3 = torch.nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 1, padding = 1)
        
        self.linearlayer1 = torch.nn.Linear(in_features = 8*8*128, out_features = 128)
        self.linearlayer2 = torch.nn.Linear(in_features = 128, out_features = 6)       
        self.dropout_rate = 0.5
       

    def forward(self, x):
        
        
        x = self.convlayer1(x)        
        x = F.relu(F.max_pool2d(x, 2))     
        x = self.convlayer2(x)      
        x = F.relu(F.max_pool2d(x, 2))    
        x = self.convlayer3(x)      
        x = F.relu(F.max_pool2d(x, 2))     

        x = x.view(-1, 8*8*128)  

        x = F.dropout(F.relu((self.linearlayer1(x))), 
        p=self.dropout_rate, training=self.training)    
        x = self.linearlayer2(x)                                    

        return F.log_softmax(x, dim=1)




def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), map_location='cpu'))
    return r
