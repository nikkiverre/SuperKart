import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        """
        m = torch.nn.LogSoftmax(dim = 1)
        loss = torch.nn.NLLLoss()
        return loss(m(input), target)
        #raise NotImplementedError('ClassificationLoss.forward')


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear = torch.nn.Linear(3*64*64, 6)
        #raise NotImplementedError('LinearClassifier.__init__')

    def forward(self, x):
        """
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """ 
        
        x = x.view(-1, 3*64*64)
        return self.linear(x)
        #raise NotImplementedError('LinearClassifier.forward')


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        #self.activation = torch.nn.ReLU()
        #self.linearlayer1 = torch.nn.Linear(3 * 64 * 64, 2048) 
        #self.linearlayer2 = torch.nn.Linear(2048, 6)
        self.layers = torch.nn.Sequential(torch.nn.Linear(3*64*64, 100), torch.nn.ReLU(), torch.nn.Linear(100, 6))
        
        #raise NotImplementedError('MLPClassifier.__init__')

    def forward(self, x):
        #@x: torch.Tensor((B,3,64,64))
        #@return: torch.Tensor((B,6))
        x = x.view(x.size(0), -1)
        #x = x.view(-1, 3*64*64)
        #return torch.nn.Sequential(self.linearlayer1(x), self.activation(x), self.linearlayer2(x))
        return self.layers(x)
        #raise NotImplementedError('MLPClassifier.forward')


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
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
