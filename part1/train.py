from .models import ClassificationLoss, model_factory, save_model
from .utils import accuracy, load_data
import torch

def train(args):
    model = model_factory[args.model]()

    train_dataset = load_data("data/train", num_workers=2, batch_size=4)
    valid_dataset = load_data("data/valid", num_workers=2, batch_size=4)
    train_dataloader = train_dataset
    valid_dataloader = valid_dataset
    l = ClassificationLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.01)
    
    for epoch in range(20):
        for data in enumerate(train_dataloader):
            i, (inputs, labels) = data

            optimizer.zero_grad()

        # forward, backward, optimize
            outputs = model(inputs)
            loss = l.forward(outputs, labels) #forward, make prediction
            loss.backward() #backward, take derivative
            optimizer.step()





    #raise NotImplementedError('train')

    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model', choices=['linear', 'mlp'], default='linear')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
