from .models import CNNClassifier, save_model
from .utils import accuracy, load_data
import torch
import torch.utils.tensorboard as tb


def train(args):
    from os import path
    model = CNNClassifier()
    train_logger, valid_logger = None, None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
    
    train_dataset = load_data("data/train", num_workers=2, batch_size=4)
    valid_dataset = load_data("data/valid", num_workers=2, batch_size=4)
    train_dataloader = train_dataset
    valid_dataloader = valid_dataset
    l = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.01)
    for epoch in range(20):
        for data in enumerate(train_dataloader):
            i, (inputs, labels) = data

            optimizer.zero_grad()

        # forward, backward, optimize
            outputs = model(inputs)
            #loss = l(outputs, labels)
            loss = l.forward(outputs, labels) #forward, make prediction
            loss.backward() #backward, take derivative
            optimizer.step()





    save_model(model)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)
