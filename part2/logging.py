from os import path
import torch
import torch.utils.tensorboard as tb
import numpy as np

def test_logging(train_logger, valid_logger):

    """
    logging the loss every iteration and the accuracy only after each epoch
    epoch=0, iteration=0: global_step=0
    """

    global_step = 0
    acc_list = []
    acc_val_list = []
    #acc_list = np.array([])
    #acc_val_list = np.array([])
    # This is a strongly simplified training loop
    for epoch in range(10):
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            
            #log the training loss
            train_logger.add_scalar('loss', dummy_train_loss, global_step = global_step)
            global_step += 1

            #append the training accuracy to a list
            acc_list.append(dummy_train_accuracy)
        
        #take the average of the training accuract    
        #average = torch.mean(torch.stack(acc_list))
        #log the taining accuracy
        acc_new = [x.cpu().detach().numpy() for x in acc_list]
        train_logger.add_scalar('accuracy', np.mean(acc_new), global_step = global_step)
        
        torch.manual_seed(epoch)
        for iteration in range(10):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            
            #append the accuracy to a list
            acc_val_list.append(dummy_validation_accuracy)
        
        #take the average and log the accuracy
        averageValid = torch.mean(torch.stack(acc_val_list))
        valid_logger.add_scalar('accuracy', averageValid, global_step = global_step)
        
    
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('log_dir')
    args = parser.parse_args()
    train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
    valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'test'))
    test_logging(train_logger, valid_logger)
