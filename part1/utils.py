from PIL import Image
import csv
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']


class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        
        
        self.data_path = dataset_path
        with open(dataset_path + '/labels.csv', "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter = ',')
            #self.supertuxdata = list(csv.reader(csv_file, delimiter = ','))
            next(csv_reader)
            self.supertuxdata = list(csv_reader)
        


        #raise NotImplementedError('SuperTuxDataset.__init__')

    def __len__(self):
        """
        Returns the size of the dataset
        """
        return len(self.supertuxdata)

        #raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
        """
        returns a tuple: img, label
        """

        I = Image.open(self.data_path + '/' + str(self.supertuxdata[idx][0]))
        image_to_tensor = transforms.ToTensor()
        image_tensor = image_to_tensor(I)
        return (image_tensor, LABEL_NAMES.index(self.supertuxdata[idx][1]))
        ###
            #for each in self.supertuxdata:
            #I = Image.open(each[0])
            #image_to_tensor = transforms.ToTensor(I)
            #image_tensor = image_to_tensor(I)
            #print('image_tensor.shape', image_tensor.shape)
            #return (image_tensor, LABEL_NAMES.index(each[1]))
        ###
        #raise NotImplementedError('SuperTuxDataset.__getitem__')


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=False)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
