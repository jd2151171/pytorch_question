"""PyTorch implementation of the LSTM model."""

import torch
from torch.nn import Module
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torchvision import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import ToTensor
from torch.autograd import Variable
from torchvision import transforms

import datetime
import time

#We try to minimize the randomness as much as possible by setting the random seeds
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

class PyTorchLSTMMod(torch.nn.Module):
    """This class implements the LSTM model using PyTorch.

    Arguments
    ---------
    initializer: function
        The weight initialization function from the torch.nn.init module that is used to initialize
        the initial weights of the models.
    vocabulary_size: int
        The number of words that are to be considered among the words that used most frequently.
    embedding_size: int
        The number of dimensions to which the words will be mapped to.
    hidden_size: int
        The number of features of the hidden state.
    dropout: float
        The dropout rate that will be considered during training.
    """
    def __init__(self, initializer, vocabulary_size, embedding_size, hidden_size, dropout):
        super().__init__()
        
        self.embed = torch.nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=embedding_size)
        # initializer(self.embed.weight)

        self.dropout1 = torch.nn.Dropout(dropout)

        self.lstm = torch.nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, batch_first=True)
        initializer(self.lstm.weight_ih_l0)
        torch.nn.init.orthogonal_(self.lstm.weight_hh_l0)
        
        self.dropout2 = torch.nn.Dropout(dropout)
        
        self.fc = torch.nn.Linear(in_features=hidden_size, out_features=1)
        # initializer(self.fc.weight)

        


    def forward(self, inputs, is_training=False):
        """This function implements the forward pass of the model.
        
        Arguments
        ---------
        inputs: Tensor
            The set of samples the model is to infer.
        is_training: boolean
            This indicates whether the forward pass is occuring during training
            (i.e., if we should consider dropout).
        """
        x = inputs
        x = self.embed(x)
        if is_training:
            x = self.dropout1(x)

        o, (h, c) = self.lstm(x)
        out = h[-1]
        if is_training:
            out = self.dropout2(out)
        f = self.fc(out) 
        return f.flatten()#torch.sigmoid(f).flatten()

    def train_pytorch(self, optimizer, epoch, train_loader, device, data_type, log_interval):
        """This function implements a single epoch of the training process of the PyTorch model.

        Arguments
        ---------
        self: PyTorchLSTMMod
            The model that is to be trained.
        optimizer: torch.nn.optim
            The optimizer to be used during the training process.
        epoch: int
            The epoch associated with the training process.
        train_loader: DataLoader
            The DataLoader that is used to load the training data during the training process.
            Note that the DataLoader loads the data according to the batch size
            defined with it was initialized.
        device: string
            The string that indicates which device is to be used at runtime (i.e., GPU or CPU).
        data_type: string
            This string indicates whether mixed precision is to be used or not.
        log_interval: int
            The interval at which the model logs the process of the training process
            in terms of number of batches passed through the model.
        """
        self.train()

        epoch_start = time.time()
        
        loss_fn = torch.nn.BCEWithLogitsLoss()

        if data_type == 'mixed':
            scaler = torch.cuda.amp.GradScaler()

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            if data_type == 'mixed':
                with torch.cuda.amp.autocast():
                    output = self(data, is_training=True)


                    loss = loss_fn(output, target)

                scaler.scale(loss).backward()

                scaler.step(optimizer)
                scaler.update()
            else:

                output = self(data, is_training=True)

                loss = loss_fn(output, target)

                loss.backward()


                optimizer.step()

            if log_interval == -1:
                continue

            if batch_idx % log_interval == 0:
                print('Train set, Epoch {}\tLoss: {:.6f}'.format(
                    epoch, loss.item()))
        print("-PyTorch: Epoch {} done in {}s\n".format(epoch, time.time() - epoch_start))

    def test_pytorch(self, test_loader, device, data_type):
        """This function implements the testing process of the PyTorch model and returns the accuracy
        obtained on the testing dataset.

        Arguments
        ---------
        model: torch.nn.Module
            The model that is to be tested.
        test_loader: DataLoader
            The DataLoader that is used to load the testing data during the testing process.
            Note that the DataLoader loads the data according to the batch size
            defined with it was initialized.
        device: string
            The string that indicates which device is to be used at runtime (i.e., GPU or CPU).
        data_type: string
            This string indicates whether mixed precision is to be used or not.

        """
        
        
        self.eval()

        with torch.no_grad():

            #Loss and correct prediction accumulators
            test_loss = 0
            correct = 0
            total = 0

            loss_fn = torch.nn.BCEWithLogitsLoss()


            for data, target in test_loader:

                data, target = data.to(device), target.to(device)

                if data_type == 'mixed':
                    with torch.cuda.amp.autocast():

                        outputs = self(data).detach()

                        test_loss += loss_fn(outputs, target).detach()


                        preds = (outputs >= 0.5).float() == target
                        correct += preds.sum().item()
                        total += preds.size(0)

                else:
                    outputs = self(data).detach()

                    test_loss += loss_fn(outputs, target).detach()

                    preds = (outputs >= 0.5).float() == target

                    correct += preds.sum().item()
                    total += preds.size(0)

            #Print log
            test_loss /= len(test_loader.dataset)
            print('\nTest set, Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * (correct / total)))

            return 100. * (correct / total)
        
def generate_pytorch_dataloader(X_train_padded, X_test_padded, X_test_padded_ext, y_train,
                                y_test, y_test_ext, batch_size, device, review_length=500):
    """This functions generate the dataset loaders for PyTorch. The function returns the
    training dataloader (i.e., pytorch_train_loader), the testing dataloader (i.e.,
    pytorch_test_loader) and the larger testing dataset dataloader (i.e.,
    pytorch_test_loader_ext) used during the inference phase.  
    
    Arguments
    ---------
    X_train_padded: numpy array
        Padded training dataset.
    X_test_padded: numpy array
        Padded testing dataset.
    X_test_padded_ext: numpy array
        Padded larger testing dataset.
    y_train: numpy array
        Labels of the training set.
    y_test: numpy array
        Labels of the testing set.
    y_test_ext: numpy array
        Labels of the larger testing dataset.
    batch_size: int
        The batch size that will be used for training
        and testing the model.
    device: string
        The string that indicates which device is to be used at runtime (i.e., GPU or CPU).
    review_lenght: int
        The maximum lenght of the movie reviews loaded.
    """
    X_torch_train = torch.from_numpy(X_train_padded).view(X_train_padded.shape[0], review_length).to(device)
    X_torch_test = torch.from_numpy(X_test_padded).view(X_test_padded.shape[0], review_length).to(device)
    X_torch_test_ext = torch.from_numpy(X_test_padded_ext).view(X_test_padded_ext.shape[0], review_length).to(device)

    y_torch_train = torch.FloatTensor(y_train).to(device)
    y_torch_test = torch.FloatTensor(y_test).to(device)
    y_torch_test_ext = torch.FloatTensor(y_test_ext).to(device)


    pytorch_train_dataset = TensorDataset(X_torch_train, y_torch_train)
    pytorch_test_dataset = TensorDataset(X_torch_test, y_torch_test)
    pytorch_test_dataset_ext = TensorDataset(X_torch_test_ext, y_torch_test_ext)

    pytorch_train_loader = DataLoader(pytorch_train_dataset, batch_size=batch_size, shuffle=False)
    pytorch_test_loader = DataLoader(pytorch_test_dataset, batch_size=batch_size, shuffle=False)
    pytorch_test_loader_ext = DataLoader(pytorch_test_dataset_ext, batch_size=batch_size, shuffle=False)
    
    return pytorch_train_loader, pytorch_test_loader, pytorch_test_loader_ext
        
def pytorch_training_phase(model, optimizer, train_loader, test_loader, n_epochs, device, data_type, experiment):
    """"This function mplements the training phase of the PyTorch implementation of the LSTM model
    and returns the training time, the training timestamps (corresponding to when the training
    process began and when it ended) and the accuracy obtained on the testing dataset. The function
    also saves the model. 
    
    Arguments
    ---------
    model: torch.nn.Module
        The model that is to be trained.
    optimizer: torch.nn.optim
        The optimizer to be used during the training process.
    train_loader: DataLoader
        The DataLoader that is used to load the testing data during the testing process.
        Note that the DataLoader loads the data according to the batch size
        defined when the DataLoader was initialized.
    test_loader: DataLoader
        The DataLoader that is used to load the testing data during the testing process.
        Note that the DataLoader loads the data according to the batch size
        defined when the DataLoader was initialized.
    n_epochs: int
        The number of epochs for the training process.
    device: string
        The string that indicates which device is to be used at runtime (i.e., GPU or CPU).
    data_type: string
        This string indicates whether mixed precision is to be used or not.
    experiment: string
        The string that is used to identify the model (i.e., the set of configurations the model uses).
    
    """

    train_start_timestamp = datetime.datetime.now()
    start = time.time()

    for epoch in range(1, n_epochs+1):
        model.train_pytorch(optimizer, epoch, train_loader, device, data_type, log_interval=-1)
        

    training_time = time.time() - start
    train_end_timestamp = datetime.datetime.now()

    
    start = time.time()
    accuracy = model.test_pytorch(test_loader, device, data_type)
    inference_time = (time.time() - start)

    #Save the model
    #torch.save(model.state_dict(), './models/lstm/{}/model'.format(experiment))

    return training_time, inference_time, accuracy, train_start_timestamp, train_end_timestamp

def pytorch_inference_phase(model, experiment, pytorch_test_loader_ext, device, data_type):
    """This function implements the inference phase of the PyTorch implementation of the LSTM model.
    The function returns the inference timestamps (corresponding to when the inference began and when
    it ended). 
    
    Arguments
    ---------
    model: torch.nn.Module
        The model that is to be evaluated (the model acts as a placeholder into which the weights of
        the trained model will be loaded).
    experiment: string
        The string that is used to identify the model (i.e., the set of configurations the model uses).
    pytorch_test_loader_ext: DataLoader
        The DataLoader that is used to load the larger testing data during the inference phase.
        Note that the DataLoader loads the data according to the batch size
        defined when the DataLoader was initialized.
    device: string
        The string that indicates which device is to be used at runtime (i.e., GPU or CPU).
    data_type: string
        This string indicates whether mixed precision is to be used or not.
    """
    
    #Load the weigths of the trained model.
    model.load_state_dict(torch.load('./models/lstm/{}/model'.format(experiment)))
    model.eval()

    inference_start_timestamp = datetime.datetime.now()
    accuracy = model.test_pytorch(pytorch_test_loader_ext, device, data_type)
    inference_end_timestamp = datetime.datetime.now()
    print('Accuracy: {}'.format(accuracy))
    
    return inference_start_timestamp, inference_end_timestamp