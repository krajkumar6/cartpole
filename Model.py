import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class DQNNet(nn.Module):
    '''
    class that defines the architecture of the neural network
    that is used for learning Q function
    '''
    def __init__(self,input_size,output_size,lr=1e-3):
        super(DQNNet, self).__init__()
        self.dense1 = nn.Linear(input_size,400)
        self.dense2 = nn.Linear(400,300)
        self.dense3 = nn.Linear(300,output_size)
        
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        
    def forward(self,x):
        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)
        return x
    
    def save_model(self, filename):
        """
        Function to save model parameters
        Parameters
        ---
        filename: str
            Location of the file where the model is to be saved
        Returns
        ---
        none
        """
        torch.save(self.state_dict(), filename)
        
    def load_model(self, filename, device):
        """
        Function to load model parameters
        Parameters
        ---
        filename: str
            Location of the file from where the model is to be loaded
        device:
            Device in use - CPU or GPU
        Returns
        ---
        none
        """
        # map_location is required to ensure that a model that is trained on GPU can be run even on CPU
        self.load_state_dict(torch.load(filename, map_location=device))