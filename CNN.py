import torch.nn as nn

class Mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=2,out_channels=32,kernel_size=3,padding='same'),
            nn.BatchNorm1d(num_features=32),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=16*900,out_features=256),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(in_features=128, out_features=2),
        )

    def forward(self, input):
        return self.model(input)