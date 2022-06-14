# imports
import torch
import torch.nn as nn
import torch.optim as optim  # optimization algorithms
import torch.nn.functional as F  # activation functions
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# create a simple fully connected neural network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)  # in_features are equal to input size and out features to 50
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):  # input x
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


# create a CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        # formula to find size of output
        # n_out = [(n_in + 2p - k)/s] + 1, [] is floor , in our case [28+2-3] + 1 = 28
        # n_out : number of output features
        # n_in : number of input features
        # p : padding size
        # k : kernel size
        # s : stride size
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 28 becomes 14
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1))
        # we apply pooling twice, so now 14 becomes 7, so we have 16*7*7 values after flattening as we are taking 16
        # channels
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)  # Flatten x to pass to fully connected later
        x = self.fc1(x)
        return x


# model = CNN()
# x = torch.randn(64, 1, 28, 28)
# print(model(x).shape)
# exit()

# model = NN(784,10)
# x = torch.rand((64,784))
# print(model(x).shape) # should be 64x10, and it is

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper-parameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# load data
train_dataset = datasets.MNIST(root="./dataset/", train=True, transform=transforms.ToTensor(), download=True)
trainLoader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root="./dataset/", train=False, transform=transforms.ToTensor(), download=True)
testLoader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)

# initialize network
model = CNN().to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network
for epoch in range(num_epochs):  # one epoch means the network has seen all images in dataset
    for batch_index, (data, targets) in enumerate(trainLoader):
        data = data.to(device)
        targets = targets.to(device)
        # print(data.shape) # This gives torch.Size([64, 1, 28, 28]), number of images, channels,height,width

        # Commenting this because we already flattened it in forward function
        # # Get to correct shape
        # data = data.reshape(data.shape[0], -1)  # Flatten all images in batch

        # forward
        scores = model.forward(data)
        loss = criterion(scores, targets)

        # back-propagation
        optimizer.zero_grad()
        loss.backward()

        # gradient descent step
        optimizer.step()


# check accuracy on training and test
def check_accuracy(dataloader, model):
    if dataloader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            # x = x.reshape(x.shape[0], -1)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct) / float(num_samples)}")
    model.train()


check_accuracy(trainLoader, model)
check_accuracy(testLoader, model)


# with epochs=5
# Checking accuracy on training data
# Got 59202/60000 with accuracy 0.9867
# Checking accuracy on test data
# Got 9858/10000 with accuracy 0.9858
