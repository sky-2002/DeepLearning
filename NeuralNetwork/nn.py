# imports
import torch
import torch.nn as nn
import torch.optim as optim  # optimization algorithms
import torch.nn.functional as F  # activation functions
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# create a fully connected neural network
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


# model = NN(784,10)
# x = torch.rand((64,784))
# print(model(x).shape) # should be 64x10, and it is

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# hyper-parameters
input_size = 784
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
model = NN(input_size=input_size, num_classes=num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train network
for epoch in range(num_epochs):  # one epoch means the network has seen all images in dataset
    for batch_index, (data, targets) in enumerate(trainLoader):
        data = data.to(device)
        targets = targets.to(device)
        # print(data.shape) # This gives torch.Size([64, 1, 28, 28]), number of images, channels,height,width

        # Get to correct shape
        data = data.reshape(data.shape[0], -1)  # Flatten all images in batch

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
            x = x.reshape(x.shape[0], -1)
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(f"Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)}")
    model.train()


check_accuracy(trainLoader,model)
check_accuracy(testLoader,model)

# With epochs=1
# Checking accuracy on training data
# Got 55989/60000 with accuracy 0.93315
# Checking accuracy on test data
# Got 9326/10000 with accuracy 0.9326

# with epochs=5
# Checking accuracy on training data
# Got 58437/60000 with accuracy 0.97395
# Checking accuracy on test data
# Got 9673/10000 with accuracy 0.9673
