from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import os.path
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
writer = None

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # MaxPool(Relu(x)) = Relu(MaxPool(x)) so we can use F.relu() before or after F.max_pool2d()
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        print("self.fc2: ", x.shape)
        output = F.log_softmax(x, dim=1)
        return output
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    total_acc = 0
    # in tqdm print progress bar with current batch number and loss
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # negative log likelihood loss
        loss = F.nll_loss(output, target)

        # backpropagation
        loss.backward()

        # update parameters
        optimizer.step()

        # print following information every log_interval on progress bar
        if batch_idx % args.log_interval == 0:
            writer.add_scalar('Loss/Train', loss.item(), epoch)
            writer.add_scalar('Accuracy/Train', (output.argmax(dim=1) == target).sum().item() / len(data), epoch)

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    pred_list = []
    label_list = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

            # scikit-learn compatible pred and label to reprot recall, f1, acc, and precision
            pred_list += pred.view_as(target).cpu().numpy().tolist()
            label_list += target.cpu().numpy().tolist()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    writer.add_scalar('Loss/Test', test_loss, epoch)
    writer.add_scalar('Accuracy/Test', 100. * correct / len(test_loader.dataset), epoch)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--no-mps', action='store_true', default=False,
                        help='disables macOS GPU training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # data augmentation unit
    transform=transforms.Compose([
        transforms.ToTensor(),
        ])
    dataset1 = datasets.CIFAR10('./data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.CIFAR10('./data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    directory = "./result/cifar10/MLP/epoch{}_batch{}_lr{:.3f}_seed{}".format(args.epochs, args.batch_size, args.lr, args.seed)
    location = "./result/cifar10/MLP/epoch{}_batch{}_lr{:.3f}_seed{}/checkpoints/single_mlp.pt".format(args.epochs, args.batch_size, args.lr, args.seed)
    os.makedirs("./result/cifar10/MLP/epoch{}_batch{}_lr{:.3f}_seed{}".format(args.epochs, args.batch_size, args.lr, args.seed), exist_ok=True)

    global writer
    writer = SummaryWriter(directory)

    if(os.path.isfile(location) and not args.save_model):
       model.load_state_dict(torch.load(location))
    else:
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            scheduler.step()
            test(model, device, test_loader, epoch)

        # save model
        if args.save_model:
            os.makedirs(directory + "/checkpoints", exist_ok=True)
            torch.save(model.state_dict(), location)
            print("Model saved at " + location)
    writer.close()
    count = 0
    for param in model.parameters():
        if param.requires_grad:
            count += 1
    print("Number of trainable parameters: {}".format(count))
if __name__ == '__main__':
    main()