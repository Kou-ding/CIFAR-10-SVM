import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time

# Specify the classes
classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

########################## Useful Functions ########################### 
# Training function
def train(model, trainloader, loss_function, optimizer, device):
    model.train() # Set model to training mode
    running_loss = 0.0 #
    correct = 0 # Number of correctly predicted images
    total = 0 # Total number of images
    
    # Iterate over the training dataset
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device) #data[0] is the image, data[1] is the label
        
        optimizer.zero_grad() # Reset the gradients of model parameters
        outputs = model(inputs) # Forward pass
        loss = loss_function(outputs, labels) # Calculate loss
        loss.backward() # Backward pass
        optimizer.step() # Update weights
        
        # Track training statistics
        running_loss += loss.item() # Add loss to running loss
        _, predicted = outputs.max(1) # Get the class index with the highest probability
        total += labels.size(0) # Add batch size to total
        correct += predicted.eq(labels).sum().item() # Add number of correct predictions to correct
        
        # Print statistics every 100 mini-batches
        if i % 100 == 99:
            print(f'[{i + 1}] loss: {running_loss / 100:.3f} | acc: {100.*correct/total:.2f}%')
            running_loss = 0.0 # Reset running loss
    
    return 100. * correct / total

# Evaluation function
def evaluate(model, testloader, loss_function, device):
    model.eval() # Set model to evaluation mode
    test_loss = 0 # Total loss
    correct = 0 # Number of correctly predicted images
    total = 0 # Total number of images
    
    # Compute per-class accuracy
    class_correct = [0]*10
    class_total = [0]*10

    with torch.no_grad(): # Disable gradient calculation
        # Iterate over the test dataset
        for data in testloader: 
            images, labels = data[0].to(device), data[1].to(device) # data[0] is the image, data[1] is the label
            outputs = model(images) # Forward pass
            loss = loss_function(outputs, labels) # Calculate loss
            
            # Track test statistics
            test_loss += loss.item() # Add loss to test loss
            _, predicted = outputs.max(1) # Get the class index with the highest probability
            total += labels.size(0) # Add batch size to total
            correct += predicted.eq(labels).sum().item() # Add number of correct predictions to correct

            # Update per-class statistics
            for i in range(len(labels)):  # Loop through each image in the batch
                label = labels[i]
                class_correct[label] += (predicted[i] == label).item()  # Increment correct count for this class
                class_total[label] += 1  # Increment total count for this class

    # Calculate overall test accuracy and average loss            
    accuracy = 100. * correct / total # Calculate accuracy
    avg_loss = test_loss / len(testloader) # Calculate average loss
    print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Calculate per-class accuracy
    print('\nPer-class accuracy:')
    for i in range(10):
        if class_total[i] > 0: # Avoid division by zero
            class_accuracy = 100. * class_correct[i] / class_total[i]
            print(f'{classes[i]:>5s}: {class_accuracy:.2f}%')

    return accuracy, avg_loss

####################### Define the MLP architecture #######################
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # Input size: 32x32x3 = 3072 (flattened RGB image)
        self.flatten = nn.Flatten()
        
        self.layers = nn.Sequential(
            # First hidden layer
            nn.Linear(3072, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Output layer
            nn.Linear(2048, 10)  # 10 classes for CIFAR-10
        )
    
    def forward(self, x):
        x = self.flatten(x)  # Flatten the input image
        x = self.layers(x)   # Pass through MLP layers
        return x

############################## Main Function ###############################
def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # The variable "device" specifies the computational device 
    # This is where we run our neural network on (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ######################### Data Preprocessing #########################
    # Tranformations for the training dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4), # Randomly crops image with padding
        transforms.RandomHorizontalFlip(), # Randomly flips image horizontally
        transforms.ToTensor(), # Converts image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # Tranformations for the test dataset
    transform_test = transforms.Compose([
        transforms.ToTensor(), # Converts image to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    ############################# Our Dataset #############################
    # Load CIFAR-10 training dataset
    trainset = torchvision.datasets.CIFAR10(
        root='./dataset', train=True,
        download=True,
        transform=transform_train
    )
    trainloader = DataLoader(
        trainset, 
        batch_size=128,
        shuffle=True, 
        num_workers=2
    )
    # Load CIFAR-10 test dataset
    testset = torchvision.datasets.CIFAR10(
        root='./dataset',
        train=False,
        download=True, 
        transform=transform_test
    )
    testloader = DataLoader(
        testset, 
        batch_size=100,
        shuffle=False,
        num_workers=2
    )
        
    ######################### Model Initialization ##########################
    # MLP: Send model to device for training
    model = MLP().to(device)
    # Loss function
    loss_function = torch.nn.MultiMarginLoss(p=1, margin=1.0, weight=None, size_average=None, reduce=None, reduction='mean')
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5) # Learning rate scheduler

    # Main loop
    num_epochs = 30 # Number of epochs
    best_acc = 0 # Best accuracy
    train_acc_history = [] # Training accuracy history
    test_acc_history = [] # Test accuracy history

    print(f"Training on {device}") # Print device
    start_time = time.time() # Start time

    # Iterate over the epochs
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}') # Print epoch
        
        # Train and evaluate the model
        train_acc = train(model, trainloader, loss_function, optimizer, device)
        test_acc, test_loss = evaluate(model, testloader, loss_function, device)
        
        # Track accuracy history
        train_acc_history.append(train_acc)
        test_acc_history.append(test_acc)
        
        # Learning rate scheduling
        scheduler.step(test_loss) # Adjust learning rate based on loss
        
        # Save best model
        if test_acc > best_acc:
            print(f'Saving best model with accuracy: {test_acc:.2f}%') # Print current best accuracy
            # Save the model with the best accuracy in the filename
            #torch.save(model.state_dict(), f'./cifar10_acc_{test_acc:.2f}.pth')
            best_acc = test_acc

    # Calculate training time
    training_time = time.time() - start_time # Calculate training time
    print(f'\nTraining completed in {training_time/60:.2f} minutes') # Print training time
    print(f'Best accuracy: {best_acc:.2f}%') # Print final best accuracy

    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc_history, label='Train Accuracy')
    plt.plot(test_acc_history, label='Test Accuracy')
    plt.title('MLP Accuracy over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.savefig('report/media/cifar10_mlp_accuracy.png')
    plt.show()

if __name__ == '__main__':
    main()