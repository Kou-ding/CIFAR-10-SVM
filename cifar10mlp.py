import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import time

class MLP(nn.Module):
    def __init__(self, num_classes=10):
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
            nn.Linear(2048, num_classes)
        )
        
        # Class attributes for training
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck')
        self.train_acc_history = []
        self.test_acc_history = []
        
    def forward(self, x):
        x = self.flatten(x)  # Flatten the input image
        x = self.layers(x)   # Pass through MLP layers
        return x
    
    def preprocessing(self):
        """Prepare training and test data loaders"""
        # Transformations for the training dataset
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # Transformations for the test dataset
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Download and load datasets
        trainset = torchvision.datasets.CIFAR10(
            root='./dataset', 
            train=True,
            download=True, 
            transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root='./dataset', 
            train=False,
            download=True, 
            transform=transform_test
        )

        # Create data loaders
        trainloader = DataLoader(
            trainset, 
            batch_size=64,
            shuffle=True, 
            num_workers=2
        )
        testloader = DataLoader(
            testset, 
            batch_size=64,
            shuffle=False,
            num_workers=2
        )
        return trainloader, testloader
    
    def train(self, trainloader, loss_function, optimizer, device):
        """Train the model for one epoch"""
        self.train()  # Set model to training mode
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            
            optimizer.zero_grad()
            outputs = self(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        epoch_accuracy = 100. * correct / total
        return epoch_accuracy
    
    def evaluate(self, testloader, loss_function, device):
        """Evaluate the model on test dataset"""
        self.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        # Compute per-class accuracy
        class_correct = [0]*10
        class_total = [0]*10

        with torch.no_grad():
            for data in testloader: 
                images, labels = data[0].to(device), data[1].to(device)
                outputs = self(images)
                loss = loss_function(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                # Update per-class statistics
                for i in range(len(labels)):
                    label = labels[i]
                    class_correct[label] += (predicted[i] == label).item()
                    class_total[label] += 1

        # Calculate overall test accuracy and average loss            
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(testloader)
        print(f'Test set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        return accuracy, avg_loss
    
    def visualize_training(self):
        """Plot training and test accuracy history"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_acc_history, label='Train Accuracy')
        plt.plot(self.test_acc_history, label='Test Accuracy')
        plt.title('MLP Accuracy over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        plt.savefig('report/media/cifar10_mlp_accuracy.png')
        plt.show()
    
    def train_model(self, num_epochs=30, learning_rate=0.001):
        """Complete training process"""
        # Set random seed for reproducibility
        torch.manual_seed(42)

        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)

        # Prepare data loaders
        trainloader, testloader = self.preprocessing()

        # Loss function
        loss_function = torch.nn.MultiMarginLoss(
            p=1, margin=1.0, weight=None, 
            size_average=None, reduce=None, 
            reduction='mean'
        )
        
        # Optimizer
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', patience=3, factor=0.5
        )

        best_acc = 0
        start_time = time.time()

        # Training loop
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            
            # Train and evaluate
            train_acc = self.train(trainloader, loss_function, optimizer, device)
            test_acc, test_loss = self.evaluate(testloader, loss_function, device)
            
            # Track accuracy history
            self.train_acc_history.append(train_acc)
            self.test_acc_history.append(test_acc)
            
            # Learning rate scheduling
            scheduler.step(test_loss)
            
            # Save best model
            if test_acc > best_acc:
                print(f'Saving best model with accuracy: {test_acc:.2f}%')
                best_acc = test_acc

        # Calculate and print training time
        training_time = time.time() - start_time
        print(f'\nTraining completed in {training_time/60:.2f} minutes')
        print(f'Best accuracy: {best_acc:.2f}%')

        # Visualize training history
        self.visualize_training()

def main():
    # Create and train the model
    model = MLP()
    model.train_model()

if __name__ == '__main__':
    main()