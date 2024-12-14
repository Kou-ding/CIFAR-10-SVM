import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class CIFARSVM:
    # Initialize the model
    def __init__(self, num_classes=10, input_size=3072):
        # Send model to device (GPU or CPU) for computation
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # SVM-like model using linear layer
        self.model = nn.Linear(input_size, num_classes).to(self.device)
        
        # Use hinge loss (standard for SVMs)
        self.criterion = nn.MultiMarginLoss()
        
        # SGD optimizer with high learning rate and momentum
        self.optimizer = optim.SGD(
            self.model.parameters(), 
            lr=0.01, 
            momentum=0.9, 
            weight_decay=1e-5
        )
        
        # Tracking metrics
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []

        # Preprocess datasets only once
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten images
        ])

        # Download datasets only once
        self.train_dataset = datasets.CIFAR10(
            root='./dataset', 
            train=True, 
            download=True, 
            transform=self.transform
        )
        self.test_dataset = datasets.CIFAR10(
            root='./dataset', 
            train=False, 
            download=True, 
            transform=self.transform
        )

    def get_data_loaders(self, batch_size=64):
        # Create data loaders
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2
        )
        test_loader = DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=2
        )
        return train_loader, test_loader

    def train(self, epochs=10):
        train_loader, test_loader = self.get_data_loaders()
        
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            total = 0
            correct = 0
            
            for batch_images, batch_labels in train_loader:
                batch_images = batch_images.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_images)
                
                # Compute loss
                loss = self.criterion(outputs, batch_labels)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()

                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

            # Store training metrics
            train_loss = total_loss / len(train_loader)
            train_accuracy = 100 * correct / total
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_accuracy)
            
            # Evaluate on test set for each epoch
            test_accuracy = self.evaluate(test_loader)
            self.test_accuracies.append(test_accuracy)
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, "
                  f"Train Accuracy: {train_accuracy:.2f}%, "
                  f"Test Accuracy: {test_accuracy:.2f}%")

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy

    def predict(self, image):
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)
            return torch.argmax(output).item()
    
    def plot_metrics(self):
        """
        Plot training and testing metrics
        """
        plt.figure(figsize=(12, 5))
        
        # Training Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.title('Training Loss over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        
        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Training Accuracy')
        plt.plot(self.test_accuracies, label='Testing Accuracy')
        plt.title('Training vs Testing Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()


# CIFAR-10 class names for reference
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck')

def main():
    # Initialize and train SVM
    svm = CIFARSVM()
    svm.train(epochs=10)

    # Plot training and testing metrics
    svm.plot_metrics()

if __name__ == "__main__":
    main()