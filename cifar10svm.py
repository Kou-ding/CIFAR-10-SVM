import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class CIFARSVM:
    def __init__(self, num_classes=10, input_size=3072):
        """
        Initialize Support Vector Machine for CIFAR-10 classification
        
        Args:
            num_classes (int): Number of classes in CIFAR-10 (default 10)
            input_size (int): Flattened image size (32x32x3 = 3072)
        """
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

    def _preprocess_data(self):
        """
        Prepare CIFAR-10 dataset with transformations
        
        Returns:
            train_loader, test_loader
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Lambda(lambda x: x.view(-1))  # Flatten images
        ])

        # Download and load training data
        train_dataset = datasets.CIFAR10(
            root='./dataset', 
            train=True, 
            download=True, 
            transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root='./dataset', 
            train=False, 
            download=True, 
            transform=transform
        )

        train_loader = DataLoader(
            train_dataset, 
            batch_size=64, 
            shuffle=True, 
            num_workers=2
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=64, 
            shuffle=False, 
            num_workers=2
        )

        return train_loader, test_loader

    def train(self, epochs=50):
        """
        Train the SVM model on CIFAR-10
        
        Args:
            epochs (int): Number of training epochs
        
        Returns:
            train_accuracies, test_accuracies
        """
        train_loader, test_loader = self._preprocess_data()
        
        train_accuracies = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            total_loss = 0.0
            
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
            
            # Evaluate training accuracy
            train_accuracy = self._calculate_accuracy(train_loader)
            train_accuracies.append(train_accuracy)
            
            # Evaluate test accuracy
            test_accuracy = self._calculate_accuracy(test_loader)
            test_accuracies.append(test_accuracy)
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, "
                  f"Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%")
        
        return train_accuracies, test_accuracies

    def _calculate_accuracy(self, data_loader):
        """
        Calculate accuracy for a given data loader
        
        Args:
            data_loader (DataLoader): Dataset to evaluate
        
        Returns:
            Accuracy percentage
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return 100 * correct / total

    def predict(self, image):
        """
        Predict class for a single image
        
        Args:
            image (torch.Tensor): Preprocessed image tensor
        
        Returns:
            Predicted class
        """
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            output = self.model(image)
            return torch.argmax(output).item()

# CIFAR-10 class names for reference
classes = ('plane', 'car', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck')

def main():
    # Initialize and train SVM
    svm = CIFARSVM()
    train_accuracies, test_accuracies = svm.train(epochs=30)
    
    # Plot accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, len(test_accuracies) + 1), test_accuracies, label='Testing Accuracy')
    plt.title('Training and Testing Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()