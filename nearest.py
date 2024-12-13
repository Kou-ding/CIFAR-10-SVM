import matplotlib.pyplot as plt
import numpy as np # For numerical operations and array handling
import torch # PyTorch main library
import torchvision # For accessing datasets like CIFAR-10
import torchvision.transforms as transforms # For image transformations
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid # The classifiers we'll use
from sklearn.metrics import accuracy_score, classification_report # For evaluation metrics
import time # For timing our training process

def preprocessing():
    # Define the transform to normalize the data
    transform = transforms.Compose([
        transforms.ToTensor(), # Converts PIL images to tensors (0-1 range)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalizes each channel
    ])

    # Load CIFAR-10 dataset
    # Training set
    trainset = torchvision.datasets.CIFAR10(
        root='./dataset', 
        train=True,                                  
        download=True, 
        transform=transform
    )
    # Test set
    testset = torchvision.datasets.CIFAR10(
        root='./dataset', 
        train=False,
        download=True, 
        transform=transform
    )
    
    # Convert datasets to numpy arrays
    # This uses a DataLoader to load the entire dataset into memory as numpy arrays
    x_train = []
    y_train = []
    for images, labels in torch.utils.data.DataLoader(
        trainset, 
        batch_size=len(trainset)
    ):
        x_train = images.numpy()
        y_train = labels.numpy()

    x_test = []
    y_test = []
    for images, labels in torch.utils.data.DataLoader(
        testset, 
        batch_size=len(testset)
    ):
        x_test = images.numpy()
        y_test = labels.numpy()
    
    # Reshape the data: flatten the 32x32x3 images into 1D vectors (with 3072 features)
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_test = x_test.reshape(x_test.shape[0], -1)
    
    return x_train, x_test, y_train, y_test

def evaluate(clf, x_train, x_test, y_train, y_test, name):
    # Record start time
    start_time = time.time()
    
    # Train the classifier
    clf.fit(x_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(x_test)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Calculate prediction accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Print results
    print(f"\n{name} Results:")
    print(f"Training time: {training_time:.2f} seconds")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy, training_time

def plot_results(results):
    # Extract data for plotting
    classifiers = [result['name'] for result in results]
    accuracies = [result['accuracy'] for result in results]
    times = [result['time'] for result in results]

    # Create subplots for accuracy and training time
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    
    # Accuracy bar chart
    ax[0].bar(classifiers, accuracies, color='skyblue')
    ax[0].set_title('Classifier Accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_ylim(0, 1)  # Accuracy is between 0 and 1
    ax[0].set_xlabel('Classifier')

    # Training time bar chart
    ax[1].bar(classifiers, times, color='salmon')
    ax[1].set_title('Training Time')
    ax[1].set_ylabel('Time (seconds)')
    ax[1].set_xlabel('Classifier')

    # Display the plots
    plt.tight_layout()
    plt.savefig('report/media/knn_centroid.png')
    plt.show()

def main():
    # User feedback
    print("Loading and preparing CIFAR-10 dataset...")
    x_train, x_test, y_train, y_test = preprocessing()
    
    # Initialize classifiers
    classifiers = [
        {'name': 'KNN (k=1)', 'clf': KNeighborsClassifier(n_neighbors=1)},
        {'name': 'KNN (k=3)', 'clf': KNeighborsClassifier(n_neighbors=3)},
        {'name': 'Nearest Centroid', 'clf': NearestCentroid()}
    ]
    
    results = []

    for classifier in classifiers:
        name = classifier['name']
        clf = classifier['clf']
        print(f"\nEvaluating {name}...")
        accuracy, training_time = evaluate(clf, x_train, x_test, y_train, y_test, name)
        results.append({'name': name, 'accuracy': accuracy, 'time': training_time})
    
    # Compare results
    print("\nPerformance Comparison:")
    print(f"{'Classifier':<20} {'Accuracy':<10} {'Training Time (s)':<15}")
    print("-" * 45)
    for result in results:
        print(f"{result['name']:<20} {result['accuracy']:.4f}    {result['time']:.2f}")

    # Visualize results
    plot_results(results)


if __name__ == "__main__":
    main()
