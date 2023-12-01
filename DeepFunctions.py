import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score
import torch

def plot_confusion_matrix(cm, accuracy=0, num_classes=9, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    # Print confusion matrix
    print("Confusion Matrix:")

    classes = list(range(num_classes))
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 12))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(f'{title}\nAccuracy: {accuracy:.2%}', fontsize=20)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=15, rotation=45)
    plt.yticks(tick_marks, classes, fontsize=15)
    
    plt.xlabel('Predicted label', fontsize=20)
    plt.ylabel('True label', fontsize=20)

    # Use white text if squares are dark; otherwise black
    threshold = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = 'white' if cm[i, j] > threshold else 'black'
        plt.text(j, i, cm[i, j], horizontalalignment='center', color=color, fontsize=15)

    plt.tight_layout()
    plt.show()
    #plot_confusion_matrix(cm1,acc1, num_classes, title='Confusion matrix for validation set')

def plot_3_confusion_matrices(cm1, cm2, cm3, accuracy1=0, accuracy2=0, accuracy3=0, num_classes=9, title1='Confusion matrix 1', title2='Confusion matrix 2', title3='Confusion matrix 3', cmap=plt.cm.Blues):
    classes = list(range(num_classes))
    
    # Plot confusion matrices side by side
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Plot for the first confusion matrix
    axes[0].imshow(cm1, interpolation='nearest', cmap=cmap)
    axes[0].set_title(f'{title1}\nAccuracy: {accuracy1:.2%}', fontsize=15)
    axes[0].set_xlabel('Predicted label', fontsize=15)
    axes[0].set_ylabel('True label', fontsize=15)

    # Plot for the second confusion matrix
    axes[1].imshow(cm2, interpolation='nearest', cmap=cmap)
    axes[1].set_title(f'{title2}\nAccuracy: {accuracy2:.2%}', fontsize=15)
    axes[1].set_xlabel('Predicted label', fontsize=15)
    axes[1].set_ylabel('True label', fontsize=15)

    # Plot for the third confusion matrix
    axes[2].imshow(cm3, interpolation='nearest', cmap=cmap)
    axes[2].set_title(f'{title3}\nAccuracy: {accuracy3:.2%}', fontsize=15)
    axes[2].set_xlabel('Predicted label', fontsize=15)
    axes[2].set_ylabel('True label', fontsize=15)

    for ax in axes:
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=12, rotation=45)
        ax.set_yticklabels(classes, fontsize=12)

        # Use white text if squares are dark; otherwise black
        threshold = cm1.max() / 2
        for i, j in itertools.product(range(cm1.shape[0]), range(cm1.shape[1])):
            color = 'white' if cm1[i, j] > threshold else 'black'
            ax.text(j, i, cm1[i, j], horizontalalignment='center', color=color, fontsize=12)

    plt.tight_layout()
    plt.show()

# Example usage:
# plot_confusion_matrices(cm1, cm2, cm3, accuracy1, accuracy2, accuracy3, num_classes, title1, title2, title3)


def create_confusion_matrix(dataloader, net, num_classes=9, device="cpu"):
    # Set the model to evaluation mode
    net.eval()
    # Lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []
    for i, data in enumerate(dataloader, 0):
        net.eval()
        # Get the inputs and labels
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # Forward pass
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)  # Assuming a classification task with softmax activation
        # Move tensors to CPU before extending lists
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

    # Convert lists to NumPy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    return cm, accuracy_score(true_labels, predicted_labels)