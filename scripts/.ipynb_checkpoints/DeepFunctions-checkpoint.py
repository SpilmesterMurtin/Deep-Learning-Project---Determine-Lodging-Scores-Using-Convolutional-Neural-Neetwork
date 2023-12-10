import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, accuracy_score
import torch
from torchvision import transforms
import os
import json
import numpy as np


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

    for i, ax in enumerate(axes):
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=12, rotation=45)
        ax.set_yticklabels(classes, fontsize=12)

        # Use white text if squares are dark; otherwise black
        threshold = [cm1, cm2, cm3][i].max() / 2
        for j, k in itertools.product(range([cm1, cm2, cm3][i].shape[0]), range([cm1, cm2, cm3][i].shape[1])):
            color = 'white' if [cm1, cm2, cm3][i][j, k] > threshold else 'black'
            ax.text(k, j, [cm1, cm2, cm3][i][j, k], horizontalalignment='center', color=color, fontsize=12)

    plt.tight_layout()
    plt.show()

# Example usage:
# plot_3_confusion_matrices(cm1, cm2, cm3, accuracy1, accuracy2, accuracy3, num_classes, title1, title2, title3)


def create_confusion_matrix(dataloader, net, num_classes=9, device="cpu"):
    # Set the model to evaluation mode
    net.eval()
    # Lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []
    totalAcc = 0.0
    for i, data in enumerate(dataloader, 0):
        # Get the inputs and labels
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # Forward pass
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)  # Assuming a classification task with softmax activation
        # Move tensors to CPU before extending lists
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())
        totalAcc += torch.eq(torch.argmax(outputs, dim=1), labels).sum()

    # Convert lists to NumPy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    return cm, accuracy_score(true_labels, predicted_labels)


def plot_loss_and_accuracy(trainLossList, trainAccList, valLossList, valAccList):
    # Create a single subplot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Plot training loss
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='blue')
    ax1.plot(trainLossList, label='Training Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # Create a second y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel('Accuracy', color='orange')
    ax2.plot(trainAccList, label='Training Accuracy', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.legend(loc='upper right')

    plt.title('Training Loss and Accuracy')
    plt.show()

    # Create a single subplot with two y-axes for validation
    fig, ax3 = plt.subplots(figsize=(12, 5))

    # Plot validation loss
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss', color='blue')
    ax3.plot(valLossList, label='Validation Loss', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3.legend(loc='upper left')

    # Create a second y-axis for accuracy
    ax4 = ax3.twinx()
    ax4.set_ylabel('Accuracy', color='orange')
    ax4.plot(valAccList, label='Validation Accuracy', color='orange')
    ax4.tick_params(axis='y', labelcolor='orange')
    ax4.legend(loc='upper right')

    plt.title('Validation Loss and Accuracy')
    plt.show()

def transfer_to_cpu(data_list):
    cpu_data_list = []
    for data in data_list:
        if isinstance(data, torch.Tensor):
            cpu_data_list.append(data.to('cpu'))
        else:
            cpu_data_list.append(data)
    return cpu_data_list

#Keep only 500 zeros in the dataset. In attempt to balance the dataset where the zeors are the majority by far.
def filter_data(dataSet):
    filtered_data = []
    count = 0
    for data in dataSet:
        if data[1] == 0 and count < 500:
            filtered_data.append(data)
            count += 1
        elif data[1] != 0 and data[1] != 9:
            filtered_data.append(data)
    return filtered_data

def list_gpu_names():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")

        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")
    else:
        print("No GPUs available on this machine.")
        
def augment_data(dataSet):
    horizontal_flip = transforms.RandomHorizontalFlip(p=1.0)  # Set probability to 1.0 for always flipping
    vertical_flip = transforms.RandomVerticalFlip(p=1.0)  # Set probability to 1.0 for always flipping
    augmented_data = []
    for data in dataSet:
        img, label = data
        augmented_img = horizontal_flip(img)
        augmented_data.append((augmented_img, label))
        augmented_img = vertical_flip(img)
        augmented_data.append((augmented_img, label))
        augmented_img = vertical_flip(horizontal_flip(img))
        augmented_data.append((augmented_img, label))
        augmented_data.append((img, label))
    return augmented_data


def save_results_to_file(layer_sizes, train_cm, train_accuracy, val_cm, val_accuracy, test_cm, test_accuracy):
    # Create a directory if it doesn't exist
    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)

    # Convert confusion matrices to lists for JSON serialization
    train_cm_list = train_cm.tolist() if isinstance(train_cm, np.ndarray) else train_cm
    val_cm_list = val_cm.tolist() if isinstance(val_cm, np.ndarray) else val_cm
    test_cm_list = test_cm.tolist() if isinstance(test_cm, np.ndarray) else test_cm
    numpy_array = tensor_data.numpy()

    results_dict = {
        'layer_sizes': layer_sizes,
        'train': {
            'confusion_matrix': train_cm_list,
            'accuracy': train_accuracy
        },
        'validation': {
            'confusion_matrix': val_cm_list,
            'accuracy': val_accuracy
        },
        'test': {
            'confusion_matrix': test_cm_list,
            'accuracy': test_accuracy
        }
    }

    # Convert the dictionary to JSON format
    results_json = json.dumps(results_dict, indent=4)

    # Create a filename based on the layer sizes and accuracies
    filename = '_'.join(map(str, layer_sizes))
    filename += f'_train_acc_{train_accuracy:.2%}_val_acc_{val_accuracy:.2%}_test_acc_{test_accuracy:.2%}_results.json'

    # Save the results to a file in the 'results' folder
    filepath = os.path.join(results_folder, filename)
    with open(filepath, 'w') as file:
        file.write(results_json)

        
def convert_to_json_serializable(obj):
    if isinstance(obj, torch.Tensor):
        # Move the tensor to the CPU before converting to NumPy array
        return obj.detach().numpy().tolist()
    elif isinstance(obj, (list, np.ndarray)):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj


def save_training_data_to_file(hyperparams, train_loss, train_accuracy, val_loss, val_accuracy):
    # Create a directory if it doesn't exist
    results_folder = 'results'
    os.makedirs(results_folder, exist_ok=True)
    
     # Convert the non-serializable objects to serializable format
    train_loss_serializable = convert_to_json_serializable(train_loss)
    train_accuracy_serializable = convert_to_json_serializable(train_accuracy)
    val_loss_serializable = convert_to_json_serializable(val_loss)
    val_accuracy_serializable = convert_to_json_serializable(val_accuracy)

    # Create a dictionary with the results
    results_dict = {
        'train': {
            'loss': train_loss_serializable,
            'accuracy': train_accuracy_serializable
        },
        'validation': {
            'loss': val_loss_serializable,
            'accuracy': val_accuracy_serializable
        },
        'hyperparams':{
            'learning_rate': hyperparams[0],
            'weight_decay': hyperparams[1],
            'patience': hyperparams[2],
            'momentum_bn2': hyperparams[3],
            'labelsmooth': hyperparams[4],
            "filename": hyperparams[5]
        }

    }

    # Convert the dictionary to JSON format
    results_json = json.dumps(results_dict, default=convert_to_json_serializable, indent=4)


    # Create a filename based on the layer sizes and accuracies
    filename = '_'.join(map(str, hyperparams[:-1]))
    filename += f'_{np.max(val_accuracy):.2%}_results.json'

    # Save the results to a file in the 'results' folder
    filepath = os.path.join(results_folder, filename)
    with open(filepath, 'w') as file:
        file.write(results_json)