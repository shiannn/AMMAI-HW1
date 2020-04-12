import matplotlib.pyplot as plt
from sklearn.metrics import auc


def plot_roc_lfw(false_positive_rate, true_positive_rate, figure_name, epochNum):
    """Plots the Receiver Operating Characteristic (ROC) curve.

    Args:
        false_positive_rate: False positive rate
        true_positive_rate: True positive rate
        figure_name (str): Name of the image file of the resulting ROC curve plot.
    """
    roc_auc = auc(false_positive_rate, true_positive_rate)
    #fig = plt.figure()
    plt.clf()
    plt.plot(
        false_positive_rate, true_positive_rate, color="red", lw=2, label="ROC Curve (area = {:.2f}), epoch={}".format(roc_auc, epochNum)
    )
    plt.plot([0, 1], [0, 1], color="blue", lw=2, linestyle="--", label="Random")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(figure_name)


def plot_accuracy_lfw(log_dir, epochs, figure_name="lfw_accuracies.png"):
    """Plots the accuracies on the Labeled Faces in the Wild dataset over the training epochs.

    Args:
        log_dir (str): Directory of the log file containing the lfw accuracy values to be plotted.
        epochs (int): Number of training epochs finished.
        figure_name (str): Name of the image file of the resulting lfw accuracies plot.
    """
    with open(log_dir, 'r') as f:
        lines = f.readlines()
        epoch_list = [int(line.split('\t')[0]) for line in lines]
        accuracy_list = [round(float(line.split('\t')[1]), 2) for line in lines]

        #fig = plt.figure()
        plt.plot(epoch_list, accuracy_list, color='red', label='LFW Accuracy')
        plt.ylim([0.0, 1.05])
        plt.xlim([1, epochs + 1])
        plt.xlabel('Epoch')
        plt.ylabel('LFW Accuracy')
        plt.title('LFW Accuracies plot')
        plt.legend(loc='lower right')
        plt.savefig(figure_name)


def plot_training_validation_losses_center(log_dir, epochs, figure_name="training_validation_losses_center.png"):
    """Plots the Training/Validation losses plot for Cross Entropy Loss with Center Loss over the training epochs.

    Args:
        log_dir (str): Directory of the training log file containing the loss values to be plotted.
        epochs (int): Number of training epochs finished.
        figure_name (str): Name of the image file of the resulting Training/Validation losses plot.
    """
    with open(log_dir, 'r') as f:
        lines = f.readlines()
        epoch_list = [int(line.split('\t')[0]) for line in lines]
        train_loss_list = [float(round(float(line.split('\t')[1]), 2)) for line in lines]
        #valid_loss_list = [float(round(float(line.split('\t')[2]), 2)) for line in lines]

        #fig = plt.figure()
        plt.plot(epoch_list, train_loss_list, color='blue', label='Training loss')
        #plt.plot(epoch_list, valid_loss_list, color='red', label='Validation loss')
        plt.ylim([0.0, max(train_loss_list)])
        plt.xlim([1, epochs + 1])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training/Validation losses plot (Cross Entropy loss with Center loss)')
        plt.legend(loc='upper left')
        #fig.savefig(figure_name, dpi=fig.dpi)
        plt.savefig(figure_name)


def plot_triplet_losses(log_dir, epochs, figure_name="triplet_losses.png"):
    """PLots the Triplet loss over the training epochs.

    Args:
        log_dir (str): Directory of the training log file containing the loss values to be plotted.
        epochs (int): Number of training epochs finished.
        figure_name (str): Name of the image file of the resulting Triplet losses plot.
    """
    with open(log_dir, 'r') as f:
        lines = f.readlines()
        epoch_list = [int(line.split('\t')[0]) for line in lines]
        triplet_loss_list = [float(round(float(line.split('\t')[1]), 2)) for line in lines]
        print('epoch_list', epoch_list)
        print('triplet_loss_list', triplet_loss_list)

        #fig = plt.figure()
        #, label='Triplet loss'
        plt.plot(epoch_list, triplet_loss_list, color='red')
        #plt.ylim([0.0, max(triplet_loss_list)])
        plt.xlim([1, epochs + 1])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Triplet losses plot')
        #plt.legend(loc='upper left')
        #fig.savefig(figure_name, dpi=fig.dpi)
        plt.savefig(figure_name)
