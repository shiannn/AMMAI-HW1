import numpy as np
import math
import time
import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn import CosineSimilarity
from torch.nn.modules.distance import PairwiseDistance
from torch.utils.data import DataLoader, Subset
from losses.center_loss import CenterLoss
from dataloaders.APDDataset import APDDataset
from validate_on_LFW_m import evaluate_lfw
from plots import plot_roc_lfw, plot_accuracy_lfw, plot_training_validation_losses_center
from tqdm import tqdm
from models.resnet18 import Resnet18Center
from models.resnet34 import Resnet34Center
from models.resnet50 import Resnet50Center
from models.resnet101 import Resnet101Center
from models.inceptionresnetv2 import InceptionResnetV2Center
from ArcMarginModel import ArcMarginModel
from PIL import ImageFile
from pathlib import Path
import shutil


ImageFile.LOAD_TRUNCATED_IMAGES = True

logging.basicConfig(level=logging.INFO)
shutil.rmtree("plots/roc_plots_margin")
rocFolder = Path("plots") / Path("roc_plots_margin")
rocFolder.mkdir(exist_ok=True)

try:
    os.remove("plots/training_validation_losses_resnet34_margin.png")
    os.remove("logs/resnet34_log_margin.txt")
except OSError:
    pass

parser = argparse.ArgumentParser(description="Training FaceNet facial recognition model using Cross Entropy Loss with Center Loss.")
# Dataset
parser.add_argument('--dataroot', '-d', type=str, required=True,
                    help="(REQUIRED) Absolute path to the dataset folder"
                    )
# LFW
parser.add_argument('--apd', type=str, required=True,
                    help="(REQUIRED) Absolute path to the labeled faces in the wild dataset folder"
                    )
parser.add_argument('--apd_batch_size', default=32, type=int,
                    help="Batch size for APD dataset (default: 64)"
                    )
parser.add_argument('--apd_validation_epoch_interval', default=5, type=int,
                    help="Perform APD validation every n epoch interval (default: every 5 epochs)"
                    )
# Training settings
parser.add_argument('--model', type=str, default="resnet34", choices=["resnet18", "resnet34", "resnet50", "resnet101", "inceptionresnetv2"],
    help="The required model architecture for training: ('resnet18','resnet34', 'resnet50', 'resnet101', 'inceptionresnetv2'), (default: 'resnet34')"
                    )
parser.add_argument('--epochs', default=20, type=int,
                    help="Required training epochs (default: 275)"
                    )
parser.add_argument('--resume_path', default='',  type=str,
    help='path to latest model checkpoint: (Model_training_checkpoints/model_resnet34_epoch_0.pt file) (default: None)'
                    )
parser.add_argument('--batch_size', default=64, type=int,
                    help="Batch size (default: 128)"
                    )
parser.add_argument('--num_workers', default=4, type=int,
                    help="Number of workers for data loaders (default: 4)"
                    )
parser.add_argument('--valid_split', default=0.01, type=float,
                    help="Validation dataset percentage to be used from the dataset (default: 0.01)"
                    )
parser.add_argument('--embedding_dim', default=128, type=int,
                    help="Dimension of the embedding vector (default: 128)"
                    )
parser.add_argument('--pretrained', default=False, type=bool,
                    help="Download a model pretrained on the ImageNet dataset (Default: False)"
                    )
parser.add_argument('--optimizer', type=str, default="sgd", choices=["sgd", "adagrad", "rmsprop", "adam"],
    help="Required optimizer for training the model: ('sgd','adagrad','rmsprop','adam'), (default: 'sgd')"
                    )
parser.add_argument('--lr', default=0.1, type=float,
                    help="Learning rate for the optimizer (default: 0.1)"
                    )
parser.add_argument('--center_loss_lr', default=0.5, type=float,
                    help="Learning rate for center loss (default: 0.5)"
                    )
parser.add_argument('--center_loss_weight', default=0.007, type=float,
                    help="Center loss weight (default: 0.007)"
                    )
args = parser.parse_args()


def main():
    dataroot = args.dataroot
    apd_dataroot = args.apd
    apd_batch_size = args.apd_batch_size
    apd_validation_epoch_interval = args.apd_validation_epoch_interval
    model_architecture = args.model
    epochs = args.epochs
    resume_path = args.resume_path
    batch_size = args.batch_size
    num_workers = args.num_workers
    validation_dataset_split_ratio = args.valid_split
    embedding_dimension = args.embedding_dim
    pretrained = args.pretrained
    optimizer = args.optimizer
    learning_rate = args.lr
    learning_rate_center_loss = args.center_loss_lr
    center_loss_weight = args.center_loss_weight
    start_epoch = 0

    # Define image data pre-processing transforms
    #   ToTensor() normalizes pixel values between [0, 1]
    #   Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) normalizes pixel values between [-1, 1]

    #  Size 182x182 RGB image -> Center crop size 160x160 RGB image for more model generalization
    data_transforms = transforms.Compose([
        #transforms.RandomCrop(size=50),
        transforms.Resize(size=(160,160)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])
    # Size 160x160 RGB image
    apd_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    # Load the dataset
    dataset = torchvision.datasets.ImageFolder(
        root=dataroot,
        transform=data_transforms
    )

    # Subset the dataset into training and validation datasets
    num_classes = len(dataset.classes)
    print("\nNumber of classes in dataset: {}".format(num_classes))
    num_validation = int(num_classes * validation_dataset_split_ratio)
    num_train = num_classes - num_validation
    indices = list(range(num_classes))
    np.random.seed(420)
    np.random.shuffle(indices)
    train_indices = indices[:num_train]
    validation_indices = indices[num_train:]

    # Define the dataloaders
    train_dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    logging.info('prepare APD dataset')
    apd_dataroot = '../APD'
    apdDataset=APDDataset(
        directory=apd_dataroot,
        #pairs_path='negative_pairs.txt',
        #pairs_path='positive_pairs.txt',
        pairs_path='full.txt',
        transform=apd_transforms
    )
    print('apdDataset', apdDataset)

    apd_dataloader = torch.utils.data.DataLoader(
        dataset=apdDataset,
        batch_size=apd_batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    # Instantiate model
    if model_architecture == "resnet18":
        model = Resnet18Center(
            num_classes=num_classes,
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet34":
        model = Resnet34Center(
            num_classes=num_classes,
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet50":
        model = Resnet50Center(
            num_classes=num_classes,
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet101":
        model = Resnet101Center(
            num_classes=num_classes,
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "inceptionresnetv2":
        model = InceptionResnetV2Center(
            num_classes=num_classes,
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    print("\nUsing {} model architecture.".format(model_architecture))

    # Load model to GPU or multiple GPUs if available
    flag_train_gpu = torch.cuda.is_available()
    flag_train_multi_gpu = False

    if flag_train_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()
        flag_train_multi_gpu = True
        print('Using multi-gpu training.')
    elif flag_train_gpu and torch.cuda.device_count() == 1:
        model.cuda()
        print('Using single-gpu training.')

    # Set loss functions
    criterion_crossentropy = nn.CrossEntropyLoss().cuda()
    criterion_centerloss = CenterLoss(num_classes=num_classes, feat_dim=embedding_dimension).cuda()

    metric_fc = ArcMarginModel(num_classes, embedding_dimension).cuda()

    # Set optimizers
    if optimizer == "sgd":
        optimizer_model = torch.optim.SGD([{'params':model.parameters()}, {'params':metric_fc.parameters()}], lr=learning_rate)
        #optimizer_metric = torch.optim.SGD(metric_fc.parameters(), lr=learning_rate_center_loss)

    elif optimizer == "adagrad":
        optimizer_model = torch.optim.Adagrad([{'params':model.parameters()}, {'params':metric_fc.parameters()}], lr=learning_rate)
        #optimizer_metric = torch.optim.Adagrad(metric_fc.parameters(), lr=learning_rate_center_loss)

    elif optimizer == "rmsprop":
        optimizer_model = torch.optim.RMSprop([{'params':model.parameters()}, {'params':metric_fc.parameters()}], lr=learning_rate)
        #optimizer_metric = torch.optim.RMSprop(metric_fc.parameters(), lr=learning_rate_center_loss)

    elif optimizer == "adam":
        optimizer_model = torch.optim.Adam([{'params':model.parameters()}, {'params':metric_fc.parameters()}], lr=learning_rate)
        #optimizer_metric = torch.optim.Adam(metric_fc.parameters(), lr=learning_rate_center_loss)

    # Set learning rate decay scheduler
    learning_rate_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer_model,
        milestones=[150, 225],
        gamma=0.1
    )

    # Optionally resume from a checkpoint
    if resume_path:

        if os.path.isfile(resume_path):
            print("\nLoading checkpoint {} ...".format(resume_path))

            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']

            # In order to load state dict for optimizers correctly, model has to be loaded to gpu first
            if flag_train_multi_gpu:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])

            optimizer_model.load_state_dict(checkpoint['optimizer_model_state_dict'])
            #optimizer_metric.load_state_dict(checkpoint['optimizer_metric_state_dict'])
            learning_rate_scheduler.load_state_dict(checkpoint['learning_rate_scheduler_state_dict'])

            print("\nCheckpoint loaded: start epoch from checkpoint = {}\nRunning for {} epochs.\n".format(
                    start_epoch,
                    epochs-start_epoch
                )
            )
        else:
            print("WARNING: No checkpoint found at {}!\nTraining from scratch.".format(resume_path))

    # Start Training loop
    print("\nTraining using cross entropy loss with center loss starting for {} epochs:\n".format(epochs-start_epoch))

    total_time_start = time.time()
    start_epoch = start_epoch
    end_epoch = start_epoch + epochs

    BATCH_NUM = len(train_dataloader.dataset) / batch_size
    APD_BATCH_NUM = len(apd_dataloader.dataset) / apd_batch_size

    for epoch in range(start_epoch, end_epoch):
        epoch_time_start = time.time()

        #flag_validate_apd = (epoch + 1) % lfw_validation_epoch_interval == 0 or (epoch + 1) % epochs == 0
        flag_validate_apd = True
        train_loss_sum = 0
        validation_loss_sum = 0

        # Training the model
        model.train()
        #metric_fc.train()
        learning_rate_scheduler.step()
        #progress_bar = enumerate(tqdm(train_dataloader))
        progress_bar = enumerate(train_dataloader)
        
        for batch_index, (data, labels) in progress_bar:
            
            data, labels = data.cuda(), labels.cuda()
            print(data.shape)
            #print('label', labels)

            # Forward pass
            #if flag_train_multi_gpu:
            #    embedding, logits = model.module.forward_training(data)
            #else:
            #    embedding, logits = model.forward_training(data)

            feature = model(data) ### embedding size 
            ### use ArcMargin to get output
            output = metric_fc(feature, labels)
            ### use CrossEntropy to calculate loss between output and label
            # Calculate losses
            cross_entropy_loss = criterion_crossentropy(output.cuda(), labels.cuda())


            #center_loss = criterion_centerloss(embedding, labels)
            #loss = (center_loss * center_loss_weight) + cross_entropy_loss
            loss = cross_entropy_loss
            logging.info("epoch:{}/{} batch_idx:{}/{} loss:{}".format(epoch, end_epoch, batch_index, BATCH_NUM, loss))

            # Backward pass
            #optimizer_metric.zero_grad()
            optimizer_model.zero_grad()
            loss.backward()
            #optimizer_metric.step()
            optimizer_model.step()

            # Remove center_loss_weight impact on the learning of center vectors
            #for param in criterion_centerloss.parameters():
            #    param.grad.data *= (1. / center_loss_weight)

            # Update training loss sum
            train_loss_sum += loss.item()*data.size(0)
            #if batch_index == 20:
            #    break

        # Calculate average losses in epoch
        avg_train_loss = train_loss_sum / len(train_dataloader.dataset)
        """
        avg_validation_loss = validation_loss_sum / len(validation_dataloader.dataset)
        """

        # Calculate training performance statistics in epoch
        #classification_accuracy = correct * 100. / total
        #classification_error = 100. - classification_accuracy

        epoch_time_end = time.time()

        print('Epoch {}:\t Average Training Loss: {:.4f}\t'.format(epoch+1, avg_train_loss))

        with open('logs/{}_log_margin.txt'.format(model_architecture), 'a') as f:
            val_list = [
                epoch+1,
                avg_train_loss
                #avg_validation_loss,
                #classification_accuracy.item(),
                #classification_error.item()
            ]
            log = '\t'.join(str(value) for value in val_list)
            f.writelines(log + '\n')

        try:
            # Plot plot for Cross Entropy Loss and Center Loss on training and validation sets
            plot_training_validation_losses_center(
                log_dir="logs/{}_log_margin.txt".format(model_architecture),
                epochs=epochs,
                figure_name="plots/training_validation_losses_{}_margin.png".format(model_architecture)
            )
        except Exception as e:
            print(e)

        # Validating on LFW dataset using KFold based on Euclidean distance metric
        if flag_validate_apd:

            model.eval()
            with torch.no_grad():
                l2_distance = PairwiseDistance(2).cuda()
                distances, labels = [], []

                print("Validating on APD! ...")
                #progress_bar = enumerate(tqdm(apd_dataloader))
                progress_bar = enumerate(apd_dataloader)


                for batch_index, (data_a, data_b, label) in progress_bar:
                    logging.info("epoch:{}/{} batch_idx:{}/{}".format(epoch, end_epoch, batch_index, APD_BATCH_NUM))
                    data_a, data_b, label = data_a.cuda(), data_b.cuda(), label.cuda()

                    output_a, output_b = model(data_a), model(data_b)
                    ### using cosine similarity
                    
                    cos = CosineSimilarity(dim=1)
                    cosine = cos(output_a, output_b)
                    #print('cosine', cosine.shape)
                    #print('cosine',cosine)

                    theta = torch.acos(cosine)
                    #print('theta', theta)
                    
                    distance = theta * 180 / math.pi

                    #print('output_a.shape', output_a.shape,'output_b.shape', output_b.shape)
                    #distancel2 = l2_distance.forward(output_a, output_b)  # Euclidean distance
                    #distance = distance.squeeze()
                    #print(distancel2)
                    #print(distance)
                    
                    #print(label)

                    distances.append(distance.cpu().detach().numpy())
                    #distances.append(distancel2.cpu().detach().numpy())
                    labels.append(label.cpu().detach().numpy())
                #exit(0)
                labels = np.array([sublabel for label in labels for sublabel in label])
                distances = np.array([subdist for distance in distances for subdist in distance])
                
                true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
                    tar, far = evaluate_lfw(
                        distances=distances,
                        labels=labels
                     )
                # Print statistics and add to log
                print("Accuracy on LFW: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\tROC Area Under Curve: {:.4f}\tBest distance threshold: {:.2f}+-{:.2f}\tTAR: {:.4f}+-{:.4f} @ FAR: {:.4f}".format(
                        np.mean(accuracy),
                        np.std(accuracy),
                        np.mean(precision),
                        np.std(precision),
                        np.mean(recall),
                        np.std(recall),
                        roc_auc,
                        np.mean(best_distances),
                        np.std(best_distances),
                        np.mean(tar),
                        np.std(tar),
                        np.mean(far)
                    )
                )
                with open('logs/lfw_{}_log_margin.txt'.format(model_architecture), 'a') as f:
                    val_list = [
                        epoch + 1,
                        np.mean(accuracy),
                        np.std(accuracy),
                        np.mean(precision),
                        np.std(precision),
                        np.mean(recall),
                        np.std(recall),
                        roc_auc,
                        np.mean(best_distances),
                        np.std(best_distances),
                        np.mean(tar)
                    ]
                    log = '\t'.join(str(value) for value in val_list)
                    f.writelines(log + '\n')

            try:
                # Plot ROC curve
                plot_roc_lfw(
                    false_positive_rate=false_positive_rate,
                    true_positive_rate=true_positive_rate,
                    figure_name="plots/roc_plots_margin/roc_{}_epoch_{}_margin.png".format(model_architecture, epoch+1),
                    epochNum = epoch
                )
                # Plot LFW accuracies plot
                plot_accuracy_lfw(
                    log_dir="logs/lfw_{}_log_margin.txt".format(model_architecture),
                    epochs=epochs,
                    figure_name="plots/lfw_accuracies_{}_margin.png".format(model_architecture)
                )
            except Exception as e:
                print(e)

        # Save model checkpoint
        state = {
            'epoch': epoch+1,
            'num_classes': num_classes,
            'embedding_dimension': embedding_dimension,
            'batch_size_training': batch_size,
            'model_state_dict': model.state_dict(),
            'model_architecture': model_architecture,
            'optimizer_model_state_dict': optimizer_model.state_dict(),
            #'optimizer_metric_state_dict': optimizer_metric.state_dict(),
            'learning_rate_scheduler_state_dict': learning_rate_scheduler.state_dict()
        }

        # For storing data parallel model's state dictionary without 'module' parameter
        if flag_train_multi_gpu:
            state['model_state_dict'] = model.module.state_dict()

        # For storing best euclidean distance threshold during LFW validation
        if flag_validate_apd:
            state['best_distance_threshold'] = np.mean(best_distances)

        # Save model checkpoint
        torch.save(state, 'margin_checkpoints/model_{}_margin_epoch_{}.pt'.format(model_architecture, epoch+1))

    # Training loop end
    total_time_end = time.time()
    total_time_elapsed = total_time_end - total_time_start
    print("\nTraining finished: total time elapsed: {:.2f} hours.".format(total_time_elapsed/3600))


if __name__ == '__main__':
    main()
