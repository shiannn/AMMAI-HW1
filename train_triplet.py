import logging
import numpy as np
import time
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance
from dataloaders.APDDataset import APDDataset
from losses.triplet_loss import TripletLoss
from dataloaders.triplet_loss_dataloader import TripletFaceDataset
from validate_on_LFW import evaluate_lfw
from plots import plot_roc_lfw, plot_accuracy_lfw, plot_triplet_losses
from tqdm import tqdm
from models.resnet18 import Resnet18Triplet
from models.resnet34 import Resnet34Triplet
from models.resnet50 import Resnet50Triplet
from models.resnet101 import Resnet101Triplet
from models.inceptionresnetv2 import InceptionResnetV2Triplet
from pathlib import Path
import shutil


logging.basicConfig(filename="training.log", level=logging.INFO)
shutil.rmtree("plots/roc_plots_triplet")
rocFolder = Path("plots") / Path("roc_plots_triplet")
rocFolder.mkdir(exist_ok=True)

try:
    os.remove("plots/triplet_losses_resnet34.png")
    os.remove("plots/apd_accuracies_resnet34_triplet.png")
    os.remove("logs/resnet34_log_triplet.txt")
except OSError:
    pass

parser = argparse.ArgumentParser(description="Training FaceNet facial recognition model using Triplet Loss.")
# Dataset
parser.add_argument('--dataroot', '-d', type=str, required=True,
                    help="(REQUIRED) Absolute path to the dataset folder"
                    )
# APD
parser.add_argument('--apd', type=str, required=True,
                    help="(REQUIRED) Absolute path to the labeled faces in the wild dataset folder"
                    )
parser.add_argument('--dataset_csv', type=str, default='datasets/vggface2_full.csv',
                    help="Path to the csv file containing the image paths of the training dataset."
                    )
parser.add_argument('--apd_batch_size', default=32, type=int,
                    help="Batch size for APD dataset (default: 64)"
                    )
parser.add_argument('--apd_validation_epoch_interval', default=1, type=int,
                    help="Perform APD validation every n epoch interval (default: every 1 epoch)"
                    )
# Training settings
parser.add_argument('--model', type=str, default="resnet34", choices=["resnet18", "resnet34", "resnet50", "resnet101", "inceptionresnetv2"],
    help="The required model architecture for training: ('resnet18','resnet34', 'resnet50', 'resnet101', 'inceptionresnetv2'), (default: 'resnet34')"
                    )
parser.add_argument('--epochs', default=30, type=int,
                    help="Required training epochs (default: 30)"
                    )
parser.add_argument('--training_triplets_path', default=None, type=str,
    help="Path to training triplets numpy file in 'datasets/' folder to skip training triplet generation step."
                    )
parser.add_argument('--num_triplets_train', default=100000, type=int,
                    help="Number of triplets for training (default: 100000)"
                    )
parser.add_argument('--resume_path', default='',  type=str,
    help='path to latest model checkpoint: (Model_training_checkpoints/model_resnet34_epoch_0.pt file) (default: None)'
                    )
parser.add_argument('--batch_size', default=64, type=int,
                    help="Batch size (default: 64)"
                    )
parser.add_argument('--num_workers', default=4, type=int,
                    help="Number of workers for data loaders (default: 4)"
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
parser.add_argument('--margin', default=0.5, type=float,
                    help='margin for triplet loss (default: 0.5)'
                    )
args = parser.parse_args()


def main():
    dataroot = args.dataroot
    apd_dataroot = args.apd
    dataset_csv = args.dataset_csv
    apd_batch_size = args.apd_batch_size
    apd_validation_epoch_interval = args.apd_validation_epoch_interval
    model_architecture = args.model
    epochs = args.epochs
    training_triplets_path = args.training_triplets_path
    num_triplets_train = args.num_triplets_train
    resume_path = args.resume_path
    batch_size = args.batch_size
    num_workers = args.num_workers
    embedding_dimension = args.embedding_dim
    pretrained = args.pretrained
    optimizer = args.optimizer
    learning_rate = args.lr
    margin = args.margin
    start_epoch = 0

    # Define image data pre-processing transforms
    #   ToTensor() normalizes pixel values between [0, 1]
    #   Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) normalizes pixel values between [-1, 1]

    #  Size 182x182 RGB image -> Center crop size 160x160 RGB image for more model generalization
    data_transforms = transforms.Compose([
        #transforms.RandomCrop(size=10),
        transforms.Resize(size=(50,50)),
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

    logging.info('prepare VGGNet dataloader...')
    # Set dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        dataset=TripletFaceDataset(
            root_dir=dataroot,
            csv_name=dataset_csv,
            num_triplets=num_triplets_train,
            training_triplets_path=training_triplets_path,
            transform=data_transforms
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False
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

    logging.info('prepare apd loader')
    apd_dataloader = torch.utils.data.DataLoader(
        dataset=apdDataset,
        batch_size=apd_batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    print('apd_dataloader', apd_dataloader)
    
    logging.info('prepare model')
    # Instantiate model
    if model_architecture == "resnet18":
        model = Resnet18Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet34":
        model = Resnet34Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet50":
        model = Resnet50Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet101":
        model = Resnet101Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "inceptionresnetv2":
        model = InceptionResnetV2Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    print("Using {} model architecture.".format(model_architecture))

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

    logging.info('set optimizer')
    # Set optimizers
    if optimizer == "sgd":
        optimizer_model = torch.optim.SGD(model.parameters(), lr=learning_rate)
        
    elif optimizer == "adagrad":
        optimizer_model = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
        
    elif optimizer == "rmsprop":
        optimizer_model = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        
    elif optimizer == "adam":
        optimizer_model = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
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

            print("\nCheckpoint loaded: start epoch from checkpoint = {}\nRunning for {} epochs.\n".format(
                    start_epoch,
                    epochs-start_epoch
                )
            )
        else:
            print("WARNING: No checkpoint found at {}!\nTraining from scratch.".format(resume_path))

    # Start Training loop
    print("\nTraining using triplet loss on {} triplets starting for {} epochs:\n".format(
            num_triplets_train,
            epochs-start_epoch
        )
    )

    total_time_start = time.time()
    start_epoch = start_epoch
    end_epoch = start_epoch + epochs
    l2_distance = PairwiseDistance(2).cuda()

    BATCH_NUM = len(train_dataloader.dataset) / batch_size
    for epoch in range(start_epoch, end_epoch):
        epoch_time_start = time.time()

        #flag_validate_apd = (epoch + 1) % lfw_validation_epoch_interval == 0 or (epoch + 1) % epochs == 0
        flag_validate_apd = True

        triplet_loss_sum = 0
        num_valid_training_triplets = 0

        # Training pass
        model.train()
        #progress_bar = enumerate(tqdm(train_dataloader))
        progress_bar = enumerate(train_dataloader)

        for batch_idx, (batch_sample) in progress_bar:
            #break
            logging.info("epoch:{}/{} batch_idx:{}/{}".format(epoch, end_epoch, batch_idx, BATCH_NUM))
            #print('batch_idx',batch_idx)

            anc_img = batch_sample['anc_img'].cuda()
            pos_img = batch_sample['pos_img'].cuda()
            neg_img = batch_sample['neg_img'].cuda()

            # Forward pass - compute embeddings
            anc_embedding, pos_embedding, neg_embedding = model(anc_img), model(pos_img), model(neg_img)

            # Forward pass - choose hard negatives only for training
            pos_dist = l2_distance.forward(anc_embedding, pos_embedding)
            neg_dist = l2_distance.forward(anc_embedding, neg_embedding)

            all = (neg_dist - pos_dist < margin).cpu().numpy().flatten()

            hard_triplets = np.where(all == 1)
            if len(hard_triplets[0]) == 0:
                continue

            anc_hard_embedding = anc_embedding[hard_triplets].cuda()
            pos_hard_embedding = pos_embedding[hard_triplets].cuda()
            neg_hard_embedding = neg_embedding[hard_triplets].cuda()

            # Calculate triplet loss
            triplet_loss = TripletLoss(margin=margin).forward(
                anchor=anc_hard_embedding,
                positive=pos_hard_embedding,
                negative=neg_hard_embedding
            ).cuda()

            # Calculating loss
            triplet_loss_sum += triplet_loss.item()
            num_valid_training_triplets += len(anc_hard_embedding)

            # Backward pass
            optimizer_model.zero_grad()
            triplet_loss.backward()
            optimizer_model.step()

            #if batch_idx == 20:
            #    break

        # Model only trains on hard negative triplets
        avg_triplet_loss = 0 if (num_valid_training_triplets == 0) else triplet_loss_sum / num_valid_training_triplets
        epoch_time_end = time.time()

        # Print training statistics and add to log
        print('Epoch {}:\tAverage Triplet Loss: {:.4f}\tEpoch Time: {:.3f} hours\tNumber of valid training triplets in epoch: {}'.format(
                epoch+1,
                avg_triplet_loss,
                (epoch_time_end - epoch_time_start)/3600,
                num_valid_training_triplets
            )
        )
        
        with open('logs/{}_log_triplet.txt'.format(model_architecture), 'a') as f:
            val_list = [
                epoch+1,
                avg_triplet_loss,
                num_valid_training_triplets
            ]
            log = '\t'.join(str(value) for value in val_list)
            f.writelines(log + '\n')

        try:
            # Plot Triplet losses plot
            plot_triplet_losses(
                log_dir="logs/{}_log_triplet.txt".format(model_architecture),
                epochs=epochs,
                figure_name="plots/triplet_losses_{}.png".format(model_architecture)
            )
        except Exception as e:
            print(e)

        # Evaluation pass on LFW dataset
        if flag_validate_apd:

            model.eval()
            with torch.no_grad():
                distances, labels = [], []

                print("Validating on APD! ...")
                progress_bar = enumerate(tqdm(apd_dataloader))

                for batch_index, (data_a, data_b, label) in progress_bar:
                    data_a, data_b, label = data_a.cuda(), data_b.cuda(), label.cuda()
                    #print('data_a', data_a.shape)
                    #print('data_b', data_b.shape)
                    #print('label', label)

                    output_a, output_b = model(data_a), model(data_b)
                    #print('output_a',output_a)
                    #print('output_b',output_b)
                    
                    distance = l2_distance.forward(output_a, output_b)  # Euclidean distance
                    #print('distance',distance)
                    
                    distances.append(distance.cpu().detach().numpy())
                    labels.append(label.cpu().detach().numpy())
                    #if batch_index == 20:
                    #    break

                labels = np.array([sublabel for label in labels for sublabel in label])
                distances = np.array([subdist for distance in distances for subdist in distance])
                print('len(labels)', len(labels))
                print('len(distances)', len(distances))

                
                true_positive_rate, false_positive_rate, precision, recall, accuracy, roc_auc, best_distances, \
                    tar, far = evaluate_lfw(
                        distances=distances,
                        labels=labels
                    )

                # Print statistics and add to log
                print("Accuracy on APD: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\tROC Area Under Curve: {:.4f}\tBest distance threshold: {:.2f}+-{:.2f}\tTAR: {:.4f}+-{:.4f} @ FAR: {:.4f}".format(
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

                with open('logs/lfw_{}_log_triplet.txt'.format(model_architecture), 'a') as f:
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
                    figure_name="plots/roc_plots_triplet/roc_{}_epoch_{}_triplet.png".format(model_architecture, epoch+1),
                    epochNum = epoch
                )
                # Plot LFW accuracies plot
                plot_accuracy_lfw(
                    log_dir="logs/lfw_{}_log_triplet.txt".format(model_architecture),
                    epochs=epochs,
                    figure_name="plots/apd_accuracies_{}_triplet.png".format(model_architecture)
                )
            except Exception as e:
                print(e)

        # Save model checkpoint
        state = {
            'epoch': epoch+1,
            'embedding_dimension': embedding_dimension,
            'batch_size_training': batch_size,
            'model_state_dict': model.state_dict(),
            'model_architecture': model_architecture,
            'optimizer_model_state_dict': optimizer_model.state_dict()
        }

        # For storing data parallel model's state dictionary without 'module' parameter
        if flag_train_multi_gpu:
            state['model_state_dict'] = model.module.state_dict()

        # For storing best euclidean distance threshold during LFW validation
        if flag_validate_apd:
            state['best_distance_threshold'] = np.mean(best_distances)

        # Save model checkpoint
        torch.save(state, 'Model_training_checkpoints/model_{}_triplet_epoch_{}.pt'.format(model_architecture, epoch+1))

    # Training loop end
    total_time_end = time.time()
    total_time_elapsed = total_time_end - total_time_start
    print("\nTraining finished: total time elapsed: {:.2f} hours.".format(total_time_elapsed/3600))


if __name__ == '__main__':
    main()
