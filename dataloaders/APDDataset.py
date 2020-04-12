import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import os
import numpy as np
from tqdm import tqdm

usingFolder = 'C'

class APDDataset(datasets.ImageFolder):
    def __init__(self, directory, pairs_path, transform=None):

        super(APDDataset, self).__init__(directory, transform)

        self.pairs_path = pairs_path

        # APD dir contains 2 folders: faces and lists
        self.validation_images = self.get_apd_paths(directory)

    def read_apd_pairs(self, pairs_filename):
        pairs = []
        print('pairs_filename',pairs_filename)
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[0:]:
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs)

    def get_apd_paths(self, apd_dir):
        #pairs = self.read_apd_pairs(self.pairs_path)
        pairs = self.read_apd_pairs(os.path.join(apd_dir, self.pairs_path))
        #print(pairs[:10])

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []
        for pair in pairs:
            if len(pair) == 3:
                ### positive pairs
                path0 = self.add_extension(os.path.join(apd_dir, usingFolder, pair[0] + '_' + pair[1]))
                path1 = self.add_extension(os.path.join(apd_dir, usingFolder, pair[0] + '_' + pair[2]))
                issame = True
                #exit(0)
            elif len(pair) == 4:
                ### negative pairs
                #print(pair)
                path0 = self.add_extension(os.path.join(apd_dir, usingFolder, pair[0] + '_' + pair[1]))
                path1 = self.add_extension(os.path.join(apd_dir, usingFolder, pair[2] + '_' + pair[3]))
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                path_list.append((path0, path1, issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list

    # Modified here
    def add_extension(self, path):
        #print(path)
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)

    def __getitem__(self, index):
        """
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        """

        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            img = self.loader(img_path)
            return self.transform(img)

        (path_1, path_2, issame) = self.validation_images[index]
        img1, img2 = transform(path_1), transform(path_2)
        return img1, img2, issame

    def __len__(self):
        return len(self.validation_images)

if __name__ == '__main__':
    apd_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ])

    imgFolder = '.'
    apddataset=APDDataset(
        directory=imgFolder,
        pairs_path='negative_pairs.txt',
        transform=apd_transforms
    )
    print(apddataset)

    apd_dataloader = torch.utils.data.DataLoader(
        dataset=apddataset,
        batch_size=2,
        num_workers=2,
        shuffle=False
    )

    progress_bar = enumerate(apd_dataloader)
    for batch_index, (data_a, data_b, label) in progress_bar:
        if batch_index == 4:
            break
        print(data_a.shape)
        print(data_b.shape)