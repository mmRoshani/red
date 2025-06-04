import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Subset

import torch.utils.data as data

from constants.datasets_constants import DATA_SET_FMNIST, DATA_SET_MNIST, DATA_SET_CIFAR_10, DATA_SET_CIFAR_100, \
    DATA_SET_SVHN, DATA_SET_TINY_IMAGE_NET
from datasets.image.add_gaussian_noise import AddGaussianNoise
from datasets.image.datasets_generator import MNIST_truncated, FEMNIST, FashionMNIST_truncated, SVHN_custom, CIFAR10_truncated, \
    CIFAR100_truncated, ImageFolder_custom, Generated
from utils.check_train_test_class_mismatch import check_train_test_class_mismatch


def get_dataloader(dataset, datadir, train_bs, test_bs, dataidxs=None, noise_level=0, net_id=None, total=0):
    if dataset in (
            DATA_SET_MNIST, 'femnist', DATA_SET_FMNIST, DATA_SET_CIFAR_10, DATA_SET_SVHN, 'generated', 'covtype', 'a9a',
            'rcv1', 'SUSY', DATA_SET_CIFAR_100, DATA_SET_TINY_IMAGE_NET):
        if dataset == DATA_SET_MNIST:
            dl_obj = MNIST_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)
            ])

        elif dataset == 'femnist':  # TODO: add in datasets constants
            dl_obj = FEMNIST
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)
            ])

        elif dataset == DATA_SET_FMNIST:
            dl_obj = FashionMNIST_truncated
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)
            ])

        elif dataset == DATA_SET_SVHN:
            dl_obj = SVHN_custom
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)
            ])


        elif dataset == DATA_SET_CIFAR_10:
            dl_obj = CIFAR10_truncated

            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: F.pad(
                    Variable(x.unsqueeze(0), requires_grad=False),
                    (4, 4, 4, 4), mode='reflect').data.squeeze()),
                transforms.ToPILImage(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)

            ])
            # datasets prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                AddGaussianNoise(0., noise_level, net_id, total)
            ])

        elif dataset == DATA_SET_CIFAR_100:
            dl_obj = CIFAR100_truncated

            normalize = transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                             std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404])
            # transform_train = transforms.Compose([
            #     transforms.RandomCrop(32),
            #     transforms.RandomHorizontalFlip(),
            #     transforms.ToTensor(),
            #     normalize
            # ])
            transform_train = transforms.Compose([
                # transforms.ToPILImage(),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize
            ])
            # datasets prep for test set
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                normalize])
        elif dataset == DATA_SET_TINY_IMAGE_NET:
            dl_obj = ImageFolder_custom
            transform_train = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        else:
            dl_obj = Generated
            transform_train = None
            transform_test = None

        if dataset == DATA_SET_TINY_IMAGE_NET:
            train_ds = dl_obj(datadir + './train/', dataidxs=dataidxs, transform=transform_train)
            test_ds = dl_obj(datadir + './val/', transform=transform_test)
        else:
            train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=transform_train, download=True)
            test_ds = dl_obj(datadir, train=False, transform=transform_test, download=True)

        train_labels = np.array(train_ds.target)

        # Get unique classes present in this partition
        unique_classes = np.unique(train_labels)

        # Filter test set to only include these classes
        test_labels = np.array(test_ds.target)
        test_mask = np.isin(test_labels, unique_classes)
        test_indices = np.where(test_mask)[0]

        # Create subset of test datasets with matching classes
        test_ds_subset = Subset(test_ds, test_indices)

        train_dl = data.DataLoader(dataset=train_ds,
                                   batch_size=train_bs,
                                   shuffle=True,
                                   drop_last=False
                                   )

        test_dl = data.DataLoader(dataset=test_ds_subset,
                                  batch_size=test_bs,
                                  shuffle=False,
                                  drop_last=False
                                  )

        print(f'checking train and test class mismatch')
        mismatch_classes = check_train_test_class_mismatch(train_ds, test_ds_subset)

    return train_dl, test_dl, train_ds, test_ds_subset
