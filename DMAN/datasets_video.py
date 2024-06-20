import os
import torch
import torchvision
import torchvision.datasets as datasets


def return_ucf101():
    root_data = '/share/UCF-101'
    filename_imglist_train = 'train_ucf101.csv'
    filename_imglist_val = 'val_ucf101.csv'
    prefix = "image_{:05d}.jpg"

    return filename_imglist_train, filename_imglist_val, root_data, prefix

#/share2 for shuoshi fuwuqi
def return_something():
    root_data = '/share/something_v1/20bn-something-something-v1'
    filename_imglist_train = '/share/something_v1/train_videofolder.txt'
    filename_imglist_val = '/share/something_v1/val_videofolder.txt'
    prefix = '{:05d}.jpg'

    return filename_imglist_train, filename_imglist_val, root_data, prefix

def return_something_v2():
    root_data = '/share/something-v2_images'
    filename_imglist_train = '/share/something-v2/train_videofolder.txt'
    filename_imglist_val = '/share/something-v2/val_videofolder.txt'
    prefix = '{:05d}.jpg'

    return filename_imglist_train, filename_imglist_val, root_data, prefix
def return_diving48():
    root_data = '/share/diving48_images'
    filename_imglist_train = '/share/diving48_images/train_videofolder.txt'
    filename_imglist_val = '/share/diving48_images/val_videofolder.txt'
    prefix = "img_{:05d}.jpg"

    return filename_imglist_train, filename_imglist_val, root_data, prefix

def return_dataset(dataset):
    dict_single = {'something-v1': return_something, 'diving48':return_diving48, 'ucf101':return_ucf101}
    if dataset in dict_single:
            file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset]()
    else:
        raise ValueError('Unknown dataset '+dataset)

    return file_imglist_train, file_imglist_val, root_data, prefix
