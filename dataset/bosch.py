import torch, torchvision
import numpy as np
from torch.utils.data import Dataset
import yaml, os, sys, time, io
from PIL import Image

# BOSCH_ROOT = os.path.join(HOME, 'Bosch_Dataset/')
INDEX2LABEL = {0: 'None', 1: 'Green', 2: 'GreenStraightLeft', 3: 'GreenLeft', 4: 'RedLeft', 5: 'GreenStraightRight',
               6: 'Red', 7: 'off', \
               8: 'GreenRight', 9: 'GreenStraight', 10: 'Yellow', 11: 'RedRight', 12: 'RedStraight',
               13: 'RedStraightLeft'}
BOSCH_CLASSES = ['None', 'Green', 'GreenStraightLeft', 'GreenLeft', 'RedLeft', 'GreenStraightRight', 'Red', 'off', \
                 'GreenRight', 'GreenStraight', 'Yellow', 'RedRight', 'RedStraight', 'RedStraightLeft']


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)


class BoschTrainDetection(Dataset):
    decode_yaml_file = None
    color2index = {}
    index2color = {}
    class_names = BOSCH_CLASSES

    def __init__(self, root_dir, yaml_file, transform=None, target_transform=None):
        self.root_dir = root_dir
        self.yaml_file = yaml_file
        self.transform = transform
        self.target_transform = target_transform

        if not os.path.exists(os.path.join(self.root_dir, self.yaml_file)):
            print(os.path.join(self.root_dir, self.yaml_file))
            print("input path parameters not valid.")
            return None

        self.color2index = {}
        self.index2color = {}
        color_array = []

        # return as a dict
        self.decode_yaml_file = yaml.load(open(os.path.join(self.root_dir, self.yaml_file)))
        self.decode_yaml_file = list(filter(lambda x: len(x['boxes']) != 0, self.decode_yaml_file))
        for item in self.decode_yaml_file:
            for box in item['boxes']:
                if box['label'] not in color_array:
                    color_array.append(box['label'])

        for index in range(len(color_array)):
            self.color2index[color_array[index]] = index + 1
            self.index2color[index + 1] = color_array[index]
        self.color2index['bg'] = 0
        self.index2color[0] = 'bg'

        self.threshold = 0.5
        self.variance = [0.1, 0.2]

        self.decode_dataset = []

        count = 0

        for element in self.decode_yaml_file:
            if len(element['boxes']) == 0:
                continue
            dataset_element = {}
            dataset_element['path'] = element['path']
            dataset_element['annotation'] = []
            for box_info in element['boxes']:
                dataset_element['annotation'].append(np.array(
                    [box_info['x_min'], box_info['y_min'], box_info['x_max'], box_info['y_max'],
                     self.color2index[box_info['label']]]).astype(np.float32))
            dataset_element['annotation'] = np.array(dataset_element['annotation']).astype(np.float32)
            # dataset_element['annotation'] = np.transpose(np.array(dataset_element['annotation']), (1, 0))
            self.decode_dataset.append(dataset_element)

            count += 1
            if count == 100:
                break

        image_name = os.path.join(self.root_dir, self.decode_dataset[0]['path'])
        #print("image shape:", np.array(Image.open(image_name)).shape)

        del self.decode_yaml_file

    def __len__(self):
        return len(self.decode_dataset)

    def __getitem__(self, index):
        image_name = os.path.join(self.root_dir, self.decode_dataset[index]['path'])
        image = Image.open(image_name)
        image = np.array(image)
        boxes = self.decode_dataset[index]['annotation'][:, : 4]
        labels = self.decode_dataset[index]['annotation'][:, 4]
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            """
            print(self.priors)
            loc_t = torch.Tensor(1, self.priors.shape[0], 4)
            conf_t = torch.LongTensor(1, self.priors.shape[0])
            match(self.threshold, torch.Tensor(boxes), torch.Tensor(self.priors).float(), self.variance, torch.Tensor(labels), loc_t, conf_t, 0)
            boxes, labels = loc_t[0], conf_t[0]
            """
            boxes, labels = self.target_transform(boxes, labels)
        return image, np.concatenate((boxes, np.expand_dims(labels, -1)), 1)


class BoschTestDetection(Dataset):

    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, index):
        pass
em__(self, index):
        pass
