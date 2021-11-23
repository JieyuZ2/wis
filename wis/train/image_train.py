import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from PIL import Image
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

from ..dataset import ImageDataset
from ..utils import cross_entropy_with_probs

logger = logging.getLogger(__name__)

MODEL_LIST = ["resnet", "alexnet", "vgg", "squeezenet", "googlenet"]


def pil_loader(path: [str, Path]):
    """pil loader very simple

    Args:
        path: default picture loader for CV datasets

    Returns:
        image object
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def _format_transform_repr(transform, head):
    lines = transform.__repr__().splitlines()
    return (["{}{}".format(head, lines[0])] +
            ["{}{}".format(" " * len(head), line) for line in lines[1:]])


class StandardTransform(object):
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

    def __call__(self, x, target):
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return x, target

    def __repr__(self):
        body = [self.__class__.__name__]
        if self.transform is not None:
            body += _format_transform_repr(self.transform, "Transform: ")
        if self.target_transform is not None:
            body += _format_transform_repr(self.target_transform, "Target transform: ")

        return '\n'.join(body)


class TrainingDataset(Dataset):
    """
    Training Dataset similar to VisionDataset

    Attributes:
        data:           list of data
        classes:        list of labels
        class_to_idx:   dictionary label:label index in classes
        loader:         loader to load the data
        transform:      transformer
        targets:        list of numbered labels
    """

    def __init__(self,
                 dataset: ImageDataset,
                 imagenet_root,
                 y: np.ndarray = None,
                 loader=pil_loader,
                 transform=None,
                 stage='train',
                 preload=True
                 ):
        self.stage = stage
        imagenet_root = Path(imagenet_root)
        image2path = dataset.image2path
        self.images = [imagenet_root / image2path[i] for i in dataset.raw_data]
        self.imagenet_root = imagenet_root
        self.loader = loader
        self.transform = transform
        self.transforms = StandardTransform(transform, None)

        if y is None:
            self.targets = dataset.y
        else:
            self.targets = y

        self.preload = preload
        if preload:
            self.samples = [self.loader(image) for image in self.images]

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        target = self.targets[index]
        if self.preload:
            sample = self.samples[index]
        else:
            image = self.images[index]
            sample = self.loader(image)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.targets)


def initialize_model(model_name, num_classes):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet34()
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 244

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet()
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn()
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_1()
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "googlenet":
        """ GoogLeNet
        """
        model_ft = models.googlenet()
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the auxilary net 1
        num_ftrs = model_ft.aux1.fc2.in_features
        model_ft.aux1.fc2 = nn.Linear(num_ftrs, num_classes)
        # Handle the auxilary net 2
        num_ftrs = model_ft.aux2.fc2.in_features
        model_ft.aux2.fc2 = nn.Linear(num_ftrs, num_classes)
        input_size = 224


    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121()
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3()
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        logger.info("Invalid model name, exiting...")
        exit()
    # logger.info("Model Structure")
    # logger.info(model_ft)
    return model_ft, input_size


def train_model(model, device, dataloaders, optimizer, lr_scheduler, num_epochs=200, patience=200, is_inception=False):
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    not_improve_cnt = 0
    early_stop_flag = False
    for epoch in range(num_epochs):
        log = f'epoch {epoch}: '
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            # for inputs, labels in tqdm(dataloaders[phase],desc=f"{epoch_info} {phase}"):
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux1_outputs, aux2_outputs = model(inputs)
                        loss1 = cross_entropy_with_probs(outputs, labels)
                        loss2 = cross_entropy_with_probs(aux1_outputs, labels)
                        loss3 = cross_entropy_with_probs(aux2_outputs, labels)
                        loss = loss1 + 0.4 * loss2 + 0.4 * loss3
                    else:
                        outputs = model(inputs)
                        loss = cross_entropy_with_probs(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                if labels.ndim == 2:
                    running_corrects += torch.sum(preds == labels.data.argmax(dim=1))
                else:
                    running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            log += f'{phase} loss: {epoch_loss:.4f} {phase} acc: {epoch_acc:.4f} |'

            # deep copy the model
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                if epoch_acc > best_acc:
                    best_epoch = epoch
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    not_improve_cnt = 0
                else:
                    not_improve_cnt += 1
                    if not_improve_cnt > patience:
                        early_stop_flag = True
        logger.info(log)
        if early_stop_flag:
            break

    logger.info('Best val Acc: {:4f} @ {:d}'.format(best_acc, best_epoch))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def train_imagenet(
        data_train: ImageDataset,
        data_val: ImageDataset,
        imagenet_root,
        y: np.ndarray = None,
        gpu=0,
        lr=0.001,
        batch_size=256,
        num_epochs=200,
        input_size=224,
        model_name='resnet'):
    device = torch.device(f"cuda:{gpu}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val'  : transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_dataset = TrainingDataset(
        dataset=data_train,
        y=y,
        imagenet_root=imagenet_root,
        transform=data_transforms['train'],
        stage='train')
    val_dataset = TrainingDataset(
        dataset=data_val,
        imagenet_root=imagenet_root,
        transform=data_transforms['val'], stage='val')
    dataloaders_dict = {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8),
        'val'  : DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
    }

    num_classes = len(data_train.classes)
    model_ft, input_size = initialize_model(model_name, num_classes)
    model_ft = model_ft.to(device)
    params_to_update = model_ft.parameters()

    optimizer_ft = optim.Adam(params_to_update, lr=lr, weight_decay=5e-5)
    scheduler = StepLR(optimizer_ft, step_size=1, gamma=0.7)
    model_ft, hist = train_model(model_ft, device, dataloaders_dict, optimizer_ft, scheduler,
                                 num_epochs=num_epochs, is_inception=(model_name == "googlenet"))

    return model_ft


def train_imagenet_multiple(
        data_train: ImageDataset,
        data_val: ImageDataset,
        imagenet_root,
        y_train: np.ndarray,
        y_valid: np.ndarray,
        gpu=0,
        lr=0.001,
        batch_size=256,
        num_epochs=100,
        input_size=224,
        model_name='resnet'):
    device = torch.device(f"cuda:{gpu}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val'  : transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    classifier = []
    m = y_train.shape[1]

    for i in range(m):
        train_dataset = TrainingDataset(
            dataset=data_train,
            y=y_train[:, i],
            imagenet_root=imagenet_root,
            transform=data_transforms['train'],
            stage='train')
        data_val_i = copy.deepcopy(data_val)
        data_val_i.y = y_valid[:, i]
        val_dataset = TrainingDataset(
            dataset=data_val,
            imagenet_root=imagenet_root,
            transform=data_transforms['val'], stage='val')
        dataloaders_dict = {
            'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=8),
            'val'  : DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8)
        }

        num_classes = len(data_train.classes)
        model_ft, input_size = initialize_model(model_name, num_classes)
        model_ft = model_ft.to(device)
        params_to_update = model_ft.parameters()

        optimizer_ft = optim.Adam(params_to_update, lr=lr, weight_decay=5e-5)
        scheduler = StepLR(optimizer_ft, step_size=1, gamma=0.7)
        model_ft, hist = train_model(model_ft, device, dataloaders_dict, optimizer_ft, scheduler,
                                     num_epochs=num_epochs, patience=10, is_inception=(model_name == "googlenet"))

        classifier.append(model_ft)

    return classifier


def apply_image_multiple(models, dataset, desired_class_to_attributes, imagenet_root):
    n_class = len(desired_class_to_attributes)
    n_data = len(dataset)
    A = np.zeros((n_data, len(models)))
    for i, model in enumerate(models):
        if model is not None:
            probas, preds = test_imagenet(model, dataset, imagenet_root)
            A[:, i] = probas[:, 1]
        else:
            A[:, i] = 0.5

    proba = np.zeros((n_data, n_class))
    for i, a in enumerate(desired_class_to_attributes):
        Ai = np.prod(A[:, a.astype(bool)], axis=1) * np.prod((1 - A)[:, (1 - a).astype(bool)], axis=1)
        proba[:, i] = Ai

    proba /= proba.sum(1, keepdims=True)
    proba = np.nan_to_num(proba)
    for i, p in enumerate(proba):
        if p.sum() == 0:
            proba[i] = 1 / n_class

    return proba


def test_imagenet(
        model,
        data_test: ImageDataset,
        imagenet_root,
        gpu=0,
        batch_size=256,
        input_size=224,
        convert_pred=False,
):
    device = torch.device(f"cuda:{gpu}")
    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    with torch.no_grad():

        test_dataset = TrainingDataset(
            dataset=data_test,
            imagenet_root=imagenet_root,
            transform=transform,
            stage='test')
        dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)

        model_ft = model.to(device)

        model_ft.eval()
        predicts = []
        for data, label in dataloader:
            data = data.to(device)
            logits = model_ft(data)
            predicts.append(logits)
        predicts = torch.cat(predicts).tolist()

    probas = np.array(predicts)
    preds = np.argmax(probas, axis=1)
    if convert_pred:
        preds = np.array(data_test.classes)[preds]
    return probas, preds
