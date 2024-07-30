import sys
import argparse

from torchvision import transforms

from segmentation.data_loader.segmentation_dataset import SegmentationDataset
from segmentation.data_loader.transform import Rescale, ToTensor
from segmentation.trainer import Trainer
from segmentation.predict import *
from segmentation.models import all_models
from util.logger import Logger

'''
Reference Code for Dataset Folder

train_images = r'dataset/cityspaces/images/train'
test_images = r'dataset/cityspaces/images/test'
train_labled = r'dataset/cityspaces/labeled/train'
test_labeled = r'dataset/cityspaces/labeled/test'
'''

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--modelname', default="fcn8_vgg16",
                        help='model name')
    parser.add_argument('--cuda', default="cuda:0", type=str,
                        help='cuda:0 or cuda:1')
    parser.add_argument('--batchsize', default=4, type=int,
                        help='batch size')
    parser.add_argument('--nclasses', default=34, type=int,
                        help='number of classes')
    parser.add_argument('--epoch', default=101, type=int,
                        help='eopch to train')
    parser.add_argument('--dataset', default="sperm", type=str,
                        help='sperm: Sperm or city: CitySpaces')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    
    '''
        Config of CitySpaces Dataset
    '''
    model_name = args.modelname
    device = args.cuda
    batch_size = args.batchsize
    # 34 for CitySpaces; 5 for Sperm
    n_classes = args.nclasses
    num_epochs = args.epoch
    image_axis_minimum_size = 200
    pretrained = True
    fixed_feature = False
    
    '''
        Select Dataset
    '''
    if args.dataset == "sperm":
        '''
            Sperm in/out-focus head 5 classes Dataset
        '''
        train_images = r'data/sperm/sperm-inout-focus-splited-dataset/images/train'
        test_images = r'data/sperm/sperm-inout-focus-splited-dataset/images/val'
        train_labled = r'data/sperm/sperm-inout-focus-splited-dataset/labeled/train'
        test_labeled = r'data/sperm/sperm-inout-focus-splited-dataset/labeled/val'
    elif args.dataset == "city":
        '''
            CitySpace Dataset
        '''
        train_images = r'data/cityscapes/cityscapes_dataset/images/train'
        test_images = r'data/cityscapes/cityscapes_dataset/images/val'
        train_labled = r'data/cityscapes/cityscapes_dataset/labeled/train'
        test_labeled = r'data/cityscapes/cityscapes_dataset/labeled/val'
    else:
        print("args.dataset is invaild!!!!")
        sys.exit(1)

    
    """
        Save check point.
        Please check the runs folder, ./segmentation/runs/models
    """
    check_point_stride = 30 # the checkpoint is saved for every 30 epochs.

    logger = Logger(model_name=model_name, data_name='example')

    ### Loader
    compose = transforms.Compose([
        # Rescale(image_axis_minimum_size),
        Rescale((1200, 1920)),
        # transforms.Resize((960, 1280)),  # 调整图像大小到 1920x1200
        ToTensor()
         ])

    train_datasets = SegmentationDataset(train_images, train_labled, n_classes, compose)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    test_datasets = SegmentationDataset(test_images, test_labeled, n_classes, compose)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    ### Model
    model = all_models.model_from_name[model_name](n_classes, batch_size,
                                                   pretrained=pretrained,
                                                   fixed_feature=fixed_feature)
    model.to(device)

    ###Load model
    ###please check the foloder: (.segmentation/test/runs/models)
    #logger.load_model(model, 'epoch_15')

    ### Optimizers
    if pretrained and fixed_feature: #fine tunning
        params_to_update = model.parameters()
        print("Params to learn:")
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
        optimizer = torch.optim.Adadelta(params_to_update)
    else:
        optimizer = torch.optim.Adadelta(model.parameters())

    ### Train
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    trainer = Trainer(model, optimizer, logger, num_epochs, train_loader, test_loader, check_point_epoch_stride=check_point_stride, scheduler=scheduler)
    trainer.train()


    #### Writing the predict result.
    # predict(model, r'segmentation/test/dataset/cityspaces/input.png',
    #          r'segmentation/test/dataset/cityspaces/output.png')
    
    predict(model, r"data/sperm/sperm-inout-focus-splited-dataset/images/test/143.png",
            r"data/sperm/sperm-inout-focus-splited-dataset/143-output.png")
    