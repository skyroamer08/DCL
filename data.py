import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import clip
from PIL import Image

def id2name(id_list, dataset=None, subset=None, dataset_root=None):
    # id_list中包含按num_ann次数重复的图片id
    name_list=[]
    if dataset == 'coco':
        for id in id_list:
                name_list.append('COCO_2014_'+str(id).rjust(12, '0') + '.jpg')
    elif dataset =='f30k':
        image_name=[]
        for line in open(os.path.join(dataset_root, f'annotations/scan_split/image_name.txt'), 'r', encoding='utf-8'):
            image_name.append(line.strip())
        for id in id_list:
                name_list.append(image_name[int(id)])
    elif dataset == 'cc':
         for id in id_list:
              name_list.append(str(id)+'.jpg')
    return name_list


class LoadDataset(Dataset):
    def __init__(self, args, dataset_root, preprocess, subset=None, logger=None):
        logger.info("========== Initial the %s set ==========", subset)
        self.args = args
        self.dataset = args.dataset #coco/f30k/cc
        self.dataset_root = os.path.join(dataset_root, self.dataset)
        self.image_root = os.path.join(self.dataset_root, 'images/')
        self.noise_ratio = args.noise_ratio
        self.noise_file = os.path.join(self.dataset_root, f'annotations/scan_split/'+str(self.noise_ratio)+'_noise_train_caps.txt')


        self.preprocess = preprocess
        self.subset = subset
        self.num_anns = args.num_anns
        
        
        self.images_id = []
        self.captions = []
        self.label = []

        self.cap_length = 0

        if subset == 'dev':
            self.length = args.dev_length
        
        if subset == 'dev' or subset == 'test':
            logger.info('Loading %s set.', self.dataset)
            # get images' id(repeat 5 times cause each image has 5 annotation caps)
            for line in open(os.path.join(self.dataset_root, 'annotations/scan_split/%s_ids.txt' % subset), 'r'):
                if self.dataset == 'cc':
                     self.images_id.append(str(line.strip()))
                else:
                    self.images_id.append(int(line.strip())) # "int(line.strip())" for MSCOCO and Flickr30K; "str(line.strip())" for CC120K
            # get annotation sentences
            for line in open(os.path.join(self.dataset_root, 'annotations/scan_split/%s_caps.txt' % subset), 'r', encoding='utf-8'):
                    self.captions.append(line.strip())
            logger.info('%d captions have been loaded.', len(self.captions))
            # get images' name list
            self.image_name = id2name(self.images_id, self.dataset, 'dev', self.dataset_root)
            # tokenize annotation captions
            self.cap_token = clip.tokenize(self.captions,truncate=True)
            self.img_length = len(set(self.images_id))
            self.cap_length = len(self.captions)
            logger.info('%d images have been loaded.', self.img_length)
            logger.info("%s set initialization completed!", subset)

        elif subset == 'train' or subset == 'train_base':
            if self.noise_ratio != 0:
                if os.path.exists(self.noise_file) == False:
                    logger.info('No noise file! Preparing {} {} set.'.format(self.noise_ratio, self.dataset))
                    # get original captions
                    captions = []
                    for line in open(os.path.join(self.dataset_root, 'annotations/scan_split/train_caps.txt'), 'r', encoding='utf-8'):
                            captions.append(line.strip())
                    self.length = len(captions)
                    idx = np.arange(self.length)
                    np.random.shuffle(idx)
                    # get noise's number
                    noise_length = int(self.noise_ratio * self.length)
                    shuffle_cap= np.array(captions)[idx[:noise_length]]
                    np.random.shuffle(shuffle_cap)
                    noise_cap = np.array(captions)
                    noise_cap[idx[:noise_length]] = shuffle_cap
                    with open(os.path.join(self.dataset_root, 'annotations/scan_split/{}_noise_train_caps.txt' .format(self.noise_ratio)), mode='a', encoding='utf-8') as f:
                         for cap in list(noise_cap):
                              f.write(cap + '\n')

                logger.info('Loading {} {} set.'.format(self.noise_ratio, self.dataset))
                # get images' id(repeat 5 times cause each image has 5 annotation caps)
                for line in open(os.path.join(self.dataset_root, 'annotations/scan_split/train_ids.txt'), 'r'):
                    for i in range(self.num_anns):
                        if self.dataset == 'cc':
                            self.images_id.append(str(line.strip()))
                        else:
                            self.images_id.append(int(line.strip())) # "int(line.strip())" for MSCOCO and Flickr30K; "str(line.strip())" for CC120K
                # get noisy annotation sentences
                for line in open(os.path.join(self.dataset_root, 'annotations/scan_split/{}_noise_train_caps.txt' .format(self.noise_ratio)), 'r', encoding='utf-8'):
                        self.captions.append(line.strip())
                logger.info('%d captions have been loaded.', len(self.captions))

                # get images' name list
                self.image_name = id2name(self.images_id, self.dataset,'train', self.dataset_root)
                
                # tokenize annotation captions
                self.cap_token = clip.tokenize(self.captions,truncate=True)
                self.img_length = len(set(self.images_id))
                self.cap_length = len(self.captions)
                logger.info('%d images have been loaded.', self.img_length)
                logger.info("%s set initialization completed!", subset)
            else:
                logger.info('Loading %s set.', self.dataset)
                # get images' id(repeat 5 times cause each image has 5 annotation caps)
                for line in open(os.path.join(self.dataset_root, 'annotations/scan_split/%s_ids.txt' % subset), 'r'):
                    for i in range(self.num_anns):
                        if self.dataset == 'cc':
                            self.images_id.append(str(line.strip()))
                        else:
                            self.images_id.append(int(line.strip())) # "int(line.strip())" for MSCOCO and Flickr30K; "str(line.strip())" for CC120K
                logger.info('%d images have been loaded.', len(set(self.images_id)))
                # get annotation sentences
                for line in open(os.path.join(self.dataset_root, 'annotations/scan_split/%s_caps.txt' % subset), 'r', encoding='utf-8'):
                        self.captions.append(line.strip())
                logger.info('%d captions have been loaded.', len(self.captions))
                # get images' name list 
                self.image_name = id2name(self.images_id, self.dataset, 'train', self.dataset_root)
                # tokenize annotation captions
                self.cap_token = clip.tokenize(self.captions,truncate=True)
                self.img_length = len(set(self.images_id))
                self.cap_length = len(self.captions)
                logger.info('%d images have been loaded.', self.img_length)
                logger.info("%s set initialization completed!", subset) 
        
    def __len__(self):
        return self.cap_length

    def __getitem__(self, idx):
        if self.subset == 'train':
            image = self.preprocess(Image.open(os.path.join(self.image_root, self.image_name[idx]))) # Image from PIL module
            text = self.cap_token[idx]
            img_id = self.images_id[idx]
            return image, text, idx
        elif self.subset == 'train_base':
            image = self.preprocess(Image.open(os.path.join(self.image_root, self.image_name[idx]))) # Image from PIL module
            text = self.cap_token[idx]
            img_id = self.images_id[idx]
            return image, text, idx       
        else:
            image = self.preprocess(Image.open(os.path.join(self.image_root, self.image_name[idx]))) # Image from PIL module
            text = self.cap_token[idx]
            img_id = self.images_id[idx]
            return image, text, img_id
    
def load_data(args, data, subset, logger):
    
    if subset == 'train':
        dataloader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=(subset == 'train'), 
        )
    elif subset == 'train_base':
        dataloader = DataLoader(
        data,
        batch_size=args.batch_size,
        shuffle=(subset == 'train_base'),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False, 
        )
    else:
        dataloader = DataLoader(
        data,
        batch_size=args.eval_batch_size,
        shuffle=(subset == 'train'),
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=(subset == 'train'), 
        )
    return dataloader, len(data)

def prepare_dataloader(args, dataset_root, preprocess, logger, subset):
    dataloders = {}
    data = LoadDataset(args, dataset_root, preprocess, subset=subset, logger=logger)
    dataloders[subset] = load_data(args, data, subset, logger)
    return dataloders
     