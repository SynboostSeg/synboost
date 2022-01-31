import torch.utils.data as data
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from natsort import natsorted
from torchvision import transforms
import torch
import random

import sys
sys.path.append("..")
import image_dissimilarity.data.cityscapes_labels as cityscapes_labels
from image_dissimilarity.data.augmentations import get_transform


trainid_to_name = cityscapes_labels.trainId2name
id_to_trainid = cityscapes_labels.label2trainid

# invalid frames are those where np.count_nonzero(labels_source) is 0 for Lost and Found Dataset
INVALID_LABELED_FRAMES = [17,  37,  55,  72,  91, 110, 129, 153, 174, 197, 218, 353, 490, 618, 686, 792, 793]

class CityscapesDataset(Dataset):
    
    def __init__(self, dataroot, preprocess_mode, crop_size=512, aspect_ratio= 0.5, flip=False, normalize=False,
                 prior = True, only_valid = False, roi = False, light_data= True, void = False, num_semantic_classes = 19, is_train = True):
        
        dataroot = '/kaggle/input/cityscapes-synboost/final_dataset/road_anomaly/original'
        self.original_paths = [os.path.join(dataroot, image)
                               for image in os.listdir(os.path.join(dataroot))]
        if light_data:
            dataroot = '/kaggle/input/road-anomaly-icnet-original-synthesis/results/image-synthesis/test_latest/images/synthesized_image'
            self.synthesis_paths = [os.path.join(dataroot, image)
                                    for image in os.listdir(os.path.join(dataroot))]
        else:
            self.semantic_paths = [os.path.join(dataroot, 'semantic', image)
                                   for image in os.listdir(os.path.join(dataroot, 'semantic'))]
            self.synthesis_paths = [os.path.join(dataroot, 'synthesis', image)
                                    for image in os.listdir(os.path.join(dataroot, 'synthesis'))]
        
        
        
        # We need to sort the images to ensure all the pairs match with each other
        self.original_paths = natsorted(self.original_paths)
        
        self.synthesis_paths = natsorted(self.synthesis_paths)
        
        
               
        
        self.dataset_size = len(self.original_paths)
        self.preprocess_mode = preprocess_mode
        self.crop_size = crop_size
        self.aspect_ratio = aspect_ratio
        self.num_semantic_classes = num_semantic_classes
        self.is_train = is_train
        self.void = void
        self.flip = flip
        self.prior = prior
        self.normalize = normalize

    def __getitem__(self, index):
        
        # get and open all images
        

        image_path = self.original_paths[index]
        image = Image.open(image_path)

        syn_image_path = self.synthesis_paths[index]
        syn_image = Image.open(syn_image_path)
        
        
    
           

        # get input for transformations
        w = self.crop_size
        h = round(self.crop_size / self.aspect_ratio)
        image_size = (h, w)
        
        if self.flip:
            flip_ran = random.random() > 0.5
            
            
            image = _flip(image, flip_ran)
            syn_image = _flip(syn_image, flip_ran)
            

        # get augmentations
        base_transforms, augmentations = get_transform(image_size, self.preprocess_mode)

        # apply base transformations
        
        
        syn_image_tensor = base_transforms(syn_image)
        
        

        if self.is_train and self.preprocess_mode != 'none':
            image_tensor = augmentations(image)
        else:
            image_tensor = base_transforms(image)
            
        if self.normalize:
            norm_transform = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) #imageNet normamlization
            syn_image_tensor = norm_transform(syn_image_tensor)
            image_tensor = norm_transform(image_tensor)
            
        print(syn_image_tensor.shape)
        input_dict = {
                      'original': image_tensor,
                      
                      'synthesis': syn_image_tensor,
                      
                      'original_path': image_path,
                      
                      'syn_image_path': syn_image_path,
                      
                      }

        return input_dict
        
    def __len__(self):
        return self.dataset_size
    
def normalize():
    return

def _flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def one_hot_encoding(semantic, num_classes=20):
    one_hot = torch.zeros(num_classes, semantic.size(1), semantic.size(2))
    for class_id in range(num_classes):
        one_hot[class_id,:,:] = (semantic.squeeze(0)==class_id)
    one_hot = one_hot[:num_classes-1,:,:]
    return one_hot

# ----------- FOR TESTING --------------

def test(dataset_args, dataloader_args, save_imgs=False, path='./visualization'):
    norm_mean = [0.485, 0.456, 0.406]
    norm_std = [0.229, 0.224, 0.225]

    if save_imgs and not os.path.exists(path):
        os.makedirs(path)

    dataset = CityscapesDataset(**dataset_args)
    dataloader = torch.utils.data.DataLoader(dataset, **dataloader_args)
    for counter, sample in enumerate(dataloader):
        print('Images Saved: ', sample['original'].shape[0] * counter)
        if save_imgs:
            transform = ToPILImage()
            decoder = DenormalizeImage(norm_mean, norm_std)
            for idx, (original, label, semantic, synthesis) in \
            enumerate(zip(sample['original'], sample['label'], sample['semantic'], sample['synthesis'])):
                # get original image
                original = original.squeeze().cpu()
                original = decoder(original)
                original = np.asarray(transform(original))
                original = Image.fromarray(original)
                original.save(os.path.join(path, 'Original_%i_%i' % (counter, idx) + '.png'))

                # get label image
                label = label.squeeze().cpu().numpy()
                label = np.asarray(transform(label))
                label = Image.fromarray(label).convert('RGB')
                label.save(os.path.join(path, 'Label_%i_%i' % (counter, idx) + '.png'))

                # get semantic image
                semantic = semantic.squeeze().cpu().numpy()
                semantic = np.asarray(transform(semantic))
                semantic = visualization.colorize_mask(semantic)
                semantic = semantic.convert('RGB')
                semantic.save(os.path.join(path, 'Semantic_%i_%i' % (counter, idx) + '.png'))

                # get original image
                synthesis = synthesis.squeeze().cpu()
                synthesis = decoder(synthesis)
                synthesis = np.asarray(transform(synthesis))
                synthesis = Image.fromarray(synthesis)
                synthesis.save(os.path.join(path, 'Synthesis_%i_%i' % (counter, idx) + '.png'))
        

if __name__ == '__main__':
    from torchvision.transforms import ToPILImage
    import torch
    
    import sys
    sys.path.append("..")
    from util.image_decoders import DenormalizeImage
    from util import visualization
    
    dataset_args = {
        'dataroot': '/media/giancarlo/Samsung_T5/master_thesis/data/fs_static',
        'preprocess_mode': 'none',
        'crop_size': 512,
        'aspect_ratio': 2,
        'flip': True,
        'normalize': True,
        'void': False,
        'num_semantic_classes': 19,
        'is_train': False
    }
    
    dataloader_args = {
        'batch_size': 1,
        'num_workers': 1,
        'shuffle': False
    }
    
    test(dataset_args, dataloader_args, save_imgs=True)
