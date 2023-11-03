import os
import pickle

from tqdm import tqdm

import numpy as np
import pandas as pd

import cv2

import torch
from torch.utils.data import Dataset

from transformers import ViTModel

from data_helper import convert_semanticID_to_text




class FlickrDataset(Dataset):
    def __init__(self,
                 caption_data_path='flickr-30k/results_20130124.token',
                 figure_dir_path='flickr-30k/flickr30k-images/',
                 debug_mode=False,
                 save_figs_repr=False,
                 use_size=10000) -> None:
        super().__init__()
        with open(caption_data_path, 'r', encoding='utf-8') as fc:
            self.caption_data = fc.readlines()
            if use_size:
                self.caption_data = self.caption_data[: use_size*5]

        if debug_mode:
            self.caption_data = self.caption_data[: 500]

        self.figure_ids = []
        self.figure_captions = []
        self.figures = []

        # load mae model
        model = ViTModel.from_pretrained('ptms/vit-mae-base/').cuda()

        for one_data in tqdm(self.caption_data):
            one_data = one_data.rstrip()
            one_id, one_caption = one_data.split('\t')
            

            if not self.figure_ids:
                self.figure_ids.append(one_id[:-6])
                temp_captions = [one_caption]

                # read figure and repr
                img_cv = cv2.imread(os.path.join(figure_dir_path, one_id[:-2]))
                inputs = torch.tensor(self.resize_for_vit(img_cv), device='cuda', dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
                outputs = model(inputs)
                self.figures.append(outputs.pooler_output.squeeze().detach().cpu().numpy())
                
            else:
                if one_id[:-6] == self.figure_ids[-1]:
                    temp_captions.append(one_caption)
                else:
                    self.figure_ids.append(one_id[:-6])
                    self.figure_captions.append(temp_captions)
                    temp_captions = [one_caption]
                    # read figure and repr
                    img_cv = cv2.imread(os.path.join(figure_dir_path, one_id[:-2]))
                    inputs = torch.tensor(self.resize_for_vit(img_cv), device='cuda', dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2)
                    outputs = model(inputs)
                    self.figures.append(outputs.pooler_output.squeeze().detach().cpu().numpy())

            self.figures_repr_mat = np.vstack(self.figures)
            
        if save_figs_repr:
            with open('flickr-30k/figures_repr_mat.npy', 'wb') as f:
                pickle.dump(self.figures_repr_mat, f)


    def __getitem__(self, index):
        return self.figure_ids[index], self.figure_captions[index], self.figures[index]


    def __len__(self):
        return len(self.figure_ids)
    

    @staticmethod
    def resize_for_vit(img):
        H, W, _ = img.shape
        if H >= 224 and W >= 224:
            mid_H = H // 2
            mid_W = W // 2
            return img[mid_H-112: mid_H+112, mid_W-112: mid_W+112, :]
        
        elif H < 224 and W < 224:
            gap_H = 224 - H
            gap_W = 224 - W
            gap_side_H, left_H = divmod(gap_H, 2)
            gap_side_W, left_W = divmod(gap_W, 2)
            return np.pad(array=img,
                        pad_width=((gap_side_H, gap_side_H+left_H), (gap_side_W, gap_side_W+left_W), (0, 0)),
                        mode='constant',
                        constant_values=(0, 0))
        else:
            if H < 224:
                mid_W = W // 2
                img = img[:, mid_W-112: mid_W+112, :]
                gap_H = 224 - H
                gap_side_H, left_H = divmod(gap_H, 2)
                return np.pad(array=img,
                            pad_width=((gap_side_H, gap_side_H+left_H), (0, 0), (0, 0)),
                            mode='constant',
                            constant_values=0)
            if W < 224:
                mid_H = H // 2
                img = img[mid_H-112: mid_H+112, :, :]
                gap_W = 224 - W
                gap_side_W, left_W = divmod(gap_W, 2)
                return np.pad(array=img,
                            pad_width=((0, 0), (gap_side_W, gap_side_W+left_W), (0, 0)),
                            mode='constant',
                            constant_values=0)
            



class PureFlickr(Dataset):
    def __init__(self,
                 repr_npy_path='flickr-30k/figures_repr_mat.npy',
                 caption_data_path='flickr-30k/results_20130124.token',
                 use_size=10000) -> None:
        super().__init__()
        
        self.use_size = use_size
        # load feas
        self.feas = np.load(repr_npy_path, allow_pickle=True)

        # load caption
        with open(caption_data_path, 'r', encoding='utf-8') as fc:
            self.caption_data = fc.readlines()
            if use_size:
                self.caption_data = self.caption_data[: use_size*5]

        # load mapper
        with open('flickr-30k/ids_related/old2new_id_mapper_k5_c30.pkl', 'rb') as f:
            self.old2new_mapper = pickle.load(f)

        self.figure_ids = []
        self.figure_captions = []
        for one_data in tqdm(self.caption_data):
            one_data = one_data.rstrip()
            one_id, one_caption = one_data.split('\t')
            one_id = one_id[:-6]

            if not self.figure_ids:
                self.figure_ids.append(one_id)
                temp_captions = [one_caption]
            else:
                if one_id == self.figure_ids[-1]:
                    temp_captions.append(one_caption)
                else:
                    self.figure_ids.append(one_id)
                    self.figure_captions.append(temp_captions)
                    temp_captions = [one_caption]
    

    def __getitem__(self, index):
        return self.figure_ids[index], self.figure_captions[index], self.feas[index, :]
    

    def __len__(self):
        if self.use_size:
            assert self.use_size == self.feas.shape[0]
        return self.feas.shape[0]
    

    def collate_fn_4_pf(self, batch_data):
        # old_ids, captions, feas = batch_data
        # collated_new_ids = []
        # for oid in old_ids:
        #     new_id = convert_semanticID_to_text(self.old2new_mapper)
        #     collated_new_ids.extend([new_id] * 5)
        # collated_captions = []
        # for i in range(len(captions[0])):
        #     for j in range(len(captions)):
        #         collated_captions.append(captions[i][j])
        # collated_feas = torch.vstack([feas[i, :] for i in range(feas.shape[0]) for _ in range(5)])
        # return collated_new_ids, collated_captions, collated_feas
        collated_new_ids = []
        collated_captions = []
        collated_feas = []

        for data in batch_data:
            collated_new_ids.extend([convert_semanticID_to_text(self.old2new_mapper[data[0]])] * 5)
            collated_captions.extend(data[1])
            collated_feas.extend([torch.tensor(data[2], device='cuda')] * 5)
        collated_feas = torch.vstack(collated_feas)
        return collated_new_ids, collated_captions, collated_feas




        

if __name__ == '__main__':
    # myset = FlickrDataset(save_figs_repr=True)
    myset = PureFlickr()
    print('done')