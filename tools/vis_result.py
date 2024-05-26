import cv2
import torch
import json
from collections import defaultdict
from tqdm import tqdm
import os
import random
import xml.etree.ElementTree as ET


class Vis_tool():
    def __init__(self, wb_file, gm_file, fomo_file, eval_model, 
                 known_classes, dataset_name, data_root='/home/xx/FOMO/data/RWD'):
        self.wb_file = wb_file
        self.gm_file = gm_file
        self.fomo_file = fomo_file
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.wb_dict = self.load_dict(self.wb_file)
        self.gm_dict = self.load_dict(self.gm_file)
        self.fomo_dict = self.load_dict(self.fomo_file)
        self.att_text = self.get_text(eval_model)
        self.class_names = self.get_class_names()
        self.known_classes = self.class_names[:known_classes]
        
    def get_class_names(self):
        txt_root = os.path.join(self.data_root, 'ImageSets', self.dataset_name, 'classnames.txt')
        with open(txt_root, 'r') as f:
            lines = f.readlines()
        return [line.strip() for line in lines]
        
    def load_dict(self, file):
        results = torch.load(file)
        lines = results['lines']
        lines_cls = results['lines_cls']
        att_contribtion = results['att_sim']
        cate_sim = results['cate_sim']
        imageId2prediction = defaultdict(list)
        for i, line in tqdm(enumerate(lines)):
            imageId, score, x1, y1, x2, y2 = line.split(' ')
            imageId2prediction[imageId].append(
                [float(x1), float(y1), float(x2), float(y2), float(score), lines_cls[i].int().item(), 
                 (cate_sim[i] if len(cate_sim) != 0 else None),
                 (att_contribtion[i] if len(att_contribtion) != 0 else None)]
            )
        return imageId2prediction
        
    def get_text(self, eval_model):
        return torch.load(eval_model)['attributes_texts']

    def get_image(self, imageId):
        img_path = os.path.join(self.data_root, 'JPEGImages', self.dataset_name, imageId.split('-^-')[-1]+'.jpg')
        if not os.path.exists(img_path):
            img_path = os.path.join(self.data_root, 'JPEGImages', self.dataset_name, imageId.split('-^-')[-1]+'.png')
        return cv2.imread(img_path)
        
    def get_annotations(self, imageId):
        tree = ET.parse(os.path.join(self.data_root, 'Annotations', self.dataset_name, imageId.split('-^-')[-1] + '.xml'))
        root = tree.getroot()
        annotations = []
        for obj in root.iter('object'):
            cls = obj.find('name').text
            bbox = obj.find('bndbox')
            x1 = int(bbox.find('xmin').text)
            y1 = int(bbox.find('ymin').text)
            x2 = int(bbox.find('xmax').text)
            y2 = int(bbox.find('ymax').text)
            annotations.append([x1, y1, x2, y2, cls])
        return annotations

    def get_vaild_id(self):
        wb_id = set(self.wb_dict.keys())
        gm_id = set(self.gm_dict.keys())
        fomo_id = set(self.fomo_dict.keys())
        return wb_id & gm_id & fomo_id

    def plot_gt(self, imageId, image, known_color=(0, 255, 0), unknown_color=(255, 0, 0)):
        gt_annotations = self.get_annotations(imageId)
        for x1, y1, x2, y2, cls in gt_annotations:
            color = known_color if cls in self.known_classes else unknown_color
            print(cls if cls in self.known_classes else f'unknown: {cls}')
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        return image
        
    def plot_predictions(self, image, predictions, known_color=(0, 255, 0), unknown_color=(255, 0, 0),
                         known_thr=0.3, unknown_thr=0.3, plot_known=True, top_k=-1):
        other_plot = []
        for x1, y1, x2, y2, score, cls, cate_sim, att_contribtion in predictions:
            if top_k == 0:
                break
            top_k -= 1
            cate_name = self.class_names[cls]
            if cate_name in self.known_classes and score < known_thr:
                continue
            if cate_name not in self.known_classes and score < unknown_thr:
                continue
            if cate_name not in self.known_classes:
                cate_name = 'unknown'
            if plot_known == True and cate_name == 'unknown':
                other_plot.append([x1, y1, x2, y2, score, cls, cate_sim, att_contribtion])
                continue
            
            if cate_sim is not None and att_contribtion is not None and cate_name == 'unknown':
                cate_sim_index = cate_sim.index(max(cate_sim))
                cate_sim_value = cate_sim[cate_sim_index]
                att_index = att_contribtion.index(max(att_contribtion[:-1]))
                att_value = att_contribtion[att_index]
                print(f'sim: {self.known_classes[cate_sim_index]} {cate_sim_value}, att: {self.att_text[att_index]} {att_value}')
                
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            color = known_color if self.class_names[cls] in self.known_classes else unknown_color
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            print(cate_name, score)
            
        if plot_known == True:
            image = self.plot_predictions(image, other_plot, known_color, unknown_color, known_thr, unknown_thr, plot_known=False, top_k=3)
        return image

    def fine_thr(self, image, prediction, known_color=(0, 255, 0), unknown_color=(255, 0, 0),
                       known_thr=0.6, unknown_thr=0.6, save_name='fomo.jpg', img_id=None):
        cv2.imwrite(save_name, self.plot_predictions(image.copy(), prediction, known_color, unknown_color, known_thr, unknown_thr))
        while(True):
            print(save_name)
            print('img_id:', img_id)    
            key = input('un_thr: a+, d-. thr: z+, c-:. q to quit. un_int or un_float: set thr')
            if key == 'q':
                break
            if key == 'a':
                unknown_thr += 0.01
            elif key == 'd':
                unknown_thr -= 0.01
            elif key == 'z':
                known_thr += 0.01
            elif key == 'c':    
                known_thr -= 0.01
            elif key[0:2] == 'un':
                unknown_thr = float(key[2:])
            elif key[0:2] == 'kn':
                known_thr = float(key[2:])
            print('unknown_thr:', unknown_thr, 'known_thr:', known_thr)
            cv2.imwrite(save_name, self.plot_predictions(image.copy(), prediction, known_color, unknown_color, known_thr, unknown_thr))
                

Aquatic_class_names = ['jellyfish' ,'penguin','puffin','shark'
,'stingray'
,'starfish'
,'fish']


Aerial_class_names = ['Expressway-Service-area',
'Expressway-toll-station',
'airplane',
'airport',
'baseballfield',
'basketballcourt',
'bridge',
'chimney',
'dam',
'golffield',
'groundtrackfield',
'harbor',
'overpass',
'ship',
'stadium',
'storagetank',
'tenniscourt',
'trainstation',
'vehicle',
'windmill']
known_dataset_num = {
    'Aquatic': 4,
    'Aerial': 10,
    'Game': 30,
    'Medical': 6,
    'Surgical': 6
}
dataset_name = 'Aerial'

tool = Vis_tool(f'tools/results/{dataset_name}_eval_wb.pth', 
         f'tools/results/{dataset_name}_eval_gm.pth', 
         f'tools/results/{dataset_name}_eval_fomo.pth',
         f'/home/xx/FOMO/experiments/full_repeat/owlvit-large-patch14/t1/{dataset_name}_bast.pth',
         known_dataset_num[dataset_name],
         dataset_name)
vaild_ids = tool.get_vaild_id()
save_dir = f'tools/results/{dataset_name}_vis/'
display_conf = {
    'known_color': (21, 201, 253),
    'unknown_color': (0, 0, 255),
}

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

config = '4302-^-04087'
for vaild_id in vaild_ids:
    if (vaild_id != config and config != None):
        continue
    config = None
    print(vaild_id)
    image = tool.get_image(vaild_id)
    cv2.imwrite(os.path.join(save_dir, 'gt.jpg'), tool.plot_gt(vaild_id, image.copy(), **display_conf))
    wb_predictions = tool.wb_dict[vaild_id]
    gm_predictions = tool.gm_dict[vaild_id]
    fomo_predictions = tool.fomo_dict[vaild_id]
    tool.fine_thr(image.copy(), wb_predictions, save_name=os.path.join(save_dir, 'wb.jpg'), **display_conf, img_id=vaild_id)
    tool.fine_thr(image.copy(), gm_predictions, save_name=os.path.join(save_dir, 'gm.jpg'), **display_conf, img_id=vaild_id)
    tool.fine_thr(image.copy(), fomo_predictions, save_name=os.path.join(save_dir, 'fomo.jpg'), **display_conf, img_id=vaild_id)
    
    # unknown_thr = input('fomo unknown_thr:')
    # known_thr = input('fomo known_thr:') 
    # while(input('continue?') == 'y'):
    #     cv2.imwrite(os.path.join(save_dir, 'fomo.jpg'), tool.plot_predictions(image.copy(), fomo_predictions))
    #     unknown_thr = input('fomo unknown_thr:')
    #     known_thr = input('fomo known_thr:') 
        
    # unknown_thr = input('gm unknown_thr:')
    # known_thr = input('gm known_thr:') 
    # while(input('continue?') == 'y'):
    #     cv2.imwrite(os.path.join(save_dir, 'gm.jpg'), tool.plot_predictions(image.copy(), gm_predictions))
    #     unknown_thr = input('gm unknown_thr:')
    #     known_thr = input('gm known_thr:') 
        
    # unknown_thr = input('wb unknown_thr:')
    # known_thr = input('wb known_thr:') 
    # while(input('continue?') == 'y'):
    #     cv2.imwrite(os.path.join(save_dir, 'fomo.jpg'), tool.plot_predictions(image.copy(), wb_predictions))
    #     unknown_thr = input('wb unknown_thr:')
    #     known_thr = input('wb known_thr:')         

    

    
