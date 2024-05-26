import os
import json
import xml.etree.ElementTree as ET
import collections
import torch
from collections import defaultdict
from tqdm import tqdm
from torchvision.ops.boxes import box_iou
from func import write_to_csv


datasets = ['Aquatic', 'Aerial', 'Game', 'Surgical', 'Medical']
dataset_root = '/home/xx/FOMO/data/RWD'
result_path = '/home/xx/FOMO/tools/results'
fit_method = 'wb'



def prase_xml(file_path):
    def parse_voc_xml(node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict
    
    tree = ET.parse(file_path[:-4] + '.xml')
    target = parse_voc_xml(tree.getroot())
    return target['annotation']['object']

def prase_result(results):
    unknown_id = 0
    lines = results['lines']
    lines_cls = results['lines_cls']
    att_contribtion = results['att_sim']
    cate_sim = results['cate_sim']
    imageId2prediction = defaultdict(list)
    for i, line in tqdm(enumerate(lines)):
        imageId, score, x1, y1, x2, y2 = line.split(' ')
        imageId = imageId.split('-^-')[-1]
        imageId2prediction[imageId].append(
            [float(x1), float(y1), float(x2), float(y2), float(score), lines_cls[i].int().item(), 
                (cate_sim[i] if len(cate_sim) != 0 else None),
                (att_contribtion[i] if len(att_contribtion) != 0 else None)]
        )
        unknown_id = max(unknown_id, lines_cls[i].int().item())
    return imageId2prediction, unknown_id


for dataset in datasets:
    att_texts = torch.load(f'/home/xx/FOMO/experiments/full_repeat/owlvit-large-patch14/t1/{dataset}_bast.pth')['attributes_texts']
    print(dataset)
    annotations = os.path.join(dataset_root, 'Annotations', dataset)
    # 测试集所有文件名
    with open(os.path.join(dataset_root, 'ImageSets', dataset, 'test.txt'), 'r') as f:
        test_data_names = [line.strip() for line in f.readlines()]
    # 所有类名
    with open(os.path.join(dataset_root, 'ImageSets', dataset, 'classnames.txt'), 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    # 已知类名
    with open(os.path.join(dataset_root, 'ImageSets', dataset, 'known_classnames.txt'), 'r') as f:
        known_class_names = [line.strip() for line in f.readlines()]
    
    test_instances = {class_name: 0  for class_name in class_names}
    results = torch.load(os.path.join(result_path, f'{dataset}_eval_{fit_method}.pth'))
    imageId2prediction, unknown_id = prase_result(results)
    unknown_class = [class_name for class_name in class_names if class_name not in known_class_names]
    un_matched_num = 0
    att_text = []
    num_att = 0
    class2att = {class_name: [] for class_name in unknown_class}
    for test_data_name in test_data_names:
        img_id = test_data_name[:-4]
        objects = prase_xml(os.path.join(annotations, test_data_name.strip()))
        gt_unboxes = [[int(obj['bndbox']['xmin']), int(obj['bndbox']['ymin']),
                     int(obj['bndbox']['xmax']), int(obj['bndbox']['ymax'])] for obj in objects if obj['name'] in unknown_class]
        if len(gt_unboxes) == 0:
            continue
        
        gt_unboxes = torch.as_tensor(gt_unboxes, dtype=torch.float32)
        gt_unclasses = [obj['name'] for obj in objects if obj['name'] in unknown_class]
        
        predictions = imageId2prediction[img_id]
        un_predictions = [prediction for prediction in predictions if prediction[5] == unknown_id]
        if len(un_predictions) == 0:
            continue
        un_boxes = torch.stack([torch.tensor(prediction[:4]) for prediction in un_predictions])
        un_att_scores = [max(prediction[7][:-1]) for prediction in un_predictions]
        un_att_indx = [prediction[7].index(score) for prediction, score in zip(un_predictions, un_att_scores)]
                
        iou = box_iou(un_boxes, gt_unboxes)
        matched_unknown_classes = [gt_unclasses[match_id] for match_id in iou.max(dim=1)[1]]
        iou_mask = iou.max(dim=1)[0]> 0.5 
        un_matched_num += iou_mask.sum().item()
        
        vaild_matched_classes = [class_name for class_name, mask in zip(matched_unknown_classes, iou_mask) if mask]
        vaild_matched_att_scores = [score for score, mask in zip(un_att_scores, iou_mask) if mask]
        vaild_matched_att_indx = [indx for indx, mask in zip(un_att_indx, iou_mask) if mask]
        for class_name, att_score, att_indx in zip(vaild_matched_classes, vaild_matched_att_scores, vaild_matched_att_indx):
            class2att[class_name].append([att_score, att_indx])
    
    num_att = len(att_texts)
    top_k = 3
    # 根据预测的数量找前top k的类别
    top_k_classes = sorted(class2att.items(), key=lambda x: len(x[1]), reverse=True)[:top_k]
    top_k_att = 5
    csv_data_num, csv_data_score = [], []
    # 所有类的前topk排序（错写为了dataset）
    all_dataset_matched_att = [0]*num_att
    all_dataset_scores = [[] for _ in range(num_att)]
    for top_k_class in top_k_classes:
        class_name, atts = top_k_class
        for att in atts:
            all_dataset_matched_att[att[1]] += 1
            all_dataset_scores[att[1]].append(att[0])
    all_dataset_sorted_att = sorted(enumerate(all_dataset_matched_att), key=lambda x: x[1], reverse=True)[: top_k_att]
    all_dataset_scores = [(att_i, sum(att_score)/(len(att_score)+1e-5)) for att_i, att_score in enumerate(all_dataset_scores)]
    all_dataset_sorted_scores = sorted(all_dataset_scores, key=lambda x: x[1], reverse=True)[:top_k_att]
    
    # 在每一类中查看排序对应的数据
    for top_k_class in top_k_classes:
        class_name, atts = top_k_class
        
        matched_att = [0]*num_att
        scores = [[] for _ in range(num_att)]
        for att in atts:
            matched_att[att[1]] += 1
            scores[att[1]].append(att[0])
        print(f'{class_name} has {len(atts)} prediction')
        csv_data_one_num = []
        for att_i, _ in all_dataset_sorted_att:
            print(f'{att_texts[att_i]} dominent {matched_att[att_i]} prediction')
            csv_data_one_num.append(matched_att[att_i])
    
        csv_data_one_score = []
        for att_i, att_score in all_dataset_sorted_scores:
            print(f'{att_texts[att_i]} dominent avg_score: {sum(scores[att_i])/(len(scores[att_i])+1e-5)}')
            csv_data_one_score.append(sum(scores[att_i])/(len(scores[att_i])+1e-5))
        csv_data_num.append(csv_data_one_num)
        csv_data_score.append(csv_data_one_score)
    write_to_csv(f'/home/xx/FOMO/tools/heat_map/{dataset}_{fit_method}_num.csv', csv_data_num)
    write_to_csv(f'/home/xx/FOMO/tools/heat_map/{dataset}_{fit_method}_score.csv', csv_data_score)
    # print(f'un_matched_num: {class2att}')
    
        