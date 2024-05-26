import os
import json
import xml.etree.ElementTree as ET
import collections

datasets = ['Aquatic', 'Aerial', 'Game', 'Surgical', 'Medical']
dataset_root = '/home/xx/FOMO/data/RWD'

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

for dataset in datasets:
    print(dataset)
    annotations = os.path.join(dataset_root, 'Annotations', dataset)
    few_shot_json = os.path.join(dataset_root, 'ImageSets', dataset, 'few_shot_data.json')
    with open(os.path.join(dataset_root, 'ImageSets', dataset, 'known_classnames.txt'), 'r') as f:
        known_class_names = [line.strip() for line in f.readlines()]
    with open(os.path.join(dataset_root, 'ImageSets', dataset, 'classnames.txt'), 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    with open(few_shot_json) as f:
        few_shot_datas = json.loads(f.read())
    t1_instances = {class_name: min(len(few_shot_datas[class_name]), 100) for class_name in known_class_names}
    t2_instances = {class_name: min(len(few_shot_datas[class_name]), 100) for class_name in class_names}
    
    with open(os.path.join(dataset_root, 'ImageSets', dataset, 'test.txt'), 'r') as f:
        test_data_names = [line.strip() for line in f.readlines()]
    test_instances = {class_name: 0  for class_name in class_names}
    for test_data_name in test_data_names:
        objects = prase_xml(os.path.join(annotations, test_data_name.strip()))
        for obj in objects:
            test_instances[obj['name']] += 1
    lines = ''
    for i in range(0 , len(class_names)):
        if class_names[i] in known_class_names:
            name = class_names[i]
            class_names.pop(i)
            class_names.insert(0, name)
            
    for id, class_name in enumerate(class_names):
        if len(lines):
            lines += ', '
        lines += f'{class_name}({t1_instances[class_name] if class_name in t1_instances else 0}, {t2_instances[class_name] if class_name in t2_instances else 0}, {test_instances[class_name]})'
        if id % 7 == 0 and id != 0:
            print(lines)
            lines = ''
    if len(lines):
        print(lines)
        
    # print('---------------t1 instance---------------')
    # for i, (t1_data_name, count) in enumerate(t1_instances.items()):
    #     print(f'{i+1}: {t1_data_name} {count}')
    # print('---------------t2 instance---------------')
    # for i, (t2_data_name, count) in enumerate(t2_instances.items()):
    #     print(f'{i+1}: {t2_data_name} {count}')
    # # 测试集细节
    # print('--------------test instance: known--------')
    # for i, (test_data_name, count) in enumerate(test_instances.items()):
    #     if test_data_name in known_class_names:
    #         print(f'{i+1}: {test_data_name} {count}')
    # print('--------------test instance: unknown------')
    # for i, (test_data_name, count) in enumerate(test_instances.items()):
    #     if test_data_name not in known_class_names:
    #         print(f'{i+1}: {test_data_name} {count}')
        
        
    