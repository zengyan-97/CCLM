import os
import json


def wit_preproc(split, anno_path, img_path, opath):
    if not os.path.exists(opath):
        os.makedirs(opath)
    for ann_file in os.listdir(anno_path):
        if not ann_file.endswith('.jsonl'):
            continue
        if split == "train":
            if "test" in ann_file:
                continue
        else:
            if "test" not in ann_file:
                continue
        anns = {}
        with open(os.path.join(anno_path, ann_file)) as f:
            for l in f:
                item = json.loads(l)
                if item['image_url'] not in anns:
                    anns[item['image_url']] = [item]
                else:
                    anns[item['image_url']].append(item)
        with open(os.path.join(opath, ann_file), 'w') as wf:
            for i, img_file in enumerate(os.listdir(img_path)):
                if not img_file.endswith('.csv'):
                    continue
                print(img_file, i)
                with open(os.path.join(img_path, img_file)) as f:
                    for l in f:
                        if not len(anns):
                            break
                        item = l[:-1].split('\t')
                        if item[0] in anns:
                            for i in anns[item[0]]:
                                i['image_content'] = item[1]
                                wf.write(json.dumps(i)+'\n')
                            anns.pop(item[0])
