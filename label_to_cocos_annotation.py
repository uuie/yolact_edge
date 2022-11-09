import itertools
import json
import os


def label_to_cocos(working_dir, validation_ratio=0.1):
    images = []
    annotations = []
    categories = []
    img_id = 0
    ann_id = 0
    all_files = []
    for root, dirs, files in os.walk(working_dir, topdown=False):
        for name in files:
            if not name.endswith('.jpg'):
                continue
            label_json = os.path.relpath(os.path.join(root, name.replace('.jpg', '.json')), working_dir)
            image_file = os.path.relpath(os.path.join(root, name), working_dir)
            abs_json_path = os.path.join(working_dir, label_json)
            if not os.path.exists(abs_json_path):
                continue
            all_files.append([abs_json_path, label_json, image_file])
    traing_imgs = int(len(all_files) * (1 - validation_ratio))
    processed_imgs = 0
    training_data, valid_data = {}, {}
    for (abs_json_path, label_json, image_file) in all_files:
        img_id += 1
        with open(abs_json_path) as fp:
            label_data = json.loads(fp.read())
        img_w, img_h = label_data.get('imageWidth'), label_data.get('imageHeight')
        for s in label_data.get('shapes'):
            ann_id += 1
            label = s.get('label')
            if label not in categories:
                categories.append(label)
            cat_id = categories.index(label) + 1
            points = s.get('points')

            min_x, max_x, min_y, max_y = img_w, 0, img_h, 0
            for p in points:
                min_x = min(p[0], min_x)
                max_x = max(p[0], max_x)
                min_y = min(p[1], min_y)
                max_y = max(p[1], max_y)

            annotations.append({
                "keypoints": [],
                "num_keypoints": 0,
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "segmentation": [list(itertools.chain(*points))],
                "bbox": [min_x, min_y, max_x - min_x, max_y - min_y],
                "iscrowd": 0,
            })
        images.append({
            "file_name": image_file,
            "height": img_h,
            "width": img_w,
            "id": img_id
        })
        processed_imgs += 1
        if processed_imgs == traing_imgs:
            training_data = dict(
                categories=[{
                    "id": i + 1,
                    "name": categories[i],
                    "supercategory": categories[i],
                } for i in range(len(categories))],
                images=images,
                annotations=annotations
            )
            images = []
            annotations = []
    valid_data = dict(
        categories=[{
            "id": i + 1,
            "name": categories[i],
            "supercategory": categories[i],
        } for i in range(len(categories))],
        images=images,
        annotations=annotations
    )
    return training_data, valid_data


if __name__ == '__main__':
    training_data, valid_data = label_to_cocos('/home/chris/Downloads/bucket')
    with open(os.path.join('/home/chris/Downloads/bucket', 'training.json'), 'w') as fp:
        fp.write(json.dumps(training_data, indent=2))
    with open(os.path.join('/home/chris/Downloads/bucket', 'validation.json'), 'w') as fp:
        fp.write(json.dumps(valid_data, indent=2))
