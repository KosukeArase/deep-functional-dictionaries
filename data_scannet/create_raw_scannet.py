import argparse
import json
import pathlib
import os
import h5py
import joblib
import numpy as np
import pandas as pd
import open3d
import collections


label_names = [  # https://github.com/ScanNet/ScanNet/blob/83c2196a16c385aeabeac87b949f68d4594e2e8b/Tasks/Benchmark/classes_SemVoxLabel-nyu40id.txt
    "unannotated",
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refridgerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture"
]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', default="/data/unagi0/dataset/ScanNet/specific", type=str, help='Path to your dataset')
    parser.add_argument('--output_dir', required=True, type=str, help='Path to output directory')

    args = parser.parse_args()

    return args


def process(scene, scene_id, label_map):
    point_file = str(scene.joinpath("{scene_id}_vh_clean_2.ply".format(scene_id=scene_id)))
    instance_file = str(scene.joinpath("{scene_id}.aggregation.json".format(scene_id=scene_id)))
    segment_file = str(scene.joinpath("{scene_id}_vh_clean_2.0.010000.segs.json".format(scene_id=scene_id)))

    pc_data = open3d.read_point_cloud(point_file)
    all_points = np.asarray(pc_data.points)  # XYZ, float
    all_colors = np.asarray(pc_data.colors)  # RGB, float

    with open(instance_file, "r") as f:
        instances = json.load(f)["segGroups"]
    with open(segment_file, "r") as f:
        segment_indices = json.load(f)["segIndices"]

    seg_to_point = collections.defaultdict(list)
    for point_id, seg_id in enumerate(segment_indices):
        seg_to_point[seg_id].append(point_id)

    seg_to_point = {k: np.array(v) for k, v in seg_to_point.items()}

    XYZ = []
    RGB = []
    label_ids = []
    instance_ids = []

    for instance_id, instance in enumerate(instances):
        label_name = instance["label"]
        if label_name in label_map.keys() and label_map[label_name] in label_names:
            label_id = label_names.index(label_map[label_name])
        else:
            label_id = 0
        segments = instance["segments"]
        for seg_id in segments:
            point_ids = seg_to_point[seg_id]
            xyz = all_points[point_ids]
            rgb = all_colors[point_ids]
            XYZ.append(xyz)
            RGB.append(rgb)
            label_ids.append([label_id] * len(point_ids))
            instance_ids.append([instance_id] * len(point_ids))

    XYZ = np.array(np.concatenate(XYZ, axis=0), dtype=np.float32)
    RGB = np.array(np.concatenate(RGB, axis=0), dtype=np.float32)
    label_ids = np.expand_dims(np.array(np.concatenate(label_ids, axis=0), dtype=np.int64), 1)
    instance_ids = np.expand_dims(np.array(np.concatenate(instance_ids, axis=0), dtype=np.int64), 1)

    assert XYZ.shape[0] == RGB.shape[0] == label_ids.shape[0] == instance_ids.shape[0]
    assert XYZ.shape[1] == RGB.shape[1] == 3
    assert label_ids.shape[1] == instance_ids.shape[1] == 1

    return np.concatenate([XYZ, RGB, label_ids, instance_ids], axis=1)  # [[xyzrgb, sem, ins], ...]


def main():
    dataset_root = pathlib.Path(args.dataset_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve().joinpath("raw_scannet")
    output_dir.mkdir(exist_ok=True)

    label_map_df = pd.read_csv(dataset_root.joinpath("./scannetv2-labels.combined.tsv"), delimiter='\t')
    label_map = {row["raw_category"]: row["nyu40class"] for index, row in label_map_df.iterrows()}

    scenes = dataset_root.glob("{}/scene*_*".format('scans'))

    for i, scene in enumerate(scenes):
        try:
            scene_name = str(scene).split("/")[-1]
            output_path = output_dir.joinpath("{}.npy".format(scene_name))
            if output_path.exists():
                print('{}.npy exists, skip'.format(scene_name))
                continue
            data_list = process(scene, scene_name, label_map)
            data_min = np.min(data_list, axis=0)[:3]
            data_list[:, :3] -= data_min  # set origin zero
            data_max = np.max(data_list, axis=0)
            print(i, scene_name, data_list.shape)
            np.save(output_path, data_list)
        except Exception as e:
            print("an error has occured while processing {}, skip".format(scene))
            print(e)
            continue

    print("=============================================")

if __name__ == "__main__":
    args = parse_args()
    print(args)

    main()
