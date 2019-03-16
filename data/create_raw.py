import argparse
import pathlib
import os
import h5py
import joblib
import numpy as np


label_names = ["ceiling", "floor", "wall", "beam", "column", "window", "door", "table", "chair", "sofa", "bookcase", "board", "clutter"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', default="/data/umihebi0/dataset/S3DIS/Stanford3dDataset_v1.2_Aligned_Version", type=str, help='Path to your dataset')
    parser.add_argument('--output_dir', required=True, type=str, help='Path to output directory')

    args = parser.parse_args()

    return args


def process(file, ins):
    label_name = str(file).split("/")[-1].split("_")[0]
    data = np.loadtxt(file)
    label_id = label_names.index(label_name) if label_name in label_names else label_names.index("clutter")
    return np.concatenate([data, np.ones((len(data), 1)) * np.array([label_id, ins])], axis=1)


def main():
    dataset_root = pathlib.Path(args.dataset_dir).resolve()
    output_dir = pathlib.Path(args.output_dir).resolve().joinpath("raw_s3dis")
    output_dir.mkdir(exist_ok=True)

    scenes = dataset_root.glob("Area_*/*")
    for scene in scenes:
        try:
            scene_name = "_".join(str(scene).split("/")[-2:])
            output_path = output_dir.joinpath("{}.npy".format(scene_name))
            if output_path.exists() or str(scene).split("/")[-1] == ".DS_Store":
                continue
            files = list(scene.glob("Annotations/*.txt"))
            data_list = [process(file, ins) for ins, file in enumerate(files)]
            data_list = np.vstack(data_list)
            data_min = np.min(data_list, axis=0)[:3]
            data_list[:, :3] -= data_min
            data_max = np.max(data_list, axis=0)
            print(scene_name, data_list.shape)
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
