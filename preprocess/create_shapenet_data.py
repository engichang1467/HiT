"""This script creates train/val/test splits for the ShapeNetCore v2 dataset
and samples point clouds from the meshes.
"""

import argparse
import sys
import os
import json
from typing import List, Tuple
import tqdm
import numpy as np
import trimesh
import h5py
import open3d as o3d
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import read_text, get_split_shape_ids, convert_to_torch_tensor, convert_to_np_array
from utils.mesh import scale_mesh_unit_sphere, sample_surface_torch


def process_shape_id(shape_id_path: str, npoints: int) -> Tuple[np.ndarray, float]:
    """Load mesh, normalize to unit sphere, sample point cloud.
    Returns:
        pc: sampled point cloud (npoints, 3)
        scale: the diagonal extent used for normalization
    """
    mesh_path = os.path.join(shape_id_path, "models", "model_normalized.obj")
    mesh_obj = trimesh.load(mesh_path, force="mesh", process=False)

    # compute scale before normalizing
    vertices = mesh_obj.vertices
    extent = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    scale = np.sqrt(np.sum(extent ** 2))

    mesh_obj = scale_mesh_unit_sphere(mesh=mesh_obj)

    vertices, faces = mesh_obj.vertices, mesh_obj.faces
    vertices = convert_to_torch_tensor(np_arr=vertices, data_type="fp32")
    faces = convert_to_torch_tensor(np_arr=faces, data_type="int64")

    pc, f_indxs, normals = sample_surface_torch(
        faces=faces,
        vs=vertices.unsqueeze(0),
        count=npoints
    )
    pc = convert_to_np_array(pc)
    return pc, scale


def create_splits(data_root: str, splits: List[str]=["train", "val", "test"],
                  ratios: List[float]=[0.8, 0.1, 0.1], seed: int=42) -> None:
    """Create train/val/test splits by listing shape directories.
    Args:
        data_root (str): path to the data root (used as both in and out)
        splits (List[str]): split names
        ratios (List[float]): split ratios (must sum to 1.0)
        seed (int): random seed for reproducibility
    """
    cats = read_text(f"{data_root}/cats.txt")
    rng = np.random.default_rng(seed)

    for cat in cats:
        cat_path = os.path.join(data_root, cat)
        os.makedirs(cat_path, exist_ok=True)

        # list all shape directories that contain models/model_normalized.obj
        all_shape_ids = sorted([
            d for d in os.listdir(cat_path)
            if os.path.isdir(os.path.join(cat_path, d)) and
               os.path.exists(os.path.join(cat_path, d, "models", "model_normalized.obj"))
        ])

        n_total = len(all_shape_ids)
        print(f"Category {cat}: found {n_total} shapes")

        # shuffle and split
        rng.shuffle(all_shape_ids)
        n_train = int(n_total * ratios[0])
        n_val = int(n_total * ratios[1])

        split_list = {
            splits[0]: all_shape_ids[:n_train],
            splits[1]: all_shape_ids[n_train:n_train + n_val],
            splits[2]: all_shape_ids[n_train + n_val:],
        }

        for s in splits:
            print(f"  {s}: {len(split_list[s])} shapes")

        with open(f"{cat_path}/split.json", 'w') as f:
            json.dump(split_list, f)


def sample_pc(data_root: str, splits: List[str]=["train"], n_points: int=2048) -> None:
    """Sample point clouds from meshes and save to HDF5.
    Args:
        data_root (str): path to the data root (used as both in and out)
        splits (List[str]): splits to process
        n_points (int): number of points to sample per shape
    """
    cats = read_text(f"{data_root}/cats.txt")
    for cat in cats:
        cat_path = os.path.join(data_root, cat)
        for split in splits:
            shape_ids = get_split_shape_ids(path=f"{cat_path}/split.json", split_type=split)
            num_shapes = len(shape_ids)

            out_filepath = f"{cat_path}/{cat}_{split}_pc.hdf5"
            with h5py.File(out_filepath, 'w') as f:
                pts_hdf5 = f.create_dataset(
                    "pc",
                    shape=(num_shapes, n_points, 3),
                    dtype=np.float32,
                    compression="gzip",
                    chunks=(1, n_points, 3)
                )
                scale_hdf5 = f.create_dataset(
                    "scale",
                    shape=(num_shapes,),
                    dtype=np.float32,
                )

                with tqdm.tqdm(total=num_shapes, desc=f"Sampling PCs for {cat}/{split} ({n_points} pts)") as pbar:
                    for i, shape_id in enumerate(shape_ids):
                        shape_id_path = os.path.join(cat_path, shape_id)
                        pc, scale = process_shape_id(shape_id_path=shape_id_path, npoints=n_points)
                        pts_hdf5[i] = pc
                        scale_hdf5[i] = scale
                        pbar.update(1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="Path to ShapeNetCore v2 data root")
    parser.add_argument("--n_points", type=int, default=2048, help="Number of points to sample")
    parser.add_argument("--splits", type=str, nargs="+", default=["train", "val", "test"], help="Splits to process")
    parser.add_argument("--step", type=str, choices=["split", "sample_pc", "all"], default="all", help="Which step to run")
    args = parser.parse_args()

    if args.step in ["split", "all"]:
        create_splits(data_root=args.data_root)

    if args.step in ["sample_pc", "all"]:
        sample_pc(data_root=args.data_root, splits=args.splits, n_points=args.n_points)
