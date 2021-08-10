import os
import numpy as np
import msgpack_numpy


"""filter for only long trajectories"""
good_traj = []
base_dir = "/srv/share/datasets/RealEstate10k/RealEstate10K/"
for d in ["train/", "test/"]:
    for fname in os.listdir(base_dir + d):
        f = open(base_dir + d + fname, "r")
        last_pose = None
        total_len = 0.0
        for frame in f.readlines()[1:]:
            line = frame.strip().split(" ")
            timestamp = line[0]
            camera_intrinsics = line[1:7]
            camera_pose = np.asarray(list(map(float, line[7:19])))

            values = [float(v) for j, v in enumerate(line) if j > 6]

            R = np.array(
                [
                    [values[0], values[1], values[2]],
                    [values[4], values[5], values[6]],
                    [values[8], values[9], values[10]],
                ]
            )
            t = np.array([values[3], values[7], values[11]])
            R_inv = np.linalg.inv(R)
            T = -R_inv @ t
            transform_matrix = np.array(
                [
                    [R[0, 0], R[0, 1], R[0, 2], t[0]],
                    [R[1, 0], R[1, 1], R[1, 2], t[1]],
                    [R[2, 0], R[2, 1], R[2, 2], t[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            inv_transform_matrix = np.array(
                [
                    [R_inv[0, 0], R_inv[0, 1], R_inv[0, 2], T[0]],
                    [R_inv[1, 0], R_inv[1, 1], R_inv[1, 2], T[1]],
                    [R_inv[2, 0], R_inv[2, 1], R_inv[2, 2], T[2]],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
            inv_transform_matrix = transform_matrix @ inv_transform_matrix

            pose = transform_matrix[0:3, 3]
            if last_pose is not None:
                total_len += np.linalg.norm(last_pose - pose)
            last_pose = pose
        if total_len > 5:
            good_traj.append(d + fname)
good_traj = np.asarray(good_traj)
msgpack_numpy.pack(
    good_traj,
    open("/srv/share/datasets/RealEstate10k/trajectory_names.msg", "wb"),
    use_bin_type=True,
)