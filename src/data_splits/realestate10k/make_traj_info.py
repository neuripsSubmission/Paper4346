import os
import numpy as np
import msgpack_numpy
import quaternion
import json

base_dir = "/srv/share/datasets/RealEstate10k/RealEstate10K/"
file = "/srv/share/datasets/RealEstate10k/trajectory_names.msg"
good_trajs = msgpack_numpy.unpack(open(file, "rb"), raw=False)
print("total trajectories " + len(good_trajs))
infos = {}
for fname in good_trajs:
    f = open(base_dir + fname, "r")
    last_pose = None
    total_len = 0.0
    poses, rots = [], []
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
        state_matrix = np.array(
            [
                [R[0, 0], R[0, 1], R[0, 2], t[0]],
                [R[1, 0], R[1, 1], R[1, 2], t[1]],
                [R[2, 0], R[2, 1], R[2, 2], t[2]],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        rotation = quaternion.as_float_array(
            quaternion.from_rotation_matrix(state_matrix[0:3, 0:3])
        )
        rots.append(rotation)
        pose = state_matrix[0:3, 3]
        poses.append(pose)
        if last_pose is not None:
            total_len += np.linalg.norm(last_pose - pose)
        last_pose = pose
    infos[fname] = {"poses": poses, "rots": rots}

with open("/srv/flash1/userid/toponav/RE10/traj_infos.json", "r") as f:
    json.dump(infos, f)
