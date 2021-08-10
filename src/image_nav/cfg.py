import argparse
import datetime

parser = argparse.ArgumentParser(description="Image Nav Task")


# What you are doing
parser.add_argument("--dataset", type=str, default="gibson")
parser.add_argument("--run_type", type=str, default="straight")
parser.add_argument("--visualize", default=False, action="store_true")
parser.add_argument(
    "--single", default=False, action="store_true"
)  # how many episodes to run

parser.add_argument("--processed_data_file", type=str, default="test_hard.pkl")

# Baselines
parser.add_argument("--behavioral_cloning", default=False, action="store_true")
parser.add_argument("--bc_type", type=str, default="gru", help="gru, map, random")

# Abalations/GT
parser.add_argument("--switch_func", default=False, action="store_true")
parser.add_argument("--distance_func", default=False, action="store_true")
parser.add_argument("--feat_prediction", default=False, action="store_true")
parser.add_argument("--validity_prediction", default=False, action="store_true")
parser.add_argument("--point_nav", default=False, action="store_true")

# NOISE
parser.add_argument("--pose_noise", default=False, action="store_true")
parser.add_argument("--actuation_noise", default=False, action="store_true")

# RANDOM
parser.add_argument("--sample_used", type=float, default=1.0)
parser.add_argument("--max_steps", type=int, default=500)

# Data/Input Paths
parser.add_argument(
    "--base_dir",
    type=str,
)
parser.add_argument(
    "--sim_dir", type=str, default="/srv/share/datasets/habitat-sim-datasets/"
)
parser.add_argument("--test_dir", type=str, default="image_nav_episodes/")

parser.add_argument(
    "--scene_dir",
    type=str,
    default="../data_splits/matterport/scenes_",
)
parser.add_argument(
    "--graph_dir",
    type=str,
)

parser.add_argument("--floorplan_dir", type=str, default="floorplans/")
parser.add_argument("--visualization_dir", type=str)

parser.add_argument("--habitat_dir", type=str)


# Models
parser.add_argument("--model_dir", type=str)
parser.add_argument(
    "--mp3d_feat_model_path",
    type=str,
    default="feat_pred/deepgcn_CGCon_dl_unseenAcc0.47_epoch2.pt",  # NO NOISE
    # default="feat_pred/deepgcn_CGCon_dl_noise_mp3d_unseenAcc0.42_epoch23.pt",  # NOISY
    # default="feat_pred/deepgcn_RE10.pt",  # NO NOISE Realestate
)
parser.add_argument(
    "--gibson_feat_model_path",
    type=str,
    default="feat_pred/deepgcn_CGCon_dl_gibson_unseenAcc0.38_epoch22.pt",  # NO NOISE
    # default="feat_pred/deepgcn_CGCon_dl_noise_gibson_unseenAcc0.41_epoch13.pt",  # NOISY
    # default="feat_pred/CGConv_mpnn_gibson_unseenAcc0.40_epoch5.pt",  # NO NOISE MPNN
    # default="feat_pred/deepgcn_RE10.pt",  # NO NOISE Realestate
)

parser.add_argument(
    "--mp3d_goal_model_path",
    type=str,
    default="switch_func/goal_mlp_noisy_mp3d_distAcc0.80_epoch13.pt",  # NO NOISE
    # default="switch_func/goal_mlp_noisy_mp3d_distAcc0.01_epoch9.pt",  # NOISY
    # default="switch_func/goal_mlp_RE10.pt",  # NO NOISE Realestate
)
parser.add_argument(
    "--gibson_goal_model_path",
    type=str,
    default="switch_func/goal_mlp_gibson_distAcc0.94_epoch10.pt",  # NO NOISE
    # default="switch_func/goal_mlp_noisy_gibson_distAcc1.00_epoch1.pt",  # NOISY
    # default="switch_func/goal_mlp_RE10.pt",  # NO NOISE Realestate
)

parser.add_argument(  # Behavioral Cloning
    "--mp3d_bc_model_path",
    type=str,
    default="action_func/action_network_transformer_Acc0.50_epoch0.pt",  # ResNet + Prev action + GRU + weighting
    # default="action_func/action_network_map_mp3d_Acc1.12_epoch3.pt", #Metric Map + GRU + weighting
    # default="action_func/action_network_mlp_Acc0.51_epoch4.pt",  # MLP
    # default="action_func/action_network_gru_Acc0.65_epoch18.pt",  # GRU
)

parser.add_argument(  # Behavioral Cloning
    "--gibson_bc_model_path",
    type=str,
    # default="action_func/action_network_map_gibson_Loss1.17_Epoch4.pt"  # Metric Map + GRU + weighting
    default="action_func/action_network_gru_gibson_Loss1.05_Epoch3.pt",  # ResNet + Prev action + GRU + weighting
)


def parse_args():
    args = parser.parse_args()
    args.base_dir += f"{args.dataset}/"
    args.test_dir = f"{args.base_dir}{args.test_dir}"

    if args.dataset == "mp3d":
        args.sim_dir += "mp3d/"
        args.bc_model_path = args.mp3d_bc_model_path
        args.switch_model_path = args.mp3d_switch_model_path
        args.goal_model_path = args.mp3d_goal_model_path
        args.feat_model_path = args.mp3d_feat_model_path
        args.floorplan_dir = f"{args.base_dir}{args.floorplan_dir}"
    else:
        args.sim_dir += "gibson_train_val/"
        args.switch_model_path = args.gibson_switch_model_path
        args.bc_model_path = args.gibson_bc_model_path
        args.goal_model_path = args.gibson_goal_model_path
        args.feat_model_path = args.gibson_feat_model_path
    return args
