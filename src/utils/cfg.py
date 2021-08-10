import argparse
import datetime

# Data/Input Paths
def input_paths(parser):
    parser.add_argument(
        "--noise",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gibson",
        # default="mp3d",
        # default="realestate10",
    )
    parser.add_argument(
        "--sim_dir",
        type=str,
    )
    parser.add_argument(
        "--data_splits",
        type=str,
    )
    parser.add_argument(
        "--base_dir",
        type=str,
    )
    parser.add_argument(
        "--floorplans",
        type=str,
        default="floorplans/",
    )
    # generated data
    parser.add_argument(
        "--trajectory_data_dir",
        type=str,
        default="trajectory_data/",
    )
    parser.add_argument(
        "--clustered_graph_dir",
        type=str,
        default="clustered_graph/",
    )
    parser.add_argument(
        "--distance_data_dir",
        type=str,
        default="distance_data_straight/",
    )
    parser.add_argument(
        "--action_dir",
        type=str,
        default="behavioral_cloning/",
    )
    # save folders
    parser.add_argument(
        "--visualization_dir",
        type=str,
    )
    parser.add_argument(
        "--submitit_log_dir",
        type=str,
    )
    parser.add_argument(
        "--saved_model_dir",
        type=str,
    )

    return parser
