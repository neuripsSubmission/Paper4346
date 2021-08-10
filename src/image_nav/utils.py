import numpy as np
import torch
from src.utils.sim_utils import get_geodesic_dist
from src.functions.target_func.goal_mlp import GoalMLP
from src.functions.distance_func.deepgcn import TopoGCN


"""Evaluate Episode"""


def evaluate_episode(agent, args, length_shortest):
    success = False
    dist_thres = 1.0
    dist_to_goal = np.linalg.norm(
        np.asarray(agent.goal_pos.tolist()) - np.asarray(agent.current_pos.tolist())
    )
    # test oracle stop within 3m
    if not args.switch_func and not args.behavioral_cloning:
        dist_thres = 3.0
        agent.length_taken += dist_to_goal
    if dist_to_goal <= dist_thres and agent.steps < args.max_steps:
        success = True
    episode_spl = calculate_spl(success, length_shortest, agent.length_taken)
    return dist_to_goal, episode_spl, success


def calculate_spl(success, length_shortest, length_taken):
    spl = (length_shortest * 1.0 / max(length_shortest, length_taken)) * success
    return spl


"""Load Models"""


def load_models(args):
    """Action Pred function"""
    if args.bc_type == "map":
        from src.functions.bc_func.bc_map_network import ActionNetwork
    else:
        from src.functions.bc_func.bc_gru_network import ActionNetwork
    model_action = ActionNetwork()
    model_action.load_state_dict(torch.load(args.model_dir + args.bc_model_path))
    model_action.to(args.device)
    model_action.eval()

    """Load Target function"""
    model_goal = GoalMLP()
    model_goal.load_state_dict(torch.load(args.model_dir + args.goal_model_path))
    model_goal.to(args.device)
    model_goal.eval()

    """Load Global Policy (distance function)"""
    model_feat_pred = torch.load(args.model_dir + args.feat_model_path)
    print(sum(p.numel() for p in model_feat_pred.parameters()))

    model_feat_pred.to(args.device)
    model_feat_pred.eval()

    return model_goal, model_goal, model_feat_pred, model_action
