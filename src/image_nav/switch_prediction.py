import torch
import numpy as np
import quaternion
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_sim.geo import UP
from src.utils.sim_utils import get_geodesic_dist, se3_to_mat
from src.functions.validity_func.local_nav import LocalAgent, loop_nav

"""
predict if you want to switch to local navigation
"""


def predict_switch(args, agent, visualizer):
    switchThreshold = 0.5
    switch = False
    if args.switch_func:
        # use trained distance function
        with torch.no_grad():
            batch_goal = agent.goal_feat.clone().detach()
            batch_nodes = (
                agent.node_feats[agent.current_node, :].clone().detach().unsqueeze(0)
            )

            switch_pred, rho, phi = agent.goal_model(
                batch_nodes.to(args.device), batch_goal.to(args.device)
            )
            switch_pred = switch_pred.detach().cpu()[0].item()
            if switch_pred >= switchThreshold:
                rho = rho.cpu().detach().item()
                phi = phi.cpu().detach().item()
                if agent.point_nav:
                    localnav(agent, rho, phi, visualizer)
                else:
                    localnav_check(agent, rho, phi)
            else:
                switch = False

    else:
        # use ground truth switch (for ablation experiments)
        true_dist = get_geodesic_dist(
            agent.pathfinder, agent.current_pos.tolist(), agent.goal_pos.tolist()
        )
        if true_dist <= 3:
            switch = True
    return switch


def localnav_check(agent, rho, phi):
    stateA = se3_to_mat(
        quaternion.from_float_array(agent.current_rot.numpy()),
        np.asarray(agent.current_pos.numpy()),
    )
    stateB = (
        stateA
        @ se3_to_mat(
            quat_from_angle_axis(phi, UP),
            np.asarray([0, 0, 0]),
        )
        @ se3_to_mat(
            quaternion.from_float_array([1, 0, 0, 0]),
            np.asarray([0, 0, -1 * rho]),
        )
    )
    final_pos = stateB[0:3, 3]
    final_rot = quaternion.as_float_array(
        quaternion.from_rotation_matrix(stateB[0:3, 0:3])
    )
    if not agent.point_nav:
        agent.current_pos = torch.tensor(final_pos)
        agent.current_rot = torch.tensor(final_rot)
        agent.length_taken += rho
    else:
        pass
        # print("nav check", np.linalg.norm(agent.goal_pos - final_pos))

    return np.linalg.norm(agent.goal_pos - final_pos)


def run_vis(agent, visualizer, prev_poses):
    for p in prev_poses:
        img = agent.sim.get_observations_at(p[0], quaternion.from_float_array(p[1]),)[
            "rgb"
        ][:, :, :3]
        agent.current_pos, agent.current_rot = torch.tensor(p[0]), torch.tensor(p[1])
        visualizer.seen_images.append(img)
        visualizer.current_graph(agent, switch=True)


def localnav(agent, rho, phi, visualizer):
    agent.sim.set_agent_state(
        agent.current_pos.numpy(),
        quaternion.from_float_array(agent.current_rot.numpy()),
    )
    agent.switch_index = len(agent.prev_poses)
    agent.prev_poses.append([agent.current_pos.numpy(), agent.current_rot.numpy()])
    try:
        agent.sim.set_agent_state(
            agent.current_pos.numpy(),
            quaternion.from_float_array(agent.current_rot.numpy()),
        )
        local_agent = LocalAgent(
            agent.actuation_noise,
            agent.pose_noise,
            agent.current_pos.numpy(),
            agent.current_rot.numpy(),
            map_size_cm=1200,
            map_resolution=5,
        )
        final_pos, final_rot, nav_length, prev_poses = loop_nav(
            agent.sim,
            local_agent,
            agent.current_pos.numpy(),
            agent.current_rot.numpy(),
            rho,
            phi,
            min(100, 499 - agent.steps),
        )
        if agent.visualize:
            run_vis(agent, visualizer, prev_poses)
        agent.prev_poses.extend(prev_poses)
        agent.current_pos = torch.tensor(final_pos)
        agent.current_rot = torch.tensor(final_rot)
        agent.length_taken += nav_length
        return np.linalg.norm(agent.goal_pos - final_pos)
    except:
        print("ERROR: local navigation through error")

    return np.linalg.norm(agent.goal_pos - agent.current_pos.numpy())
