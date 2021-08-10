from src.utils.sim_utils import get_num_steps, diff_rotation_signed
from src.image_nav.gen_ghost_nodes import generate_ghost
from src.image_nav.feat_prediction import predict_features
from src.image_nav.switch_prediction import predict_switch

import torch
import numpy as np
import quaternion
import math
import networkx as nx

turn_angle = 15


def single_image_nav(agent, visualizer, args):
    # set vars
    reached_goal = False
    prev_pos = None

    switch = predict_switch(args, agent, visualizer)
    if switch:
        reached_goal = True

    if agent.visualize:
        visualizer.set_start_images(agent.rgb_img, agent.goal_img)
        visualizer.current_graph(agent)

    """While not reached goal location OR exceed max steps:"""
    step = 0
    while not reached_goal and agent.steps <= args.max_steps:

        """
        1) Graph Expansion (G_EA)
            # uses validity function to determine where unexplored nodes can be added
            # adds unexplore nodes to graph
        """
        generate_ghost(agent)

        """ 
        2) End the navigation if no more unexplored nodes
        """
        if len(agent.unexplored_nodes) == 0:
            print("no more to explore")
            break

        """
        3) Global Policy (G_D)
                # adds pred dist to dist from cur node to ghost nodes
                # selects subgoal as node with lowest cost
        """
        next_node = predict_features(agent)

        """
        4) Navigate to subgoal location
            # 
            # update graph and agent with new observations at subgoal
            # add steps to counter
        """
        prev_pos = agent.current_pos.clone().detach().numpy()
        prev_rot = agent.current_rot.clone().detach().numpy()
        agent.node_poses[next_node] = torch.tensor(
            agent.pathfinder.snap_point(agent.node_poses[next_node])
        )
        next_pos = agent.node_poses[next_node]
        next_rot = agent.node_rots[next_node]
        if agent.validity_prediction:
            closest_dist = math.inf
            closest_connected = None
            angle_connected = None

            # Find the best parent of unexplored noded
            for edge in [list(e) for e in agent.graph.edges]:
                if agent.graph.nodes[edge[0]]["status"] == "explored":
                    if edge[1] == next_node:
                        closest_pose = agent.node_poses[edge[0]].numpy()
                        edge_distance = np.linalg.norm(closest_pose - next_pos.numpy())
                        if edge_distance < closest_dist:
                            closest_connected = edge[0]
                            closest_dist = edge_distance

            closest_pose = agent.node_poses[closest_connected].numpy()
            closest_rot = quaternion.from_float_array(
                agent.node_rots[closest_connected].numpy()
            )

            obs, closest_pose, closest_rot = single_nav(
                agent,
                obs,
                prev_pos,
                prev_rot,
                closest_pose,
                closest_connected,
                visualizer,
            )

            angle_connected = round(
                diff_rotation_signed(
                    closest_rot,
                    quaternion.from_float_array(next_rot.numpy()),
                ).item()
            )

            if closest_dist < 0.1:
                turns = abs(round(angle_connected / turn_angle))
                for _ in range(turns):
                    obs, pose, rotation = agent.take_step("left")
                    agent.steps += 1
            else:
                obs, pose, rotation = single_nav(
                    agent,
                    obs,
                    closest_pose,
                    closest_rot,
                    next_pos,
                    next_node,
                    visualizer,
                )
            agent.update_agent(next_node, pose, rotation, obs)
        else:
            """snap to grid and teleport"""
            agent.node_poses[next_node] = torch.tensor(
                agent.pathfinder.snap_point(agent.node_poses[next_node])
            )
            agent.update_agent(next_node)
            agent.length_taken += np.linalg.norm(agent.current_pos.numpy() - prev_pos)
            agent.steps += (
                get_num_steps(
                    agent.sim,
                    prev_pos,
                    quaternion.from_float_array(prev_rot),
                    agent.current_pos.numpy(),
                )
                + 5
            )

        # Visualization
        if args.visualize:
            agent.map_images.append(agent.topdown_grid)
            visualizer.seen_images.append(agent.rgb_img.copy())
            visualizer.current_graph(agent)

        agent.localize_ue()

        # Add Steps to Counter
        if agent.steps >= args.max_steps or step >= 500:
            print("max steps")
            break
        step += 1

        """
        7) Target Function
            # predicts location of target
            # predicts if target within sight and close
            # if so, switches to local nav 
        """
        switch = predict_switch(args, agent, visualizer)
        if switch:
            reached_goal = True
            break


from src.functions.validity_func.validity_utils import (
    get_relative_location,
    get_sim_location,
)
from src.functions.validity_func.local_nav import LocalAgent


def single_nav(agent, obs, start_pos, start_rot, goal_pos, next_node, visualizer):
    local_agent = LocalAgent(
        actuation_noise=False,
        pose_noise=False,
        curr_pos=start_pos,
        curr_rot=start_rot,
        map_size_cm=1200,
        map_resolution=5,
    )
    prev_pos = start_pos
    terminate_local = 0
    delta_dist, delta_rot = get_relative_location(start_pos, start_rot, goal_pos)
    agent.prev_poses.append([start_pos, start_rot])
    local_agent.update_local_map(obs["depth"])
    local_agent.set_goal(delta_dist, delta_rot)
    action, terminate_local = local_agent.navigate_local()
    for _ in range(15):
        obs = agent.sim.step(action)
        if agent.visualize:
            visualizer.seen_images.append(obs["rgb"].copy())
            visualizer.current_graph(agent, switch=True)
        curr_depth_img = obs["depth"]
        curr_pose = agent.sim.get_agent_state().position
        curr_rot = quaternion.as_float_array(agent.sim.get_agent_state().rotation)
        delta_dist, delta_rot = get_relative_location(curr_pose, curr_rot, goal_pos)
        local_agent.new_sim_origin = get_sim_location(
            curr_pose, agent.sim.get_agent_state().rotation
        )
        local_agent.update_local_map(curr_depth_img)
        action, terminate_local = local_agent.navigate_local()
        agent.steps += 1

        agent.length_taken += np.linalg.norm(curr_pose - prev_pos)
        prev_pos = curr_pose
        agent.prev_poses.append([curr_pose, curr_rot])
        if terminate_local == 1:
            break

    angle_connected = round(
        diff_rotation_signed(
            quaternion.from_float_array(curr_rot),
            quaternion.from_float_array(agent.node_rots[next_node].numpy()),
        ).item()
    )
    turns = abs(round(angle_connected / turn_angle))
    for _ in range(turns):
        if angle_connected >= 0:
            obs, curr_pose, curr_rot = agent.take_step("left")
        else:
            obs, curr_pose, curr_rot = agent.take_step("right")
        agent.prev_poses.append([curr_pose, curr_rot])
        agent.steps += 1
    return obs, curr_pose, curr_rot
