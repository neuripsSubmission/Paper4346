# Data Generation: Trajectories & Graph 

This directory contains the code for creating the passive dataset for training our image nav modules


## Trajectories

The folder `trajectory_data/` contains code generating train and test epsiodes. Uses submitit to generate faster over multi gpu:
1. randomly generating non shortest path trajectories through the scenes for nrns
    1.1 run `generate_nrns_train_trajectories.py`
2. creating random train episodes for the habitat rl image nav agent
    2.1 run `generate_habitat_train_instances.py`
    2.2 run `combine_habitat_train_instances.py`
3. creating random test episodes to be run on habitat and nrns
    1.1 run `generate_test_instances.py`


## Graphs

The folder `graph_data/` contains code for:

1. Taking the info from the simulated trajectories, storing it in a graph form and clustering the step-wise graph to a topologolgical graph.
    For each trajectories it saves a clustered graph with the following info:
    * nodes
    * edges
    * edge_attrs - (pose difference)
    * invalid_points - (per each node -list of invalid points)
    * valid_points - (per each node -list of valid points)
    * floor
    * scene
    * scan_name



