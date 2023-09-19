### Demo: `plan_ex.ipynb`

- Loads
    - Pre-sampled points
    - Pre-computed reachability sets
- Initializes the Safe Planner
- Plans from $(0.2,0.4)$ to $(0.6,0.8)$
- Plots generated tree and trajectory

### Safe Planner

- Planning Algorithm
    - `plan`: main planning function
        - Inputs:
            
            `state`: current 4-dimensional state $(x,y,v_x,v_y)$
            
            `*boxes`: predicted bounding boxes
            
        - Outputs:
            
            `idx_solution`: node indices of the solution trajectory
            
            `x_waypoints`: states to visit at each discretized time
            
            `u_waypoints`: control inputs at each discretized time
            
    - `solve`: Main FMT*
        - extend until goal found or maximum iteration reached
        - If goal found, trace back tree to find path
    - `extend`: inner loop of FMT*
        - Find lowest-cost node
        - Find forward-reachable nodes $X_\text{near}$
        - For each forward-reachble node $x$
            - Find backward-reachable nodes $Y_\text{near}$
            - Find lowest-cost node $y_\text{min}$
            - Check collision
            - Check ICS before next sensor update
            - Connect if free
- Preparation
    - `find_all_reachable`: should be executed offline to find all reachability connections
    - `load_reachable`: loads pre-computed reachability sets to planner
    - `filter_neighbors`: non-dynamics version of reachability connection
    - `goal_inter`: finds intermediate goal if final goal cannot be reached
- Plots
    - `plot_reachable`: connects reachable pairs of points with optimal trajectories
    - `show`: plots solution trajectory
    - `show_connection`: plots all connection drawn and solution trajectory