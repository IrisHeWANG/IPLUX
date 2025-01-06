# IPLUX
**IPLUX**, is designed to solve distributed convex optimization problems with both inequality and equality constraints, where the objective function can be a general nonsmooth convex function, and all the constraints can exhibit both sparse and dense coupling. 

## Main results
By strategically integrating ideas from primal-dual, proximal, and virtual-queue optimization methods, we develop a novel distributed algorithm, referred to as IPLUX, to address the problem over a connected, undirected graph. We show that IPLUX achieves an $O(1/K)$ rate of convergence in terms of optimality and feasibility, which is stronger than the convergence results of the alternative methods and eliminates the standard assumption on the compactness of the feasible region. Finally, IPLUX exhibits faster convergence and higher efficiency than several state-of-the-art methods in the simulation.

## Simulation files
- 'QP.py': main program.

  - Change problem types by setting the **part** in line 19-21, QP.py

    ```python
    #part = 'l1'
    part = 'smooth' #currently running the smooth problem
    #part = 0
    ```
- Implemented algorithms can be found in:
  - IPLUX.py
  - otheralg.py: subgradient and primal dual methods
- network_generator.py:
  - generate the graph based on problem and role of each node (network_generator_role.py).
  - plot the graph: general graph and graph with roles.
  - adjacency matrix
- problem generator.py:
  - generate the problem
- opt.py:
  - calculate the optimal solution.
- utils.py
  - error evaluation function
  - check assumption functions: strong Slater's constraint, parameter assumption....


## Reference
Please cite IPLUX in your publications if it helps your research:
```
@ARTICLE{9762539,
  author={Wu, Xuyang and Wang, He and Lu, Jie},
  journal={IEEE Transactions on Automatic Control}, 
  title={Distributed Optimization With Coupling Constraints}, 
  year={2023},
  volume={68},
  number={3},
  pages={1847-1854},
  doi={10.1109/TAC.2022.3169955}}
```
