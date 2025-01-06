# IPLUX

- 'QP.py': main program.

  - Change problem types by setting the **part** in line 19-21, QP.py

    ```python
    #part = 'l1'
    part = 'smooth' #currently running the smooth problem
    #part = 0
    ```

- Algorithms:
  - IPLUX
  - otheralg: subgradient and primal dual methods
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


## Cite our paper:
@ARTICLE{9762539,
  author={Wu, Xuyang and Wang, He and Lu, Jie},
  journal={IEEE Transactions on Automatic Control}, 
  title={Distributed Optimization With Coupling Constraints}, 
  year={2023},
  volume={68},
  number={3},
  pages={1847-1854},
  doi={10.1109/TAC.2022.3169955}}
