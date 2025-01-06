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
