# NFM2021_Static_Test_Synthesis

This repository contains the code required to reproduce the figures for "Synthesizing Static Test Environments for Observing Sequence-like Behaviors in Autonomy".
<p align="center">
<img src="https://github.com/abadithela/NFM2021_Static_Test_Synthesis/blob/main/examples/movie_ex2.gif" width="500" />
</p>

In the above simulation, we see how constraints on gridworld transitions are iteratively synthesized such that any agent starting at p1 must visit p2 before it can navigate to p3.
### Prerequisites

The following packeges needed for running the code can be installed using pip:

```
pip install networkx
pip install numpy
pip install matplotlib
pip install scipy
```

The code also requires Gurobi for Python to run successfully. Instructions for installation can be found [here](https://www.gurobi.com/academia/academic-program-and-licenses/). This code was implemented in Anaconda on a MacOS Catalina. Instructions for installing setting up Gurobi for Anaconda can be found [here](https://www.gurobi.com/gurobi-and-anaconda-for-mac/).
## Instructions

To re-create the figures, navigate to the examples folder and run the scripts: simple_automaton.py, gridworld.py, and parametered_grid.py for the simple automaton example, the gridworld example, and the average runtime for random gridworlds.
