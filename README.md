# NFM2021_Static_Test_Synthesis

This repository contains the code required to reproduce the figures for "Synthesizing Static Test Environments for Observing Sequence-like Behaviors in Autonomy".
<p align="center">
<img src="https://github.com/abadithela/NFM2021_Static_Test_Synthesis/blob/main/examples/movie_ex2.gif" width="500" />
</p>

In the above simulation, we see how constraints on gridworld transitions are iteratively synthesized such that any agent starting at p1 must visit p2 before it can navigate to p3.
### Prerequisites

The packeges needed for running the code can be installed using pip:

```
pip install networkx
pip install numpy
pip install matplotlib
pip install scipy
pip install pickle
```
## Instructions

To re-create the figures, navigate to the examples folder and run the scripts: simple_automaton.py, gridworld.py, and parametered_grid.py for the simple automaton example, the gridworld example, and the average runtime for random gridworlds.
