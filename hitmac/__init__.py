"""HiT-MAC: Hierarchical Target-oriented Multi-Agent Coordination.

A hierarchical multi-agent reinforcement-learning agent for the MATE
(Multi-Agent Tracking Environment) distributed target-coverage task.

The agent decomposes cooperative multi-camera target coverage into two
levels that are trained jointly with asynchronous advantage actor-critic
(A3C):

* a **Coordinator** that decides *which* targets each camera should attend
  to -- a discrete goal-assignment policy whose multi-agent credit is
  estimated with a Shapley-value critic, and
* an **Executor** that decides *how* each camera should move -- a continuous
  pan/zoom control policy.

Reference: Fangwei Zhong et al., "Towards Distributed Multi-Agent
Coordination via Hierarchical Target-oriented Communication" (HiT-MAC).
"""

__version__ = "0.1.0"
