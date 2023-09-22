![](https://github.com/zakaria-narjis/Emergent-flocking-with-reinforcement-learning/scatter.gif)

# Emergent Flocking Behaviour using Reinforcement Learning

This is the Python codes related to the following article: Zakaria Narjis

# Abstract of the article

Flocking behaviour, a widespread phenomenon in the natural world, represents coordination and collective motion observed among diverse species. Traditional approaches are mostly used to model this behaviour. However, these approaches rely on static flocking rules, limiting their adaptability to dynamic real-world scenarios. The challenge lies in effectively understanding and using this complex behaviour for practical applications. In this study, we present an approach using reinforcement learning to address this challenge. Our aim is to train autonomous agents to replicate flocking behaviour within a continuous 2D environment. The approach involves using a reward function to imitate flocking behaviour with an artificially generated flock. By overcoming these limitations, our study offers a deeper understanding of natural systems and broadens the scope for controlling swarming behaviours in various domains and environments.

# Dependencies 
1. Agentpy 0.0.23
2. Gymnasium 0.29.1
3. pfrl 0.4.0
4. pytorch 2.0.1
5. pandas 2.1.1
6. matplotlib 3.8.0

# Future Work
1. Exploring of continuous action space algorithms like DDPG for improved performance and reward convergence. Further optimization of the reward function is also suggested.
2. Addressing partial observability challenges by integrating Deep Recurrent Q-Learning Networks (DRQN) or other memory architectures like HCAM (Hierarchical Chunk Attention Memory) and extending research to include variable inner and outer radius parameters to enhance the agent's adaptability and understanding of distant flock behaviors, potentially advancing our understanding and emulation of complex flocking dynamics.
   
# Citation
