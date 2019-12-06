"""Module for the Risk-Exploiting Cargo Transport Scenario
 
- Network of agents are to be used to transport packages from a known origin terminal to a 
known destination terminal
- Cargo packages appear at origin terminal periodically and/or stochastically.
- Any agent my pick up package by visiting origin terminal when package has appeared. That 
same agent must then deliver it to the destination, no hand offs are possible
- Rewards are recieved per package delivered to destination terminal
- Hazardous landmarks and/or adversarial agents are present in the environment that can cause
good agents to fail which comes with an associated cost.
- Failure of the package carrying agent (i.e. loss of package) is associated with very large
cost, significantly greater than normal agent failure
- Landmarks can also have a risk associated with them, i.e. probability of causing a nearby
agent to fail.
- Agents actions are their movements. Collection of package happens automatically if agent visits
origin terminal when package is present
- General case: 
    - Location of landmarks and adversaries are not globally known and can only be locally 
    sensed locations and unknown nature (i.e. risk,
    - Package carry has reduced maneuvering capabilities and speed
"""

import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario


