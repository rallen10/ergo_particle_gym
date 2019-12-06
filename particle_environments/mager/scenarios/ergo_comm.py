"""Module for the Risk-Exploiting Comm Relay Scenario
 
- Network of agents are to send messages between origin and destination terminals of known positions.
- Messages are in the form of bit strings.
- The aggregate reward depends on the message quality received by the destination terminal, 
i.e. how much it differs from the message broadcast at the origin terminal
- The message quality (i.e. probability of flipping a bit) is a function of the 
distance between agents (larger distances, higher probability of flipping a bit), and 
proximity to non-terminal landmarks which can increase or decrease probability of bit flips
- Landmarks can also have a risk associated with them, i.e. probability of causing a nearby
agent to fail.
- Agents actions are their movements as well as what message to transmit best on messages received
- Most general case: landmarks are at unknown locations and unknown nature (i.e. risk,
signal degredation) and part of the problem is to explore for landmarks and learn their nature
- Simplified case: to accelerate testing and learning, a simplified case has the landmarks
at known locations with known nature. 
"""

import numpy as np
from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
