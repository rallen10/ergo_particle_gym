# Scenarios
This module when imported allows the integrated test script to either load scenarios from this directory or scenarios in the multiagent-particle-envs package.

**NOTE**: The \_\_init\_\_.py file assumes that the multiagent-particle-envs directory has been appended to the `sys.path` from the integrated test script.

Scenarios prefixed with `simple` imply there is no form of hazard, risk-taking, or unit health inherent to the problem. In contrast, scenarios prefixed with `ergo` imply that hazards are present in the environment that can cause agents to be damaged and/or terminated
