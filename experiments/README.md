# Expirements

This directory holds training results and trained policies in experiment directories labeled `expdata.YYYYMMDD`. Note that this directories are ignored by default in .gitignore in order to prevent large amounts of data being pushed to the remote repo.

To visualize a trained policy, you can run a command from the root project directory:
```
python train.py --environment MultiAgentRiskEnv --scenario ergo_hazards --load-dir ./experiments/expdata.20180823/policy/ --display
```

To visualize the learning curves, you can run the following in python within the relevant expdata directory:
```
import pickle
import matplotlib.pyplot as plt
a = pickle.load(open("default_rewards.pkl", "rb"))
plt.plot(a)
plt.show()
```

For MAPPO or other baselines-derived algorithms, you can visualize policy and value loss with:
```
l = pickle.load(open("default_losses.pkl", "rb"))
t = np.arange(len(l[1]))
_,ax1=plt.subplots();ax1.plot(t,[v[3]for v in l[1]],'b');ax1.set_ylabel('policy_loss',color='b');ax2=ax1.twinx();ax2.plot(t,[v[4]for v in l[1]],'r');ax2.set_ylabel('value_loss',color='r')
plt.show()
```
