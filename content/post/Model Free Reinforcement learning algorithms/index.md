---
title: "Model Free Reinforcement Learning Algorithms"
date: 2014-01-21T11:54:26-08:00
tags: [reinforcement learning, sutton, monte carlo]
---

{{< figure
img="image0.png" 
command="Resize" 
options="700x" >}}

I struggled with the intuition behind Sutton’s [Learning to Predict by the Methods of Temporal Differences paper](https://www.semanticscholar.org/paper/Learning-to-predict-by-the-methods-of-temporal-Sutton/094ca99cc94e38984823776158da738e5bc3963d). I hit the wall early on the “A Random-Walk” example (Page 19 3.2). I read Chapter 6 and 12 of Sutton’s Reinforcement Learning textbook to gain more intuition. [(available for free)](http://incompleteideas.net/book/the-book.html). To work through my reproduction of the "A Random Walk" below, I recommend at minimum the reader has a basic understanding of [Value-Functions](https://en.wikipedia.org/wiki/Reinforcement_learning#Value_function) and [Q-Learning](https://en.wikipedia.org/wiki/Q-learning).

A critical aspect of research is the reproduction of previously published results. Yet most will find reproduction of research challenging since important parameters needed to reproduce results are often not stated in the papers. I’ve noticed in the past 5 years there has been a sort of catharsis regarding the lack of reproducibility [[1]](https://www.nature.com/collections/prbfkwmwvz/)[[2]](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5220388/)[[3]](https://www.sciencemag.org/news/2016/02/if-you-fail-reproduce-another-scientist-s-results-journal-wants-know?r3f_986=http://kldavenport.com/suttons-temporal-difference-learning/). This isn’t an issue for wetlab science alone [[4]](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.1002165). The obvious benefit of reproduction is to aid in your own understanding of the results. This then enables one to extend and compare new contributions to existing publications.

Reproducibility means different things to many people working the applied sciences space. The continuum appears to be sharing:
1. Code (.py, .cpp files)
2. Jupyter Notebook
3. Docker Container 

With #1 and #2 you are putting the onus on the recepient to have the same environment as you, maybe you point them to the same Anaconda distribution. With #3 you get an easy way to share your working environments including libraries and drivers. On a related note please check out this entertaining and education talk on ["I Don't Like Notebooks"](https://docs.google.com/presentation/d/1n2RlMdmv1p25Xy5thJUhkKGvjtV-dkAIsUXP-AL4ffI/edit) by Joel Grus.

## TD-Learning
Temporal-Difference (TD) algorithms work without an explicit model and learn from the experience and outcomes of iterating over multiple episodes (or sequences). TD Learning is similar to Monte Carlo methods, however TD can learn from individual steps without needing the final result. These are among other differences:


<table width="950">
<tbody>
<tr>
<td style="text-align: left;" width="355"><strong>Monte Carlo methods</strong></td>
<td style="text-align: left;" width="355"><strong>TD learning</strong></td>
</tr>
<tr>
<td style="text-align: left;">MC must wait until the end of the episode before the return is known.</td>
<td style="text-align: left;">TD can learn online after every step and does not need to wait until the end of episode.</td>
</tr>
<tr>
<td style="text-align: left;">MC has high variance and low bias.</td>
<td style="text-align: left;">TD has low variance and some decent bias.</td>
</tr>
<tr>
<td style="text-align: left;">MC does not exploit the Markov property.</td>
<td style="text-align: left;">TD exploits the Markov property.</td>
</tr>
</tbody>
</table>

Rather than computing the estimate of a next state, TD can estimate n-steps into future. In the case of TD-λ , we use lambda to set the myopicness of our reward emphasis.  The value of λ can be optimized to for a performance/speed tradeoff. The λ parameter is also called the trace decay parameter, with 0 ≤ λ ≤ 1. The higher the value, the longer lasting the traces. In this case, a larger proportion of credit from a reward can be given to more distant states and actions. λ = 1 is essentially Monte Carlo.

{{< figure
img="image1.png" 
command="Resize" 
options="700x" >}}

Figure 2 from the book. A generator of bounded random walks. This Markov process generated the data sequences in the example. All walks begin in state D. From states B, C, D, E, and F, the walk has a 50% chance of moving either to the right or to the left. If either edge state, A or G, is entered, then the walk terminates.

## Setting up our environment


```python
import math, sys, json, random 
import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.style
import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.externals import joblib # Still better than pickle in 2018?
# To protect from IPython kernel switching mistakes
from __future__ import division
%matplotlib inline
```

## Utility Functions

We implement the TD Lambda algorithm below as `tdlEstimate` the image below is from chapter 6 of the Sutton textbook.

{{< figure
img="image2.png" 
command="Resize" 
options="700x" >}}

```python
# TD Lambda
def tdlEstimate(alpha, _lambda, state_sequence, values):
    """
    alphas: array of arbitrary values (e.g. 0.005, 0.01, 0.015)
    _lambda: chosen from an arbitrary array (e.g. 0.1, 1)
    state sequence: an array chosen from an arbitrary set of sequence simulations such as [3, 4, 5, 6] or \
    [3, 4, 3, 2, 3, 4, 3, 4, 5, 6] per the MDP figure 2 above.
    returns: """
    
    # Per figure 2, we have 7 possible states, with two of them being end states (A,G)
    
    eligibility = np.zeros(7)
    updates     = np.zeros(7)

    for t in range(0, len(state_sequence) - 1):
        current_state = state_sequence[t]
        next_state = state_sequence[t+1]

        eligibility[current_state] += 1.0

        td = alpha * (values[next_state] - values[current_state])

        updates += td * eligibility
        eligibility *= _lambda

    return updates

# Simulator to generate random walk sequences in our MDP defined in fig 2 above
states = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

def simulate():
    """returns: a sequence of states picked from a uniform random sample such as 
    [3, 2, 1, 0] or [3, 4, 3, 4, 3, 4, 5, 6]
    """
    states = [3] # Start in center at "D"
    while states[-1] not in [0, 6]:
        states.append(states[-1] +  (1 if random.choice([True, False]) else -1)) # go left or right randomly

    return states


# Setup data for plots

random.seed(101)
# pg.20 gives true probabilities for states B, C, D, E, F
# truth = np.arange(1, 6) / 6.0
truth = [1 / 6, 1 / 3, 1 / 2, 2 / 3, 5 / 6]

dtype = np.float

num_train_sets = 100
num_sequences   = 10 # or episodes

training_sets = [[simulate() for i in range(num_sequences)] for i in range(num_train_sets)]
```

## Figure 3

```python
# Figure 3
alphas  = np.array([0.005, 0.01, 0.015], dtype=dtype)
lambdas = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0], dtype=dtype) # given in figure 3 caption

results = []

for _lambda in lambdas:
    for alpha in alphas:
        rmses = []
        for training_set in training_sets:
            # values initialized to zero and updates via tdlEstimate
            values = np.zeros(7, dtype=dtype)
            iterations = 0
            
            while True:
                iterations += 1
                before  = np.copy(values)
                updates = np.zeros(7, dtype=dtype)
                # The reward for reaching state "G" (element 7)
                values[6] = 1.0

                for sequence in training_set: 
                    updates += tdlEstimate(alpha, _lambda, sequence, values)

                values += updates
                diff = np.sum(np.absolute(before - values))

                if diff < .000001:
                    break

            estimate = np.array(values[1:-1], dtype=dtype)
            error = (truth - estimate)
            rms   = np.sqrt(np.average(np.power(error, 2)))
            rmses.append(rms)

        result = [_lambda, alpha, np.mean(rmses), np.std(rmses)]
        results.append(result)

# outputs
# joblib.dump(results, 'results.pkl')

# results = joblib.load('results.pkl') 
data = pd.DataFrame(results)
data.columns = ["lambda", "alpha", "rms", "rmsstd"]
data.head()
```

<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>lambda</th>
      <th>alpha</th>
      <th>rms</th>
      <th>rmsstd</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.235702</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.05</td>
      <td>0.175010</td>
      <td>0.002556</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.10</td>
      <td>0.131192</td>
      <td>0.009537</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.15</td>
      <td>0.103448</td>
      <td>0.020459</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.20</td>
      <td>0.091540</td>
      <td>0.033838</td>
    </tr>
  </tbody>
</table>

```python
data = data[data.groupby(['lambda'])['rms'].transform(min) == data['rms']].set_index(keys=['lambda'])
data.drop('alpha', 1, inplace=True,)
data.drop('rmsstd', 1,inplace=True,)
```

I repeatedly calculated the TD equation on a given set until the RMSE between the weights (new values) and the ideal probabilities was less than my arbitrary threshold.
The values and trend are very similar to the original but more tweaking to my environment assumptions might have improved the similarity. Given more time and intellect I’d like to do a random search of a space of hyper-parameters and see what most closely approximates the original. RMS error increases with non-linearity in relation to lambda. The initial curtailing isn’t the same as Sutton with lambda 0 performing the best out right.

If we choose to imitate figure 3 literally we get the below. I would prefer grid lines though, which is just a matter of using default seaborn aesthetics.

```python
# sns.set_style("white")
plt.figure(num=None, figsize=(10, 6), dpi=72)
plt.margins(.05)
plt.xlabel(r"$\lambda$")
plt.ylabel("RMS")
plt.title("Figure 3")
plt.xticks([i * .1 for i in range(0, 10)])
plt.yticks([i * .01 for i in range(10, 19)])
plt.text(.79, .17, "Widrow-Hoff", ha="center", va="center", rotation=0,size=15)
plt.text(-.22, .174, "ERROR\nUSING\nBEST α",size=15)
plt.plot(data,marker='o');
```

{{< figure
img="image3.png" 
command="Resize" 
options="700x" >}}

## Figure 5
```python
%time 
alphas  = [0.05 * i for i in range(0,16)]
lambdas = [0.05 * i for i in range(0, 21)]

results = []

for _lambda in lambdas:
    for alpha in alphas:
        rms_vals = []
        for training_set in training_sets:

            values = np.array([0.5 for i in range(7)])

            for sequence in training_set:
                values[0] = 0.0
                values[6] = 1.0
                values += tdlEstimate(alpha, _lambda, sequence, values)

            estimate = np.array(values[1:-1])
            error = (truth - estimate)
            rms   = np.sqrt(np.average(np.power(error, 2)))

            rms_vals.append(rms)

        result = [_lambda, alpha, np.mean(rms_vals), np.std(rms_vals)]
        results.append(result)
```

```bash
CPU times: user 3 µs, sys: 1 µs, total: 4 µs
Wall time: 5.96 µs
```

Here we reiterate that larger lambda values perform better via smaller learning rates. I believe this is due to the larger lambda values emphasizing weight on larger steps and the final output.

```python
data = pd.DataFrame(results)

data.columns = ["lambda", "alpha", "rms", "rmsstd"]

data = data[data.groupby(['lambda'])['rms'].transform(min) == \
            data['rms']].set_index(keys=['lambda'])

data = data.drop('alpha', 1)
data = data.drop('rmsstd', 1)

plt.figure(num=None, figsize=(10, 6), dpi=80)
plt.plot(data, marker='o') 
plt.margins(.10)
plt.xlabel(r"$\lambda$")
plt.ylabel("RMS")
plt.title("Figure 5 ")
plt.text(.7, .185, "Widrow-Hoff", ha="center", va="center", rotation=0,size=15)
plt.text(-.25,.204, "ERROR\nUSING\nBEST α",size=12)
```
{{< figure
img="image4.png" 
command="Resize" 
options="700x" >}}

To conclude, [here is a short simple video](https://www.youtube.com/embed/DZzffdHNqtQ) from Peter Norvig that provides good intuition on TD learning.

