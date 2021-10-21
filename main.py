import tensorflow as tf
from tensorflow_probability import distributions as tfpd
import matplotlib.pyplot as plt
import numpy as np

#tensors flatten
def evaluate(tensors):
    if tf.executing_eagerly():
        return tf.nest.pack_sequence_as(
            tensors,
            [t.numpy() if tf.is_tensor(t) else t
             for t in tf.nest.flatten(tensors)])
    with tf.compat.v1.Session() as sess:
        return sess.run(tensors)

#parameter
beta = 0.25
alpha = -15

#challenge temperature
T_ = [66, 70, 69, 68, 67, 72, 73, 70, 57, 63, 70, 78, 67, 53, 67, 75, 70, 81, 76, 79, 75, 76, 58]

#disaster probability function
def p(t):
    prob = 1 / (1 + np.exp(beta * t + alpha))
    return prob

#disaster probability of each temperature
p_ = []
i = 0
for t in T_:
    p_.append(p(t))
    i = i + 1
#Bernoulli distribution simulation
[p_deterministic_] = evaluate([p_])
simulated_data = tfpd.Bernoulli(name="bernoulli_sim", probs=p_deterministic_).sample(sample_shape=10000)

[bernoulli_sim_samples_] = evaluate([simulated_data])
simulations_ = bernoulli_sim_samples_

#simulation result
plt.figure(figsize=(12.5, 12))
for i in range(4):
    ax = plt.subplot(4, 1, i + 1)
    plt.scatter(T_, simulations_[1000 * i, :], color="k", s=50, alpha=0.6)
plt.show()
