from __future__ import absolute_import, division, print_function, unicode_literals
import os

import numpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

overall_data = np.asarray(pd.read_csv('Resources/DogeCoinInitialSet.csv')).astype(np.float32)
train_data = overall_data[:round(len(overall_data) / 2)]
eval_data = overall_data[round(len(overall_data) / 2):len(overall_data)]

train_features = train_data.copy()
train_labels = train_features[:, 0]

eval_features = eval_data.copy()
eval_labels = eval_features[:, 0]

print((train_features == eval_features))

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(1),
])

lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=600 * 1000,
    decay_rate=1,
    staircase=False)

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.MeanSquaredError()
)

history = model.fit(train_features, train_labels, batch_size=1, epochs=10, verbose=2,
                    validation_data=(eval_features, eval_labels))

predictions = model.predict(eval_features)
print(predictions[-10:])

plt.figure(figsize=(10,10))
plt.scatter(eval_labels, predictions, c='crimson')
plt.yscale('log')
plt.xscale('log')

p1 = max(max(predictions), max(eval_labels))
p2 = min(min(predictions), min(eval_labels))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.savefig('Resources/fig.png')
