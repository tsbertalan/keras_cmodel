from subprocess import check_output
import matplotlib.pyplot as plt
import numpy as np, tensorflow as tf

from keras_cmodel import CModel

# We make a little test model and train it on a parabola.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='tanh'),
    tf.keras.layers.Dense(10, activation='tanh'),
    tf.keras.layers.Dense(1, activation='linear'),
])
model.compile(optimizer=tf.keras.optimizers.Adam(
    learning_rate=1e-3), loss='mse')

x = np.random.uniform(size=(1000, 1), low=-1, high=1)
true_func = lambda x: x ** 2
y = true_func(x)
history = model.fit(x, y, 
    epochs=500, verbose=0, batch_size=500,
).history
fig, ax = plt.subplots()
ax.plot(history['loss'])
ax.set_yscale('log')
ax.set_ylabel('Loss')
ax.set_xlabel('Batch')
fig.savefig('mlp_training.png')

test_batch_size=128
cmodel = CModel(model, batch_size=test_batch_size)
cmodel.save()
x_test =  np.linspace(-1, 1, test_batch_size)
inputs = ','.join([str(f) for f in x_test])

# In real applications, you would create some other
# wrapper program like this around your MLP; e.g.
# to adapt the interface to Auto.
test_program = '''#include "MLP.h"
#include "mm_utils.h"

int main() {{
setup();
double inputs[] = {{ {inputs} }};
double outputs[{batchsize}];
MLP(inputs, outputs);
print_array("outputs", outputs, {batchsize}, 1);
}}'''.format(
    inputs=inputs,
    batchsize=test_batch_size,
)
with open('MLP_test.c', 'w') as fp:
    fp.write(test_program)

# Obviously, you wouldn't really compile and run from Python,
# (well, that, you might)
# and then receive results by importing a generated .py file.
# Here, we do so just to get a standalone demo script.
build_result = check_output('gcc MLP_test.c MLP.c -lm -g -o MLP_test'.split())
result = check_output(['./MLP_test']).decode('utf-8')
with open('MLP_result.py', 'w') as fp:
    fp.write(result)
from MLP_result import outputs
outputs = np.array(outputs)

fig, ax = plt.subplots()
ax.plot(x_test, true_func(x_test), label='Truth')
ax.scatter(x_test, outputs.ravel(), label='Net')
ax.legend()
fig.savefig('mlp_result.png')
