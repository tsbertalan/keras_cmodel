Generate C-code for wrapping simple Keras models, by the simple expedient of putting the weights directly into the c files.

With a trained model:
```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='tanh'),
    tf.keras.layers.Dense(10, activation='tanh'),
    tf.keras.layers.Dense(1, activation='linear'),
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mse')

x = np.random.uniform(size=(1000, 1), low=-1, high=1)
y = x ** 2
model.fit(x, y, epochs=500, batch_size=500)
```
<img src="mlp_training.png" width=500px />

we create a C wrapper (for a fixed `usage_batch_size`) like this
```python
from keras_cmodel import CModel

usage_batch_size = 5
cmodel = CModel(model, usage_batch_size)
cmodel.save(name='MLP')  # Writes MLP.c and MLP.h
```

Then, you can have user code like e.g.
```C
#include "MLP.h"

int main() {
    setup();
    double inputs[] = {-1, -.5, 0, .5, 1};
    double outputs[5];
    MLP(inputs, outputs);
    print_array("outputs", outputs, 5, 1);
}
```

(Doing this for a larger `np.linspace` batch of inputs and plotting:)
<img src="mlp_result.png" width=500px />
