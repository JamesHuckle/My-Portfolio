# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + tags=["active-ipynb"]
# import wandb
#
# sweep_config = {
#     'name': 'keras-1',
#     'program': 'wandb_test.py',
#     'method': 'random',
#     'metric': {
#         'name': 'val_loss',
#         'goal': 'minimize',
#     },
#     'parameters': {
#         'layers': {'values': [32, 64, 96]},
#         'epochs': {'values': [1, 2, 3, 4]},
#         'window_len': {'values': [100, 200, 400]},
#     }
# }
#
# sweep_id = wandb.sweep(sweep_config, project='timeseries')
# sweep_id

# +
# #%%wandb

def train():
    import numpy as np
    import tensorflow as tf
    import wandb
    config = {
        'layers': 32,
        'epochs': 2,
        'window_len': 10,
    }
    wandb.init(config=config, magic=False)
    # You can override values if you want
    # wandb.config.update({'layers':1}, allow_val_change=True)

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    train_images = train_images[:wandb.config.window_len]
    train_labels = train_labels[:wandb.config.window_len]
    train_images.shape
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(wandb.config.layers, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(train_images, train_labels, epochs=wandb.config.epochs,
                  validation_data=(test_images, test_labels), verbose=1)
    
    data = [[x, y] for (x, y) in zip(list(range(10)), np.random.random(10).cumsum().tolist())]
    table = wandb.Table(data=data, columns = ["x", "y"])   
    hist = model.history.history
    metrics = {
        'max_accuracy': max(hist['accuracy']),
        'max_val_accuracy': max(hist['val_accuracy']),
        'ROMAD': np.random.randint(-5,5),
        'my_custom_plot_id': wandb.plot.line(table, "x", "y", title="Profit plot"),
    }
    wandb.log(metrics)
    
if __name__ == '__main__':
    train()

# +
#wandb.agent(sweep_id, function=train)
