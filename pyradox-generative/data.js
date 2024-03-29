const nav_data = [
  "Vanilla GAN",
  "Conditional GAN",
  "Wesserstein GAN",
  "Variational Autoencoder",
  "Style GAN",
  "Cycle GAN",
];

const references = [
  [
    "Generative Adversarial Networks (2014, June)",
    "https://arxiv.org/abs/1406.2661",
  ],
  [
    "Conditional Generative Adversarial Nets (2014, November)",
    "https://arxiv.org/abs/1411.1784",
  ],
  [
    "Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (2015, November)",
    "https://arxiv.org/abs/1511.06434",
  ],
  [
    "Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (2017, March)",
    "https://arxiv.org/abs/1703.10593",
  ],

  ["Wasserstein GAN (2017, January)", "https://arxiv.org/abs/1701.07875"],
  [
    "Improved Training of Wasserstein GANs (2017, April)",
    "https://arxiv.org/abs/1704.00028",
  ],
  [
    "A Style-Based Generator Architecture for Generative Adversarial Networks (2018, December)",
    "https://arxiv.org/abs/1812.04948",
  ],
  [
    "An Introduction to Variational Autoencoders (2019, June)",
    "https://arxiv.org/abs/1906.02691",
  ],
];

const usage = [
  {
    title: "Vanilla GAN",
    content: [
      {
        type: "p",
        text: "Just provide your genrator and discriminator and train your GAN <br /> Data Preparation:",
      },
      {
        type: "code",
        text: `from pyradox_generative import GAN
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255
x_train = x_train.reshape(-1, 28, 28, 1) * 2.0 - 1.0

dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(1024)
dataset = dataset.batch(32, drop_remainder=True).prefetch(1)`,
      },
      {
        type: "p",
        text: "Define the generator and discriminator models:",
      },
      {
        type: "code",
        text: `generator = keras.models.Sequential(
    [
        keras.Input(shape=[28]),
        keras.layers.Dense(7 * 7 * 3),
        keras.layers.Reshape([7, 7, 3]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(
            32, kernel_size=3, strides=2, padding="same", activation="selu"
        ),
        keras.layers.Conv2DTranspose(
            1, kernel_size=3, strides=2, padding="same", activation="tanh"
        ),
    ],
    name="generator",
)

discriminator = keras.models.Sequential(
    [
        keras.layers.Conv2D(
            32,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=keras.layers.LeakyReLU(0.2),
            input_shape=[28, 28, 1],
        ),
        keras.layers.Conv2D(
            3,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=keras.layers.LeakyReLU(0.2),
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)`,
      },
      {
        type: "p",
        text: "Plug in the models to the trainer class and train them using the very familiar compile and fit methods:",
      },
      {
        type: "code",
        text: `gan = GAN(discriminator=discriminator, generator=generator, latent_dim=28)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

history = gan.fit(dataset)`,
      },
    ],
  },
  {
    title: "Conditional GAN",
    content: [
      {
        type: "p",
        text: "Just provide your genrator and discriminator and train your GAN <br /> Data Preparation and calculate the input and output dimensions of generator and discriminator:",
      },
      {
        type: "code",
        text: `from pyradox_generative import ConditionalGAN
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

CODINGS_SIZE = 28
N_CHANNELS = 1
N_CLASSES = 10
G_INP_CHANNELS = CODINGS_SIZE + N_CLASSES
D_INP_CHANNELS = N_CHANNELS + N_CLASSES

(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train
x_train = x_train.astype(np.float32) / 255
x_train = x_train.reshape(-1, 28, 28, 1) * 2.0 - 1.0
y_train = y_train
y_train = keras.utils.to_categorical(y_train, 10)

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.shuffle(1024)
dataset = dataset.batch(32, drop_remainder=True).prefetch(1)`,
      },
      {
        type: "p",
        text: "Define the generator and discriminator models:",
      },
      {
        type: "code",
        text: `generator = keras.models.Sequential(
    [
        keras.Input(shape=[G_INP_CHANNELS]),
        keras.layers.Dense(7 * 7 * 3),
        keras.layers.Reshape([7, 7, 3]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(
            32, kernel_size=3, strides=2, padding="same", activation="selu"
        ),
        keras.layers.Conv2DTranspose(
            1, kernel_size=3, strides=2, padding="same", activation="tanh"
        ),
    ],
    name="generator",
)

discriminator = keras.models.Sequential(
    [
        keras.layers.Conv2D(
            32,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=keras.layers.LeakyReLU(0.2),
            input_shape=[28, 28, D_INP_CHANNELS],
        ),
        keras.layers.Conv2D(
            3,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=keras.layers.LeakyReLU(0.2),
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)`,
      },
      {
        type: "p",
        text: "Plug in the models to the trainer class and train them using the very familiar compile and fit methods:",
      },
      {
        type: "code",
        text: `gan = ConditionalGAN(
    discriminator=discriminator, generator=generator, latent_dim=CODINGS_SIZE
)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss_fn=keras.losses.BinaryCrossentropy(),
)

history = gan.fit(dataset)`,
      },
    ],
  },
  {
    title: "Wasserstein GAN",
    content: [
      {
        type: "p",
        text: "Just provide your genrator and discriminator and train your GAN <br /> Data Preparation:",
      },
      {
        type: "code",
        text: `from pyradox_generative import WGANGP
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255
x_train = x_train.reshape(-1, 28, 28, 1) * 2.0 - 1.0

dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(1024)
dataset = dataset.batch(32, drop_remainder=True).prefetch(1)`,
      },
      {
        type: "p",
        text: "Define the generator and discriminator models:",
      },
      {
        type: "code",
        text: `generator = keras.models.Sequential(
    [
        keras.Input(shape=[28]),
        keras.layers.Dense(7 * 7 * 3),
        keras.layers.Reshape([7, 7, 3]),
        keras.layers.BatchNormalization(),
        keras.layers.Conv2DTranspose(
            32, kernel_size=3, strides=2, padding="same", activation="selu"
        ),
        keras.layers.Conv2DTranspose(
            1, kernel_size=3, strides=2, padding="same", activation="tanh"
        ),
    ],
    name="generator",
)

discriminator = keras.models.Sequential(
    [
        keras.layers.Conv2D(
            32,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=keras.layers.LeakyReLU(0.2),
            input_shape=[28, 28, 1],
        ),
        keras.layers.Conv2D(
            3,
            kernel_size=3,
            strides=2,
            padding="same",
            activation=keras.layers.LeakyReLU(0.2),
        ),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator",
)`,
      },
      {
        type: "p",
        text: "Plug in the models to the trainer class and train them using the very familiar compile and fit methods:",
      },
      {
        type: "code",
        text: `gan = WGANGP(
    discriminator=discriminator,
    generator=generator,
    latent_dim=28,
    discriminator_extra_steps=1,
    gp_weight=10,
)
gan.compile(
    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
)

history = gan.fit(dataset)`,
      },
    ],
  },
  {
    title: "Variational Auto Encoder",
    content: [
      {
        type: "p",
        text: "Just provide your encoder and decoder and train your VAE (Sampling is done internally) <br /> Data Preparation:",
      },
      {
        type: "code",
        text: `from pyradox_generative import VAE
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255
x_train = x_train.reshape(-1, 28, 28, 1) * 2.0 - 1.0

dataset = tf.data.Dataset.from_tensor_slices(x_train)
dataset = dataset.shuffle(1024)
dataset = dataset.batch(32, drop_remainder=True).prefetch(1)`,
      },
      {
        type: "p",
        text: "Define the encoder and decoder models",
      },
      {
        type: "code",
        text: `encoder = keras.models.Sequential(
    [
        keras.Input(shape=(28, 28, 1)),
        keras.layers.Conv2D(32, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2D(64, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Flatten(),
        keras.layers.Dense(16, activation="relu"),
    ],
    name="encoder",
)

decoder = keras.models.Sequential(
    [
        keras.Input(shape=(28,)),
        keras.layers.Dense(7 * 7 * 64, activation="relu"),
        keras.layers.Reshape((7, 7, 64)),
        keras.layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same"),
        keras.layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same"),
    ],
    name="decoder",
)`,
      },
      {
        type: "p",
        text: "Plug in the models to the trainer class and train them using the very familiar compile and fit methods:",
      },
      {
        type: "code",
        text: `vae = VAE(encoder=encoder, decoder=decoder, latent_dim=28)
vae.compile(keras.optimizers.Adam(learning_rate=0.001))
history = vae.fit(dataset)`,
      },
    ],
  },
  {
    title: "Style GAN",
    content: [
      {
        type: "p",
        text: "Just provide your genrator and discriminator and train your GAN <br /> Data Preparation:",
      },
      {
        type: "code",
        text: `from pyradox_generative import StyleGAN
import numpy as np
import tensorflow as tf
from functools import partial

def resize_image(res, image):
    # only donwsampling, so use nearest neighbor that is faster to run
    image = tf.image.resize(
        image, (res, res), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    image = tf.cast(image, tf.float32) / 127.5 - 1.0
    return image


def create_dataloader(res):
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train[:100, :, :]
    x_train = np.pad(x_train, [(0, 0), (2, 2), (2, 2)], mode="constant")
    x_train = tf.image.grayscale_to_rgb(tf.expand_dims(x_train, axis=3), name=None)
    x_train = tf.data.Dataset.from_tensor_slices(x_train)

    batch_size = 32
    dl = x_train.map(partial(resize_image, res), num_parallel_calls=tf.data.AUTOTUNE)
    dl = dl.shuffle(200).batch(batch_size, drop_remainder=True).prefetch(1).repeat()
    return dl`,
      },
      {
        type: "p",
        text: "Define the model by providing number of filters for each resolution (log 2):",
      },
      {
        type: "code",
        text: `gan = StyleGAN(
    target_res=32,
    start_res=4,
    filter_nums={0: 32, 1: 32, 2: 32, 3: 32, 4: 32, 5: 32},
)
opt_cfg = {"learning_rate": 1e-3, "beta_1": 0.0, "beta_2": 0.99, "epsilon": 1e-8}

start_res_log2 = 2
target_res_log2 = 5`,
      },
      {
        type: "p",
        text: "Train the Style GAN:",
      },
      {
        type: "code",
        text: `for res_log2 in range(start_res_log2, target_res_log2 + 1):
    res = 2 ** res_log2
    for phase in ["TRANSITION", "STABLE"]:
        if res == 4 and phase == "TRANSITION":
            continue

        train_dl = create_dataloader(res)

        steps = 10

        gan.compile(
            d_optimizer=tf.keras.optimizers.Adam(**opt_cfg),
            g_optimizer=tf.keras.optimizers.Adam(**opt_cfg),
            loss_weights={"gradient_penalty": 10, "drift": 0.001},
            steps_per_epoch=steps,
            res=res,
            phase=phase,
            run_eagerly=False,
        )

        print(phase)
        history = gan.fit(train_dl, epochs=1, steps_per_epoch=steps)`,
      },
    ],
  },
  {
    title: "Cycle GAN",
    content: [
      {
        type: "p",
        text: "Just provide your genrator and discriminator and train your GAN <br /> Data Preparation:",
      },
      {
        type: "code",
        text: `import tensorflow_datasets as tfds
import tensorflow as tf
from tensorflow import keras
from pyradox_generative import CycleGAN

tfds.disable_progress_bar()
autotune = tf.data.AUTOTUNE
orig_img_size = (286, 286)
input_img_size = (256, 256, 3)


def normalize_img(img):
    img = tf.cast(img, dtype=tf.float32)
    return (img / 127.5) - 1.0


def preprocess_train_image(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.resize(img, [*orig_img_size])
    img = tf.image.random_crop(img, size=[*input_img_size])
    img = normalize_img(img)
    return img


def preprocess_test_image(img, label):
    img = tf.image.resize(img, [input_img_size[0], input_img_size[1]])
    img = normalize_img(img)
    return img

train_horses, _ = tfds.load(
    "cycle_gan/horse2zebra", with_info=True, as_supervised=True, split="trainA[:5%]"
)
train_zebras, _ = tfds.load(
    "cycle_gan/horse2zebra", with_info=True, as_supervised=True, split="trainB[:5%]"
)

buffer_size = 256
batch_size = 1

train_horses = (
    train_horses.map(preprocess_train_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)
)
train_zebras = (
    train_zebras.map(preprocess_train_image, num_parallel_calls=autotune)
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)
)`,
      },
      {
        type: "p",
        text: "Define the generator and discriminator models:",
      },
      {
        type: "code",
        text: `def build_generator(name):
    return keras.models.Sequential(
        [
            keras.layers.Input(shape=input_img_size),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.Conv2D(3, 3, activation="tanh", padding="same"),
        ],
        name=name,
    )


def build_discriminator(name):
    return keras.models.Sequential(
        [
            keras.layers.Input(shape=input_img_size),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(pool_size=2, strides=2),
            keras.layers.Conv2D(1, 3, activation="relu", padding="same"),
        ],
        name=name,
    )`,
      },
      {
        type: "p",
        text: "Plug in the models to the trainer class and train them using the very familiar compile and fit methods:",
      },
      {
        type: "code",
        text: `gan = CycleGAN(
    generator_g=build_generator("gen_G"),
    generator_f=build_generator("gen_F"),
    discriminator_x=build_discriminator("disc_X"),
    discriminator_y=build_discriminator("disc_Y"),
)

gan.compile(
    gen_g_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    gen_f_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_x_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
    disc_y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),
)

history = gan.fit(
    tf.data.Dataset.zip((train_horses, train_zebras)),
)`,
      },
    ],
  },
];
