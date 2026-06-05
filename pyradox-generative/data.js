let title = "pyradox-generative";
let project_date = "Open Source"
let links = {
    "paper": "",
    "demo": "",
    "code": "https://github.com/Ritvik19/pyradox-generative",
    "model": "",
    "data": ""
}
let link2icon = {
    "code": "fas fa-code",
    "demo": "fas fa-terminal",
    "model": "fas fa-cogs",
    "data": "fas fa-database",
    "paper": "fas fa-file-pdf",
}
let project_contents = {
    "Overview": [
        {
            "type": "text",
            "content": "Lightweight trainers for various state-of-the-art Generative Adversarial Networks. Part of the <a href=\"/pyradox/\">pyradox</a> ecosystem."
        }
    ],
    "Installation": [
        {
            "type": "code",
            "content": "pip install pyradox-generative"
        }
    ],
    "Vanilla GAN": [
        {
            "type": "text",
            "content": "Just provide your genrator and discriminator and train your GAN <br /> Data Preparation:"
        },
        {
            "type": "code",
            "content": "from pyradox_generative import GAN\nimport numpy as np\nimport tensorflow as tf\nimport tensorflow.keras as keras\n\n(x_train, y_train), _ = keras.datasets.mnist.load_data()\nx_train = x_train.astype(np.float32) / 255\nx_train = x_train.reshape(-1, 28, 28, 1) * 2.0 - 1.0\n\ndataset = tf.data.Dataset.from_tensor_slices(x_train)\ndataset = dataset.shuffle(1024)\ndataset = dataset.batch(32, drop_remainder=True).prefetch(1)"
        },
        {
            "type": "text",
            "content": "Define the generator and discriminator models:"
        },
        {
            "type": "code",
            "content": "generator = keras.models.Sequential(\n    [\n        keras.Input(shape=[28]),\n        keras.layers.Dense(7 * 7 * 3),\n        keras.layers.Reshape([7, 7, 3]),\n        keras.layers.BatchNormalization(),\n        keras.layers.Conv2DTranspose(\n            32, kernel_size=3, strides=2, padding=\"same\", activation=\"selu\"\n        ),\n        keras.layers.Conv2DTranspose(\n            1, kernel_size=3, strides=2, padding=\"same\", activation=\"tanh\"\n        ),\n    ],\n    name=\"generator\",\n)\n\ndiscriminator = keras.models.Sequential(\n    [\n        keras.layers.Conv2D(\n            32,\n            kernel_size=3,\n            strides=2,\n            padding=\"same\",\n            activation=keras.layers.LeakyReLU(0.2),\n            input_shape=[28, 28, 1],\n        ),\n        keras.layers.Conv2D(\n            3,\n            kernel_size=3,\n            strides=2,\n            padding=\"same\",\n            activation=keras.layers.LeakyReLU(0.2),\n        ),\n        keras.layers.Flatten(),\n        keras.layers.Dense(1, activation=\"sigmoid\"),\n    ],\n    name=\"discriminator\",\n)"
        },
        {
            "type": "text",
            "content": "Plug in the models to the trainer class and train them using the very familiar compile and fit methods:"
        },
        {
            "type": "code",
            "content": "gan = GAN(discriminator=discriminator, generator=generator, latent_dim=28)\ngan.compile(\n    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),\n    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),\n    loss_fn=keras.losses.BinaryCrossentropy(),\n)\n\nhistory = gan.fit(dataset)"
        }
    ],
    "Conditional GAN": [
        {
            "type": "text",
            "content": "Just provide your genrator and discriminator and train your GAN <br /> Data Preparation and calculate the input and output dimensions of generator and discriminator:"
        },
        {
            "type": "code",
            "content": "from pyradox_generative import ConditionalGAN\nimport numpy as np\nimport tensorflow as tf\nimport tensorflow.keras as keras\n\nCODINGS_SIZE = 28\nN_CHANNELS = 1\nN_CLASSES = 10\nG_INP_CHANNELS = CODINGS_SIZE + N_CLASSES\nD_INP_CHANNELS = N_CHANNELS + N_CLASSES\n\n(x_train, y_train), _ = keras.datasets.mnist.load_data()\nx_train = x_train\nx_train = x_train.astype(np.float32) / 255\nx_train = x_train.reshape(-1, 28, 28, 1) * 2.0 - 1.0\ny_train = y_train\ny_train = keras.utils.to_categorical(y_train, 10)\n\ndataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))\ndataset = dataset.shuffle(1024)\ndataset = dataset.batch(32, drop_remainder=True).prefetch(1)"
        },
        {
            "type": "text",
            "content": "Define the generator and discriminator models:"
        },
        {
            "type": "code",
            "content": "generator = keras.models.Sequential(\n    [\n        keras.Input(shape=[G_INP_CHANNELS]),\n        keras.layers.Dense(7 * 7 * 3),\n        keras.layers.Reshape([7, 7, 3]),\n        keras.layers.BatchNormalization(),\n        keras.layers.Conv2DTranspose(\n            32, kernel_size=3, strides=2, padding=\"same\", activation=\"selu\"\n        ),\n        keras.layers.Conv2DTranspose(\n            1, kernel_size=3, strides=2, padding=\"same\", activation=\"tanh\"\n        ),\n    ],\n    name=\"generator\",\n)\n\ndiscriminator = keras.models.Sequential(\n    [\n        keras.layers.Conv2D(\n            32,\n            kernel_size=3,\n            strides=2,\n            padding=\"same\",\n            activation=keras.layers.LeakyReLU(0.2),\n            input_shape=[28, 28, D_INP_CHANNELS],\n        ),\n        keras.layers.Conv2D(\n            3,\n            kernel_size=3,\n            strides=2,\n            padding=\"same\",\n            activation=keras.layers.LeakyReLU(0.2),\n        ),\n        keras.layers.Flatten(),\n        keras.layers.Dense(1, activation=\"sigmoid\"),\n    ],\n    name=\"discriminator\",\n)"
        },
        {
            "type": "text",
            "content": "Plug in the models to the trainer class and train them using the very familiar compile and fit methods:"
        },
        {
            "type": "code",
            "content": "gan = ConditionalGAN(\n    discriminator=discriminator, generator=generator, latent_dim=CODINGS_SIZE\n)\ngan.compile(\n    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),\n    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),\n    loss_fn=keras.losses.BinaryCrossentropy(),\n)\n\nhistory = gan.fit(dataset)"
        }
    ],
    "Wasserstein GAN": [
        {
            "type": "text",
            "content": "Just provide your genrator and discriminator and train your GAN <br /> Data Preparation:"
        },
        {
            "type": "code",
            "content": "from pyradox_generative import WGANGP\nimport numpy as np\nimport tensorflow as tf\nimport tensorflow.keras as keras\n\n(x_train, y_train), _ = keras.datasets.mnist.load_data()\nx_train = x_train.astype(np.float32) / 255\nx_train = x_train.reshape(-1, 28, 28, 1) * 2.0 - 1.0\n\ndataset = tf.data.Dataset.from_tensor_slices(x_train)\ndataset = dataset.shuffle(1024)\ndataset = dataset.batch(32, drop_remainder=True).prefetch(1)"
        },
        {
            "type": "text",
            "content": "Define the generator and discriminator models:"
        },
        {
            "type": "code",
            "content": "generator = keras.models.Sequential(\n    [\n        keras.Input(shape=[28]),\n        keras.layers.Dense(7 * 7 * 3),\n        keras.layers.Reshape([7, 7, 3]),\n        keras.layers.BatchNormalization(),\n        keras.layers.Conv2DTranspose(\n            32, kernel_size=3, strides=2, padding=\"same\", activation=\"selu\"\n        ),\n        keras.layers.Conv2DTranspose(\n            1, kernel_size=3, strides=2, padding=\"same\", activation=\"tanh\"\n        ),\n    ],\n    name=\"generator\",\n)\n\ndiscriminator = keras.models.Sequential(\n    [\n        keras.layers.Conv2D(\n            32,\n            kernel_size=3,\n            strides=2,\n            padding=\"same\",\n            activation=keras.layers.LeakyReLU(0.2),\n            input_shape=[28, 28, 1],\n        ),\n        keras.layers.Conv2D(\n            3,\n            kernel_size=3,\n            strides=2,\n            padding=\"same\",\n            activation=keras.layers.LeakyReLU(0.2),\n        ),\n        keras.layers.Flatten(),\n        keras.layers.Dense(1, activation=\"sigmoid\"),\n    ],\n    name=\"discriminator\",\n)"
        },
        {
            "type": "text",
            "content": "Plug in the models to the trainer class and train them using the very familiar compile and fit methods:"
        },
        {
            "type": "code",
            "content": "gan = WGANGP(\n    discriminator=discriminator,\n    generator=generator,\n    latent_dim=28,\n    discriminator_extra_steps=1,\n    gp_weight=10,\n)\ngan.compile(\n    d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),\n    g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),\n)\n\nhistory = gan.fit(dataset)"
        }
    ],
    "Variational Auto Encoder": [
        {
            "type": "text",
            "content": "Just provide your encoder and decoder and train your VAE (Sampling is done internally) <br /> Data Preparation:"
        },
        {
            "type": "code",
            "content": "from pyradox_generative import VAE\nimport numpy as np\nimport tensorflow as tf\nimport tensorflow.keras as keras\n\n(x_train, y_train), _ = keras.datasets.mnist.load_data()\nx_train = x_train.astype(np.float32) / 255\nx_train = x_train.reshape(-1, 28, 28, 1) * 2.0 - 1.0\n\ndataset = tf.data.Dataset.from_tensor_slices(x_train)\ndataset = dataset.shuffle(1024)\ndataset = dataset.batch(32, drop_remainder=True).prefetch(1)"
        },
        {
            "type": "text",
            "content": "Define the encoder and decoder models"
        },
        {
            "type": "code",
            "content": "encoder = keras.models.Sequential(\n    [\n        keras.Input(shape=(28, 28, 1)),\n        keras.layers.Conv2D(32, 3, activation=\"relu\", strides=2, padding=\"same\"),\n        keras.layers.Conv2D(64, 3, activation=\"relu\", strides=2, padding=\"same\"),\n        keras.layers.Flatten(),\n        keras.layers.Dense(16, activation=\"relu\"),\n    ],\n    name=\"encoder\",\n)\n\ndecoder = keras.models.Sequential(\n    [\n        keras.Input(shape=(28,)),\n        keras.layers.Dense(7 * 7 * 64, activation=\"relu\"),\n        keras.layers.Reshape((7, 7, 64)),\n        keras.layers.Conv2DTranspose(64, 3, activation=\"relu\", strides=2, padding=\"same\"),\n        keras.layers.Conv2DTranspose(32, 3, activation=\"relu\", strides=2, padding=\"same\"),\n        keras.layers.Conv2DTranspose(1, 3, activation=\"sigmoid\", padding=\"same\"),\n    ],\n    name=\"decoder\",\n)"
        },
        {
            "type": "text",
            "content": "Plug in the models to the trainer class and train them using the very familiar compile and fit methods:"
        },
        {
            "type": "code",
            "content": "vae = VAE(encoder=encoder, decoder=decoder, latent_dim=28)\nvae.compile(keras.optimizers.Adam(learning_rate=0.001))\nhistory = vae.fit(dataset)"
        }
    ],
    "Style GAN": [
        {
            "type": "text",
            "content": "Just provide your genrator and discriminator and train your GAN <br /> Data Preparation:"
        },
        {
            "type": "code",
            "content": "from pyradox_generative import StyleGAN\nimport numpy as np\nimport tensorflow as tf\nfrom functools import partial\n\ndef resize_image(res, image):\n    # only donwsampling, so use nearest neighbor that is faster to run\n    image = tf.image.resize(\n        image, (res, res), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR\n    )\n    image = tf.cast(image, tf.float32) / 127.5 - 1.0\n    return image\n\n\ndef create_dataloader(res):\n    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()\n    x_train = x_train[:100, :, :]\n    x_train = np.pad(x_train, [(0, 0), (2, 2), (2, 2)], mode=\"constant\")\n    x_train = tf.image.grayscale_to_rgb(tf.expand_dims(x_train, axis=3), name=None)\n    x_train = tf.data.Dataset.from_tensor_slices(x_train)\n\n    batch_size = 32\n    dl = x_train.map(partial(resize_image, res), num_parallel_calls=tf.data.AUTOTUNE)\n    dl = dl.shuffle(200).batch(batch_size, drop_remainder=True).prefetch(1).repeat()\n    return dl"
        },
        {
            "type": "text",
            "content": "Define the model by providing number of filters for each resolution (log 2):"
        },
        {
            "type": "code",
            "content": "gan = StyleGAN(\n    target_res=32,\n    start_res=4,\n    filter_nums={0: 32, 1: 32, 2: 32, 3: 32, 4: 32, 5: 32},\n)\nopt_cfg = {\"learning_rate\": 1e-3, \"beta_1\": 0.0, \"beta_2\": 0.99, \"epsilon\": 1e-8}\n\nstart_res_log2 = 2\ntarget_res_log2 = 5"
        },
        {
            "type": "text",
            "content": "Train the Style GAN:"
        },
        {
            "type": "code",
            "content": "for res_log2 in range(start_res_log2, target_res_log2 + 1):\n    res = 2 ** res_log2\n    for phase in [\"TRANSITION\", \"STABLE\"]:\n        if res == 4 and phase == \"TRANSITION\":\n            continue\n\n        train_dl = create_dataloader(res)\n\n        steps = 10\n\n        gan.compile(\n            d_optimizer=tf.keras.optimizers.Adam(**opt_cfg),\n            g_optimizer=tf.keras.optimizers.Adam(**opt_cfg),\n            loss_weights={\"gradient_penalty\": 10, \"drift\": 0.001},\n            steps_per_epoch=steps,\n            res=res,\n            phase=phase,\n            run_eagerly=False,\n        )\n\n        print(phase)\n        history = gan.fit(train_dl, epochs=1, steps_per_epoch=steps)"
        }
    ],
    "Cycle GAN": [
        {
            "type": "text",
            "content": "Just provide your genrator and discriminator and train your GAN <br /> Data Preparation:"
        },
        {
            "type": "code",
            "content": "import tensorflow_datasets as tfds\nimport tensorflow as tf\nfrom tensorflow import keras\nfrom pyradox_generative import CycleGAN\n\ntfds.disable_progress_bar()\nautotune = tf.data.AUTOTUNE\norig_img_size = (286, 286)\ninput_img_size = (256, 256, 3)\n\n\ndef normalize_img(img):\n    img = tf.cast(img, dtype=tf.float32)\n    return (img / 127.5) - 1.0\n\n\ndef preprocess_train_image(img, label):\n    img = tf.image.random_flip_left_right(img)\n    img = tf.image.resize(img, [*orig_img_size])\n    img = tf.image.random_crop(img, size=[*input_img_size])\n    img = normalize_img(img)\n    return img\n\n\ndef preprocess_test_image(img, label):\n    img = tf.image.resize(img, [input_img_size[0], input_img_size[1]])\n    img = normalize_img(img)\n    return img\n\ntrain_horses, _ = tfds.load(\n    \"cycle_gan/horse2zebra\", with_info=True, as_supervised=True, split=\"trainA[:5%]\"\n)\ntrain_zebras, _ = tfds.load(\n    \"cycle_gan/horse2zebra\", with_info=True, as_supervised=True, split=\"trainB[:5%]\"\n)\n\nbuffer_size = 256\nbatch_size = 1\n\ntrain_horses = (\n    train_horses.map(preprocess_train_image, num_parallel_calls=autotune)\n    .cache()\n    .shuffle(buffer_size)\n    .batch(batch_size)\n)\ntrain_zebras = (\n    train_zebras.map(preprocess_train_image, num_parallel_calls=autotune)\n    .cache()\n    .shuffle(buffer_size)\n    .batch(batch_size)\n)"
        },
        {
            "type": "text",
            "content": "Define the generator and discriminator models:"
        },
        {
            "type": "code",
            "content": "def build_generator(name):\n    return keras.models.Sequential(\n        [\n            keras.layers.Input(shape=input_img_size),\n            keras.layers.Conv2D(32, 3, activation=\"relu\", padding=\"same\"),\n            keras.layers.Conv2D(32, 3, activation=\"relu\", padding=\"same\"),\n            keras.layers.Conv2D(3, 3, activation=\"tanh\", padding=\"same\"),\n        ],\n        name=name,\n    )\n\n\ndef build_discriminator(name):\n    return keras.models.Sequential(\n        [\n            keras.layers.Input(shape=input_img_size),\n            keras.layers.Conv2D(32, 3, activation=\"relu\", padding=\"same\"),\n            keras.layers.MaxPooling2D(pool_size=2, strides=2),\n            keras.layers.Conv2D(32, 3, activation=\"relu\", padding=\"same\"),\n            keras.layers.MaxPooling2D(pool_size=2, strides=2),\n            keras.layers.Conv2D(32, 3, activation=\"relu\", padding=\"same\"),\n            keras.layers.MaxPooling2D(pool_size=2, strides=2),\n            keras.layers.Conv2D(1, 3, activation=\"relu\", padding=\"same\"),\n        ],\n        name=name,\n    )"
        },
        {
            "type": "text",
            "content": "Plug in the models to the trainer class and train them using the very familiar compile and fit methods:"
        },
        {
            "type": "code",
            "content": "gan = CycleGAN(\n    generator_g=build_generator(\"gen_G\"),\n    generator_f=build_generator(\"gen_F\"),\n    discriminator_x=build_discriminator(\"disc_X\"),\n    discriminator_y=build_discriminator(\"disc_Y\"),\n)\n\ngan.compile(\n    gen_g_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),\n    gen_f_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),\n    disc_x_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),\n    disc_y_optimizer=keras.optimizers.Adam(learning_rate=2e-4, beta_1=0.5),\n)\n\nhistory = gan.fit(\n    tf.data.Dataset.zip((train_horses, train_zebras)),\n)"
        }
    ],
    "References": [
        {
            "type": "list",
            "content": [
                "<a href=\"https://arxiv.org/abs/1406.2661\" target=\"_blank\" rel=\"noopener\">Generative Adversarial Networks (2014, June)</a>",
                "<a href=\"https://arxiv.org/abs/1411.1784\" target=\"_blank\" rel=\"noopener\">Conditional Generative Adversarial Nets (2014, November)</a>",
                "<a href=\"https://arxiv.org/abs/1511.06434\" target=\"_blank\" rel=\"noopener\">Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (2015, November)</a>",
                "<a href=\"https://arxiv.org/abs/1703.10593\" target=\"_blank\" rel=\"noopener\">Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (2017, March)</a>",
                "<a href=\"https://arxiv.org/abs/1701.07875\" target=\"_blank\" rel=\"noopener\">Wasserstein GAN (2017, January)</a>",
                "<a href=\"https://arxiv.org/abs/1704.00028\" target=\"_blank\" rel=\"noopener\">Improved Training of Wasserstein GANs (2017, April)</a>",
                "<a href=\"https://arxiv.org/abs/1812.04948\" target=\"_blank\" rel=\"noopener\">A Style-Based Generator Architecture for Generative Adversarial Networks (2018, December)</a>",
                "<a href=\"https://arxiv.org/abs/1906.02691\" target=\"_blank\" rel=\"noopener\">An Introduction to Variational Autoencoders (2019, June)</a>"
            ]
        }
    ],
};
