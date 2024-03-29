const nav_data = [
  "Data Preparation",
  "Deep Tabular Network",
  "Wide and Deep Tabular Network",
  "Deep and Cross Tabular Network",
  "TabTansformer",
  "TabNet",
  "Deep Neural Decision Tree",
  "Deep Neural Decision Forest",
  "Neural Oblivious Decision Tree",
  "Neural Oblivious Decision Ensemble",
  "Feature Tokenizer Transformer",
  "Tabular ResNet",
];

const references = [
  [
    "Entity Embeddings of Categorical Variables (2016, April)",
    "https://arxiv.org/abs/1604.06737",
  ],
  [
    "Wide & Deep Learning: Better Together with TensorFlow (2016, June)",
    "https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html",
  ],
  [
    "Deep & Cross Network for Ad Click Predictions (2017, August)",
    "https://arxiv.org/pdf/1708.05123.pdf",
  ],
  [
    "TabTransformer: Tabular Data Modeling Using Contextual Embeddings (2020, December)",
    "https://arxiv.org/pdf/2012.06678.pdf",
  ],
  [
    "TabNet: Attentive Interpretable Tabular Learning (2020, December)",
    "https://arxiv.org/pdf/1908.07442.pdf",
  ],
  [
    "Deep Neural Decision Forests (2015, December)",
    "https://ieeexplore.ieee.org/document/7410529",
  ],
  [
    "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data (2019, September)",
    "https://arxiv.org/pdf/1909.06312.pdf",
  ],
  [
    "Revisiting Deep Learning Models for Tabular Data (2021, June)",
    "https://arxiv.org/abs/2106.11959",
  ],
];

const usage = [
  {
    title: "Data Preparation",
    content: [
      {
        type: "p",
        text:
          "pyradox-tabular comes with its own DataLoader Class which can be used to load data from a " +
          "pandas DataFrame. <br> We provide a utility DataConfig class which stores the configuration of " +
          "the data, which are then required by the model for feature preprocessing. <br> We also provide " +
          "seperate ModelConfig classes for the different models, which ae required to store the model " +
          "hyperparamers.",
      },
      {
        type: "code",
        text: `from pyradox_tabular.data import DataLoader
from pyradox_tabular.data_config import DataConfig

data_config = DataConfig(
    numeric_feature_names=["numerical", "column","names"],
    categorical_features_with_vocabulary={
        "column": ["label", "encoded", "unique", "values", "as", "strings"],
    },
)

data_train = DataLoader.from_df(x_train, y_train, batch_size=1024)
data_valid = DataLoader.from_df(x_valid, y_valid, batch_size=1024)
data_test = DataLoader.from_df(x_test, batch_size=1024)`,
      },
    ],
  },
  {
    title: "Deep Tabular Network",
    content: [
      {
        type: "p",
        text:
          "In principle a neural network can approximate any continuous function and piece wise continuous " +
          "function. However, it is not suitable to approximate arbitrary non-continuous functions as it " +
          "assumes certain level of continuity in its general form.<br>" +
          "Unlike unstructured data found in nature, structured data with categorical features may not have " +
          "continuity at all and even if it has it may not be so obvious.<br>" +
          "Deep Tabular Network use the entity embedding method to automatically learn the representation of " +
          "categorical features in multi-dimensional spaces which reveals the intrinsic continuity of the data " +
          "and helps neural networks to solve the problem.",
      },
      {
        type: "code",
        text: `from pyradox_tabular.model_config import DeepNetworkConfig
from pyradox_tabular.nn import DeepTabularNetwork

model_config = DeepNetworkConfig(num_outputs=1, out_activation='sigmoid', hidden_units=[64, 64])
model = DeepTabularNetwork.from_config(data_config, model_config, name="deep_network")
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(data_train, validation_data=data_valid)
preds = model.predict(data_test)`,
      },
    ],
  },
  {
    title: "Wide and Deep Tabular Network",
    content: [
      {
        type: "p",
        text:
          "The human brain is a sophisticated learning machine, forming rules by memorizing everyday events " +
          "and generalizing those learnings to apply tothings we haven't seen before. Perhaps more powerfully, " +
          "memorization also allows us to further refine our generalized rules with exceptions.<br>" +
          "By jointly training a wide linear model (for memorization) alongside a deep neural network (for " +
          "generalization) Wide and Deep Tabular Networks combine the strengths of both to bring us one step " +
          "closer to teach computers to learn like humans do.",
      },
      {
        type: "code",
        text: `from pyradox_tabular.model_config import WideAndDeepNetworkConfig
from pyradox_tabular.nn import WideAndDeepTabularNetwork

model_config = WideAndDeepNetworkConfig(num_outputs=1, out_activation='sigmoid', hidden_units=[64, 64])
model = WideAndDeepTabularNetwork.from_config(data_config, model_config, name="wide_deep_network")
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(data_train, validation_data=data_valid)
preds = model.predict(data_test)`,
      },
    ],
  },
  {
    title: "Deep and Cross Tabular Network",
    content: [
      {
        type: "p",
        text:
          "Feature engineering has been the key to the success of many prediction models. However, " +
          "the process is nontrivial and often requires manual feature engineering or exhaustive searching. " +
          "DNNs are able to automatically learn feature interactions; however, they generate all the " +
          "interactions implicitly, and are not necessarily efficient in learning all types of cross " +
          "features.<br>" +
          "Deep and Cross Tabular Network explicitly applies feature crossing at each layer, requires no " +
          "manual feature engineering, and adds negligible extra complexity to the DNN model.",
      },
      {
        type: "code",
        text: `from pyradox_tabular.model_config import DeepAndCrossNetworkConfig
from pyradox_tabular.nn import DeepAndCrossTabularNetwork

model_config = DeepAndCrossNetworkConfig(num_outputs=1, out_activation='sigmoid', hidden_units=[64, 64], n_cross=2)
model = DeepAndCrossTabularNetwork.from_config(data_config, model_config, name="deep_cross_network")
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(data_train, validation_data=data_valid)
preds = model.predict(data_test)`,
      },
    ],
  },
  {
    title: "TabTansformer",
    content: [
      {
        type: "p",
        text:
          "TabTransformer is built upon self-attention based on Transformers. The Transformer layers " +
          "transform the embeddings of categorical features into robust contextual embeddings to achieve " +
          "higher prediction accuracy.<br>" +
          "The contextual embeddings learned from TabTransformer are highly robust against both missing and " +
          "noisy data features, and provide better interpretability.",
      },
      {
        type: "code",
        text: `from pyradox_tabular.model_config import TabTransformerConfig
from pyradox_tabular.nn import TabTransformer

model_config = TabTransformerConfig(num_outputs=1, out_activation='sigmoid', num_transformer_blocks=3, num_heads=4, mlp_hidden_units_factors=[2, 1])
model = TabTransformer.from_config(data_config, model_config, name="tab_transformer")
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(data_train, validation_data=data_valid)
preds = model.predict(data_test)`,
      },
    ],
  },
  {
    title: "TabNet",
    content: [
      {
        type: "p",
        text:
          "TabNet uses sequential attention to choose which features to reason from at each decision " +
          "step, enabling interpretability and better learning as the learning capacity is used for the " +
          "most salient features.<br>" +
          "It employs a single deep learning architecture for feature selection and reasoning.",
      },
      {
        type: "code",
        text: `from pyradox_tabular.model_config import TabNetConfig
from pyradox_tabular.nn import TabNet

model_config = TabNetConfig(num_outputs=1, out_activation='sigmoid',feature_dim=16, output_dim=12, num_decision_steps=5)
model = TabNet.from_config(data_config, model_config, name="tabnet")
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(data_train, validation_data=data_valid)
preds = model.predict(data_test)`,
      },
    ],
  },
  {
    title: "Deep Neural Decision Tree",
    content: [
      {
        type: "p",
        text:
          "Deep Neural Decision Trees unifies classification trees with the representation learning " +
          "functionality known from deep convolutional network. These are essentially a stochastic and " +
          "differentiable decision tree model.",
      },
      {
        type: "code",
        text: `from pyradox_tabular.model_config import NeuralDecisionTreeConfig
from pyradox_tabular.nn import NeuralDecisionTree

model_config = NeuralDecisionTreeConfig(depth=2, used_features_rate=1, num_classes=2)
model = NeuralDecisionTree.from_config(data_config, model_config, name="deep_neural_decision_tree")
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(data_train, validation_data=data_valid)
preds = model.predict(data_test)`,
      },
    ],
  },
  {
    title: "Deep Neural Decision Forest",
    content: [
      {
        type: "p",
        text: "A Deep Neural Decision Forest is an bagging ensemble of Deep Neural Decision Trees.",
      },
      {
        type: "code",
        text: `from pyradox_tabular.model_config import NeuralDecisionForestConfig
from pyradox_tabular.nn import NeuralDecisionForest

model_config = NeuralDecisionForestConfig(num_trees=10, depth=2, used_features_rate=0.8, num_classes=2)
model = NeuralDecisionForest.from_config(data_config, model_config, name="deep_neural_decision_forest")
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(data_train, validation_data=data_valid)
preds = model.predict(data_test)`,
      },
    ],
  },
  {
    title: "Neural Oblivious Decision Tree",
    content: [
      {
        type: "p",
        text: "",
      },
      {
        type: "code",
        text: `from pyradox_tabular.model_config import NeuralObliviousDecisionTreeConfig
from pyradox_tabular.nn import NeuralObliviousDecisionTree

model_config = NeuralObliviousDecisionTreeConfig()
model = NeuralObliviousDecisionTree.from_config(data_config, model_config, name="neural_oblivious_decision_tree")
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(data_train, validation_data=data_valid)
preds = model.predict(data_test)`,
      },
    ],
  },
  {
    title: "Neural Oblivious Decision Ensemble",
    content: [
      {
        type: "p",
        text:
          "NODE architecture generalizes ensembles of oblivious decision trees, but benefits from both " +
          "end-to-end gradient-based optimization and the power of multi-layer hierarchical representation " +
          "learning.",
      },
      {
        type: "code",
        text: `from pyradox_tabular.model_config import NeuralObliviousDecisionEnsembleConfig
from pyradox_tabular.nn import NeuralObliviousDecisionEnsemble

model_config = NeuralObliviousDecisionEnsembleConfig()
model = NeuralObliviousDecisionEnsemble.from_config(data_config, model_config, name="neural_oblivious_decision_ensemble")
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(data_train, validation_data=data_valid)
preds = model.predict(data_test)`,
      },
    ],
  },
  {
    title: "Feature Tokenizer Transformer",
    content: [
      {
        type: "p",
        text:
          "It is a simple adaptation of the Transformer architecture for the tabular domain. In a nutshell, " +
          "Feature Tokenizer Transformer transforms all features (categorical and numerical) to embeddings " +
          "and applies a stack of Transformer layers to the embeddings. <br>" +
          "Thus, every Transformer layer operates on the feature level of one object.",
      },
      {
        type: "code",
        text: `from pyradox_tabular.model_config import FeatureTokenizerTransformerConfig
from pyradox_tabular.nn import FeatureTokenizerTransformer

model_config = FeatureTokenizerTransformerConfig(num_outputs=1, out_activation='sigmoid', num_transformer_blocks=2, num_heads=8, embedding_dim=32, dense_dim=16)
model = FeatureTokenizerTransformer.from_config(data_config, model_config, name="feature_tokenizer_transformer")
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(data_train, validation_data=data_valid)
preds = model.predict(data_test)`,
      },
    ],
  },
  {
    title: "Tabular ResNet",
    content: [
      {
        type: "p",
        text:
          "Tabular Resnet is a ResNet like architecture containing skip connection but instead of " +
          "Convolutional Layers, it consists of Linear Layers.",
      },
      {
        type: "code",
        text: `from pyradox_tabular.model_config import TabularResNetConfig
from pyradox_tabular.nn import TabularResNet

model_config = TabularResNetConfig(num_outputs=1, out_activation='sigmoid', hidden_units=[64, 64])
model = TabularResNet.from_config(data_config, model_config, name="deep_network")
model.compile(optimizer="adam", loss="binary_crossentropy")
model.fit(data_train, validation_data=data_valid)
preds = model.predict(data_test)`,
      },
    ],
  },
];
