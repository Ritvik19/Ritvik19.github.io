let title = "pyradox-tabular";
let project_date = "Open Source"
let links = {
    "paper": "",
    "demo": "",
    "code": "https://github.com/Ritvik19/pyradox-tabular",
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
            "content": "Implementations for various state-of-the-art neural networks for tabular data. Part of the <a href=\"/pyradox/\">pyradox</a> ecosystem."
        }
    ],
    "Installation": [
        {
            "type": "code",
            "content": "pip install pyradox-tabular"
        }
    ],
    "Data Preparation": [
        {
            "type": "text",
            "content": "pyradox-tabular comes with its own DataLoader Class which can be used to load data from a pandas DataFrame. <br> We provide a utility DataConfig class which stores the configuration of the data, which are then required by the model for feature preprocessing. <br> We also provide seperate ModelConfig classes for the different models, which ae required to store the model hyperparamers."
        },
        {
            "type": "code",
            "content": "from pyradox_tabular.data import DataLoader\nfrom pyradox_tabular.data_config import DataConfig\n\ndata_config = DataConfig(\n    numeric_feature_names=[\"numerical\", \"column\",\"names\"],\n    categorical_features_with_vocabulary={\n        \"column\": [\"label\", \"encoded\", \"unique\", \"values\", \"as\", \"strings\"],\n    },\n)\n\ndata_train = DataLoader.from_df(x_train, y_train, batch_size=1024)\ndata_valid = DataLoader.from_df(x_valid, y_valid, batch_size=1024)\ndata_test = DataLoader.from_df(x_test, batch_size=1024)"
        }
    ],
    "Deep Tabular Network": [
        {
            "type": "text",
            "content": "In principle a neural network can approximate any continuous function and piece wise continuous function. However, it is not suitable to approximate arbitrary non-continuous functions as it assumes certain level of continuity in its general form.<br>Unlike unstructured data found in nature, structured data with categorical features may not have continuity at all and even if it has it may not be so obvious.<br>Deep Tabular Network use the entity embedding method to automatically learn the representation of categorical features in multi-dimensional spaces which reveals the intrinsic continuity of the data and helps neural networks to solve the problem."
        },
        {
            "type": "code",
            "content": "from pyradox_tabular.model_config import DeepNetworkConfig\nfrom pyradox_tabular.nn import DeepTabularNetwork\n\nmodel_config = DeepNetworkConfig(num_outputs=1, out_activation='sigmoid', hidden_units=[64, 64])\nmodel = DeepTabularNetwork.from_config(data_config, model_config, name=\"deep_network\")\nmodel.compile(optimizer=\"adam\", loss=\"binary_crossentropy\")\nmodel.fit(data_train, validation_data=data_valid)\npreds = model.predict(data_test)"
        }
    ],
    "Wide and Deep Tabular Network": [
        {
            "type": "text",
            "content": "The human brain is a sophisticated learning machine, forming rules by memorizing everyday events and generalizing those learnings to apply tothings we haven't seen before. Perhaps more powerfully, memorization also allows us to further refine our generalized rules with exceptions.<br>By jointly training a wide linear model (for memorization) alongside a deep neural network (for generalization) Wide and Deep Tabular Networks combine the strengths of both to bring us one step closer to teach computers to learn like humans do."
        },
        {
            "type": "code",
            "content": "from pyradox_tabular.model_config import WideAndDeepNetworkConfig\nfrom pyradox_tabular.nn import WideAndDeepTabularNetwork\n\nmodel_config = WideAndDeepNetworkConfig(num_outputs=1, out_activation='sigmoid', hidden_units=[64, 64])\nmodel = WideAndDeepTabularNetwork.from_config(data_config, model_config, name=\"wide_deep_network\")\nmodel.compile(optimizer=\"adam\", loss=\"binary_crossentropy\")\nmodel.fit(data_train, validation_data=data_valid)\npreds = model.predict(data_test)"
        }
    ],
    "Deep and Cross Tabular Network": [
        {
            "type": "text",
            "content": "Feature engineering has been the key to the success of many prediction models. However, the process is nontrivial and often requires manual feature engineering or exhaustive searching. DNNs are able to automatically learn feature interactions; however, they generate all the interactions implicitly, and are not necessarily efficient in learning all types of cross features.<br>Deep and Cross Tabular Network explicitly applies feature crossing at each layer, requires no manual feature engineering, and adds negligible extra complexity to the DNN model."
        },
        {
            "type": "code",
            "content": "from pyradox_tabular.model_config import DeepAndCrossNetworkConfig\nfrom pyradox_tabular.nn import DeepAndCrossTabularNetwork\n\nmodel_config = DeepAndCrossNetworkConfig(num_outputs=1, out_activation='sigmoid', hidden_units=[64, 64], n_cross=2)\nmodel = DeepAndCrossTabularNetwork.from_config(data_config, model_config, name=\"deep_cross_network\")\nmodel.compile(optimizer=\"adam\", loss=\"binary_crossentropy\")\nmodel.fit(data_train, validation_data=data_valid)\npreds = model.predict(data_test)"
        }
    ],
    "TabTansformer": [
        {
            "type": "text",
            "content": "TabTransformer is built upon self-attention based on Transformers. The Transformer layers transform the embeddings of categorical features into robust contextual embeddings to achieve higher prediction accuracy.<br>The contextual embeddings learned from TabTransformer are highly robust against both missing and noisy data features, and provide better interpretability."
        },
        {
            "type": "code",
            "content": "from pyradox_tabular.model_config import TabTransformerConfig\nfrom pyradox_tabular.nn import TabTransformer\n\nmodel_config = TabTransformerConfig(num_outputs=1, out_activation='sigmoid', num_transformer_blocks=3, num_heads=4, mlp_hidden_units_factors=[2, 1])\nmodel = TabTransformer.from_config(data_config, model_config, name=\"tab_transformer\")\nmodel.compile(optimizer=\"adam\", loss=\"binary_crossentropy\")\nmodel.fit(data_train, validation_data=data_valid)\npreds = model.predict(data_test)"
        }
    ],
    "TabNet": [
        {
            "type": "text",
            "content": "TabNet uses sequential attention to choose which features to reason from at each decision step, enabling interpretability and better learning as the learning capacity is used for the most salient features.<br>It employs a single deep learning architecture for feature selection and reasoning."
        },
        {
            "type": "code",
            "content": "from pyradox_tabular.model_config import TabNetConfig\nfrom pyradox_tabular.nn import TabNet\n\nmodel_config = TabNetConfig(num_outputs=1, out_activation='sigmoid',feature_dim=16, output_dim=12, num_decision_steps=5)\nmodel = TabNet.from_config(data_config, model_config, name=\"tabnet\")\nmodel.compile(optimizer=\"adam\", loss=\"binary_crossentropy\")\nmodel.fit(data_train, validation_data=data_valid)\npreds = model.predict(data_test)"
        }
    ],
    "Deep Neural Decision Tree": [
        {
            "type": "text",
            "content": "Deep Neural Decision Trees unifies classification trees with the representation learning functionality known from deep convolutional network. These are essentially a stochastic and differentiable decision tree model."
        },
        {
            "type": "code",
            "content": "from pyradox_tabular.model_config import NeuralDecisionTreeConfig\nfrom pyradox_tabular.nn import NeuralDecisionTree\n\nmodel_config = NeuralDecisionTreeConfig(depth=2, used_features_rate=1, num_classes=2)\nmodel = NeuralDecisionTree.from_config(data_config, model_config, name=\"deep_neural_decision_tree\")\nmodel.compile(optimizer=\"adam\", loss=\"binary_crossentropy\")\nmodel.fit(data_train, validation_data=data_valid)\npreds = model.predict(data_test)"
        }
    ],
    "Deep Neural Decision Forest": [
        {
            "type": "text",
            "content": "A Deep Neural Decision Forest is an bagging ensemble of Deep Neural Decision Trees."
        },
        {
            "type": "code",
            "content": "from pyradox_tabular.model_config import NeuralDecisionForestConfig\nfrom pyradox_tabular.nn import NeuralDecisionForest\n\nmodel_config = NeuralDecisionForestConfig(num_trees=10, depth=2, used_features_rate=0.8, num_classes=2)\nmodel = NeuralDecisionForest.from_config(data_config, model_config, name=\"deep_neural_decision_forest\")\nmodel.compile(optimizer=\"adam\", loss=\"binary_crossentropy\")\nmodel.fit(data_train, validation_data=data_valid)\npreds = model.predict(data_test)"
        }
    ],
    "Neural Oblivious Decision Tree": [
        {
            "type": "text",
            "content": ""
        },
        {
            "type": "code",
            "content": "from pyradox_tabular.model_config import NeuralObliviousDecisionTreeConfig\nfrom pyradox_tabular.nn import NeuralObliviousDecisionTree\n\nmodel_config = NeuralObliviousDecisionTreeConfig()\nmodel = NeuralObliviousDecisionTree.from_config(data_config, model_config, name=\"neural_oblivious_decision_tree\")\nmodel.compile(optimizer=\"adam\", loss=\"binary_crossentropy\")\nmodel.fit(data_train, validation_data=data_valid)\npreds = model.predict(data_test)"
        }
    ],
    "Neural Oblivious Decision Ensemble": [
        {
            "type": "text",
            "content": "NODE architecture generalizes ensembles of oblivious decision trees, but benefits from both end-to-end gradient-based optimization and the power of multi-layer hierarchical representation learning."
        },
        {
            "type": "code",
            "content": "from pyradox_tabular.model_config import NeuralObliviousDecisionEnsembleConfig\nfrom pyradox_tabular.nn import NeuralObliviousDecisionEnsemble\n\nmodel_config = NeuralObliviousDecisionEnsembleConfig()\nmodel = NeuralObliviousDecisionEnsemble.from_config(data_config, model_config, name=\"neural_oblivious_decision_ensemble\")\nmodel.compile(optimizer=\"adam\", loss=\"binary_crossentropy\")\nmodel.fit(data_train, validation_data=data_valid)\npreds = model.predict(data_test)"
        }
    ],
    "Feature Tokenizer Transformer": [
        {
            "type": "text",
            "content": "It is a simple adaptation of the Transformer architecture for the tabular domain. In a nutshell, Feature Tokenizer Transformer transforms all features (categorical and numerical) to embeddings and applies a stack of Transformer layers to the embeddings. <br>Thus, every Transformer layer operates on the feature level of one object."
        },
        {
            "type": "code",
            "content": "from pyradox_tabular.model_config import FeatureTokenizerTransformerConfig\nfrom pyradox_tabular.nn import FeatureTokenizerTransformer\n\nmodel_config = FeatureTokenizerTransformerConfig(num_outputs=1, out_activation='sigmoid', num_transformer_blocks=2, num_heads=8, embedding_dim=32, dense_dim=16)\nmodel = FeatureTokenizerTransformer.from_config(data_config, model_config, name=\"feature_tokenizer_transformer\")\nmodel.compile(optimizer=\"adam\", loss=\"binary_crossentropy\")\nmodel.fit(data_train, validation_data=data_valid)\npreds = model.predict(data_test)"
        }
    ],
    "Tabular ResNet": [
        {
            "type": "text",
            "content": "Tabular Resnet is a ResNet like architecture containing skip connection but instead of Convolutional Layers, it consists of Linear Layers."
        },
        {
            "type": "code",
            "content": "from pyradox_tabular.model_config import TabularResNetConfig\nfrom pyradox_tabular.nn import TabularResNet\n\nmodel_config = TabularResNetConfig(num_outputs=1, out_activation='sigmoid', hidden_units=[64, 64])\nmodel = TabularResNet.from_config(data_config, model_config, name=\"deep_network\")\nmodel.compile(optimizer=\"adam\", loss=\"binary_crossentropy\")\nmodel.fit(data_train, validation_data=data_valid)\npreds = model.predict(data_test)"
        }
    ],
    "References": [
        {
            "type": "list",
            "content": [
                "<a href=\"https://arxiv.org/abs/1604.06737\" target=\"_blank\" rel=\"noopener\">Entity Embeddings of Categorical Variables (2016, April)</a>",
                "<a href=\"https://ai.googleblog.com/2016/06/wide-deep-learning-better-together-with.html\" target=\"_blank\" rel=\"noopener\">Wide & Deep Learning: Better Together with TensorFlow (2016, June)</a>",
                "<a href=\"https://arxiv.org/pdf/1708.05123.pdf\" target=\"_blank\" rel=\"noopener\">Deep & Cross Network for Ad Click Predictions (2017, August)</a>",
                "<a href=\"https://arxiv.org/pdf/2012.06678.pdf\" target=\"_blank\" rel=\"noopener\">TabTransformer: Tabular Data Modeling Using Contextual Embeddings (2020, December)</a>",
                "<a href=\"https://arxiv.org/pdf/1908.07442.pdf\" target=\"_blank\" rel=\"noopener\">TabNet: Attentive Interpretable Tabular Learning (2020, December)</a>",
                "<a href=\"https://ieeexplore.ieee.org/document/7410529\" target=\"_blank\" rel=\"noopener\">Deep Neural Decision Forests (2015, December)</a>",
                "<a href=\"https://arxiv.org/pdf/1909.06312.pdf\" target=\"_blank\" rel=\"noopener\">Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data (2019, September)</a>",
                "<a href=\"https://arxiv.org/abs/2106.11959\" target=\"_blank\" rel=\"noopener\">Revisiting Deep Learning Models for Tabular Data (2021, June)</a>"
            ]
        }
    ],
};
