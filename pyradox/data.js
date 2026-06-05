let title = "pyradox";
let project_date = "Open Source"
let links = {
    "paper": "",
    "demo": "",
    "code": "https://github.com/Ritvik19/pyradox",
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
            "content": "State-of-the-art neural networks for deep learning with TensorFlow 2. This library helps you implement various state-of-the-art neural networks in a fully customizable fashion. See also <a href=\"/pyradox-generative/\">pyradox-generative</a> and <a href=\"/pyradox-tabular/\">pyradox-tabular</a>."
        }
    ],
    "Installation": [
        {
            "type": "code",
            "content": "pip install pyradox"
        }
    ],
    "Usage - Modules": [
        {
            "type": "table",
            "columns": [
                "Module",
                "Description",
                "Input Shape",
                "Output Shape"
            ],
            "rows": [
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/Rescale/Rescale.md\" target=\"_blank\" rel=\"noopener\">Rescale</a>",
                    "A layer that rescales the input:  x_out = (x_in -mu) / sigma",
                    "Arbitrary",
                    "Same shape as input"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/Convolution2D/Convolution2D.md\" target=\"_blank\" rel=\"noopener\">Convolution 2D</a>",
                    "Applies 2D Convolution followed by Batch Normalization  (optional) and Dropout (optional)",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnected/DenselyConnected.md\" target=\"_blank\" rel=\"noopener\">Densely Connected</a>",
                    "Densely Connected Layer followed by Batch  Normalization (optional) and Dropout (optional)",
                    "2D tensor with shape (batch_size, input_dim)",
                    "2D tensor with shape (batch_size, n_units)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenseNetConvolutionBlock/DenseNetConvolutionBlock.md\" target=\"_blank\" rel=\"noopener\">DenseNet Convolution Block</a>",
                    "A Convolution block for DenseNets",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenseNetConvolutionBlock/DenseNetConvolutionBlock.md\" target=\"_blank\" rel=\"noopener\">DenseNet Convolution Block</a>",
                    "A Convolution block for DenseNets",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenseNetTransitionBlock/DenseNetTransitionBlock.md\" target=\"_blank\" rel=\"noopener\">DenseNet Transition Block</a>",
                    "A Transition block for DenseNets",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenseSkipConnection/DenseSkipConnection.md\" target=\"_blank\" rel=\"noopener\">Dense Skip Connection</a>",
                    "Implementation of a skip connection for densely  connected layer",
                    "2D tensor with shape (batch_size, input_dim)",
                    "2D tensor with shape (batch_size, n_units)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/VGG-Module/VGG-Module.md\" target=\"_blank\" rel=\"noopener\">VGG Module</a>",
                    "Implementation of VGG Modules with slight  modifications, Applies multiple 2D Convolution  followed by Batch Normalization (optional), Dropout  (optional) and MaxPooling",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionConv/InceptionConv.md\" target=\"_blank\" rel=\"noopener\">Inception Conv</a>",
                    "Implementation of 2D Convolution Layer for Inception  Net, Convolution Layer followed by Batch  Normalization, Activation and optional Dropout",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionBlock/InceptionBlock.md\" target=\"_blank\" rel=\"noopener\">Inception Block</a>",
                    "Implementation on Inception Mixing Block",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/XceptionBlock/XceptionBlock.md\" target=\"_blank\" rel=\"noopener\">Xception Block</a>",
                    "A customised implementation of Xception Block  (Depthwise Separable Convolutions)",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetBlock/EfficientNetBlock.md\" target=\"_blank\" rel=\"noopener\">Efficient Net Block</a>",
                    "Implementation of Efficient Net Block (Depthwise  Separable Convolutions)",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ConvSkipConnection/ConvSkipConnection.md\" target=\"_blank\" rel=\"noopener\">Conv Skip Connection</a>",
                    "Implementation of Skip Connection for Convolution  Layer",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNetBlock/ResNetBlock.md\" target=\"_blank\" rel=\"noopener\">Res Net Block</a>",
                    "Customized Implementation of ResNet Block",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNetV2Block/ResNetV2Block.md\" target=\"_blank\" rel=\"noopener\">Res Net V2 Block</a>",
                    "Customized Implementation of ResNetV2 Block",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNeXtBlock/ResNeXtBlock.md\" target=\"_blank\" rel=\"noopener\">Res NeXt Block</a>",
                    "Customized Implementation of ResNeXt Block",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionResNetConv2D/InceptionResNetConv2D.md\" target=\"_blank\" rel=\"noopener\">Inception Res Net Conv 2D</a>",
                    "Implementation of Convolution Layer for Inception Res  Net: Convolution2d followed by Batch Norm",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionResNetBlock-1/InceptionResNetBlock-1.md\" target=\"_blank\" rel=\"noopener\">Inception Res Net Block</a>",
                    "Implementation of Inception-ResNet block",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/NASNetSeparableConvBlock/NASNetSeparableConvBlock.md\" target=\"_blank\" rel=\"noopener\">NAS Net Separable Conv Block</a>",
                    "Adds 2 blocks of Separable Conv Batch Norm",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\" \" target=\"_blank\" rel=\"noopener\">NAS Net Adjust Block</a>",
                    "Adjusts the input previous path to match  the shape of the  input",
                    "",
                    ""
                ],
                [
                    "<a href=\" \" target=\"_blank\" rel=\"noopener\">NAS Net Normal A Cell</a>",
                    "Normal cell for NASNet-A",
                    "",
                    ""
                ],
                [
                    "<a href=\" \" target=\"_blank\" rel=\"noopener\">NAS Net Reduction A Cell</a>",
                    "Reduction cell for NASNet-A",
                    "",
                    ""
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNetConvBlock/MobileNetConvBlock.md\" target=\"_blank\" rel=\"noopener\">Mobile Net Conv Block</a>",
                    "Adds an initial convolution layer with batch  normalization and activation",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNetDepthWiseConvBlock/MobileNetDepthWiseConvBlock.md\" target=\"_blank\" rel=\"noopener\">Mobile Net Depth Wise Conv Block</a>",
                    "Adds a depthwise convolution block. A depthwise  convolution block consists of a depthwise conv, batch  normalization, activation, pointwise convolution,  batch normalization and activation",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InvertedResBlock/InvertedResBlock.md\" target=\"_blank\" rel=\"noopener\">Inverted Res Block</a>",
                    "Adds an Inverted ResNet block",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/SEBlock/SEBlock.md\" target=\"_blank\" rel=\"noopener\">SEBlock</a>",
                    "Adds a Squeeze Excite Block",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ]
            ]
        }
    ],
    "Usage - Conv Nets": [
        {
            "type": "table",
            "columns": [
                "Module",
                "Description",
                "Input Shape",
                "Output Shape"
            ],
            "rows": [
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/GeneralizedDenseNets/GeneralizedDenseNets.md\" target=\"_blank\" rel=\"noopener\">Generalized Dense Nets</a>",
                    "A generalization of Densely Connected Convolutional  Networks (Dense Nets)",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnectedConvolutionalNetwork121/DenselyConnectedConvolutionalNetwork121.md\" target=\"_blank\" rel=\"noopener\">Densely Connected Convolutional Network 121</a>",
                    "A modified implementation of Densely Connected  Convolutional Network 121",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnectedConvolutionalNetwork169/DenselyConnectedConvolutionalNetwork169.md\" target=\"_blank\" rel=\"noopener\">Densely Connected Convolutional Network 169</a>",
                    "A modified implementation of Densely Connected  Convolutional Network 169",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnectedConvolutionalNetwork201/DenselyConnectedConvolutionalNetwork201.md\" target=\"_blank\" rel=\"noopener\">Densely Connected Convolutional Network 201</a>",
                    "A modified implementation of Densely Connected  Convolutional Network 201",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/GeneralizedVGG-1/GeneralizedVGG-1.md\" target=\"_blank\" rel=\"noopener\">Generalized VGG</a>",
                    "A generalization of VGG network",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D or 2D tensor"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/VGG16-1/VGG16-1.md\" target=\"_blank\" rel=\"noopener\">VGG 16</a>",
                    "A modified implementation of VGG16 network",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "2D tensor with shape (batch_shape, new_dim)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/VGG19-1/VGG19-1.md\" target=\"_blank\" rel=\"noopener\">VGG 19</a>",
                    "A modified implementation of VGG19 network",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "2D tensor with shape (batch_shape, new_dim)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionV3/InceptionV3.md\" target=\"_blank\" rel=\"noopener\">Inception V3</a>",
                    "Customized Implementation of Inception Net",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/GeneralizedXception/GeneralizedXception.md\" target=\"_blank\" rel=\"noopener\">Generalized Xception</a>",
                    "Generalized Implementation of XceptionNet (Depthwise  Separable Convolutions)",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/XceptionNet/XceptionNet.md\" target=\"_blank\" rel=\"noopener\">Xception Net</a>",
                    "A Customised Implementation of XceptionNet",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNet/EfficientNet.md\" target=\"_blank\" rel=\"noopener\">Efficient Net</a>",
                    "Generalized Implementation of Effiecient Net",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB0/EfficientNetB0.md\" target=\"_blank\" rel=\"noopener\">Efficient Net B0</a>",
                    "Customized Implementation of Efficient Net B0",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB1/EfficientNetB1.md\" target=\"_blank\" rel=\"noopener\">Efficient Net B1</a>",
                    "Customized Implementation of Efficient Net B1",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB2/EfficientNetB2.md\" target=\"_blank\" rel=\"noopener\">Efficient Net B2</a>",
                    "Customized Implementation of Efficient Net B2",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB3/EfficientNetB3.md\" target=\"_blank\" rel=\"noopener\">Efficient Net B3</a>",
                    "Customized Implementation of Efficient Net B3",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB4/EfficientNetB4.md\" target=\"_blank\" rel=\"noopener\">Efficient Net B4</a>",
                    "Customized Implementation of Efficient Net B4",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB5/EfficientNetB5.md\" target=\"_blank\" rel=\"noopener\">Efficient Net B5</a>",
                    "Customized Implementation of Efficient Net B5",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB6/EfficientNetB6.md\" target=\"_blank\" rel=\"noopener\">Efficient Net B6</a>",
                    "Customized Implementation of Efficient Net B6",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB7/EfficientNetB7.md\" target=\"_blank\" rel=\"noopener\">Efficient Net B7</a>",
                    "Customized Implementation of Efficient Net B7",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet/ResNet.md\" target=\"_blank\" rel=\"noopener\">Res Net</a>",
                    "Customized Implementation of Res Net",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet50/ResNet50.md\" target=\"_blank\" rel=\"noopener\">Res Net 50</a>",
                    "Customized Implementation of Res Net 50",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet101/ResNet101.md\" target=\"_blank\" rel=\"noopener\">Res Net 101</a>",
                    "Customized Implementation of Res Net 101",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet152/ResNet152.md\" target=\"_blank\" rel=\"noopener\">Res Net 152</a>",
                    "Customized Implementation of Res Net 152",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNetV2/ResNetV2.md\" target=\"_blank\" rel=\"noopener\">Res Net V2</a>",
                    "Customized Implementation of Res Net V2",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet50V2/ResNet50V2.md\" target=\"_blank\" rel=\"noopener\">Res Net 50 V2</a>",
                    "Customized Implementation of Res Net 50 V2",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet101V2/ResNet101V2.md\" target=\"_blank\" rel=\"noopener\">Res Net 101 V2</a>",
                    "Customized Implementation of Res Net 101 V2",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet152V2/ResNet152V2.md\" target=\"_blank\" rel=\"noopener\">Res Net 152 V2</a>",
                    "Customized Implementation of Res Net 152 V2",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNeXt/ResNeXt.md\" target=\"_blank\" rel=\"noopener\">Res NeXt</a>",
                    "Customized Implementation of Res NeXt",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNeXt50/ResNeXt50.md\" target=\"_blank\" rel=\"noopener\">Res NeXt 50</a>",
                    "Customized Implementation of Res NeXt 50",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNeXt101/ResNeXt101.md\" target=\"_blank\" rel=\"noopener\">Res NeXt 101</a>",
                    "Customized Implementation of Res NeXt 101",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNeXt152/ResNeXt152.md\" target=\"_blank\" rel=\"noopener\">Res NeXt 152</a>",
                    "Customized Implementation of Res NeXt 152",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionResNetV2/InceptionResNetV2.md\" target=\"_blank\" rel=\"noopener\">Inception Res Net V2</a>",
                    "Customized Implementation of Inception Res Net V2",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/NASNet/NASNet.md\" target=\"_blank\" rel=\"noopener\">NAS Net</a>",
                    "Generalised Implementation of NAS Net",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/NASNetMobile/NASNetMobile.md\" target=\"_blank\" rel=\"noopener\">NAS Net Mobile</a>",
                    "Customized Implementation of NAS Net Mobile",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/NASNetLarge/NASNetLarge.md\" target=\"_blank\" rel=\"noopener\">NAS Net Large</a>",
                    "Customized Implementation of NAS Net Large",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNet/MobileNet-1.md\" target=\"_blank\" rel=\"noopener\">MobileNet</a>",
                    "Customized Implementation of MobileNet",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNetV2/MobileNetV2-1.md\" target=\"_blank\" rel=\"noopener\">Mobile Net V2</a>",
                    "Customized Implementation of Mobile Net V2",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNetV3/MobileNetV3-1.md\" target=\"_blank\" rel=\"noopener\">Mobile Net V3</a>",
                    "Customized Implementation of Mobile Net V3",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/SegNet/SegNet.md\" target=\"_blank\" rel=\"noopener\">Seg Net</a>",
                    "Generalised Implementation of SegNet for Image  Segmentation Applications",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/UNet/UNet.md\" target=\"_blank\" rel=\"noopener\">U Net</a>",
                    "Generalised Implementation of UNet for Image  Segmentation Applications",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)",
                    "4D tensor with shape (batch_shape, rows, cols,  channels)"
                ]
            ]
        }
    ],
    "Usage - Dense Nets": [
        {
            "type": "table",
            "columns": [
                "Module",
                "Description",
                "Input Shape",
                "Output Shape"
            ],
            "rows": [
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnectedNetwork/DenselyConnectedNetwork.md\" target=\"_blank\" rel=\"noopener\">Densely Connected Network</a>",
                    "Network of Densely Connected Layers followed by Batch  Normalization (optional) and Dropout (optional)",
                    "2D tensor with shape (batch_size, input_dim)",
                    "2D tensor with shape (batch_size, new_dim)"
                ],
                [
                    "<a href=\"https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnectedResnet/DenselyConnectedResnet.md\" target=\"_blank\" rel=\"noopener\">Densely Connected Resnet</a>",
                    "Network of skip connections for densely connected  layer",
                    "2D tensor with shape (batch_size, input_dim)",
                    "2D tensor with shape (batch_size, new_dim)"
                ]
            ]
        }
    ],
};
