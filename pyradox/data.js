const nav_data = ["Usage - Modules", "Usage - Conv Nets", "Usage - Dense Nets"];

const usage_data = [
  [
    {
      Module: "Rescale",
      Description:
        "A layer that rescales the input:  x_out = (x_in -mu) / sigma",
      "Input Shape": "Arbitrary",
      "Output Shape": "Same shape as input",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/Rescale/Rescale.md",
    },
    {
      Module: "Convolution 2D",
      Description:
        "Applies 2D Convolution followed by Batch Normalization  (optional) and Dropout (optional)",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/Convolution2D/Convolution2D.md",
    },
    {
      Module: "Densely Connected",
      Description:
        "Densely Connected Layer followed by Batch  Normalization (optional) and Dropout (optional)",
      "Input Shape": "2D tensor with shape (batch_size, input_dim)",
      "Output Shape": "2D tensor with shape (batch_size, n_units)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnected/DenselyConnected.md",
    },
    {
      Module: "DenseNet Convolution Block",
      Description: "A Convolution block for DenseNets",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenseNetConvolutionBlock/DenseNetConvolutionBlock.md",
    },
    {
      Module: "DenseNet Convolution Block",
      Description: "A Convolution block for DenseNets",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenseNetConvolutionBlock/DenseNetConvolutionBlock.md",
    },
    {
      Module: "DenseNet Transition Block",
      Description: "A Transition block for DenseNets",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenseNetTransitionBlock/DenseNetTransitionBlock.md",
    },
    {
      Module: "Dense Skip Connection",
      Description:
        "Implementation of a skip connection for densely  connected layer",
      "Input Shape": "2D tensor with shape (batch_size, input_dim)",
      "Output Shape": "2D tensor with shape (batch_size, n_units)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenseSkipConnection/DenseSkipConnection.md",
    },
    {
      Module: "VGG Module",
      Description:
        "Implementation of VGG Modules with slight  modifications, Applies multiple 2D Convolution  followed by Batch Normalization (optional), Dropout  (optional) and MaxPooling",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/VGG-Module/VGG-Module.md",
    },
    {
      Module: "Inception Conv",
      Description:
        "Implementation of 2D Convolution Layer for Inception  Net, Convolution Layer followed by Batch  Normalization, Activation and optional Dropout",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionConv/InceptionConv.md",
    },
    {
      Module: "Inception Block",
      Description: "Implementation on Inception Mixing Block",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionBlock/InceptionBlock.md",
    },
    {
      Module: "Xception Block",
      Description:
        "A customised implementation of Xception Block  (Depthwise Separable Convolutions)",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/XceptionBlock/XceptionBlock.md",
    },
    {
      Module: "Efficient Net Block",
      Description:
        "Implementation of Efficient Net Block (Depthwise  Separable Convolutions)",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetBlock/EfficientNetBlock.md",
    },
    {
      Module: "Conv Skip Connection",
      Description: "Implementation of Skip Connection for Convolution  Layer",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ConvSkipConnection/ConvSkipConnection.md",
    },
    {
      Module: "Res Net Block",
      Description: "Customized Implementation of ResNet Block",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNetBlock/ResNetBlock.md",
    },
    {
      Module: "Res Net V2 Block",
      Description: "Customized Implementation of ResNetV2 Block",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNetV2Block/ResNetV2Block.md",
    },
    {
      Module: "Res NeXt Block",
      Description: "Customized Implementation of ResNeXt Block",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNeXtBlock/ResNeXtBlock.md",
    },
    {
      Module: "Inception Res Net Conv 2D",
      Description:
        "Implementation of Convolution Layer for Inception Res  Net: Convolution2d followed by Batch Norm",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionResNetConv2D/InceptionResNetConv2D.md",
    },
    {
      Module: "Inception Res Net Block",
      Description: "Implementation of Inception-ResNet block",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionResNetBlock-1/InceptionResNetBlock-1.md",
    },
    {
      Module: "NAS Net Separable Conv Block",
      Description: "Adds 2 blocks of Separable Conv Batch Norm",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/NASNetSeparableConvBlock/NASNetSeparableConvBlock.md",
    },
    {
      Module: "NAS Net Adjust Block",
      Description:
        "Adjusts the input previous path to match  the shape of the  input",
      "Input Shape": "",
      "Output Shape": "",
      Usage: " ",
    },
    {
      Module: "NAS Net Normal A Cell",
      Description: "Normal cell for NASNet-A",
      "Input Shape": "",
      "Output Shape": "",
      Usage: " ",
    },
    {
      Module: "NAS Net Reduction A Cell",
      Description: "Reduction cell for NASNet-A",
      "Input Shape": "",
      "Output Shape": "",
      Usage: " ",
    },
    {
      Module: "Mobile Net Conv Block",
      Description:
        "Adds an initial convolution layer with batch  normalization and activation",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNetConvBlock/MobileNetConvBlock.md",
    },
    {
      Module: "Mobile Net Depth Wise Conv Block",
      Description:
        "Adds a depthwise convolution block. A depthwise  convolution block consists of a depthwise conv, batch  normalization, activation, pointwise convolution,  batch normalization and activation",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNetDepthWiseConvBlock/MobileNetDepthWiseConvBlock.md",
    },
    {
      Module: "Inverted Res Block",
      Description: "Adds an Inverted ResNet block",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InvertedResBlock/InvertedResBlock.md",
    },
    {
      Module: "SEBlock",
      Description: "Adds a Squeeze Excite Block",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/SEBlock/SEBlock.md",
    },
  ],
  [
    {
      Module: "Generalized Dense Nets",
      Description:
        "A generalization of Densely Connected Convolutional  Networks (Dense Nets)",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/GeneralizedDenseNets/GeneralizedDenseNets.md",
    },
    {
      Module: "Densely Connected Convolutional Network 121",
      Description:
        "A modified implementation of Densely Connected  Convolutional Network 121",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnectedConvolutionalNetwork121/DenselyConnectedConvolutionalNetwork121.md",
    },
    {
      Module: "Densely Connected Convolutional Network 169",
      Description:
        "A modified implementation of Densely Connected  Convolutional Network 169",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnectedConvolutionalNetwork169/DenselyConnectedConvolutionalNetwork169.md",
    },
    {
      Module: "Densely Connected Convolutional Network 201",
      Description:
        "A modified implementation of Densely Connected  Convolutional Network 201",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnectedConvolutionalNetwork201/DenselyConnectedConvolutionalNetwork201.md",
    },
    {
      Module: "Generalized VGG",
      Description: "A generalization of VGG network",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape": "4D or 2D tensor",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/GeneralizedVGG-1/GeneralizedVGG-1.md",
    },
    {
      Module: "VGG 16",
      Description: "A modified implementation of VGG16 network",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape": "2D tensor with shape (batch_shape, new_dim)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/VGG16-1/VGG16-1.md",
    },
    {
      Module: "VGG 19",
      Description: "A modified implementation of VGG19 network",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape": "2D tensor with shape (batch_shape, new_dim)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/VGG19-1/VGG19-1.md",
    },
    {
      Module: "Inception V3",
      Description: "Customized Implementation of Inception Net",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionV3/InceptionV3.md",
    },
    {
      Module: "Generalized Xception",
      Description:
        "Generalized Implementation of XceptionNet (Depthwise  Separable Convolutions)",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/GeneralizedXception/GeneralizedXception.md",
    },
    {
      Module: "Xception Net",
      Description: "A Customised Implementation of XceptionNet",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/XceptionNet/XceptionNet.md",
    },
    {
      Module: "Efficient Net",
      Description: "Generalized Implementation of Effiecient Net",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNet/EfficientNet.md",
    },
    {
      Module: "Efficient Net B0",
      Description: "Customized Implementation of Efficient Net B0",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB0/EfficientNetB0.md",
    },
    {
      Module: "Efficient Net B1",
      Description: "Customized Implementation of Efficient Net B1",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB1/EfficientNetB1.md",
    },
    {
      Module: "Efficient Net B2",
      Description: "Customized Implementation of Efficient Net B2",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB2/EfficientNetB2.md",
    },
    {
      Module: "Efficient Net B3",
      Description: "Customized Implementation of Efficient Net B3",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB3/EfficientNetB3.md",
    },
    {
      Module: "Efficient Net B4",
      Description: "Customized Implementation of Efficient Net B4",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB4/EfficientNetB4.md",
    },
    {
      Module: "Efficient Net B5",
      Description: "Customized Implementation of Efficient Net B5",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB5/EfficientNetB5.md",
    },
    {
      Module: "Efficient Net B6",
      Description: "Customized Implementation of Efficient Net B6",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB6/EfficientNetB6.md",
    },
    {
      Module: "Efficient Net B7",
      Description: "Customized Implementation of Efficient Net B7",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/EfficientNetB7/EfficientNetB7.md",
    },
    {
      Module: "Res Net",
      Description: "Customized Implementation of Res Net",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet/ResNet.md",
    },
    {
      Module: "Res Net 50",
      Description: "Customized Implementation of Res Net 50",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet50/ResNet50.md",
    },
    {
      Module: "Res Net 101",
      Description: "Customized Implementation of Res Net 101",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet101/ResNet101.md",
    },
    {
      Module: "Res Net 152",
      Description: "Customized Implementation of Res Net 152",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet152/ResNet152.md",
    },
    {
      Module: "Res Net V2",
      Description: "Customized Implementation of Res Net V2",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNetV2/ResNetV2.md",
    },
    {
      Module: "Res Net 50 V2",
      Description: "Customized Implementation of Res Net 50 V2",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet50V2/ResNet50V2.md",
    },
    {
      Module: "Res Net 101 V2",
      Description: "Customized Implementation of Res Net 101 V2",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet101V2/ResNet101V2.md",
    },
    {
      Module: "Res Net 152 V2",
      Description: "Customized Implementation of Res Net 152 V2",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNet152V2/ResNet152V2.md",
    },
    {
      Module: "Res NeXt",
      Description: "Customized Implementation of Res NeXt",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNeXt/ResNeXt.md",
    },
    {
      Module: "Res NeXt 50",
      Description: "Customized Implementation of Res NeXt 50",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNeXt50/ResNeXt50.md",
    },
    {
      Module: "Res NeXt 101",
      Description: "Customized Implementation of Res NeXt 101",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNeXt101/ResNeXt101.md",
    },
    {
      Module: "Res NeXt 152",
      Description: "Customized Implementation of Res NeXt 152",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/ResNeXt152/ResNeXt152.md",
    },
    {
      Module: "Inception Res Net V2",
      Description: "Customized Implementation of Inception Res Net V2",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/InceptionResNetV2/InceptionResNetV2.md",
    },
    {
      Module: "NAS Net",
      Description: "Generalised Implementation of NAS Net",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/NASNet/NASNet.md",
    },
    {
      Module: "NAS Net Mobile",
      Description: "Customized Implementation of NAS Net Mobile",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/NASNetMobile/NASNetMobile.md",
    },
    {
      Module: "NAS Net Large",
      Description: "Customized Implementation of NAS Net Large",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/NASNetLarge/NASNetLarge.md",
    },
    {
      Module: "MobileNet",
      Description: "Customized Implementation of MobileNet",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNet/MobileNet-1.md",
    },
    {
      Module: "Mobile Net V2",
      Description: "Customized Implementation of Mobile Net V2",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNetV2/MobileNetV2-1.md",
    },
    {
      Module: "Mobile Net V3",
      Description: "Customized Implementation of Mobile Net V3",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, new_rows, new_cols,  new_channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/MobileNetV3/MobileNetV3-1.md",
    },
    {
      Module: "Seg Net",
      Description:
        "Generalised Implementation of SegNet for Image  Segmentation Applications",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/SegNet/SegNet.md",
    },
    {
      Module: "U Net",
      Description:
        "Generalised Implementation of UNet for Image  Segmentation Applications",
      "Input Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      "Output Shape":
        "4D tensor with shape (batch_shape, rows, cols,  channels)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/UNet/UNet.md",
    },
  ],
  [
    {
      Module: "Densely Connected Network",
      Description:
        "Network of Densely Connected Layers followed by Batch  Normalization (optional) and Dropout (optional)",
      "Input Shape": "2D tensor with shape (batch_size, input_dim)",
      "Output Shape": "2D tensor with shape (batch_size, new_dim)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnectedNetwork/DenselyConnectedNetwork.md",
    },
    {
      Module: "Densely Connected Resnet",
      Description: "Network of skip connections for densely connected  layer",
      "Input Shape": "2D tensor with shape (batch_size, input_dim)",
      "Output Shape": "2D tensor with shape (batch_size, new_dim)",
      Usage:
        "https://github.com/Ritvik19/pyradox-doc/blob/main/usage/DenselyConnectedResnet/DenselyConnectedResnet.md",
    },
  ],
];
