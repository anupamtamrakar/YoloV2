# YoloV2
YoloV2 quick implementation using darknet and yad2k (Yet another darknet 2 keras) <br>
## This repo will demostrate Object detection mechanism in real time which was usually been used for finding obstacles in driverless cars <br>

Libraries been used: <br>
Darknet: https://pjreddie.com/darknet/yolo/   by(Joseph Redmon) <br>
YAD2k: https://github.com/allanzelener/YAD2K  by(allanzelener)

Tensorflow (1.3.0) <br>
Keras(2.0.3) <br>

![Input](images/images.jpg) <br>
![Output](out/images.jpg)

### Installation Steps:

Download weights and configuration file from Darknet library and generate model(.h5) file using yad2k 
```bash
python yad2k.py yolo.cfg yolo.weights model_data/yolo.h5
```

Go through the steps present in jupyter notebook (Yolov2.ipynb) and observe the output <br>

Feel free to change the image in images/ folder. Output is directly saved in out/


### Troubleshooting Steps:

While generating model (.h5), if unicode error is received -
```bash
UnicodeDecodeError: 'rawunicodeescape' codec can't decode bytes
```

Then try out steps given in (https://github.com/NLeSC/mcfly-tutorial/issues/17)

```bash
#code = marshal.dumps(func.code).decode('raw_unicode_escape')
code = marshal.dumps(func.code).replace(b'\',b'/').decode('raw_unicode_escape')
```

### Sample output of generating model from weights and cfg file:

```bash

Loading weights.
Weights Header:  [       0        1        0 32013312]
Parsing Darknet config.
Creating Keras model.
Parsing section net_0
Parsing section convolutional_0
conv2d bn leaky (3, 3, 3, 32)
2019-01-29 13:14:33.309057: W C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
2019-01-29 13:14:33.330836: W C:\tf_jenkins\home\workspace\rel-win\M\windows\PY\35\tensorflow\core\platform\cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX2 instructions, but these are available on your machine and could speed up CPU computations.
Parsing section maxpool_0
Parsing section convolutional_1
conv2d bn leaky (3, 3, 32, 64)
Parsing section maxpool_1
Parsing section convolutional_2
conv2d bn leaky (3, 3, 64, 128)
Parsing section convolutional_3
conv2d bn leaky (1, 1, 128, 64)
Parsing section convolutional_4
conv2d bn leaky (3, 3, 64, 128)
Parsing section maxpool_2
Parsing section convolutional_5
conv2d bn leaky (3, 3, 128, 256)
Parsing section convolutional_6
conv2d bn leaky (1, 1, 256, 128)
Parsing section convolutional_7
conv2d bn leaky (3, 3, 128, 256)
Parsing section maxpool_3
Parsing section convolutional_8
conv2d bn leaky (3, 3, 256, 512)
Parsing section convolutional_9
conv2d bn leaky (1, 1, 512, 256)
Parsing section convolutional_10
conv2d bn leaky (3, 3, 256, 512)
Parsing section convolutional_11
conv2d bn leaky (1, 1, 512, 256)
Parsing section convolutional_12
conv2d bn leaky (3, 3, 256, 512)
Parsing section maxpool_4
Parsing section convolutional_13
conv2d bn leaky (3, 3, 512, 1024)
Parsing section convolutional_14
conv2d bn leaky (1, 1, 1024, 512)
Parsing section convolutional_15
conv2d bn leaky (3, 3, 512, 1024)
Parsing section convolutional_16
conv2d bn leaky (1, 1, 1024, 512)
Parsing section convolutional_17
conv2d bn leaky (3, 3, 512, 1024)
Parsing section convolutional_18
conv2d bn leaky (3, 3, 1024, 1024)
Parsing section convolutional_19
conv2d bn leaky (3, 3, 1024, 1024)
Parsing section route_0
Parsing section convolutional_20
conv2d bn leaky (1, 1, 512, 64)
Parsing section reorg_0
Parsing section route_1
Concatenating route layers: [<tf.Tensor 'space_to_depth_x2/SpaceToDepth:0' shape=(?, 19, 19, 256) dtype=float32>, <tf.Tensor 'leaky_re_lu_20/sub:0' shape=(?, 19, 19, 1024) dtype=float32>]
Parsing section convolutional_21
conv2d bn leaky (3, 3, 1280, 1024)
Parsing section convolutional_22
conv2d    linear (1, 1, 1024, 425)
Parsing section region_0
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
input_1 (InputLayer)             (None, 608, 608, 3)   0
____________________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 608, 608, 32)  864
____________________________________________________________________________________________________
batch_normalization_1 (BatchNorm (None, 608, 608, 32)  128
____________________________________________________________________________________________________
leaky_re_lu_1 (LeakyReLU)        (None, 608, 608, 32)  0
____________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)   (None, 304, 304, 32)  0
____________________________________________________________________________________________________
conv2d_2 (Conv2D)                (None, 304, 304, 64)  18432
____________________________________________________________________________________________________
batch_normalization_2 (BatchNorm (None, 304, 304, 64)  256
____________________________________________________________________________________________________
leaky_re_lu_2 (LeakyReLU)        (None, 304, 304, 64)  0
____________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)   (None, 152, 152, 64)  0
____________________________________________________________________________________________________
conv2d_3 (Conv2D)                (None, 152, 152, 128) 73728
____________________________________________________________________________________________________
batch_normalization_3 (BatchNorm (None, 152, 152, 128) 512
____________________________________________________________________________________________________
leaky_re_lu_3 (LeakyReLU)        (None, 152, 152, 128) 0
____________________________________________________________________________________________________
conv2d_4 (Conv2D)                (None, 152, 152, 64)  8192
____________________________________________________________________________________________________
batch_normalization_4 (BatchNorm (None, 152, 152, 64)  256
____________________________________________________________________________________________________
leaky_re_lu_4 (LeakyReLU)        (None, 152, 152, 64)  0
____________________________________________________________________________________________________
conv2d_5 (Conv2D)                (None, 152, 152, 128) 73728
____________________________________________________________________________________________________
batch_normalization_5 (BatchNorm (None, 152, 152, 128) 512
____________________________________________________________________________________________________
leaky_re_lu_5 (LeakyReLU)        (None, 152, 152, 128) 0
____________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)   (None, 76, 76, 128)   0
____________________________________________________________________________________________________
conv2d_6 (Conv2D)                (None, 76, 76, 256)   294912
____________________________________________________________________________________________________
batch_normalization_6 (BatchNorm (None, 76, 76, 256)   1024
____________________________________________________________________________________________________
leaky_re_lu_6 (LeakyReLU)        (None, 76, 76, 256)   0
____________________________________________________________________________________________________
conv2d_7 (Conv2D)                (None, 76, 76, 128)   32768
____________________________________________________________________________________________________
batch_normalization_7 (BatchNorm (None, 76, 76, 128)   512
____________________________________________________________________________________________________
leaky_re_lu_7 (LeakyReLU)        (None, 76, 76, 128)   0
____________________________________________________________________________________________________
conv2d_8 (Conv2D)                (None, 76, 76, 256)   294912
____________________________________________________________________________________________________
batch_normalization_8 (BatchNorm (None, 76, 76, 256)   1024
____________________________________________________________________________________________________
leaky_re_lu_8 (LeakyReLU)        (None, 76, 76, 256)   0
____________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)   (None, 38, 38, 256)   0
____________________________________________________________________________________________________
conv2d_9 (Conv2D)                (None, 38, 38, 512)   1179648
____________________________________________________________________________________________________
batch_normalization_9 (BatchNorm (None, 38, 38, 512)   2048
____________________________________________________________________________________________________
leaky_re_lu_9 (LeakyReLU)        (None, 38, 38, 512)   0
____________________________________________________________________________________________________
conv2d_10 (Conv2D)               (None, 38, 38, 256)   131072
____________________________________________________________________________________________________
batch_normalization_10 (BatchNor (None, 38, 38, 256)   1024
____________________________________________________________________________________________________
leaky_re_lu_10 (LeakyReLU)       (None, 38, 38, 256)   0
____________________________________________________________________________________________________
conv2d_11 (Conv2D)               (None, 38, 38, 512)   1179648
____________________________________________________________________________________________________
batch_normalization_11 (BatchNor (None, 38, 38, 512)   2048
____________________________________________________________________________________________________
leaky_re_lu_11 (LeakyReLU)       (None, 38, 38, 512)   0
____________________________________________________________________________________________________
conv2d_12 (Conv2D)               (None, 38, 38, 256)   131072
____________________________________________________________________________________________________
batch_normalization_12 (BatchNor (None, 38, 38, 256)   1024
____________________________________________________________________________________________________
leaky_re_lu_12 (LeakyReLU)       (None, 38, 38, 256)   0
____________________________________________________________________________________________________
conv2d_13 (Conv2D)               (None, 38, 38, 512)   1179648
____________________________________________________________________________________________________
batch_normalization_13 (BatchNor (None, 38, 38, 512)   2048
____________________________________________________________________________________________________
leaky_re_lu_13 (LeakyReLU)       (None, 38, 38, 512)   0
____________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)   (None, 19, 19, 512)   0
____________________________________________________________________________________________________
conv2d_14 (Conv2D)               (None, 19, 19, 1024)  4718592
____________________________________________________________________________________________________
batch_normalization_14 (BatchNor (None, 19, 19, 1024)  4096
____________________________________________________________________________________________________
leaky_re_lu_14 (LeakyReLU)       (None, 19, 19, 1024)  0
____________________________________________________________________________________________________
conv2d_15 (Conv2D)               (None, 19, 19, 512)   524288
____________________________________________________________________________________________________
batch_normalization_15 (BatchNor (None, 19, 19, 512)   2048
____________________________________________________________________________________________________
leaky_re_lu_15 (LeakyReLU)       (None, 19, 19, 512)   0
____________________________________________________________________________________________________
conv2d_16 (Conv2D)               (None, 19, 19, 1024)  4718592
____________________________________________________________________________________________________
batch_normalization_16 (BatchNor (None, 19, 19, 1024)  4096
____________________________________________________________________________________________________
leaky_re_lu_16 (LeakyReLU)       (None, 19, 19, 1024)  0
____________________________________________________________________________________________________
conv2d_17 (Conv2D)               (None, 19, 19, 512)   524288
____________________________________________________________________________________________________
batch_normalization_17 (BatchNor (None, 19, 19, 512)   2048
____________________________________________________________________________________________________
leaky_re_lu_17 (LeakyReLU)       (None, 19, 19, 512)   0
____________________________________________________________________________________________________
conv2d_18 (Conv2D)               (None, 19, 19, 1024)  4718592
____________________________________________________________________________________________________
batch_normalization_18 (BatchNor (None, 19, 19, 1024)  4096
____________________________________________________________________________________________________
leaky_re_lu_18 (LeakyReLU)       (None, 19, 19, 1024)  0
____________________________________________________________________________________________________
conv2d_19 (Conv2D)               (None, 19, 19, 1024)  9437184
____________________________________________________________________________________________________
batch_normalization_19 (BatchNor (None, 19, 19, 1024)  4096
____________________________________________________________________________________________________
conv2d_21 (Conv2D)               (None, 38, 38, 64)    32768
____________________________________________________________________________________________________
leaky_re_lu_19 (LeakyReLU)       (None, 19, 19, 1024)  0
____________________________________________________________________________________________________
batch_normalization_21 (BatchNor (None, 38, 38, 64)    256
____________________________________________________________________________________________________
conv2d_20 (Conv2D)               (None, 19, 19, 1024)  9437184
____________________________________________________________________________________________________
leaky_re_lu_21 (LeakyReLU)       (None, 38, 38, 64)    0
____________________________________________________________________________________________________
batch_normalization_20 (BatchNor (None, 19, 19, 1024)  4096
____________________________________________________________________________________________________
space_to_depth_x2 (Lambda)       (None, 19, 19, 256)   0
____________________________________________________________________________________________________
leaky_re_lu_20 (LeakyReLU)       (None, 19, 19, 1024)  0
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 19, 19, 1280)  0
____________________________________________________________________________________________________
conv2d_22 (Conv2D)               (None, 19, 19, 1024)  11796480
____________________________________________________________________________________________________
batch_normalization_22 (BatchNor (None, 19, 19, 1024)  4096
____________________________________________________________________________________________________
leaky_re_lu_22 (LeakyReLU)       (None, 19, 19, 1024)  0
____________________________________________________________________________________________________
conv2d_23 (Conv2D)               (None, 19, 19, 425)   435625
====================================================================================================
Total params: 50,983,561
Trainable params: 50,962,889
Non-trainable params: 20,672
____________________________________________________________________________________________________
None
Saved Keras model to model_data/yolo.h5
Read 50983561 of 50983561.0 from Darknet weights.
```
