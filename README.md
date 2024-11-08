# Caffe with 3D CRF-RNN
This is a modified version of [Caffe](https://github.com/BVLC/caffe) which supports the **3D Conditional Random Field Recurrent Neural Network (3D CRF-RNN) architecture** as described in our paper [Automatic bladder segmentation from CT images using deep CNN and 3D fully connected CRF-RNN](https://doi.org/10.1007/s11548-018-1733-7). The implementation of 3D CRF-RNN is extended from the [2D CRF-RNN](https://github.com/torrvision/crfasrnn/).

This code has been compiled and passed on `Windows 7 (64 bits)` using `Visual Studio 2013`.

## How to build

**Requirements**: `Visual Studio 2013`, `ITK-4.10`, `CUDA 8.0` and `cuDNN v5`

### Pre-Build Steps
Please make sure CUDA and cuDNN have been installed correctly on your computer.

Clone the project by running:
```
git clone https://github.com/superxuang/caffe_3d_crf_rnn.git
```

In `.\windows\Caffe.bat` set `ITK_PATH` to ITK intall path (the path containing ITK `include`,`lib` folders).

### Build
Run `.\windows\Caffe.bat` and build the project `caffe` in `Visual Studio 2013`.

## License and Citation

Please cite our paper and Caffe if it is useful for your research:

    @article{Xu_2018,
      author="Xu, Xuanang and Zhou, Fugen and Liu, Bo",
      title="Automatic bladder segmentation from CT images using deep CNN and 3D fully connected CRF-RNN",
      journal="International Journal of Computer Assisted Radiology and Surgery",
      year="2018",
      month="Jul",
      day="01",
      volume="13",
      number="7",
      pages="967--975",
      issn="1861-6429",
      doi="10.1007/s11548-018-1733-7",
      url="https://doi.org/10.1007/s11548-018-1733-7"
    }

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

## Contact
You are welcome to contact us:  
  - [xux12@rpi.edu](mailto:xux12@rpi.edu) ([Dr. Xuanang Xu](https://superxuang.github.io/))
