# Ψ-Net: Stacking Densely Convolutional LSTMs for Sub-Cortical Brain Structure Segmentation

by [Lihao Liu](http://lihaoliu-cambridge.github.io), [Xiaowei Hu](https://xw-hu.github.io/), [Lei Zhu](https://appsrv.cse.cuhk.edu.hk/~lzhu/), and [Pheng-Ann Heng](http://www.cse.cuhk.edu.hk/~pheng/1.html).  


## Introduction

In this repository, we provide the Tensorflow and DLTK implementation for our TMI paper [Ψ-Net: Stacking Densely Convolutional LSTMs for Sub-Cortical Brain Structure Segmentation](https://ieeexplore.ieee.org/document/9007625). 

<img src="https://github.com/lihaoliu-cambridge/lihaoliu-cambridge.github.io/blob/master/pic/papers/psi-net-network.png">  


## Requirement

tensorflow-gpu       1.14.0  
dltk                 0.2.1   
cuda                 10.0  
cudnn                7.5


## Usage

1. Clone the repository:

   ```shell
   git clone https://github.com/lihaoliu-cambridge/psi-net.git
   cd psi-net
   ```
   
2. Download the IBSR dataset: 

   [IBSR_V2.0_nifti_stripped.tgz](https://www.nitrc.org/frs/?group_id=48)  
   
3. Unzip them in folder `dataset/IBSR`:

   `dataset/IBSR/IBSR_nifti_stripped`   
   
4. Pre-process the LPBA40 dataset:

   ```shell
   cd script
   python preprocessing_ibsr.py
   ```  
   
   output results:  
   
   `dataset/IBSR_preprocessed/IBSR_nifti_stripped`   
   
   
5. Train the model:
 
   ```shell
   cd ..
   python train_ibsr.py
   ```

6. Test the saved model:
 
   ```shell
   python test_ibsr.py 
   ```
   

## Note
1. If you are using a virtual environment, please reload cuda and cudnn before running, so you can use gpu during training. You can also add the cuda and cudnn path to your system path:

   ```shell
   source ~/tensorflow-env/bin/activate
   module load cuda/10.0 cudnn/7.5_cuda-10.0
   ```
   
2. Use pip to install Tensorflow and DLTK directly:

   ```shell
   pip install tensorflow-gpu==1.14.0
   pip install dltk
   ```
   
3. In our TMI paper, we use ``Whiten`` normalization to standardize data distributions. To better standardize data distributions and facilitate training, we try another normalization approach ``Histogram Standardization``. The results are shown in the below picture:

<img src="https://github.com/lihaoliu-cambridge/lihaoliu-cambridge.github.io/blob/master/pic/papers/psi-net-histogram_standardization.png" width="360"/>  


## Citation

If you use our code for your research, please cite our paper:

```
@article{liu2020psi,
  title={$\psi$-Net: Stacking Densely Convolutional LSTMs for Sub-cortical Brain Structure Segmentation},
  author={Liu, Lihao and Hu, Xiaowei and Zhu, Lei and Fu, Chi-Wing and Qin, Jing and Heng, Pheng-Ann},
  journal={IEEE Transactions on Medical Imaging},
  year={2020},
  publisher={IEEE}
}
```


## Question

Please open an issue or email lhliu1994@gmail.com for any questions.
