# Omniscient Video Super-Resolution
This is the official code of [OVSR (Omniscient Video Super-Resolution, ICCV 2021)](https://openaccess.thecvf.com/content/ICCV2021/html/Yi_Omniscient_Video_Super-Resolution_ICCV_2021_paper.html).
This work is based on [PFNL](https://github.com/psychopa4/PFNL).

## Datasets
Please refer to [PFNL](https://github.com/psychopa4/PFNL) for the datasets (train, eval and test).
Please modify the datapath in ./data/*.txt according to your machine. 

## Pre-Trained Models
Download the pre-trained models from [mainland China](https://pan.baidu.com/s/1-qv1Io91JtcCv0-x7Q8Auw) with password: inub, or [elsewhere](https://www.terabox.com/web/share/link?surl=4DfhKLDw9j0G6RZtHtzQzw).

## Code
It should be easy to use train.sh or main.py for training or testing, note to change the hyper-parameters in options/ovsr.yml .

## Environment
  - Python >= 3.6
  - PyTorch, tested on 1.9, but should be fine when >=1.6

## Citation
If you find our code or datasets helpful, please consider citing our related works.
```
@InProceedings{Yi_2021_ICCV_OVSR,
    author    = {Yi, Peng and Wang, Zhongyuan and Jiang, Kui and Jiang, Junjun and Lu, Tao and Tian, Xin and Ma, Jiayi},
    title     = {Omniscient Video Super-Resolution},
    booktitle = {IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {4429-4438}
}

@ARTICLE{MSHPFNL,
  author={Yi, Peng and Wang, Zhongyuan and Jiang, Kui and Jiang, Junjun and Lu, Tao and Ma, Jiayi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={A Progressive Fusion Generative Adversarial Network for Realistic and Consistent Video Super-Resolution}, 
  year={2020},
  volume={},
  number={},
  pages={},
  doi={10.1109/TPAMI.2020.3042298}
}
```

## Contact
If you have questions or suggestions, please open an issue here or send an email to yipeng@whu.edu.cn.