# Attention-in-Attention-Networks

Implementation of ICCV19: Bilinear attention networks for person retrieval and its extension for Journal paper (under review)


If you find this code useful in your research then please cite
```bash
@InProceedings{Fang_2019_ICCV,
author = {Fang, Pengfei and Zhou, Jieming and Roy, Soumava Kumar and Petersson, Lars and Harandi, Mehrtash},
title = {Bilinear Attention Networks for Person Retrieval},
booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```


## Setups
- Python 2.7.16
- torch 1.1.0
- torchvision 0.3.0  

## Datasets
- CUHK01
- CUHK03
- Market-1501
- DukeMTMC
- MSMT17

## Networks
- Inception 
- Inception + part feature extractor
- other networks...

## Training network on benchmark datasets
Here is an example-step one
```bash
python train_imgreid_xent_semitri.py --pretrained --resume 'pretrained_model/inceptionV1_bn_noModule.pth.tar' --height 256 --width 128 --max-epoch 50 --train-batch 64 --focused-parts 4 --factor-of-scale-factors 1 --stepsize 150 200 250 --gamma 0.1 --lr 0.0005 --weight-decay 0.0001 --drop-rate 0.1 --margin 1 --num_trip 10 --optim 'adam' --save-dir './market/log' --arch bilinear_baseline --gpu-devices 0,1
```
step two
```bash
python train_imgreid_xent_semitri_re.py --resume './market/log/checkpoint_ep50.pth.tar' --height 256 --width 128 --max-epoch 300 --train-batch 64 --focused-parts 4 --factor-of-scale-factors 1 --stepsize 150 200 250 --gamma 0.1 --lr 0.0005 --weight-decay 0.0001 --drop-rate 0.1 --margin 1 --num_trip 10 --optim 'adam' --save-dir './market/log_re' --arch bilinear_baseline --gpu-devices 0,1
```



