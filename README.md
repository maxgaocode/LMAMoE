#LMAMoE: Learnable model augmentation for graph contrastive learning via mixture-of-experts
Pytorch implementation for the paper "LMAMoE: Learnable model augmentation for graph contrastive learning via mixture-of-experts".

## Abstract

![Framework](CMPGNN.png)


## Training CMPGNN
Taking the dataset Photo as an example, run the following command to obtain the experimental results:
    
    python traincmpgnn.py   --dataset  Photo    --lr 0.001  --weight_decay  5e-5  --dropout 0.5      --train_rate 0.6    --K=2        --step1=2.0    --step2=0.01
##  Baselines

Traditional methods. We employ 3 traditional methods including  Node2vec, DeepWalk, and LINE. These methods belong to shallow models as they did not utilize the idea of deep learning. 

Data augmentation. We apply 6 data augmentation works including DGI, GraphCL,  COLES, GGD , GRACE  and MVGRL. As a key part of the contrastive field, these methods tend to employ various ad-hoc strategies to generate augmented graphs for shared-weighted contrastive frameworks. 

Model augmentation. We apply 3 model augmentation methods including SUGRL, BGRL, and MAGCL. 
## Codes and datasets
The implementation of this code is largely built upon [MVGRL](https://github.com/kavehhassani/mvgrl) 



## Citation
```bibtex

```