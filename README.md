# Vabs-Net
Vabs-Net: Pre-Training Protein Bi-level Representation Through Span Mask Strategy On 3D Protein Chains

The repository is an official implementation of [Pre-Training Protein Bi-level Representation Through Span Mask Strategy On 3D Protein Chains](https://arxiv.org/abs/2402.01481)
![poster](https://github.com/user-attachments/assets/06610d4d-2d67-437a-9324-c62470a8d58a)

## Environment installation

```bash
cd Vabs-Net
bash env.sh
```
## Small molecule binding site dataset
In this paper, we also proposed a new small molecule binding site dataset, which can be found in [BindingSiteDataset4VabsNet](https://huggingface.co/datasets/Heisenburger2000/BindingSiteDataset4VabsNet)

## version
We found a slight difference for EC and GO evaluation and updated the re-evaluated results in the arxiv version. The difference occurred because there was an issue with the calculation of the f1max.

## Example of reproduction

### Pre-train

```bash
cd Vabs-Net
bash reproduce_pretrain.sh
```
### Finetune
```bash
cd Vabs-Net
bash reproduce_finetune.sh
```
## Cite
If you want to cite this paper:
```bash
@inproceedings{zhuangpre,
  title={Pre-Training Protein Bi-level Representation Through Span Mask Strategy On 3D Protein Chains},
  author={Zhuang, Wanru and Song, Jia and Li, Yaqi and Lu, Shuqi and others},
  booktitle={Forty-first International Conference on Machine Learning}
}
```
