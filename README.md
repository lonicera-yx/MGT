# Meta Graph Transformer: A Novel Framework for Spatial–Temporal Traffic Prediction

This is a PyTorch implementation of the paper [Meta Graph Transformer: A Novel Framework for Spatial–Temporal Traffic Prediction](https://doi.org/10.1016/j.neucom.2021.12.033). 

If you use this code for your research, please cite:
```
@article{ye2021meta,
  title = {Meta Graph Transformer: A Novel Framework for Spatial–Temporal Traffic Prediction},
  journal = {Neurocomputing},
  year = {2021},
  issn = {0925-2312},
  doi = {https://doi.org/10.1016/j.neucom.2021.12.033},
  url = {https://www.sciencedirect.com/science/article/pii/S0925231221018725},
  author = {Xue Ye and Shen Fang and Fang Sun and Chunxia Zhang and Shiming Xiang},
  publisher={Elsevier}
}
```

## Train

- Check `requirements.txt`
- Unzip `data.zip`
- Train MGT:
  ```shell
  python main.py <dataset> MGT <experiment name> <CUDA device>
  ```
  For example, 
  ```shell
  python main.py HZMetro MGT E01 0
  ```
  means training MGT model for dataset HZMetro, the experiment name is E01, and the CUDA device number is 0.
- The experiment results will be under the directory: `exps/HZMetro/MGT/E01`
