# Federated Graph Learning under Domain Shift with Generalizable Prototypes

>Guancheng Wan, Wenke Huang, Mang Ye

## Abstract
Federated Graph Learning is a privacy-preserving collaborative approach for training a shared model on graph-structured data in the distributed environment. However, in real-world scenarios, the client graph data usually originate from diverse domains, this unavoidably hinders the generalization performance of the final global model. To address this challenge, we start the first attempt to investigate this scenario by learning a well-generalizable model. In order to improve the performance of the global model from different perspectives, we propose a novel framework called Federated Graph Learning with Generalizable Prototypes (FGGP). It decouples the global model into two levels and bridges them via prototypes.  These prototypes, which are semantic centers derived from the feature extractor, can provide valuable classification information. At the classification model level, we innovatively eschew the traditional classifiers, then instead leverage clustered prototypes to capture fruitful domain information and enhance the discriminative capability of the classes, improving the performance of multi-domain predictions. Furthermore, at the feature extractor level, we go beyond traditional approaches by implicitly injecting distinct global knowledge and employing contrastive learning to obtain more powerful prototypes while enhancing the feature extractor generalization ability. Experimental results on various datasets are presented to validate the effectiveness of the proposed method.


## Citation

``` latex
@inproceedings{wan2024federated,
  title={Federated Graph Learning under Domain Shift with Generalizable Prototypes},
  author={Wan, Guancheng and Huang, Wenke and Ye, Mang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={14},
  pages={15429--15437},
  year={2024}
}
```
