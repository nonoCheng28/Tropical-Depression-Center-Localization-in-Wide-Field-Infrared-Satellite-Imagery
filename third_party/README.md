# Third-party notices

`src/td_center/model_tclnet.py` adapts the topology and operations from
[chao-tan/TCLNet](https://github.com/chao-tan/TCLNet),
`modules/components/TCLNET.py`. The upstream MIT notice is preserved in
`TCLNet-LICENSE`, verified against upstream LICENSE blob
`f1d934e2738a41dc1b6ab73034e2018a13100d97`.

The implementation replicates the single input channel to three channels and
uses this project's heatmap supervision. It does not claim the original
paper's data protocol or reported accuracy. ResNet-18 is loaded from the
separately installed torchvision dependency without pretrained weights.

JMA documentation and NOAA satellite-data distributors are identified in the
HSD acquisition source and data documentation. Their data are not included.
