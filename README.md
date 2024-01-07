# TinyGPT

## A lightweight deep language model

![image](https://github.com/jessiepathfinder/TinyGPT/assets/55774978/47448f3f-65c8-4088-910e-d31701296108)

## Current model architecture
![image](https://github.com/jessiepathfinder/TinyGPT/assets/55774978/9b2aaf29-8235-4d43-98d0-1ea62ce2a155)



### Notes
1. TinyGPT's norm uses group norm instead of layer norm
2. TinyGPT's multi-head attention shares the same key over all heads
3. TinyGPT uses 10 attention heads
4. All Conv1D layers are causal conv



[pre-trained model (3072 batches)](https://www.mediafire.com/file/5f1ld5r2yqnsdsg/TinyGPT.Pretrained.7z/file)
