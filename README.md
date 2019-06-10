
# DynamicBlocks
PyTorch implementations of a generalized ResNet-inspired network architecture which allows for broader experimentation. 

### Getting Started

Start up your python environment and install the python packages stored in requirements.txt:

```
pip3 install -r requirements.txt 
```

Run the default network (RK4 scheme using a doubleSymLayer on the CIFAR-10 dataset):
```
python3 RKNet.py 
```

[further setup details](setup.md)

[naming convention](naming.md)

### References

The concepts behind the networks implemented by this toolbox are detailed in:

Lars Ruthotto and Eldad Haber (2018). [Deep Neural Networks Motivated by Partial Differential Equations](https://arxiv.org/abs/1804.04272). arXiv.org.

Eldad Haber and Lars Ruthotto (2017). [Stable architectures for deep neural networks](https://doi.org/10.1088/1361-6420/aa9a90). Inverse Problems, 34(1).

Bo Chang, Lili Meng, Eldad Haber, Lars Ruthotto, David Begert, and Elliot Holtham (2018). [Reversible architectures for arbitrarily deep residual neural networks](https://arxiv.org/abs/1709.03698). Presented at the Thirty-Second AAAI Conference on Artificial Intelligence.



### Acknowledgements

This material is in part based upon work supported by the National Science Foundation under Grant Number DMS-1751636. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
