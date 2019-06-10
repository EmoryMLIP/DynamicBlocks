---
title: 'DynamicBlocks: A Generalized ResNet Architecture'
tags:
  - Pytorch
  - PDE-based Machine Learning
  - Discretize-Optimize
authors:
  - name: Derek Onken
    orcid: 0000-0002-4640-767X
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Simion Novikov
    affiliation: 2
  - name: Eran Treister
    affiliation: 2
  - name: Eldad Haber
    affiliation: 3
  - name: Lars Ruthotto
    affiliation: "4,1"
affiliations:
 - name: Department of Computer Science, Emory University
   index: 1
 - name: Ben-Gurion University of the Negev
   index: 2
 - name: University of British-Columbia
   index: 3
 - name: Department of Mathematics, Emory University
   index: 4
date: 9 June 2019
bibliography: paper.bib
---

# Summary

Deep Residual Neural Networks (ResNets) have shown impressive performance on several image classification tasks [@he2016deep]. ResNets feature a skip connection, which has been observed to increase a model's robustness to vanishing and exploding gradients. ResNets also possess interesting theoretical properties; for example,  the forward propagation through a ResNet can be interpreted as an explicit Euler method applied to a nonlinear ordinary differential equation (ODE) [@weinan2017; @haber2017stable]. Similarly, @ruthotto2018deep introduces a similar interpretation of convolutional ResNets as partial differential equations (PDE). These insights provide more theoretical insight along with new network architectures motivated by different types of differential equations, e.g., reversible hyperbolic network [@chang2018reversible] or parabolic networks [@ruthotto2018deep]. Recent attention focuses on improving the time integrators used in forward propagation, e.g., higher-order single and multistep methods [@lu2018], black-box time integrators [@chen2018neural], and semi-implicit discretizations [@haber2019imexnet]. 

The primary goal of this toolbox is to facilitate further research and development in convolutional residual neural networks that are motivated by PDEs. To this end, we generalize the notation of ResNets, referring to each of its parts that amend a PDE interpretation (i.e., each set of several consecutive convolutional layers of fixed number of channels) as a ``dynamic block``. A dynamic block can then be compactly described by a layer function and parameters of the time integrator (e.g., time discretization points).  We provide the flexibility to model several state-of-the-art networks by combining several dynamic blocks (acting on different image resolutions and number of channels) through connective units (i.e., a convolutional layer to change the number of channels followed by a pooling layer).

Our ``DynamicBlocks`` toolbox provides a general framework to experiment with and apply more techniques from numerical differential equations in the context of ResNets. In its first version, we include capabilities
to obtain a more general version of ResNet based on a forward Euler (Runge-Kutta 1) module that can handle arbitrary, non-uniform time steps.
Furthermore, we include a fourth-order accurate Runge-Kuttta 4 block to demonstrate the generalizability of the architecture to accept other PDE solvers. 

In the training, we primarily focus on discretize-optimize learning methods, which are popular in optimal control and whose favorable properties for ResNets have been shown in  @gholami2019anode. However, dynamic blocks can also be employed in optimize-then-discretize approaches as in Neural ODEs [@chen2018neural].  

# Acknowledgements

This material is in part based upon work supported by the National Science Foundation under
Grant Number DMS-1751636. Any opinions, findings, and conclusions or recommendations expressed
in this material are those of the authors and do not necessarily reflect the views of the
National Science Foundation.

# References