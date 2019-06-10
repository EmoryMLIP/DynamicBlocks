---
title: 'DynamicBlocks: A Generalized ResNet Architecture'
tags:
  - Pytorch
  - PDE-based Machine Learning
  - Discretize-Optimize
authors:
  - name: Derek Onken
    orcid: 0000-0000-0000-0000
    affiliation: 1 # (Multiple affiliations must be quoted)
  - name: Simion Novikov
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Eran Treister
    orcid: 0000-0000-0000-0000
    affiliation: 2
  - name: Eldad Haber
    orcid: 0000-0000-0000-0000
    affiliation: 3
  - name: Lars Ruthotto
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
 - name: Emory University
   index: 1
 - name: Ben-Gurion University of the Negev
   index: 2
 - name: University of British-Columbia
   index: 3
date: 9 June 2019
bibliography: paper.bib
---

# Summary

The impressive performance of Residual Neural Networks (ResNet) [@he2016deep] 
on image classification tasks opened these deep learning networks to new
interpretations. Recent works view ResNet as merely a forward Euler scheme
solving a continuous partial differential equation (PDE) [@weinan2017; @HaberRuthotto2017].
We generalize the ResNet, dubbing each set of several consecutive convolutional layers of 
fixed number of channels as a ``dynamic block`` because each represents a dynamical system 
with an accompanying PDE. These dynamic blocks are glued together by connective units (a convolutional
layer to change the number of channels followed by a pooling layer).

``DynamicBlocks`` exists as a toolbox and general framework to experiment with various aspects
of ResNet. By directing the channel change through the connective units, we can effectively alter 
aspects of the dynamic blocks or piece different blocks together effectively. We include capabilities
to recreate a ResNet in this framework by using a forward Euler (Runge-Kutta 1) module for each dynamic block.

Furthermore, we include a Runge-Kuttta 4 block to demonstrate the generalizability of the architecture to
accept other PDE solvers. For our implementations of dynamic block, we take the Discretize-Optimize approach
as recommended by '@gholami2019anode'. However, this toolbox does not restrict dynamic blocks to be Discretize-Optimize.  



# Acknowledgements

We acknowledge contributions from ....NSF....TODO.

# References