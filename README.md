# Concept Attribution: Explaining CNN decisions to physicians
This repository contains the main code and link to the datasets necessary to replicate the experiments in the paper "Concept Attribution: Explaining CNN decisions to physicians" published in Computers in Biology and Medicine, Volume 123, August 2020, 103865

![Alt text](figs/abstract.jpg?raw=true "Concept Attribution summary")

### 1. Highlights

<li> Feature attribution explains CNNs in terms of the input pixels.
<li> The abstraction of feature attribution to higher level impacting factors is hard.
<li> Concept attribution explains CNNs with high-level concepts such as clinical factors.
<li> Nuclei pleomorphism is shown as a relevant factor in breast tumor classification.
<li> Concept attribution can match clinical expectations to the interpretability of CNNs.

#### 2. Usage

With this library you will be able to apply concept attribution to your task.
The main steps are:
1. Extraction of concept measures
2. Finding the vector representing the concept in the activation space
3. Generating concept-based explanations

##### 2.1. Extract basic concepts
Color and texture measures can be extracted from the images in your data to be represented as concepts.
See the functions:
<li> get_color_measure(image, mask=None, type=None, verbose=True)
<li> get_texture_measure(image, mask=None, type=None, verbose=True)

##### 2.2 Find the concept vectors
We compute RCVs by least squares linear regression of the concept measures for a set of inputs. The concept vector (RCV) represents the direction of greatest increase of the measures for a single continuous concept. Different parameters can be specified to compute the regression:  
 1. compute linear regression  
 2. compute ridge regression
 3. compute local linear regression -- not yet supported

 See the functions:
 <li> get_activations(model, layer, data, labels=None, pooling=None, param_update=False, save_fold='')
 <li> linear_regression(acts, measures, type='linear', evaluation=False, verbose=True)

#### 3. Evaluation

 The regression is evaluated in different ways:
  1. on training or held-out data, with rsquared, mse and adjusted rsquared
  2. by evaluating angle between two RCVs

 See the functions:
 <li> mse(labels, predictions)
 <li> rsquared(labels, predictions)
  
#### 4. Install  
Dependencies 

cv2.cv2	3.4.0		
keras	2.1.3		
numpy	1.13.3		
skimage	0.13.0		
tensorflow	1.5.0, 1.5.0	
statsmodels

#### 5. Cite our work

If you make use of the code, please cite the paper in resulting publications.

```
@article{graziani2020concept,
title = "Concept attribution: Explaining {{CNN}} decisions to physicians",
journal = "Computers in Biology and Medicine",
pages = "103865",
year = "2020",
issn = "0010-4825",
doi = "https://doi.org/10.1016/j.compbiomed.2020.103865",
author = "Graziani M. and Andrearczyk V. and Marchand-Maillet S. and Müller H."
}

or

@incollection{graziani2018regression,
  title={Regression concept vectors for bidirectional explanations in histopathology},
  author={Graziani, Mara and Andrearczyk, Vincent and M{\"u}ller, Henning},
  booktitle={Understanding and Interpreting Machine Learning in Medical Image Computing Applications},
  pages={124--132},
  year={2018},
  publisher={Springer, Cham}
}

```

#### 6. Datasets
Three of the four datasets used for the experiments are publicly available and can be downloaded at the following links:
<li>http://yann.lecun.com/exdb/mnist/
<li>https://camelyon17.grand-challenge.org/Data/
<li>https://nucleisegmentationbenchmark.weebly.com/dataset.html

#### 7. Acknowledgements

This work was supported by PROCESS.

### Contact

For general questions, please email mara.graziani@hevs.ch <br />
