# Concept Attribution: Explaining CNN decisions to physicians
This repository contains the main code and link to the datasets necessary to replicate the experiments in the paper "Concept Attribution: Explaining CNN decisions to physicians" published in Computers in Biology and Medicine, Volume 123, August 2020, 103865

Highlights

• Feature attribution explains CNNs in terms of the input pixels.
• The abstraction of feature attribution to higher level impacting factors is hard.
• Concept attribution explains CNNs with high-level concepts such as clinical factors.
• Nuclei pleomorphism is shown as a relevant factor in breast tumor classification.
• Concept attribution can match clinical expectations to the interpretability of CNNs.

# Datasets
Three of the four datasets used for the experiments are publicly available and can be downloaded at the following links:
<li>http://yann.lecun.com/exdb/mnist/
<li>https://camelyon17.grand-challenge.org/Data/
<li>https://nucleisegmentationbenchmark.weebly.com/dataset.html
  
# Regression Concept Vectors: RCV-tool library  
With this library you will be able to apply concept attribution to your task. 
The main steps are:
1. Extraction of concept measures
2. Finding the vector representing the concept in the activation space
3. Generating concept-based explanations

## 1. Extract basic concepts
Color and texture measures can be extracted from the images in your data to be represented as concepts. 
See the functions:
<li> get_color_measure(image, mask=None, type=None, verbose=True) 
<li> get_texture_measure(image, mask=None, type=None, verbose=True) 

## 2. Find the concept vectors
We compute RCVs by least squares linear regression ofthe concept measures for a set of inputs. The concept vector (RCV) represents the direction of greatest increase of the measures for a single continuous concept. 
 
 See the functions:
 <li> get_activations(model, layer, data, labels=None, pooling=None, param_update=False, save_fold='')
 <li>  compute linear regression  
 <li> compute ridge regression
 <li> compute local linear regression -- not yet supported

# Experiments on handwritten digits

# Experiments on breast cancer histopathology

# Experiments on Retinopathy of Prematurity
