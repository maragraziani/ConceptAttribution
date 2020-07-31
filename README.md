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
With this library you will be able to extract basic concepts representing color and texture from the images in your data. 
See the functions:
<li> get_color_measure(image, mask=None, type=None, verbose=True) 
<li> get_texture_measure(image, mask=None, type=None, verbose=True) 



# Experiments on handwritten digits

# Experiments on breast cancer histopathology

# Experiments on Retinopathy of Prematurity
