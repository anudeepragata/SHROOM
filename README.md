# SHROOM: Understanding LLM Halucinations


## Link to model.pth 
https://drive.google.com/file/d/1K8GUeEwPrHmE297gMcUnolvmt1CNleaj/view?usp=sharing

## Objective 
With the increasing use of Large Language Models
(LLMs) like ChatGPT for various tasks such as question
answering, machine translation, and text correction, ensuring
the accuracy of their outputs is crucial. Our study addresses
the challenge of detecting hallucinated outputs from LLMs,
which can be grammatically correct but factually incorrect or
grammatically incorrect altogether for three types of tasks -
definition modeling, paraphrasing and machine translation. 

## Team Members
 - Andrea Pinto 
 - Anudeep Ragata
 - Hasnain Sikora
 - Saumya Gupta

 ## File Description 

 1. clustering_kmeans.py - Consists of a score-based algorithm which uses pre-processed metrics to distinguish the hypothesis based on an ensemble-based stack.

 2. shroomformers.py - Consists of a set of the best performing Siamese networks that we had trained, along with functions to showcase its performance.

 3. ModelCard.py - Is a foundational GUI that lets the user choose between Score-based clustering and the Siamese network to predict if the given source, hypothesis and target display signs of hallucination.

 4. prompting.py - Consists of the training of LLAMA2 to distinguish between example pairs and further narrow it's responses down to a HalluScore - representational of a self-consistency based prompting loop.

 ## Usage

Install all dependencies

<code>pip install -r requirements.txt</code>

To test out Score-based clustering and the Siamese network, run ModelCard.py

<code>python ModelCard.py</code>