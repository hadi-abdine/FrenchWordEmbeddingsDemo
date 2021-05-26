# Linguistic Resources Portal <a href="http://master2-bigdata.polytechnique.fr/"><img width="10%" src='https://drive.google.com/uc?export=view&id=1n43FYop0Ea7pQA77YwCo-0k2XbGaC6rD'></a>

In this [portal](http://master2-bigdata.polytechnique.fr/) we present and make available to the research and industrial community French linguistic resources of high scale and quality for different tasks result of training on very large quantities of online text collected (by our group as well) from the Web. Soon we will integrate similar resources for other languages.
We introduce [BARthez](http://master2-bigdata.polytechnique.fr/FrenchLinguisticResources/barthez), the first french sequence to sequence pretrained model pretrained on 66GB of french raw text for roughly 60 hours on 128 Nvidia V100 GPUs.
We also introduce the French [Word2vec](http://master2-bigdata.polytechnique.fr/FrenchLinguisticResources/frWordEmbeddings) vectors of dimension 300 that were trained using CBOW on a huge 33GB French raw text that we crawled and pre-processed from the French web.

BARThez: a Skilled Pretrained French Sequence-to-Sequence Model: [https://arxiv.org/abs/2010.12321](https://arxiv.org/abs/2010.12321)<br>
Evaluation Of Word Embeddings From Large-Scale French Web Content: [https://arxiv.org/abs/2105.01990](https://arxiv.org/abs/2105.01990)

If you are interested in [downloading](http://master2-bigdata.polytechnique.fr/FrenchLinguisticResources/resources) the linguistic resources files please contact the leader of [DaSciM](http://www.lix.polytechnique.fr/dascim/software_datasets/) group via email: mvazirg\~lix.polytechnique.fr <br> 
This effort is partially funded by the [ANR HELAS chair](http://www.lix.polytechnique.fr/dascim/helas/) 

This UI is built using React, JavaScript, JQuery and Bootstrap.


![image](https://drive.google.com/uc?export=view&id=1soPERpZxR4WAmQxR_h5kW6pLBi15boQ9)
![image](https://drive.google.com/uc?export=view&id=10dscL5Qsle5sDfotAD4zw9h3wvwSRne4)
![image](https://drive.google.com/uc?export=view&id=1EZoKvU0z1MJ85TvUAHcfjaDIP7Tvk4jp)
![image](https://drive.google.com/uc?export=view&id=1byym2YaK3HIRuODofUbChRu\_G1IP\_wsJ)
![image](https://drive.google.com/uc?export=view&id=1WAKDKMU5TzaPDIfy15OFkstPiRqNyecq)
### Setup
To install NPM dependencies:
```ruby
npm install
```
To install all python dependencies:
```ruby
pip3 install -r requirements.txt
```
To run the web app:
```ruby
python3 explore.py
```
Make sure to download the [word vectors](http://master2-bigdata.polytechnique.fr/FrenchLinguisticResources/resources) you're interseting in testing under '../word2vec/dascim2.bin'
