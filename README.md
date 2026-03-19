# ECE 176 Final Project Report and Code

## Final project deliverables:

 - Report: [report.tex](report/report.tex)
 - Code: [code/](code/)
 - Video: [Final video](https://youtube.com)

*Some files such as dataset files and model parameters omitted due to size constraints.*

## A small reflection on this project. 

I think this was a very interesting undertaking. I have worked in the past on a number of machine learning projects, in fact, the reverse image search at a large Russian online fashion retailer([Lamoda](https://www.lamoda.ru/)) was implemented by me. Specifically I researched, assembled a dataset, and trained a Faster-RCNN for object detection, where each detection is then piped into an embedding similarity search model, which finds the most similar items of clothing. My implementation improved a number of KPIs by up to 3%, which was one of the most successfull A/B tests conducted by the team I was part of. 

As a result of having some experience working with, and implementing computer vision models, I decided to try my hands on a language model. I found somewhere a video talking about this new paper that tried to implement "attention-like" behaviour, but with linear, rather than quadratic scaling. That was something that caught my eye, and hence I decided to read the paper, and try to implement the approach. However once I discovered that the entire evaluation was done on a 16M parameter model, I felt that it was important to try push the boundaries of this approach. Especially considering that I recently purchased an Nvidia DGX Spark, which allowed me to train models significantly larger than what I would have been able to do otherwise.

It took a while to understand what Plucker embeddings and Grassmann manifolds are, but eventually I realized that it is not the most difficult thing to implement. Once I actually got to training and testing these models out, I was pleasantly surprised. They actually perform reasonably well. Given more time, I believe that it would be interesting to do a 100B token training run (I only did 10B runs) and see what the model is able to come up with. Then also try to do some reinforcement learning to convert it into an actual chat bot. 

I hope you enjoy this exploration! Unfortunately I did not reach any conclusive results.