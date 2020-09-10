# Crowdflower Search Relevance  

Assignment for interview process at [Edge Networks](https://edgenetworks.in/)  

Info  
https://www.kaggle.com/c/crowdflower-search-relevance  
Download data from here  
https://www.kaggle.com/c/crowdflower-search-relevance/data  

Colab notebooks   
https://drive.google.com/drive/folders/1rsvlrDNDzsADFrjaed5iSo3e4PIfnmS2  

I have worked on a similar problem in [Microsoft hackathon 2018](https://github.com/bhavsarpratik/hackathons/tree/master/microsoft_AI_challenge_2018) last December. At the time I had gone through many other available solutions and tried things which had given me an understanding that BERT>LSTM+CNN>hand made feature+classical algo.   

- TFIDF+SVD+SVM - 0.58 (all numbers are weighted kappa as per Kaggle)
- TFIDF+SVD+NN - around 0.58
- sentence-transformer embeddings+FFNN - around 0.57
- glove.6B.50d+LSTM+CNN - around 0.57  

## Slight modification to BERT      
Since BERT NSP head allows only 2 classes and the problem required 4 classes, I made another processor called [Crowdflower in glue_processors.py](https://github.com/bhavsarpratik/hackathons/blob/master/crowdflower-search-relevance/glue_processors.py#L193) to take sentence A and sentence B along with 4 classes. So the preparation of input happens like NSP and SeqClassification head allows for 4 classes.  
   
- bert-large-uncased+SeqClassification head-64l- 0.69  
- bert-base-uncased+SeqClassification head-64l - 0.70  
  I tried bert-base with max_seq_length 48, 64 and 128 and 64 gave the best result.
- bert-base-uncased+SeqClassification head-64l-neg - I also tried to add negative samples by randomly taking 20% of the rows, shuffling query and giving them median_relevance 1(which means query do not match the product) - 0.68
- Roberta and Xlnet are even better than BERT at entailment problems as per GLUE RTE metrics but huggingface transformers library kept throwing error when I ran these 2 models. I couldn't find solution to the problems.
- The new model ALBERT is smaller and superior to BERT but it is not yet integrated into huggingface transformers library


Surprisingly, the best model on kaggle 4years ago is 0.72 which is quite impressive. I think an ensemble of BERT + classical might give even better results since BERT is good with semantic while classical is good with token matching.
