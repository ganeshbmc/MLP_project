## Insights gained from the datasets  

Insights gained using "domain expertise", EDA, and feature importance scores from models.

### Insights from "Domain Expertise"  

`Disclaimer: I am not a domain expert. I am just a movie buff. Luckily, movie buffs have enough domain expertise to make some educated guesses about the data.`  

### Features from train dataset  
* ReviewText
* movieid  
* reviewerName  
* isFrequentReviewer  
* 

### "genre" with ngram_range=(1,3)
F1-score using full pipeline (GridSearchCV) on X_train: 
0.6756780003465563
Confusion matrix and f1-score for rows which have no reviewText in X_train: 
[[ 129 2276]
 [  88 3954]]
0.6333178222429037

### "genre" with ngram_range=(1,5)
F1-score using full pipeline (GridSearchCV) on X_train: 
0.6778529958824713
Confusion matrix and f1-score for rows which have no reviewText in X_train: 
[[ 148 2257]
 [  93 3949]]
0.6354893749030557

### "genreSorted" with ngram_range=(1,5)
F1-score using full pipeline (GridSearchCV) on X_train: 
0.6734231035377073
Confusion matrix and f1-score for rows which have no reviewText in X_train: 
[[ 120 2285]
 [  87 3955]]
0.6320769350085311

### "genre" with ngram_range=(1,1) AND ### "genreSorted" with ngram_range=(1,5)  
F1-score using full pipeline (GridSearchCV) on X_train: 
0.6733678073961491
Confusion matrix and f1-score for rows which have no reviewText in X_train: 
[[ 118 2287]
 [  85 3957]]
0.6320769350085311




### "ratingContents" with ngram_range=(1,3)
F1-score using full pipeline (GridSearchCV) on X_train: 
0.7150063305068974
Confusion matrix and f1-score for rows which have no reviewText in X_train: 
[[ 470 1935]
 [ 248 3794]]
0.6613928959205833

### "ratingContents" with ngram_range=(1,5)
F1-score using full pipeline (GridSearchCV) on X_train: 
0.72352203219502
Confusion matrix and f1-score for rows which have no reviewText in X_train: 
[[ 532 1873]
 [ 262 3780]]
0.6688382193268186

### "ratingContents" with ngram_range=(1,5) AND "rcSorted" with CountVectorizer ngram_range=(1,1)
F1-score using full pipeline (GridSearchCV) on X_train: 
0.7236080518128338
Confusion matrix and f1-score for rows which have no reviewText in X_train: 
[[ 533 1872]
 [ 265 3777]]
0.6685279975182256

### "ratingContents" with ngram_range=(1,1) AND "rcSorted" with CountVectorizer ngram_range=(1,1)

