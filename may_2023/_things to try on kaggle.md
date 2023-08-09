## Feedback from level 1 viva

Use median for runtimemins column
Try to replace outliers with median value in runtimemins column
Try to impute missing values using other column values
Try to use custom stop_words built from common words in positive and negative sentiment rows
Use stratified split instead of train_test_split - use k-fold crossvalidation


## For level 2 viva
- Showcase all the work in the final notebook
- Add markdown for insights, summaries and also annotate
- Try ensemble models 
- Try bagging and boosting on top of base models like logreg which are already giving good scores
- Read up theory part of MLT and MLF



# Things to try on kaggle

Imputing missing values from other columns seems to be a big problem - Better not to pursue

Drop duplicates better - Use groupby?	-	DONE
Try other solvers for LogReg		-	DONE
Try LinearSVC with GridSearchCV		-	DONE
Try LightGBM				-	DONE

Try feature selection
Try feature engineering		-	Add a column for reviewText present/absent
Try ensemble methods
Try Bagging/boosting		-	DONE	-	AdaBoost is bad!


## NOTEBOOK CLEANING
	- Annotate
	- Write insights and observations
	- More EDA and graphs
	- Write strengths of my notebook


## Radical ideas
	- Get common reviewText vocabulary of train and test and train best model only on that common vocab
	- Same as above but get common vocab by splitting train itself
	- Run the model with previous best kaggle score on
				- Limited (best) feature set
	- Change GridsearchCV scoring function to F1_micro
				- Rerun on all_dict and on a limited feature (best) set 

	- Replace numbers in reviewText

### Scores of each feature on LogReg
	- Categorical  
		- isFrequentReviewer: 0.668
		- rating: 0.668
		- originalLanguage: 0.66
	- Numerical  
		- audiencescore: 0.69		**
		- runtimeMins: 0.66
		- boxOffice: 0.66
	- Text  
		- reviewText: 0.82+			*****
		- movieid: 0.78				****
		- reviewerName: 0.67		*
		- genre: 0.66				*
		- director: 0.72			***
		- originalLanguage: 0.668		


## Do something extra
	- Try stip_accents='unicode' in TfidfVectorizer		-	DONE
	- Replace numbers in reviewText with emtpy strings	-	DONE
	- Adjust boxOffice for inflation 
	- Try transformation. For eg. Log transform on boxOffice 
	- Try polynomial features
	- Try CountVectorizer instead of TfidfVectorizer
	- Try both TfidfVectorizer and CountVectorizer on reviewText	-	DONE
	- Try CountVectorizer or DictVectorizer on genre
	- Replace "&" with "" in genre
	- MinMaxScaler vs StandardScaler on numerical features

## Last minute dash  
	- Use the remaining features from movies dataset
<<<<<<< HEAD:may_2023/things to try on kaggle.md
	- Interrelationship between 'isFrequentReviewer', 'rating', 'audienceScore'


=======
	- Drop rows with more than 7 missing values in train
	- Interrelationship between 'isFrequentReviewer', 'rating', 'audienceScore'


## Finishing touches  
	- Show minimum 3 models!
	- Annotate
	- Strengths  
    	- Modular code (DRY principle), custom functions  
    	- End to end pipeline in one function (with helper functions)  
	- Insights  
    	- Domain expertise  
    	- EDA  
  	- Remove all redundant code  
>>>>>>> b81dc289be52d11d974110af7458fce3086dc041:may_2023/_things to try on kaggle.md
