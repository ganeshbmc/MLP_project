import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

def load_csv(filename: str):
    df = pd.read_csv(f"data/{filename}.csv")
    return df

def preprocess_n_train(X_train, y_train, preprocessor, model):
    pipe = Pipeline(steps=[
                            ("preprocessor", preprocessor),
                            ("model", model)
                        ])
    
    pipe.fit(X_train, y_train)

    print("Given model has been trained. Use predict method to get predictions array.")
    return pipe


def predict_n_evaluate(pipeline, X_test, y_test):
    y_pred = pipeline.predict(X_test)
    print(f"y_pred shape: {y_pred.shape}")
    print(f"Summary of predictions: {np.unique(y_pred, return_counts=True)}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return y_pred

def evaluate(y_test, y_pred):
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    # print(f1_score(y_test, y_pred))
    # ConfusionMatrixDisplay(y_test, y_pred)
    return


def get_tfidf_vocab(traindf, 
                    ngram_range=(1,2), 
                    min_df=0.0001, 
                    max_df=0.8, 
                    stop_words='english', 
                    max_features=10000):
    """
    Function for Weighted Class TF-IDF  
    Returns a list of words that are most important for the given dataset. 

    [Link](https://www.deepwizai.com/projects/how-to-correctly-use-tf-idf-with-imbalanced-data)  
    """
    text = traindf["reviewText"]
    pos_text = traindf[traindf["sentiment"] == "POSITIVE"]["reviewText"]
    neg_text = traindf[traindf["sentiment"] == "NEGATIVE"]["reviewText"]

    n_pos_features = round((len(pos_text) / len(text)) * max_features)
    n_neg_features = round((len(neg_text) / len(text)) * max_features)

    tvec_pos = TfidfVectorizer(stop_words=stop_words, 
                                max_features=n_pos_features, 
                                ngram_range=ngram_range, 
                                min_df=min_df, 
                                max_df=max_df)
    tvec_pos.fit(pos_text)
    pos_vocab = tvec_pos.vocabulary_

    tvec_neg = TfidfVectorizer(stop_words=pos_vocab, 
                                max_features=n_neg_features, 
                                ngram_range=ngram_range, 
                                min_df=min_df, 
                                max_df=max_df)
    tvec_neg.fit(neg_text)
    neg_vocab = tvec_neg.vocabulary_

    vocab_combined_dict = pos_vocab | neg_vocab
    vocab_combined = list(vocab_combined_dict.keys())
    return vocab_combined


def get_countvec_vocab(traindf, 
                    ngram_range=(1,2), 
                    min_df=0.0001, 
                    max_df=0.8, 
                    stop_words='english', 
                    max_features=10000):
    """
    Function for Weighted Class TF-IDF  
    Returns a list of words that are most important for the given dataset. 

    [Link](https://www.deepwizai.com/projects/how-to-correctly-use-tf-idf-with-imbalanced-data)  
    """
    text = traindf["reviewText"]
    pos_text = traindf[traindf["sentiment"] == "POSITIVE"]["reviewText"]
    neg_text = traindf[traindf["sentiment"] == "NEGATIVE"]["reviewText"]

    n_pos_features = round((len(pos_text) / len(text)) * max_features)
    n_neg_features = round((len(neg_text) / len(text)) * max_features)

    cvec_pos = CountVectorizer(stop_words=stop_words, 
                                max_features=n_pos_features, 
                                ngram_range=ngram_range, 
                                min_df=min_df, 
                                max_df=max_df)
    cvec_pos.fit(pos_text)
    pos_vocab = cvec_pos.vocabulary_

    cvec_neg = CountVectorizer(stop_words=pos_vocab, 
                                max_features=n_neg_features, 
                                ngram_range=ngram_range, 
                                min_df=min_df, 
                                max_df=max_df)
    cvec_neg.fit(neg_text)
    neg_vocab = cvec_neg.vocabulary_

    vocab_combined_dict = pos_vocab | neg_vocab
    vocab_combined = list(vocab_combined_dict.keys())
    return vocab_combined


# SELECT FEATURES FUNCTION
def select_features(df: pd.DataFrame, moviesdf: pd.DataFrame):
    '''
    This function merges the given dataframes. Note that the first df must be "train" or "test" and
    the second df should be "movies".
    Note: Sentiment column is present only in "train.csv" file and not "test.csv" file.
    '''
    
    # Drop duplicates from moviesdf
#     movies_unique = moviesdf.drop_duplicates(subset=["movieid"])
    # Drop duplicates using groupby - clubs similar rows and fills in missing values better
    movies_unique = moviesdf.fillna(value=np.nan).groupby("movieid").first()

    # Merge df and movies_unique
    df_merged = pd.merge(df, movies_unique, on="movieid", how='left')
    
    # Rename "isTopCritic" column, if it exists, to "isFrequentReviewer"
    df_merged.rename(columns={"isTopCritic": "isFrequentReviewer"}, inplace=True)
    
    # Drop columns
    df_merged = df_merged.drop(columns=["title", "ratingContents", "releaseDateTheaters", "releaseDateStreaming", "distributor", "soundType"])

    # Fill missing values in "reviewText" with empty string
    final = df_merged.copy()
    final["reviewYN"] = np.where(final["reviewText"].isnull(), 1, 0)    # Feature engineering - adding a new column
    final["reviewWC"] = final.apply(lambda x: len(str(x["reviewText"]).split()), axis=1)    # Feature engineering - adding second new column
    final["reviewText"] = final["reviewText"].fillna("neutral")
    
    # Fill missing values in "rating", "genre", original columns with the word "Unknown"
    final["rating"] = final["rating"].fillna("Unknown")
    final["genre"] = final["genre"].fillna("Unknown")
    final["originalLanguage"] = final["originalLanguage"].fillna("Unknown")

    # Impute missing values for "audienceScore" and "runtimeMinutes" columns
    final["audienceScore"] = final["audienceScore"].fillna(final["audienceScore"].mean())
    final["runtimeMinutes"] = final["runtimeMinutes"].fillna(final["runtimeMinutes"].median())
    
    # Preprocess and impute missing values in "boxOffice" column
    final["boxOffice"] = final["boxOffice"].str[1:]
    final["boxOffice"] = final["boxOffice"].replace(to_replace={"M": "*1000000", "K": "*1000"}, regex=True)
    final["boxOffice"] = final["boxOffice"].loc[final["boxOffice"].notnull()].apply(lambda x: eval(str(x)))
    final["boxOffice"] = final["boxOffice"].fillna(final["boxOffice"].median())
    # (Optional) Replace outliers in boxOffice with median
    median = final["boxOffice"].describe()['50%']
    iqr = final["boxOffice"].describe()['75%'] - final["boxOffice"].describe()['25%']
    ll = median - (1.5*iqr)
    ul = median + (1.5*iqr)
    final.loc[final["boxOffice"] > ul, "boxOffice"] = median
    
    # Clean language names
    final["originalLanguage"].replace({"English (United Kingdom)": "English", 
                                            "English (Australia)" : "English",
                                            "French (France)": "French", 
                                            "French (Canada)": "French",
                                            "Portuguese (Brazil)": "Portuguese",
                                            "Spanish (Spain)": "Spanish"},                                         
                                            inplace=True)
    
    # Clean reviewerName column
    pre_post_fixes = {"Mr. ": "", "Mrs. ": "", "Ms. ": "", "Dr. ": "", 
                      " MD": "", " DDS": "", " DVM": "", " Jr.": "", " PhD": "", " II": "", " IV": ""}
    final["reviewerName"] = final["reviewerName"].replace(pre_post_fixes, regex=True)
    final["reviewerName"] = final["reviewerName"].apply(name_fl)

    return final


def inspect(df: pd.DataFrame):
    print(f"Shape of the dataframe: {df.shape}")
    print()
    print(f"Columns in the dataframe:\n{df.columns}")
    print()
    print(f"{df.info()}")
    print()
    # print(f"Summary: {df.describe()}")
    print(f"Missing values:\n{df.isna().sum()}")
    return


def split_train_predict(features, labels, pipeline, test_size=0.25, random_state=42):
    # cols = features.columns
    if len(features.shape) == 1:
        features = features.to_numpy().reshape(-1, 1)  # reshape to 2D array
    features = pd.DataFrame(features)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    return #y_pred

def get_preprocessing_pipelines():
    num_pipe = Pipeline(steps=[
                            ("imputer", SimpleImputer(strategy="mean", missing_values=np.nan)),
                            ("scaler", MinMaxScaler())
                        ])
    
    cat_pipe = Pipeline(steps=[
                            ("ohe", OneHotEncoder(handle_unknown="ignore"))
                        ])
    
    txt_pipe = Pipeline(steps=[
                            ("tvec", TfidfVectorizer())
                        ])
    return num_pipe, cat_pipe, txt_pipe



# OLDER VERSIONS OF SOME FUNCTIONS

# def select_features(df: pd.DataFrame, moviesdf: pd.DataFrame):
#     # Drop duplicates from moviesdf
#     movies_unique = moviesdf.drop_duplicates(subset=["movieid"])

#     # Merge df and movies_unique
#     df_merged = pd.merge(df, movies_unique, on="movieid")

#     # Drop columns
#     df_merged = df_merged.drop(columns=["title", "ratingContents", "releaseDateTheaters", "releaseDateStreaming", "distributor", "soundType"])

#     # Fill missing values in "reviewText", 'rating" column with empty string and "NA" respectively
#     # Clean language names

#     final = df_merged.copy()
#     final["reviewText"] = final["reviewText"].fillna("")
#     final["rating"] = final["rating"].fillna("NA")
#     final["originalLanguage"].replace({"English (United Kingdom)": "English", 
#                                             "English (Australia)" : "English",
#                                             "French (France)": "French", 
#                                             "French (Canada)": "French",
#                                             "Portuguese (Brazil)": "Portuguese",
#                                             "Spanish (Spain)": "Spanish"},                                         
#                                             inplace=True)

#     return final