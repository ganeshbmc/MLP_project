import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

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



def select_features(df: pd.DataFrame, moviesdf: pd.DataFrame):
    # Drop duplicates from moviesdf
    movies_unique = moviesdf.drop_duplicates(subset=["movieid"])

    # Merge df and movies_unique
    df_merged = pd.merge(df, movies_unique, on="movieid")

    # Drop columns
    df_merged = df_merged.drop(columns=["title", "ratingContents", "releaseDateTheaters", "releaseDateStreaming", "boxOffice", "distributor", "soundType"])

    # Fill missing values in "reviewText", 'rating" column with empty string and "NA" respectively
    # Clean language names

    final = df_merged.copy()
    final["reviewText"] = final["reviewText"].fillna("")
    final["rating"] = final["rating"].fillna("NA")
    final["originalLanguage"].replace({"English (United Kingdom)": "English", 
                                            "English (Australia)" : "English",
                                            "French (France)": "French", 
                                            "French (Canada)": "French",
                                            "Portuguese (Brazil)": "Portuguese",
                                            "Spanish (Spain)": "Spanish"},                                         
                                            inplace=True)

    return final