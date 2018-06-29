"""
Main module to perform topic analysis on QA data.
"""
import os
import sys
import pandas as pd
sys.path.append('../')
from textsim import text_analysis as ts

def load_amazon_qa_data(data_location):
    product_database = os.path.join(data_location, "product_data.csv")
    qa_database = os.path.join(data_location, "qa_data.csv")
    if not os.path.isfile(product_database):
        raise(IOError, "Database file %s could not be found." % product_database)
    if not os.path.isfile(qa_database):
        raise(IOError, "Database file %s could not be found." % qa_database)

    db_product = pd.read_csv(product_database)
    db_qa = pd.read_csv(qa_database)
    return db_product, db_qa


if __name__ == "__main__":
    data_storage = os.path.abspath("../tests/test_data/")
    db_product, db_qa = load_amazon_qa_data(data_storage)

    test_key = "B00004W4UJ"
    description = db_product[db_product.asin == test_key].description.values[0]
    qa_pairs = zip(db_qa[db_qa.asin == test_key].question.values, db_qa[db_qa.asin == test_key].answer.values)
    qa_list = [i + " " + j for i, j in qa_pairs]

    text = ts.Text(qa_list)
    text.build_vocabulary()
    text.build_tfidf_model()
    text.rank_keyphrases()

    qac_score, doc_scores = text.evaluate_query_document(description)
    print(qac_score)