"""This file is not used in creating the chatbot. It's just experiments with Indico's API"""

import indicoio
import numpy as np
import os
# from scipy.spatial.distance import cosine

indicoio.config.api_key = os.environ['INDICO_API_KEY']

def string_corr_coefs(list_of_strings):
    vecs = np.array([(vec) for vec in indicoio.text_features(list_of_strings)])
    return(np.corrcoef(vecs))


name_strings = ["How does it handle names? Let's look at Peyton Manning.",
                "How does it handle names? Let's look at Tom Brady.",
                "How does it handle names? Let's look at Barack Obama.",
                "How does it handle names? Let's look at Hillary Clinton.",
                "Who is the quarterback of the Denver Broncos"]

url_strings = ["Test to distinguish URLS, like https://google.com",
               "Test to distinguish URLS, like https://hotmal.com",
               "Test to distinguish URLs, like https://imgur.com/k293fxc",
               "Test to distinguish URLs, like https://imgur.com/kgg3xc",
               "Test to distinguish URLs, like https://imgur.com/k293fxc.html",
               "Test to distinguish URLs, like https://imgur.com/kgg3xc.html"]


string_corr_coefs(name_strings)
string_corr_coefs(url_strings)
