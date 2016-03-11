import indicoio
from keras.models import Sequential
from model import get_model, prep_X
import pickle
import numpy as np

EMBEDDING_SIZE = 300 #indico embedding has 300 elements

def get_embedding(text):
    return np.array(indicoio.text_features(my_text_1))

class Responder(object):
    def load_comments_data(self):
        with open('work/indico_embeddings.pkl', 'rb') as f:
            self.indico_embeddings = pickle.load(f)
        with open('work/filtered_comments.pkl', 'rb') as f:
            self.comments = pickle.load(f)
    def id_to_text(self, response_id):
        return self.comments[response_id]['body']

class CorrelationResponder(Responder):
    def __init__(self):
        self.load_comments_data()
        self.all_candidate_embeddings = np.array([self.indico_embeddings[candidate_id]
                                                  for candidate_id in self.indico_embeddings.keys()])
    def respond(self, user_embedding):
        dot_product_score = self.all_candidate_embeddings.dot(user_embedding)
        response_id = list(self.indico_embeddings.keys())[dot_product_score.argmax()]
        return self.id_to_text(response_id)


class NNResponder(Responder):
    def __init__(self, candidate_reply_thresh = 10):
        self.model = get_model(EMBEDDING_SIZE)
        self.model.load_weights('work/relevance_model.h5')
        self.load_comments_data()
        self.candidate_reply_thresh = candidate_reply_thresh
        self.filter_candidate_responses()

    def filter_candidate_responses(self):
        self.comment_ids = set((id for id in self.comments if self.comments[id]['score'] > self.candidate_reply_thresh))
        self.comment_ids = list(self.comment_ids.intersection(self.indico_embeddings.keys()))
        self.n_comments = len(self.comment_ids)
        self.candidate_embeddings = np.array([self.indico_embeddings[candidate_id] for candidate_id in self.comment_ids])

    def respond(self, user_embedding):
        my_embedding_arr = np.tile(user_embedding, self.n_comments).reshape([self.n_comments, EMBEDDING_SIZE])
        to_eval = np.hstack([self.candidate_embeddings, my_embedding_arr])
        _, test_X = prep_X(to_eval)
        nn_preds = self.model.predict(test_X)[:,1]
        response_id = self.comment_ids[nn_preds.argmax()]
        return self.id_to_text(response_id)

corr_responder = CorrelationResponder()
nn_responder = NNResponder()

my_text_1 = "People have always felt they were in the middle of the most politically divise time ever. It's no surprise we feel that way too"
user_embedding = get_embedding(my_text_1)
print(corr_responder.respond(user_embedding))
print("------------------------------------")
print(nn_responder.respond(user_embedding))
