import pickle
from urllib import request
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

opener = request.URLopener()
myurl = "https://s3-us-west-1.amazonaws.com/chatbot-project/filtered_comments"
with opener.open(myurl) as f:
    nested_comments_dict = pickle.load(f)

comment_body = [comment['body'] for comment in nested_comments_dict.values()]
comment_id_list = list(nested_comments_dict)

my_pipeline = Pipeline(steps = [('tf_idf', TfidfVectorizer(decode_error='replace', max_features=10000)),
                                ('svd', TruncatedSVD(n_components = 50, random_state=80401))])
sample_embeddings = my_pipeline.fit_transform(comment_body)
output = {comment_id_list[i]: sample_embeddings[i]
          for i in range(len(nested_comments_dict))}

with open('./work/tf_idf_embeddings.pkl', 'wb') as f:
    pickle.dump(output, f)
