import pickle
import pandas as pd
import random

def make_pairs(n_fake_pairs, child_score_thresh, nested_comments_dict, embeddings):
    def single_pair_data(child_id, parent_id, pair_is_true):
        pair_data = [pair_is_true]
        pair_data.append(nested_comments_dict[child_id]['score'])
        pair_data.extend(embeddings[child_id])
        pair_data.extend(embeddings[parent_id])
        return pair_data

    embedding_length = len(next(iter(embeddings.values())))
    good_comment_ids = [x for x in nested_comments_dict
                        if nested_comments_dict[x]['score'] > child_score_thresh]
    valid_parent_ids = [x['short_parent_id'] for x in nested_comments_dict.values()]
    data_accum = []
    for child_id in good_comment_ids:
        parent_id = nested_comments_dict[child_id]['short_parent_id']
        data_accum.append(single_pair_data(child_id, parent_id, True))
    for i in range(n_fake_pairs):
        child_id = random.choice(good_comment_ids)
        parent_id = random.choice(valid_parent_ids)
        data_accum.append(single_pair_data(child_id, parent_id, False))
    output = pd.DataFrame(data_accum)
    output.rename(columns={0: "true_pair", 1: "score"}, inplace=True)
    for i in range(embedding_length):
        output.rename(columns={2+i:"post_feat_"+str(i)}, inplace=True)
        output.rename(columns={2+embedding_length+i:"par_feat_"+str(i)}, inplace=True)
    return output

if __name__ == "__main__":
    with open('./work/filtered_comments.pkl', 'rb') as f:
        nested_comments_dict = pickle.load(f)
    with open('./work/indico_embeddings.pkl', 'rb') as f:
        indico_embeddings = pickle.load(f)
    with open('./work/tf_idf_embeddings.pkl', 'rb') as f:
        tf_idf_embeddings = pickle.load(f)

    indico_ids = indico_embeddings.keys()
    indico_comments = {x: y for x,y in nested_comments_dict.items()
                        if x in indico_ids and y['short_parent_id'] in indico_ids}
    indico_pairs = make_pairs(75000, 25, indico_comments, indico_embeddings)
    indico_pairs.to_csv('work/indico_pairs.csv', index=False)

    tf_idf_ids = tf_idf_embeddings.keys()
    tf_idf_comments = {x: y for x,y in nested_comments_dict.items()
                        if x in tf_idf_ids and y['short_parent_id'] in tf_idf_ids}
    tf_idf_dframe = tf_idf_pairs = make_pairs(75000, 25, tf_idf_comments, tf_idf_embeddings)
    tf_idf_dframe.to_csv('work/tf_idf_pairs.csv', index=False)
