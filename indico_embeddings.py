import indicoio
import os
import pickle

indicoio.config.api_key = os.environ['INDICO_API_KEY']

def get_batch_embeddings(comment_ids, nested_comments_dict):
    """call to indico text_features api
       comment_ids: List of comment_ids to fetch in this batch
       nested_comments_dict: Dictionary with comment_id as key, and
                            value is a dict containing 'body'
    """
    strings = [nested_comments_dict[id]['body'] for id in comment_ids]
    embeddings = indicoio.text_features(strings)
    return {comment_ids[x]: embeddings[x] for x in range(len(comment_ids))}


def get_all_embeddings(nested_comments_dict, batch_size=100):
    """Creates embeddings.pkl file containing embedding for each comment in nested_comments_dict"""

    comment_ids = list(nested_comments_dict)
    index_vals = range(0, len(nested_comments_dict), batch_size)
    output = {}
    with open('./work/indico_embeddings.pkl', 'wb') as f:
        for i in index_vals:
            try:
                output.update(get_batch_embeddings(comment_ids[i:i+batch_size], nested_comments_dict))
            except:
                print("Failed for index: %d"%i)
        pickle.dump(output, f)

if __name__ == "__main__":
    with open("./work/filtered_comments", "rb") as f:
        filtered_comments = pickle.load(f)
    get_all_embeddings(filtered_comments)
