"""Filters reddit comments by length and score. Stores parts of selected comments

Each line of input file is a dictionary"""

import pickle
import json

def first_filter_fn(comment):
    return comment['score'] > 2 and len(comment['body']) > 100

def filter_thread_endpoints(comment_gen_object):
    """Removes comments that don't have responses, or whose parent isn't in dataset"""
    # Create dictionary where comment_id indexes all other relevant info
    fields_to_keep = ['score', 'ups', 'parent_id', 'link_id', 'body']
    as_nested_dict = {comment['id']:
                                    {key: comment[key] for key in fields_to_keep}
                                    for comment in comment_gen_object}
    for key in as_nested_dict:
        as_nested_dict[key]['short_parent_id'] = as_nested_dict[key]['parent_id'][3:]
    comment_ids = set(as_nested_dict.keys())
    parent_comment_ids = set((as_nested_dict[x]['short_parent_id'] for x in as_nested_dict))
    parents_in_dataset = comment_ids.intersection(parent_comment_ids)
    output = {id: as_nested_dict[id] for id in parents_in_dataset
                if as_nested_dict[id]['short_parent_id'] in parents_in_dataset}
    return output

def filter_and_shrink_comments(input_path, output_path, filter_fn):
    with open(input_path, 'r') as input_file:
        all_comments = (json.loads(comment) for comment in input_file)
        good_comments = filter(filter_fn, all_comments)
        output = filter_thread_endpoints(good_comments)
        with open(output_path, 'wb') as out_file:
            pickle.dump(output, out_file)

if __name__ == "__main__":
    filter_and_shrink_comments(input_path="./raw/RC_2015-01",
                               output_path="./work/filtered_comments_2.pkl",
                               filter_fn=first_filter_fn)
