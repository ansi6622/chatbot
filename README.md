# Simple Chatbot

This was a weekend project, and it's less sophisticated/complex than some other chatbots.

For example, this chatbot does not use any generative model. Instead, a model chooses the most relevant response to user input from a bank of 500,000 Reddit comments.

It would be exceptionally straightforward to improve the chatbot by downloading more candidate responses from Reddit, Twitter, or any other source of human-written text.

I'm not planning to do anything more with it.
---
**User input:** <br>
*I hope this responds like Shaquille O'Neal would. Shaq wasn't just a great player.  He's a funny personality.*

**Chatbot response 1:**<br>
*Sure, he sucks as a basketball player, but he isn't a bad guy. Hope everything is alright with him and his family.*

**Chatbot response 2:**<br>
*That injury has just killed him, he was so exciting to watch play his rookie season.  I think he could have been the most dynamic QB to ever play the position until age would've caught up.  Now he looks like a guy that age has already caught up with him.*

---
The pipeline to create the chatbot is:

* __preprocess.py__: Pare down the raw data (approx 30GB) to a more manageable 248MB file containing only the data used for model training.  This process removes unnecessary fields, and it restricts data to comments with score>2 and length>100.

* __indico_embeddings.py__: Use the text features API from the wonderful people at indicio.io to represent each comment as a 300-dimensional vector. This API uses a pretrained network to create the vector representations.  

 Incidentally, I also created representations with tf_idf_embeddings, but the embeddings from Indico were better (i.e. subsequent models are more accurate when using the Indico embeddings). Another win for transfer learning.

* __make_pairs.py__: Creates both true pairs of parent-comment to child-comment as well as mismatched parent-child pairs.

* __model.py__: Creates neural network model to distinguish the real parent-child pairs from the mismatched pairs.

* __use_model.py__: Gets the vector representation of that user input, and scores candidate responses for relevance to that input, returning the most relevant response.

The result of everything up to the modeling step is publicly available on S3 [here](https://s3-us-west-1.amazonaws.com/chatbot-project/). This file is ready to be plugged into model.py if you want to play with the modeling step.
