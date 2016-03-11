# Dan's Naive/Simple Chatbot

This was a weekend project to see how quickly I could put together a very simple chatbot. Unlike more sophisticated chatbots, this projects uses no generative models.

Instead, it uses a neural network to discriminate relevant responses from irrelevant responses. When a user inputs text, it uses the model to score previous comments on reddit for relevance to the user input, and responds with the most relevant reddit comment.

Both the training data and candidate responses are Reddit comments from early 2015.

As you can see from the sample response below, this chatbot isn't great... but it was a fun and quick hack.

---
**User input:** <br>
*I hope this responds like Shaquille O'Neal would. Shaq wasn't just a great player.  He's a funny personality.*

**Chatbot response 1:**<br>
*Sure, he sucks as a basketball player, but he isn't a bad guy. Hope everything is alright with him and his family.*

**Chatbot response 2:**
*That injury has just killed him, he was so exciting to watch play his rookie season.  I think he could have been the most dynamic QB to ever play the position until age would've caught up.  Now he looks like a guy that age has already caught up with him.*

---
The pipeline to create the chatbot is as follows:

* preprocess.py: Pare down the raw data (approx 30GB) to a more manageable 248MB file containing only the data used for model training.  This process removes unnecessary fields, and it restricts data to comments with score>2 and length>100.

* indico_embeddings.py: Use the text features API from the wonderful people at indicio.io to represent each comment as a 300-dimensional vector. This API uses a pretrained network to create the vector representations.  

 Incidentally, I also created representations with tf_idf_embeddings, but the embeddings from Indico were better (i.e. subsequent models are more accurate when using the Indico embeddings). Another win for transfer learning.

* make_pairs.py: Creates both true pairs of parent-comment to child-comment as well as mismatched parent-child pairs.

* model.py: Creates neural network model to distinguish the real parent-child pairs from the mismatched pairs.

* use_model.py: Gets the vector representation of that user input, and scores candidate responses for relevance to that input, returning the most relevant response.

The result of everything up to the modeling step is publicly available on S3 [here](https://s3-us-west-1.amazonaws.com/chatbot-project/). This file is ready to be plugged into model.py if you want to play with the modeling step.
