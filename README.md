# EmoContext

In this project, we focus on solving EmoContext challenge proposed in [SemEval 2019](https://www.humanizing-ai.com/emocontext.html).

## Problem Statement

In this task, given an utterance of a dialogue and its two previous turns we try to predict the underlying emotion of the dialogue into four class labels as 
Happy, Sad, Angry and Others.

## Dataset Distribution

The dataset for the problem consists for 30159 instances of training data, 2755 and 5509 instances of test data each having an attribute characteristics of emoticon, emoji and text.
The distribution of each emotion is shown in the following Table:

| Emotion  | Happy | Sad  | Angry | Others | Total |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Train  | 4243  | 5463 | 5506 | 14948 | 30160 |
| Valid  | 142  | 125 | 150 | 2338 | 2755 |
| Test  | 284  | 250 | 298 | 4677 | 5509 |

## Model Proposed

Three sentences are encoded with the encoder separately. We try both stacked LSTM and stacked Bi-LSTM as the encoder. 
The final output are concatenated as the final representation for the given instance. 
We then pass the final representationthrough three dense layers and get the one-hot encoded output

### Architecture
![Image](https://github.com/appleternity/EmoContext/blob/master/Figures/emocontext.png)

Dataset contains informal language which affect the coverage tokens. So we have experimented three tokenization techniques [Word Tokenization](http://www.datascienceassn.org/sites/default/files/Natural%20Language%20Processing%20with%20Python.pdf), [BPE Tokenization](https://arxiv.org/pdf/1508.07909.pdf) and Character Tokenization.

Since the given data is relatively small, we try to add more similar data into the training process in order to get a better encoder. We used [Twitter sentiment 140 dataset](https://cs.stanford.edu/people/alecmgo/papers/TwitterDistantSupervision09.pdf) where each instance is annotated with 0 or 1 meaning negative and positive respectively.
Incorporated Joint Training as shown in the following diagram.

![Image](https://github.com/appleternity/EmoContext/blob/master/Figures/emocontext_joint_training.png)

We used the same encoder to encode the tweets and predict its corresponding sentiment label. Notice that to prevent the model from distracting too much, we also set up a weight 0.1 for the loss of joint training part.

Increase in digitalization of artwork illustrates the importance of
classification of paintings based on artists, styles and the genre of paintings. Classification methodologies will indeed help the visitors as well as the curators in analyzing and visualizing the paintings in any museum at their own pace. Moreover, finding the
artist of a painting is a difficult task because most artworks of an artist may have a exclusive painting style and multiple artists can have same styles of paintings.


## Installation and running the code

- Install Anaconda, then install dependencies listed in the ```dependencies.txt``` file.

- All the pickle files are generated after datapreprocessing and tokenization techniques when these files are called internally 

- Word Tokenization -> Run ```python data.py```
- Word Piece Tokenization using BERT Tokenizer -> Run ```python data_word_piece.py```
- Character Tokenization -> Run ```python emoji_embedding.py```

ConversationLSTM, ConversationBiLSTM, ConversationCNNLSTM models are defined in ```python model.py``` and run ```python train.py``` to train the corresponding model imported.

### Performance & Results

Character-based Bi-LSTM performs the best with F1 score of 0.735. However, BPE-based Bi-LSTM performs surprisingly poor though its F1 score on training set achieves around 0.90. All of the proposed model does not perform better than the baseline. Nevertheless, we show that without using fancy models like ensemble model with BERT and USE, it is still possible to achieve a reasonable score.


- Results of the proposed models

| Model  | Precision | Recall | F1 Score |
| ------------- | ------------- | ------------- | ------------- |
| Bi-LSTM(word)  | 0.818  | 0.506  | 0.625  |
| Bi-LSTM(BPE)  | 0.215  | 0.297  | 0.250  |
| Bi-LSTM(char)  | 0.752 | 0.719  | **0.735**  |
| Bi-LSTM(char) + Twitter  | 0.775 | 0.657  | 0.698  |



## References
1. [Natural language processing with Python: analyzing text with the natural language toolkit](http://www.datascienceassn.org/sites/default/files/Natural%20Language%20Processing%20with%20Python.pdf)
2. [Neural machine translation of rare words with subword units](https://arxiv.org/pdf/1508.07909.pdf)
3. [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf)
4. [SemEval-2019 Task 3: EmoContext Contextual Emotion Detection in Text](https://pdfs.semanticscholar.org/675b/b798f0cf542c0e10687c39482a8ff7e3318a.pdf?_ga=2.156466940.61356801.1592779147-1953199154.1573935763)
