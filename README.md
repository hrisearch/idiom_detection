# Understanding Idioms: Multilingual Idiomaticity Detection and Sentence Embedding

## Task A: Idiom Detection

Identify whether a sentence contains an idiom

### Data

Data is from SemEval2022: Task-2 and can be found [here](https://github.com/H-TayyarMadabushi/SemEval_2022_Task2-idiomaticity/tree/main/SubTaskA).

### Run

python main.py --model {cnn/rnn}

### Requirements

- python 3
- tensorflow 2

## Task B: Idiom Representation

Compute similarities between given pairs of sentences, one with idiom and the other with replacement

### Environment
The codes are all worked on Google Colaboratory with default settings.

### Data
Data is from SemEval2022: Task-2 and can be found [here](https://github.com/H-TayyarMadabushi/SemEval_2022_Task2-idiomaticity/tree/main/SubTaskB).

### Run
Subtask_B_1(Portuguese).ipynb is the baseline (Siamane with BiLSTM) for Portuguese.<br>
Subtask_B_2(Portuguese).ipynb is BERTimbau Base with BERTRAM for Portuguese.
