import datasets
from sklearn.model_selection import train_test_split
from collections import defaultdict
import math

def main():

    #todo load the dataset
    data = load_dataset()

    #todo add <stop> and filter all short sentences
    data = data.map(add_stop)
    data = data.filter(lambda x: len(x["sentence"].split()) >= 3)

    #todo train test split
    sentences = data["sentence"]
    train_data, test_data = train_test_split(sentences, test_size=0.2, random_state=42)

    #todo create the unigram model 
    uni_model = estimate_unigram(train_data)
    
    #todo test logp of test sentences
    test_sentences = ["the the the <stop>", "ilove computer science <stop>"]
    for sentence in test_sentences:
        sen_logp = unigram_sentence_logp(sentence, uni_model)
        print(f"logprob of '{sentence}': {sen_logp}")
    







#! Functions
def load_dataset():
    return datasets.load_dataset("ptb_text_only", split="train", trust_remote_code=True)


def add_stop(data):
    sentence = data["sentence"]
    return {"sentence": sentence + " <stop>"}


def estimate_unigram(sentences):
    """
    input: trainingsset
    return: dict of the unigram model

    use defaultdict
    """
    return_dict = defaultdict(int)
    for sentence in sentences:
        for word in sentence.split():
            return_dict[word] += 1

    total_dict = sum(return_dict.values())
    for word in return_dict:
        return_dict[word] /= total_dict

    return return_dict


def unigram_sentence_logp(sentence, uni_model):
    """
    input: sentence and unigram model
    return log-prob of sentence
    """
    sum_logp = 0
    for word in sentence.split():
        if word in uni_model:
            sum_logp += math.log2(uni_model[word])
        else:
            sum_logp += math.log2(uni_model["<unk>"])
    return sum_logp


def unigram_perplexity():
    
    pass









def estimate_bigram():
    pass

def bigram_sentence_logp():
    pass











if __name__ == "__main__":
    main()