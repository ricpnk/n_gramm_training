import datasets
from sklearn.model_selection import train_test_split
from collections import defaultdict
import math
import numpy as np

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
    print("Unigram Model:")
    uni_model = estimate_unigram(train_data)
    
    #todo test logp of test sentences
    test_sentences = ["the the the <stop>", "ilove computer science <stop>"]
    for sentence in test_sentences:
        sen_logp = unigram_sentence_logp(sentence, uni_model)
        print(f"logprob of '{sentence}': {sen_logp}")

    #todo calculate the perplexity on test_data
    test_perplexity = unigram_perplexity(test_data, uni_model)
    print(f"Perplexity of Testset: {test_perplexity}")

    #todo use rare_words function to clean the datasets
    threshold = 5
    clean_train_data = remove_rares(train_data, threshold)
    clean_test_data = remove_rares(test_data, threshold)

    #todo calculate the new perplexity with clean data
    clean_uni_model = estimate_unigram(clean_train_data)
    test_perplexity = unigram_perplexity(clean_test_data, clean_uni_model)
    print(f"Perplexity of Clean-Testset: {test_perplexity}")

    #todo get the bigram and unigram model of a dataset
    print("\nBigram Model:")
    uni_counts, bi_counts, bi_model = estimate_bigram(clean_train_data)
    for sentence in test_sentences:
        bi_logp = bigram_sentence_logp(sentence, bi_model)
        print(f"logprob for bigram of '{sentence}': {bi_logp}")

    #todo calculate the perplexity of the bigram model
    inf_counts, bi_perplexity = bigram_perplexity(test_data, bi_model)
    print(f"Perplexity of Bigram-model with Testset: {bi_perplexity}")
    print(f"The Bigram-model has: {inf_counts} sentences with a probability of zero")

    #todo get the smoothed bigram model
    smoothed_bigram_model = estimate_bigram_smoothed(uni_counts, bi_counts)

    #todo calculate the perplexity of the smoothed bigram model
    smoothed_inf_counts, smoothed_bi_perplexity = bigram_perplexity(test_data, smoothed_bigram_model)
    print(f"Perplexity of Smoothed-Bigram-model with Testset: {smoothed_bi_perplexity}")
    print(f"The Smoothed-Bigram-model has: {smoothed_inf_counts} sentences with a probability of zero")






#! Functions
def load_dataset():
    return datasets.load_dataset("ptb_text_only", split="train", trust_remote_code=True)


def add_stop(data):
    sentence = data["sentence"]
    return {"sentence": sentence + " <stop>"}


def estimate_unigram(data):
    """
    input: trainingsset
    return: dict of the unigram model

    use defaultdict
    """
    uni_dict = count_words(data)

    #calculate the probability
    total_dict = sum(uni_dict.values())
    for word in uni_dict:
        uni_dict[word] /= total_dict

    return uni_dict


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
            return np.inf
    return sum_logp


def unigram_perplexity(data, uni_model):
    """
    input: dataset and unigram_model
        compute perplexity of model
        skip sentence if 'np.inf' 
    return: perplexity
    """
    total_words = 0
    total_log_prob = 0
    for sentence in data:
        log_prob = unigram_sentence_logp(sentence, uni_model)

        if log_prob == np.inf:
            continue
        else:
            # count the totals
            total_words += len(sentence.split())
            total_log_prob += log_prob

    
    #calculate the perplexity 
    if total_words == 0:
        return float("inf")
    cross_entropy = -(total_log_prob / total_words)
    perplexity = 2**cross_entropy
    
    return perplexity


def remove_rares(data, threshold):
    """
    input: dataset, threshold value
    calculate the frequency
    remove words and set to <unk>
    return: clean dataset
    """
    frequency_dict = count_words(data)

    #efficient way to create the list with set()
    rare_words = set(word for word, count in frequency_dict.items() if count <= threshold)

    cleaned_data = []
    for sentence in data:
        cleaned_words = []
        for word in sentence.split():
            if word not in rare_words:
                cleaned_words.append(word)
            else:
                cleaned_words.append("<unk>")
        cleaned_data.append(" ".join(cleaned_words))

    return cleaned_data


def estimate_bigram(data):
    """
    input: dataset
    output: bigram probability model (not raw counts)
    """
    uni_counts = count_words(data)
    bi_counts = defaultdict(lambda: defaultdict(int))

    for sentence in data:
        words = sentence.split()
        for w1, w2 in zip(words[:-1], words[1:]):
            bi_counts[w1][w2] += 1

    bigram_probs = defaultdict(dict)
    for w1 in bi_counts:
        for w2 in bi_counts[w1]:
            bigram_probs[w1][w2] = bi_counts[w1][w2] / uni_counts[w1]

    return uni_counts, bi_counts, bigram_probs


def bigram_sentence_logp(sentence, bi_model):
    """
    input: sentence, bigram probability model
    output: log_prob of bigram
    """
    sum_logp = 0
    words = sentence.split()

    for word1, word2 in zip(words[:-1], words[1:]):
        prob = bi_model.get(word1, {}).get(word2, 0)
        if prob == 0:
            return np.inf
        sum_logp += math.log2(prob)

    return sum_logp


def estimate_bigram_smoothed(uni_counts, bi_counts, alpha=0.1):
    """
    input: dataset, scalar alpha
    consider memory efficiency
    return: alpha smoothed bigram model
    """

    # Use the vocabulary from the unigram counts
    vocab = set(uni_counts.keys())
    vocab_size = len(vocab)
    
    smoothed_bigram = defaultdict(dict)
    for w1 in uni_counts:
        for w2 in vocab:
            if w2 in bi_counts[w1]:
                count_w1_w2 = bi_counts[w1][w2] 
            else: 
                count_w1_w2 = 0

            smoothed_bigram[w1][w2] = (count_w1_w2 + alpha) / (uni_counts[w1] + alpha * vocab_size)
    
    return smoothed_bigram



def bigram_perplexity(data, bi_model):
    """
    input: dataset and bigram_model
        compute perplexity of model
        skip sentence if 'np.inf' 
    return: perplexity
    """
    total_words = 0
    total_log_prob = 0
    inf_counts = 0
    for sentence in data:
        log_prob = bigram_sentence_logp(sentence, bi_model)

        if log_prob == np.inf:
            inf_counts += 1
            continue
        else:
            # count the totals
            total_words += len(sentence.split())
            total_log_prob += log_prob

    
    #calculate the perplexity 
    if total_words == 0:
        return float("inf")
    cross_entropy = -(total_log_prob / total_words)
    perplexity = 2**cross_entropy
    
    return inf_counts, perplexity


def count_words(data):
    dict = defaultdict(int)
    for sentence in data:
        for word in sentence.split():
            dict[word] += 1
    return dict







if __name__ == "__main__":
    main()