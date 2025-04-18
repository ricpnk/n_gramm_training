import datasets
from sklearn.model_selection import train_test_split

def main():

    #todo load the dataset
    data = load_dataset()


    #todo train test split
    sentences = data["sentence"]
    train_data, test_data = train_test_split(sentences, test_size=0.2, random_state=42)


    #todo add the stop to each sentence
    train_data = train_data.map(add_stop)
    test_data = test_data.map(add_stop)








def load_dataset():
    return datasets.load_dataset("ptb_text_only", split="train", trust_remote_code=True)

def add_stop(data):
    sentence = data["sentence"]
    return {"sentence": sentence + " <stop>"}






if __name__ == "__main__":
    main()