import json, random, pickle, re
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
import nltk

nltk.data.path.append("./nltk_data")

def simple_tokenize(text):
    return re.findall(r'\w+', text.lower())

def simple_lemmatize(word):
    return word[:-1] if word.endswith('s') else word

def load_intents(path="intents.json"):
    with open(path, "r") as f:
        return json.load(f)

def prepare_data(intents):
    words, classes, documents = [], [], []

    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            tokens = simple_tokenize(pattern)
            words.extend(tokens)
            documents.append((tokens, intent["tag"]))
            if intent["tag"] not in classes:
                classes.append(intent["tag"])

    words = sorted(list(set([simple_lemmatize(w) for w in words])))
    classes = sorted(list(set(classes)))

    training = []
    output_empty = [0] * len(classes)

    for tokens, tag in documents:
        bag = [1 if w in tokens else 0 for w in words]
        output_row = list(output_empty)
        output_row[classes.index(tag)] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])
    return words, classes, train_x, train_y

def build_model(input_len, output_len):
    model = Sequential([
        Dense(128, input_shape=(input_len,), activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.5),
        Dense(output_len, activation="softmax")
    ])

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])
    return model

def main():
    intents = load_intents()
    words, classes, train_x, train_y = prepare_data(intents)

    model = build_model(len(train_x[0]), len(train_y[0]))
    model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5)

    model.save("model.h5")
    pickle.dump(words, open("words.pkl", "wb"))
    pickle.dump(classes, open("classes.pkl", "wb"))

    print("Training complete. Saved: model.h5, words.pkl, classes.pkl")

if __name__ == "__main__":
    main()
