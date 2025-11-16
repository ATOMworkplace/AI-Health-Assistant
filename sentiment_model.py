import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

class SentimentModel:
    def __init__(self):
        nltk.download("vader_lexicon", quiet=True)
        self.sia = SentimentIntensityAnalyzer()

    def get_sentiment(self, text):
        scores = self.sia.polarity_scores(text)
        compound = scores["compound"]

        if compound >= 0.34:
            return "positive"
        elif compound <= -0.34:
            return "negative"
        else:
            return "neutral"
