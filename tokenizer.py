import nltk
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import string
import neuralcoref

class Parser():
    def __init__(self):
        self.weights = {
            "CC": 0.05, #not important
            "CD": 0.1, 
            "DT": 0.05, #not important
            "EX": 0.1,
            "FW": 0.1,
            "IN": 0.05, #not important
            "JJ": 0.3,
            "JJR": 0.3,
            "JJS": 0.3,
            "LS": 0.05, #not important
            "MD": 0.1,
            "NN": 0.6,
            "NNS": 0.6,
            "NNP": 0.75,
            "NNPS": 0.75,
            "PDT": 0.1,
            "POS": 0.15,
            "PRP": 0.2,
            "PRP$": 0.2,
            "RB": 0.3,
            "RBR": 0.3,
            "RBS": 0.3,
            "RP": 0.1,
            "TO": 0.05, #not important
            "UH": 0.05, #not important
            "VB": 0.6,
            "VBD": 0.4,
            "VBG": 0.4,
            "VBN": 0.4,
            "VBP": 0.4,
            "VBZ": 0.4,
            "WDT": 0.4,
            "WP": 0.4,
            "WP$": 0.4,
            "WRB": 0.4,
            ".": 0.05,
            "''": 0.05
        }

        self.ques_weights = {
            "CC": 0.05, #not important
            "CD": 0.1, 
            "DT": 0.05, #not important
            "EX": 0.1,
            "FW": 0.1,
            "IN": 0.05, #not important
            "JJ": 0.3,
            "JJR": 0.3,
            "JJS": 0.3,
            "LS": 0.05, #not important
            "MD": 0.1,
            "NN": 0.6,
            "NNS": 0.6,
            "NNP": 0.75,
            "NNPS": 0.75,
            "PDT": 0.1,
            "POS": 0.15,
            "PRP": 0.2,
            "PRP$": 0.2,
            "RB": 0.3,
            "RBR": 0.3,
            "RBS": 0.3,
            "RP": 0.1,
            "TO": 0.05, #not important
            "UH": 0.05, #not important
            "VB": 0.6,
            "VBD": 0.5,
            "VBG": 0.5,
            "VBN": 0.5,
            "VBP": 0.5,
            "VBZ": 0.5,
            "WDT": 0.4,
            "WP": 0.4,
            "WP$": 0.4,
            "WRB": 0.4,
            ".": 0.05,
            "''": 0.05
        }
        self.nlp = en_core_web_sm.load()
    
    def tokenize(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        topTen = len(tokens)//1.5
        mappedVals = []
        # Extracts more important words from text based on manual weight
        # dictionary
        for tag in tagged:
            mappedVals.append((tag[0], self.weights[tag[1]]))
        mostImp = []
        while len(mostImp) < topTen:
            bestWord = None
            bestScore = 0
            for word in mappedVals:
                if word[1] > bestScore:
                    bestWord = word
                    bestScore = word[1]
            mostImp.append(bestWord)
            mappedVals.remove(bestWord)
        topWords = []
        for word in mostImp:
            topWords.append(word[0])         


        # Using SpaCy for named entity recognition
        ner = self.nlp(sentence)
        lemmas = [(x.orth_,x.pos_, x.lemma_) for x in [y 
                                      for y
                                      in ner 
                                      if not y.is_stop and y.pos_ != 'PUNCT']]


        print([(X.text, X.label_) for X in ner.ents])
        print(lemmas)
        print(tagged)
        print(topWords)

    def question_parser(self, question):
        tokens = nltk.word_tokenize(question)
        tagged = nltk.pos_tag(tokens)
        topTen = len(tokens)//1.5
        mappedVals = []
        # Extracts more important words from text based on manual weight
        # dictionary
        for tag in tagged:
            mappedVals.append((tag[0], self.ques_weights[tag[1]]))
        mostImp = []
        while len(mostImp) < topTen:
            bestWord = None
            bestScore = 0
            for word in mappedVals:
                if word[1] > bestScore:
                    bestWord = word
                    bestScore = word[1]
            mostImp.append(bestWord)
            mappedVals.remove(bestWord)
        topWords = []
        for word in mostImp:
            topWords.append(word[0])
        
        # Using SpaCy for named entity recognition
        ner = self.nlp(question)

        #print([(X.text, X.label_) for X in ner.ents])
        #print(tagged)
        #print(topWords)
        return topWords
    
    def preprocess_text(self, text):
        # Replaces text with lemmatized form when found
        sentences = []
        start = 0
        words = text.split()
        for i in range(len(words)):
            if words[i][-1] in string.punctuation and (i == len(words)-1 or words[i+1][0].isupper()):
                sentences.append(" ".join(words[start:i+1]))
                start = i + 1
        lemmatized_text = []
        for sentence in sentences:
            lemmatized_text.append(self.lemmatize(sentence))
        return ". ".join(lemmatized_text) + "."

    def lemmatize(self, sentence):
        lemmas = [(x.orth_,x.pos_, x.lemma_) for x in [y 
                                      for y
                                      in self.nlp(sentence) 
                                      if not y.is_stop and y.pos_ != 'PUNCT']]
        lemma_sent = []
        lemma_i = 0
        for word in sentence.split():
            if word[-1] in string.punctuation:
                word = word[:-1]
            if word[0] in string.punctuation:
                word = word[1:]
            if word != lemmas[lemma_i][0]:
                lemma_sent.append(word)
            else:
                lemma_sent.append(lemmas[lemma_i][2])
                lemma_i += 1
        return " ".join(lemma_sent)

    def extract_sentences(self, text, keywords):
        # Extracts sentences in text that contain the 75% of the keywords
        sentences = []
        start = 0
        process_text = self.preprocess_text(text)
        words = process_text.split()
        for i in range(len(words)):
            if words[i][-1] in string.punctuation and (i == len(words)-1 or words[i+1][0].isupper()):
                sentences.append(" ".join(words[start:i+1]))
                start = i + 1
        result = []
        for sentence in sentences:
            count = 0
            for word in keywords:
                if word in sentence: 
                    count += 1
            if count >= len(keywords)//(5/3):
                result.append(sentence)
        return result

    def get_possible_answers(self, text, question):
        keywords = self.question_parser(self.lemmatize(question))
        return self.extract_sentences(text, keywords)

    def pronoun_resolution(self, sentence):
        neuralcoref.add_to_pipe(self.nlp)
        doc1 = nlp(sentence)
        print(doc1._.coref_clusters)




parser = Parser()
#print(parser.tokenize("Bob went to Central Park on Wednesday."))
#print(parser.question_parser("Where did Bob go on Wednesday?"))
text = """The Old Kingdom is most commonly regarded as the 
period from the Third Dynasty through to the Sixth Dynasty (2686–2181 BC). 
The 4th-6th Dynasties of Egypt, are scarce and historians regard the history 
of the era as literally 'written in stone' and largely architectural in that it 
is through the monuments and their inscriptions that scholars have been able to 
construct a history."""

#print(parser.get_possible_answers(text, "Which era do historians call 'written in stone'?"))
print(parser.pronoun_resolution("Bob went to the park. He brought his dog."))

