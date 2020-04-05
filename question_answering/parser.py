import nltk
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import string
import math
#import neuralcoref
from nltk.corpus import wordnet

class Sentence():
    def __init__(self, original, lemmatized):
        self.original = original
        self.lemmatized = lemmatized


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
        sentences_lemm = []
        for sentence in sentences:
            sentences_lemm.append(Sentence(sentence, self.lemmatize(sentence)))
        return sentences_lemm

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
            if lemmas != [] and (lemma_i >= len(lemmas) or word != lemmas[lemma_i][0]):
                lemma_sent.append(word)
            elif len(lemmas) != 0:
                lemma_sent.append(lemmas[lemma_i][2])
                lemma_i += 1
        return " ".join(lemma_sent)

    def extract_sentences_keyword(self, text, question):
        # Extracts sentences that most likely contain the answer
        # TODO: make question into class sentence before inputting
        question = self.lemmatize(question)
        keywords = self.question_parser(question)
        process_text = self.preprocess_text(text)
        result = []
        best = None
        bestRatio = 0
        for sentence in process_text:
            count = 0
            seen = set()
            for word in sentence.lemmatized.split():
                if word in keywords:
                    count += 1
                    if word not in seen:
                        seen.add(word)
                else:
                    # similarity checking
                    for kword in keywords:
                        try:
                            w1 = wordnet.synsets(word)[0]
                            w2 = wordnet.synsets(kword)[0]
                            if w1.wup_similarity(w2) > 0.9:
                                count += 1
                                seen.add(word)
                        except:
                            continue
            bigram_count = self.count_bigram(sentence, question)
            trigram_count = self.count_trigram(sentence, question)
            unique_count = len(seen)
            final_count = count*.3 + bigram_count*.4 + unique_count*.3 + trigram_count*.2
            if final_count >= (len(keywords)*.6+len(question.split(" "))*2*.4)*.6:
                result.append(sentence.original)
                ratio = count/(len(sentence.original.split(" ")))
                if ratio > bestRatio:
                    bestRatio = ratio
                    best = sentence.original
        return result, best, len(result)

    def count_bigram(self, sentence, question):
        # counts the number of bigram occurences from the question in the sentence
        count = 0
        question = "START " + question
        q_words = question.split(" ")
        s_words = sentence.lemmatized.split(" ")
        for i in range(1, len(q_words)):
            # Possibily add in similarity checking here too
            for j in range(len(s_words)):
                if j == 0:
                    count += question.count("START " + s_words[0])
                else:
                    count += question.count(s_words[j-1] + " " + s_words[j])
        return count


    def count_trigram(self, sentence, question):
        count = 0
        question = "START START " + question
        q_words = question.split(" ")
        s_words = sentence.lemmatized.split(" ")
        for i in range(2, len(q_words)):
            for j in range(len(s_words)):
                # Possibily add in similarity checking here too
                if j == 0:
                    prev1 = "START"
                    prev2 = "START"
                elif j == 1:
                    prev1 == "START"
                    prev2 = s_words[j-1]
                else:
                    prev1 = s_words[j-2]
                    prev2 = s_words[j-1]
                count += question.count(prev1 + " " + prev2 + " " + s_words[j])
        return count


    def get_possible_answers(self, text, question):
        keywords = self.question_parser(self.lemmatize(question))
        print(keywords)
        return self.extract_sentences_keyword(text, question)

    def pronoun_resolution(self, sentence):
        neuralcoref.add_to_pipe(self.nlp)
        doc1 = nlp(sentence)
        print(doc1._.coref_clusters)


# def termFrequency(term, sentence): #how do we represent this sentence




parser = Parser()
#print(parser.tokenize("Bob went to Central Park on Wednesday."))
#print(parser.question_parser("Where did Bob go on Wednesday?"))
text = """The Old Kingdom is most commonly regarded as the 
period from the Third Dynasty through to the Sixth Dynasty (2686–2181 BC). 
The 4th-6th Dynasties of Egypt, are scarce and historians regard the history 
of the era as literally 'written in stone' and largely architectural in that it 
is through the monuments and their inscriptions that scholars have been able to 
construct a history. Egyptologists also include the Memphite Seventh and Eighth 
Dynasties in the Old Kingdom as a continuation of the administration centralized 
at Memphis. While the Old Kingdom was a period of internal security and 
prosperity, it was followed by a period of disunity and relative cultural 
decline referred to by Egyptologists as the First Intermediate Period. During 
the Old Kingdom, the king of Egypt (not called the Pharaoh until the New 
Kingdom) became a living god who ruled absolutely and could demand the services 
and wealth of his subjects."""

# def readFile(path):
#     with open(path, "rt") as f:
#         return f.read()
# text = readFile("Development_data/set1/a1.txt")

# print(parser.get_possible_answers(text, "Which era do historians regard as 'written in stone'?"))
# print(parser.get_possible_answers(text, "Who was not called the Pharaoh until the New Kingdom?"))
ans = parser.get_possible_answers(text, "Who asked for the wealth of his people?")
ans = ans[0] + ["During the Old Kingdom, the king of Egypt (not called the Pharaoh until the New Kingdom) became a living god who ruled absolutely and didn't demand the services and wealth of his subjects."]
#print(parser.pronoun_resolution("Bob went to the park. He brought his dog."))

#implementing tf-idf

#returns vector space representation for a sentence of term frequencies 
def calculateTermFrequencies(s):
    s = s.split(" ")
    d = dict()
    for word in s:
        if(word.lower() in d):
            d[word.lower()] += 1
        else:
            d[word.lower()] = 1
    vectorWords = []
    vectorDict = dict()
    total = 0
    indices = set()
    for word in s:
        if word.lower() not in indices: #individual words
            indices.add(word.lower())
            vectorWords.append(word.lower())
            total += d[word.lower()]
    
    for i, word in enumerate(vectorWords):
        vectorDict[word] = d[word] / total
        # vectorFreqs[i] = d[word] / total

    return vectorDict #dictionary for term freqs for a particular sentence

def calculateTermFreq(sentence, t):
    if(t.lower() in calculateTermFrequencies(sentence)):
        return calculateTermFrequencies(sentence)[t.lower()]
    else:
        return 0


def calcInverseDocFrequency(sentences, t):
    numDocs = 0
    for sentence in sentences:
        if t.lower() in sentence.lower().split(" "):
            numDocs += 1
    if(numDocs > 0):
        return math.log(len(sentences) / numDocs)
    else:
        return 0

def compareToOriginal(originalQ, sentences): 
    allSentences = sentences + [originalQ]
    vectorMapper = dict() #mapping sentences to their vectors for comparison 
    #get vocabulary 
    vocab = set()
    for sentence in allSentences:
        for word in sentence.lower().split(" "):
            vocab.add(word)
    vocab = list(vocab) # consistently ordered set of words
    #building vectorMapper
    for sentence in allSentences:
        #building the vector 
        vector = [0 for i in range(len(vocab))]
        vectorWords = [0 for i in range(len(vocab))]
        ind = 0
        for word in vocab:
            if(word not in vectorWords):
                vectorWords[ind] = word 
                vector[ind] = calculateTermFreq(sentence, word) * calcInverseDocFrequency(allSentences, word)
                ind += 1

        vectorMapper[sentence] = vector


    origVector = vectorMapper[originalQ]
    resultDict = dict()
    for sentence in sentences:
        #calculate dot product between original sentence and each of these sentences
        res = 0
        for i in range(len(vectorMapper[sentence])):
            res += origVector[i] * vectorMapper[sentence][i]
        res = res // (len(sentence.split(" ")) + len(originalQ.split(" ")))
        resultDict[sentence] = res

    return min(resultDict, key=resultDict.get)
    
print(compareToOriginal("Who asked for the wealth of his people?", ans))


# def generateAnswer(question):
#     questionType = question[0]


        

