import math
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("was"))
#implementing tf-idf

#returns vector space representation for a sentence of term frequencies
# need to define global variable sentences 

# nlp = spacy.load("en_core_web_sm")

def calculateTermFrequencies(s):
    words = s.split(" ")
    # print(s, words)
    # for token in nlp(s):
    #     words.append(token.text)
    total = len(words)
    d = dict() #getting the num of each word in the doc 
    for word in words:
        new = word.lower()
        if(new in d):
            d[new] += 1
        else:
            d[new] = 1
    # print(d)
    vectorDict = dict()
    for word in d:
        vectorDict[word] = float(d[word]) / float(total)

    return vectorDict #dictionary for term freqs for a particular sentence

def calculateTermFreq(t, sentence):
    word = t.lower()
    if(word in calculateTermFrequencies(sentence)):
        return calculateTermFrequencies(sentence)[word]
    else:
        return 0


def calcInverseDocFrequency(t, sentences):
    word = t.lower()
    numDocs = 0
    for sentence in sentences:
        words = calculateTermFrequencies(sentence)
        # print(words)
        if word in words:
            numDocs += 1
    print(numDocs, t.lower())
    if(numDocs > 0):
        return math.log(len(sentences) / numDocs)
    else:
        return 0


def calcDotProduct(query, document, vocab, sentences): #an ordered list of words
    res = 0
    for term in vocab:
        tf_idfQ = calculateTermFreq(term, query) * calcInverseDocFrequency(term, sentences)
        tf_idfD = calculateTermFreq(term, document) * calcInverseDocFrequency(term, sentences)
        res += tf_idfQ * tf_idfD
    return res

def findCosineSimilarity(query, document, vocab, sentences):
    dotProduct = calcDotProduct(query, document, vocab, sentences)
    #calculating ||query|| and ||document||
    q = 0
    d = 0
    for term in vocab:
        tf_idf = calculateTermFreq(term, query) * calcInverseDocFrequency(term, sentences)
        print(query, tf_idf)
        tf_idf2 = calculateTermFreq(term, document) * calcInverseDocFrequency(term, sentences)
        print(document, tf_idf2)
        q += tf_idf ** 2
        d += tf_idf2 ** 2
    if(q * d == 0):
        return 5000
    return dotProduct / (q * d)

def matchingScore(query, document, vocab, sentences):
    res = 0
    for term in vocab:
        tf_idf = calculateTermFreq(term, document) * calcInverseDocFrequency(term, sentences)
        res += tf_idf
    return res

def compareToOriginal(originalQ, sentences): 
    allSentences = sentences + [originalQ]
    #constructing vocabulary
    vocabulary = set()

    words = originalQ.split(" ")
    for word in words:
        vocabulary.add(word.lower())
    # print(vocabulary)

    #comparing query with each sentence
    resultDict = dict()
    for s in sentences:
        resultDict[s] = matchingScore(originalQ, s, vocabulary, allSentences)
        # resultDict[s] = findCosineSimilarity(originalQ, s, vocabulary, allSentences)
    print(resultDict)

    return max(resultDict, key=resultDict.get)

 

# q = "What was always a central concern for ancient Egyptians"
# sents = ["I was wanting to die.", "Cheese wasn't always a central concern for the ancient Egyptians.", "Divine pardon at judgement was always a central concern for the ancient Egyptians."]

# # q = "Why does Sirius appear bright"
# # sents = ["The brightest star in the night sky, Sirius is recorded in the earliest astronomical records.",
# #     "Sirius appears bright because of its intrinsic luminosity and its proximity to Earth.", 
# # "Sirius (, a romanization of Greek Σείριος, Seirios, lit) is a star system and the brightest star in the Earth's night sky."]
# print(compareToOriginal(q, sents))



# tests = dict()
# tests["What was always a central concern for ancient Egyptians"] = ["Cheese wasn't always a central concern for the ancient Egyptians.", "Divine pardon at judgement was always a central concern for the ancient Egyptians."]
# tests["What was the first phase of the festival"] = ["The first phase of the festival was a public drama depicting the murder and dismemberment of Osiris.", "The annual festival involved the construction of Osiris Beds formed in shape of Osiris, filled with soil and sown with seed."]
# tests["When was the government in the hands of the various nomes"] = ["The history of ancient Egypt is divided into three kingdoms and two intermediate periods.",
# "During the intermediate periods (the periods of time between kingdoms) government control was in the hands of the various nomes (provinces within Egypt) and various foreigners.",
# "For most parts of its long history, ancient Egypt was unified under one government."]




# tester(tests)
