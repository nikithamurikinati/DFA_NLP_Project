import math
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
#implementing tf-idf

#returns vector space representation for a sentence of term frequencies
# need to define global variable sentences 


def calculateTermFrequencies(s):
    words = s.split(" ")

    total = len(words)
    d = dict() #getting the num of each word in the doc 
    for word in words:
        new = word.lower()
        if(new in d):
            d[new] += 1
        else:
            d[new] = 1
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
        if word in words:
            numDocs += 1
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
        tf_idf2 = calculateTermFreq(term, document) * calcInverseDocFrequency(term, sentences)
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

    #comparing query with each sentence
    resultDict = dict()
    for s in sentences:
        resultDict[s] = matchingScore(originalQ, s, vocabulary, allSentences)

    return max(resultDict, key=resultDict.get)