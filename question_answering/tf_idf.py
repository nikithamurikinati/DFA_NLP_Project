import math
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