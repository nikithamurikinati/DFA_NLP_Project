import nltk
import os
import stanfordnlp
import string
import copy
os.environ["CORENLP_HOME"] = "./corenlp"
from stanfordnlp.server import CoreNLPClient

nlpCore = CoreNLPClient(annotators=['tokenize','coref'], 
                        endpoint='http://localhost:9001')
nlpCore.start()

text1 = "Niki Murikinati is useless. Mihika Bairathi says she knows nothing. Bob went to the market. He met Niki there."
text2 = "Her mom's food is amazing."
text3 = "Her name is M. Bairathi, and she is really nice."
text4 = "Barack Hussein Obama II (born August 4, 1961) is an American politician" +\
        " and attorney who served as the 44th president of the United States " +\
        "from 2009 to 2017. A member of the Democratic Party, he was the first " + \
        "African-American president of the United States. He previously served " +\
        "as a U.S. senator from Illinois from 2005 to 2008 and an Illinois state" +\
        " senator from 1997 to 2004. Obama was born in Honolulu, Hawaii. After "+\
        "graduating from Columbia University in 1983, he worked as a community " +\
        "organizer in Chicago. In 1988, he enrolled in Harvard Law School, where " +\
        "he was the first black person to head the Harvard Law Review. After " +\
        "graduating, he became a civil rights attorney and an academic, teaching " +\
        "constitutional law at the University of Chicago Law School from 1992 to 2004."

def collapseWhitespace(s):
    """
    Helper function that replaces multiple whitespace characters with 
    a single white space
    """
    whiteSpaceRun = False
    result = ""
    for c in s:
        if (c not in string.whitespace):
            if (whiteSpaceRun == True):
                result += " "
                whiteSpaceRun = False
            result += c
        else:
            whiteSpaceRun = True
    if (whiteSpaceRun == True):
        result += " "
    return result

def greaterThan(elem1, elem2):
    #returns True if elem1 > elem2
    if elem1.sentenceIndex > elem2.sentenceIndex:
        return True
    elif elem1.sentenceIndex < elem2. sentenceIndex:
        return False
    return elem1.beginIndex > elem2.beginIndex

def merge(a, start1, start2, end):
    index1 = start1
    index2 = start2
    length = end - start1
    aux = [None] * length
    for i in range(length):
        if ((index1 == start2) or
            ((index2 != end) and greaterThan(a[index1], a[index2]))): #item in index 1 is greater
            aux[i] = a[index2]
            index2 += 1
        else:
            aux[i] = a[index1]
            index1 += 1
    for i in range(start1, end):
        a[i] = aux[i - start1]

def mergeSort(a):
    n = len(a)
    step = 1
    while (step < n):
        for start1 in range(0, n, 2*step):
            start2 = min(start1 + step, n)
            end = min(start1 + 2*step, n)
            merge(a, start1, start2, end)
        step *= 2

def extractSentences(docCore, allSentences):
    """
    Few examples:
    Her dad's food is great. => [Her, dad, 's, food, is, great, .]
    Her name is M. Bairathi. => [Her, name, is, M., Bairathi, .]
    """
    for sent in docCore.sentence:
        thisSentence = []
        for t in sent.token:
            thisSentence.append(t.originalText)
        allSentences.append(thisSentence)

def getCorefs(text, docCore, allSentences):
    newSentences = copy.deepcopy(allSentences)
    allChains = docCore.corefChain
    for chain in allChains:
        cm = list(chain.mention)
        mergeSort(cm)
        #get the right word
        begin = cm[0].beginIndex
        end = cm[0].endIndex
        sNumber = cm[0].sentenceIndex
        rightWord = " ".join(allSentences[sNumber][begin:end])
        #replace in all the others
        for m in cm[1:]:
            b = m.beginIndex
            e = m.endIndex
            sIndex = m.sentenceIndex
            newSentences[sIndex] = newSentences[sIndex][:b] + [" "]*(e-b) + newSentences[sIndex][e:]
            newSentences[sIndex][b] = rightWord
    return newSentences

def modifyText(text, docCore):
    #docCore = nlpCore.annotate(text)
    allSentences = []
    extractSentences(docCore, allSentences)
    res = getCorefs(text, docCore, allSentences)
    modifiedText = ""
    for sentence in res:
        modifiedText += " ".join(sentence) + " "
    return collapseWhitespace(modifiedText)

#replace text with the particular type
text = text4
docCore = nlpCore.annotate(text4)
print(modifyText(text, docCore))
nlpCore.stop()
