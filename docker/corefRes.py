import os
import stanfordnlp
import string
import copy

#uncomment the stuff commented out if you want to run this file on its own.
"""
os.environ["CORENLP_HOME"] = "./corenlp"
from stanfordnlp.server import CoreNLPClient

nlpCore = CoreNLPClient(annotators=['tokenize','coref'], 
                        endpoint='http://localhost:9001')
nlpCore.start()
"""


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
        for i in range (1, len(cm)):
            m = cm[i]
            b = m.beginIndex
            e = m.endIndex
            sIndex = m.sentenceIndex
            if e-b != 1:
                continue
            if cm[i-1].sentenceIndex == sIndex or (i+1 < len(cm) and cm[i+1].sentenceIndex == sIndex):
                seen = 0
                for w in rightWord.split(" "):
                    if w in newSentences[sIndex][:b]:
                        seen = 1
                if seen == 1 or rightWord in newSentences[sIndex][:b]:
                    continue
            newSentences[sIndex] = newSentences[sIndex][:b] + [" "]*(e-b) + newSentences[sIndex][e:]
            newSentences[sIndex][b] = rightWord
    return newSentences

def modifyText(text, docCore):
    allSentences = []
    extractSentences(docCore, allSentences)
    res = getCorefs(text, docCore, allSentences)
    modifiedText = ""
    for sentence in res:
        modifiedText += " ".join(sentence) + " "
    return collapseWhitespace(modifiedText)

text4 = "The Old Kingdom is the period in the third millennium (c. 2686-2181 BC) "+\
        "also known as the 'Age of the Pyramids' or 'Age of the Pyramid Builders' " +\
        "as it includes the great 4th Dynasty when King Sneferu perfected the art "+\
        "of pyramid building and the pyramids of Giza were constructed under the "+\
        "kings Khufu, Khafre, and Menkaure. Egypt attained its first continuous peak "+\
        "of civilization – the first of three so-called 'Kingdom' periods (followed "+\
        "by the Middle Kingdom and New Kingdom) which mark the high points of "+\
        "civilization in the lower Nile Valley. The term itself was coined by "+\
        "eighteenth-century historians and the distinction between the Old Kingdom "+\
        "and the Early Dynastic Period is not one which would have been recognized "+\
        "by Ancient Egyptians. Not only was the last king of the Early Dynastic "+\
        "Period related to the first two kings of the Old Kingdom, but the 'capital', "+\
        "the royal residence, remained at Ineb-Hedg, the Ancient Egyptian name for "+\
        "Memphis. The basic justification for a separation between the two periods "+\
        "is the revolutionary change in architecture accompanied by the effects on "+\
        "Egyptian society and economy of large-scale building projects. The Old Kingdom "+\
        "is most commonly regarded as the period from the Third Dynasty through to "+\
        "the Sixth Dynasty (2686–2181 BC). The 4th-6th Dynasties of Egypt, are "+\
        "scarce and historians regard the history of the era as literally 'written "+\
        "in stone' and largely architectural in that it is through the monuments "+\
        "and their inscriptions that scholars have been able to construct a history. "

text2 = "Barack Hussein Obama II (born August 4, 1961) is an American politician" +\
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

"""
#replace text with the particular type
text = text4
docCore = nlpCore.annotate(text)
print(modifyText(text, docCore))
nlpCore.stop()
"""

