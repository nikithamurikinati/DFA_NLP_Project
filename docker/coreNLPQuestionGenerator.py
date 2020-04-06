import os
import stanfordnlp
import string
os.environ["CORENLP_HOME"] = "./corenlp"
from stanfordnlp.server import CoreNLPClient
#stanfordnlp.download('en') #uncomment if running for first time!!!!
import corefRes

text0 = "Chris Manning is a nice person. Chris wrote a simple sentence. He also gives oranges to people."

text1 = "The prospects for Britain’s orderly withdrawal from the European Union " + \
        "on March 29 have receded further, even as MPs rallied to stop a no-deal" + \
        " scenario. An amendment to the draft bill on the termination of London’s" + \
        " membership of the bloc obliges Prime Minister Theresa May to " + \
        "renegotiate her withdrawal agreement with Brussels. A Tory " + \
        "backbencher’s proposal calls on the government to come up " + \
        "with alternatives to the Irish backstop, a central " + \
        "tenet of the deal Britain agreed with the rest of the EU."

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

text3 = "The Old Kingdom is the period in the third millennium (c. 2686-2181 BC) "+\
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

def greaterThan(elem1, elem2):
    #returns True if elem1 > elem2
    if elem1.target > elem2.target:
        return True
    else:
        return False

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

def getWord(i, POSTags, NERTags, theType, sentence):
    POS = POSTags[i]
    NER = NERTags[i] 
    if POS == "RB":
        return None
    if "obj" in theType:
        return "what"
    word = "what"
    if NER == "PERSON" or NER == "TITLE":
        return "who"
    if "PRP" in POS:
        if "it" in sentence[i]:
            word = "what"
        else:
            word = "who"
    if NER in ["COUNTRY", "STATE_OR_PROVINCE", "CITY", "LOCATION"]:
        if "STATE" in NER:
            NER = "STATE"
        word = "what " + NER.lower()
        if NER == "LOCATION":
            word = "what"
    if NER == "DATE" or NER == "DURATION":
        if len(sentence[i]) == 4:
            word = "what year"
        else:
            word = "when"
    return word

def getPhraseBoundaries(i, rootNumber, dependencies, types, POSTags, NERTags):
    try:
        if i >= rootNumber:
            nextRootIndex = dependencies[i+1:].index(rootNumber) \
                + (i + 1)
        else:
            nextRootIndex = min(dependencies[i+1:].index(rootNumber) \
                + (i + 1), rootNumber-1)
    except:
        nextRootIndex = len(dependencies) - 1
    compoundIndex = i - 1
    #not case , punct for nmod
    while compoundIndex >= 0 and types[compoundIndex] == "compound" or \
                                    types[compoundIndex] == "det" or \
                                    "mod" in types[compoundIndex] and \
                                    "nmod" not in types[compoundIndex]:
        compoundIndex -= 1
    if "nsubj" in types[i]:
        while "nsubj" in types[nextRootIndex]:
            nextRootIndex += 1
    if "nmod" in types[i]:
        while "nmod" in types[nextRootIndex]:
            nextRootIndex += 1
    if "obj" in types[i]:
        try:
            nextRootIndex = len(dependencies) - dependencies[::-1].index(i+1)
        except:
            i = len(dependencies)-1
    return (nextRootIndex, compoundIndex)

def sentenceQuestionsM1(sentence, dependencies, types, POSTags, NERTags):
    rootNumber = dependencies.index(0) + 1
    questions = []
    for i in range(len(dependencies)-1):
        if dependencies[i] == rootNumber:
            if "nsubj" not in types[i] and "obj" not in types[i] and "nmod" not in types[i]:
                continue
            newQuestion = []
            phraseBefore = sentence[:i]
            phraseAfter = sentence[i+1:]
            nextRootIndex, compoundIndex = getPhraseBoundaries(i, rootNumber, 
                                        dependencies, types, POSTags, NERTags)
            word = getWord(i, POSTags, NERTags, types[i], sentence)
            if word == None:
                continue
            if compoundIndex == -1:
                word = word[0].upper() + word[1:]           
            newQuestion = (sentence[:compoundIndex+1] + [word] + \
                            sentence[nextRootIndex:])[:-1] + ["?"]
            if newQuestion[-2] in string.punctuation:
                newQuestion = newQuestion[:-2] + [newQuestion[-1]]
            questions.append(" ".join(newQuestion))
    return questions

def extractInformation(doc, allSentences, allPOSTags, allNERTags, allDependencies, allTypes):
    for sent in doc.sentence:
        thisSentence = []
        thisPOSTag = []
        thisNERTag = []
        for t in sent.token:
            thisSentence.append(t.originalText)
            thisPOSTag.append(t.pos)
            thisNERTag.append(t.ner)
        allSentences.append(thisSentence)
        allPOSTags.append(thisPOSTag)
        allNERTags.append(thisNERTag)

        thisDependency = []
        thisType = []
        dependency_parse = sent.basicDependencies
        root = dependency_parse.root[0]
        passedRoot = 0
        d = list(dependency_parse.edge)
        mergeSort(d)
        for i in range(len(d)+1):
            if i == root-1:
                thisType += ["root"]
                thisDependency += [0]
                passedRoot = 1
            else:
                e = d[i - passedRoot]
                thisType += [e.dep]
                thisDependency += [e.source]
        allDependencies.append(thisDependency)
        allTypes.append(thisType)

def sentenceQuestionsM2(sentence):
    return [" ".join(sentence[:-1] + [", right?"])]

def writeFile(path, contents):
    with open(path, "wt", encoding="utf8") as f:
        f.write(contents)

def readFile(path):
    with open(path, "rt", encoding="utf8") as f:
        s =  f.read()
        text = ""
        for line in s.splitlines():
            if line == "" or line[-1] not in string.punctuation:
                continue
            text += " " + line
        return text

def generateQuestions(text):
    #modify text
    nlpCore = CoreNLPClient(annotators=['tokenize','coref'],
    timeout = 100000, memory='16G')

    nlpCore.start()
    docCore = nlpCore.annotate(text)
    text = corefRes.modifyText(text, docCore)

    #final doc to use
    doc = nlpCore.annotate(text)

    #variables
    allSentences = [] 
    allDependencies = [] 
    allTypes = [] 
    allPOSTags = []
    allNERTags = []

    #extracting all the information
    extractInformation(doc, allSentences, allPOSTags, allNERTags, allDependencies, allTypes)

    questions = []
    for i in range(len(allTypes)):
        sentence = allSentences[i]
        dependencies = allDependencies[i]
        types = allTypes[i]
        POSTags = allPOSTags[i]
        NERTags = allNERTags[i]
        questions.extend(sentenceQuestionsM1(sentence, dependencies, \
                                                    types, POSTags, NERTags))
        questions.extend(sentenceQuestionsM2(sentence))
    nlpCore.stop()
    return "\n".join(questions)
    
finalText = readFile("Development_data/set5/a10.txt")