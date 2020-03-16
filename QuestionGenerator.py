import nltk
import os
import stanfordnlp
os.environ["CORENLP_HOME"] = "./corenlp"
from stanfordnlp.server import CoreNLPClient
#stanfordnlp.download('en') #uncomment if running for first time

nlpCore = CoreNLPClient(annotators=['ner'], endpoint='http://localhost:9001')
nlpCore.start()
nlp = stanfordnlp.Pipeline()
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

def findRelatedKey(word, NERTags):
    for tag in NERTags:
        if word in tag:
            return tag
    return None

def getWord(i, POSTags, NERTags, theType, sentence):
    POS = POSTags[i]
    try:
        NER = NERTags[findRelatedKey(sentence[i], NERTags)]
    except:
        NER = None   
    if POS == "NOUN" or "obl" in theType:
        word = "what" 
    else:
        word = "who"
    if NER in ["COUNTRY", "STATE_OR_PROVINCE", "CITY"]:
        word = "where"
    if NER == "DATE":
        word = "when"
    if NER == "ORGANIZATION":
        word = "what"
    return word

def sentenceQuestionsM1(sentence, dependencies, types, POSTags, NERTags):
    """
    Method one of producing questions. Finds words of type "nsubj", "nmod", etc
    and replaces them with a "w" question, leaving the rest intact.
    Issues: Grammar is slightly awkward, and the sentences can be long, 
    so the questions will be long too. Does not generate "how" or why" or 
    "which" or questions, and doesn't make all possible questions from 
    a given sentence. Despite POS, NER tagging, might still make mistakes in
    identifying which W-question works the best for the sentence. 
    """
    rootNumber = dependencies.index("0") + 1
    questions = []
    for i in range(len(dependencies)-1):
        if dependencies[i] == str(rootNumber):
            newQuestion = []
            phraseBefore = sentence[:i]
            phraseAfter = sentence[i+1:]
            try:
                if i >= rootNumber:
                    nextRootIndex = dependencies[i+1:].index(str(rootNumber)) \
                        + (i + 1)
                else:
                    nextRootIndex = min(dependencies[i+1:].index(str(rootNumber)) \
                        + (i + 1), rootNumber-1)
            except:
                nextRootIndex = len(dependencies) - 1
            compoundIndex = i - 1
            while compoundIndex >= 0 and types[compoundIndex] == "compound" or \
                                            types[compoundIndex] == "det":
                compoundIndex -= 1
            if "nsubj" in types[i]:
                while "nsubj" in types[nextRootIndex]:
                    nextRootIndex += 1
            elif "nmod" in types[i]:
                pass
            elif "obl" in types[i]:
                nextRootIndex = i+1
                pass
            else:
                continue
            word = getWord(i, POSTags, NERTags, types[i], sentence)
            if compoundIndex == -1:
                word = word[0].upper() + word[1:]           
            newQuestion = sentence[:compoundIndex+1] + [word] + \
                            sentence[nextRootIndex:]
            newQuestion = newQuestion[:-1] + ["?"]
            newQuestionString = " ".join(newQuestion)
            questions.append(newQuestionString)
    return questions
    

def extractInformation(doc, allSentences, allDependencies, allTypes, allPOSTags):
    for sentence in doc.sentences:
        thisSentence = ""
        thisDependency = []
        thisType = []
        thisPOSTag = []
        for token in sentence.tokens:
            thisPOSTag.append(token.words[0].upos)
        for l in sentence.dependencies_string().splitlines():
            parts = l.split(",")
            infoPerWord = []
            for p in parts:
                p = p.strip()
                if p[0] == "(":
                    p = p[1:]
                elif p[-1] == ")":
                    p = p[:-1]
                p = p[1:-1]
                infoPerWord.append(p)
            if infoPerWord[0] == "":
                infoPerWord[0] = ","
            thisSentence += infoPerWord[0] + " "
            thisDependency.append(infoPerWord[1])
            thisType.append(infoPerWord[2])
        allSentences.append(thisSentence.split(" ")[:-1])
        allDependencies.append(thisDependency)
        allTypes.append(thisType)
        allPOSTags.append(thisPOSTag)

def extractCoreInformation(docCore, allNERTags):
    for sent in docCore.sentence:
        thisNERTag = dict()
        for m in sent.mentions:
            if m.entityMentionText not in thisNERTag:
                thisNERTag[m.entityMentionText] = m.entityType
        allNERTags.append(thisNERTag)

def generateQuestions(text):
    docCore = nlpCore.annotate(text)
    doc = nlp(text)
    allSentences = [] 
    allDependencies = [] 
    allTypes = []
    allPOSTags = []
    allNERTags = []
    extractInformation(doc, allSentences, allDependencies, allTypes, allPOSTags)
    extractCoreInformation(docCore, allNERTags)
    #print(doc._text) #the original text passed in
    #print(allSentences) #2D List, each list is a sentence, split at " "
    #print(allDependencies) #2D List, each list a sentence, but numbers, not tokens
    #print(allTypes) #type of each word in each sentence, 2d list
    #print(allPOSTags) #2D List of POS tags
    #print(allNERTags) #2D List of NER tags, not same dims as other lists here
    methodOneQuestions = []
    for i in range(len(allTypes)):
        sentence = allSentences[i]
        dependencies = allDependencies[i]
        types = allTypes[i]
        POSTags = allPOSTags[i]
        NERTags = allNERTags[i]
        methodOneQuestions.extend(sentenceQuestionsM1(sentence, dependencies, \
                                                    types, POSTags, NERTags))
    print(methodOneQuestions)
    
generateQuestions(text2)
nlpCore.stop()


