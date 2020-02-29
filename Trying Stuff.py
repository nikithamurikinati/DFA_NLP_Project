import nltk
import os
java_path = "C:/Program Files/Java/jre1.8.0_241/bin/java.exe"
os.environ['JAVAHOME'] = java_path
"""
from nltk.parse.stanford import StanfordDependencyParser

path_to_jar = 'C:/Users/mihik/.spyder-py3/NLP Project/stanford-parser-full-2018-10-17/stanford-parser.jar'
path_to_models_jar = 'C:/Users/mihik/.spyder-py3/NLP Project/stanford-english-corenlp-2018-02-27-models.jar'

dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)

result = dependency_parser.raw_parse('I shot an elephant in my sleep')
dep = result.__next__()

print(list(dep.triples()))
"""

import stanfordnlp
#stanfordnlp.download('en')
nlp = stanfordnlp.Pipeline()
text = "The prospects for Britain’s orderly withdrawal from the European Union \
    on March 29 have receded further, even as MPs rallied to stop a no-deal \
        scenario. An amendment to the draft bill on the termination of London’s \
            membership of the bloc obliges Prime Minister Theresa May to \
                renegotiate her withdrawal agreement with Brussels. A Tory \
                    backbencher’s proposal calls on the government to come up \
                        with alternatives to the Irish backstop, a central \
                            tenet of the deal Britain agreed with the rest of \
                                the EU."
doc = nlp(text)

allSentences = [] #list of strings, each string is a sentence
allDependencies = [] #nested list of dependencies
allTypes = [] #nested list of word type
for sentence in doc.sentences:
    thisSentence = ""
    thisDependency = []
    thisType = []
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
        thisSentence += infoPerWord[0] + " "
        thisDependency.append(infoPerWord[1])
        thisType.append(infoPerWord[2])
    allSentences.append(thisSentence.split(" ")[:-1])
    allDependencies.append(thisDependency)
    allTypes.append(thisType)    

#print(doc._text)
print(allSentences)
print(allDependencies)
print(allTypes)

trying = []
allQuestions = []
for i in range(len(allSentences)):
    thisSentence = allSentences[i]
    thisDependency = allDependencies[i]
    thisType = allTypes[i]
    possibleQs = []
    root = thisDependency.index('0')
    for j in range(len(thisDependency)):
        if thisDependency[j] == str(root+1):
            if thisType[j] not in ["punct", "aux:pass", "aux", "advmod", "advcl", "xcomp"]:
                if thisType[j] == "nsubj" or thisType[j] == "nsubj:pass":
                    trying.append("What " + " ".join(thisSentence[root:]) + "?")
                if thisType[j] == "obj":
                    newRoot = j+1
                    newQ = []
                    seenYet = False
                    for k in range(len(thisDependency)):
                        if thisDependency[k] != str(newRoot) and k != j:
                            newQ.append(thisSentence[k])
                        elif seenYet == False:
                            newQ.append("what")
                            seenYet = True
                        else:
                            continue
                    trying.append(" ".join(newQ) + "?") 
                if thisType[j] == "obl":
                    #need to fix
                    word = thisSentence[j]
                    temp = nlp(word)
                    temp = temp.sentences[0].tokens_string()
                    print(temp)
                    newRoot = j+1
                    newQ = []
                    seenYet = False
                    for k in range(len(thisDependency)):
                        if thisDependency[k] != str(newRoot) and k != j:
                            newQ.append(thisSentence[k])
                        elif seenYet == False:
                            newQ.append("whom")
                            seenYet = True
                        else:
                            continue
                    trying.append(" ".join(newQ) + "?") 
                possibleQs.append((thisSentence[j], thisType[j], thisSentence[root], " ".join(thisSentence)))
    allQuestions.append(possibleQs)
#print(allQuestions)
print(trying)