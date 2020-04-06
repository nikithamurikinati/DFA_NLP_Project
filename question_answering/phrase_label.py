
import os
import stanfordnlp
os.environ["CORENLP_HOME"] = "./corenlp"
from stanfordnlp.server import CoreNLPClient

#stanfordnlp.download('en') #uncomment if running for first time

#nlpCore = CoreNLPClient(annotators=['ner'], endpoint='http://localhost:9001')
#nlpCore.start()
#nlp = stanfordnlp.Pipeline()
#ner

nlpCore = CoreNLPClient(annotators=['tokenize','ssplit','pos','parse','coref', 'ner'],
        timeout=30000, memory='16G')
text = "Bob went to the store in Pittsburgh on Sunday. He said hi."
'''
with CoreNLPClient(annotators=['tokenize','ssplit','pos','lemma','ner', 'parse', 'depparse','coref'], 
    timeout=30000, memory='16G') as client:
    ann = client.annotate(text)
    sentence = ann.sentence[0]
    constituency_parse = sentence.parseTree
    print(constituency_parse)
    tags  = {"CC", "CD", "DT","EX", "FW", "IN", "JJ", "JJR", "JJS",
            "LS","MD","NN", "NNS","NNP","NNPS","PDT","POS", "PRP","PRP$","RB",
            "RBR","RBS", "RP", "TO", "UH", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ", 
            "WDT", "WP", "WP$","WRB", "NP", "VP", "PP", "S", "ROOT", "ADJP", "ADVP"
        }
'''

def getSentence(parseTree, result):
    for child in parseTree.child:
        if len(child.child) != 0:
            result = getSentence(child, result)
        else:
            result = result + " " + child.value
    return result

def getPhrasesFromTree(parseTree, phraseType, result):
    for child in parseTree.child:
        #print(child.value)
        if len(child.child)==1 and len(child.child[0].child) == 0:
            if child.value == phraseType:
                result.append(child.child[0].value)
        else:
            if child.value == phraseType:
                result.append(getSentence(child, "").strip())
            result = getPhrasesFromTree(child, phraseType, result)
    nlpCore.stop()
    return result
        
def getPhrases(sentence, phraseType):
    #print(sentence)
    nlpCore.start()
    docCore = nlpCore.annotate(sentence)
    parseTree = docCore.sentence[0].parseTree
    result = []
    return getPhrasesFromTree(parseTree, phraseType, result)
    #return result

def getNounVerbPhrasesFromTree(parseTree, result):
    types = set()
    for child in parseTree.child:
        types.add(child.value)
    if "NP" in types and "VP" in types:
        toDo = []
        noun = None
        verb = None
        for child in parseTree.child:
            if child.value =="NP" and noun == None:
                noun = getSentence(child, "").strip()
            elif child.value == "VP" and verb == None:
                verb = getSentence(child, "").strip()
            else:
                toDo.append(child)
        result.append((noun, verb))
        for child in toDo:
            result = getNounVerbPhrasesFromTree(child, result)
    else:
        for child in parseTree.child:
            result = getNounVerbPhrasesFromTree(child, result)
    nlpCore.stop()
    return result

def getNounVerbPhrasePairs(sentence):
    nlpCore.start()
    docCore = nlpCore.annotate(sentence)
    parseTree = docCore.sentence[0].parseTree
    return getNounVerbPhrasesFromTree(parseTree, [])

def splitQuestion(question):
    nlpCore.start()
    docCore = nlpCore.annotate(question)
    parseTree = docCore.sentence[0].parseTree
    skip = True
    result = ""
    qverb = None
    #print(parseTree.child[0])
    for child in parseTree.child[0].child[1].child:
        if qverb == None and "VB" in child.value:
            qverb = child.child[0].value
        if child.value == "NP":
            skip = False
        if not skip:
            result = result + " " + getSentence(child, "").strip()
    nlpCore.stop()
    return result.strip(), qverb

def splitWhichQuestion(question):
    nlpCore.start()
    docCore = nlpCore.annotate(question)
    parseTree = docCore.sentence[0].parseTree
    item = getSentence(parseTree.child[0].child[1], "")
    skip = True
    result = ""
    qverb = None
    for child in parseTree.child[1].child:
        if qverb == None and "VB" in child.value:
            qverb = child.child[0].value
        if child.value == "NP":
            skip = False
        if not skip:
            result = result + " " + getSentence(child, "").strip()
    nlpCore.stop()
    return result.strip(), item, qverb

def splitWhatQuestion(question):
    nlpCore.start()
    docCore = nlpCore.annotate(question)
    parseTree = docCore.sentence[0].parseTree
    beginning = getSentence(parseTree.child[0].child[0], "").strip()
    nlpCore.stop()
    return beginning

'''
text2 = "Where did Bob go to on Sunday?"
text3 = "During the Old Kingdom , the king of Egypt ( not called the Pharaoh until the New Kingdom ) became a living god who ruled absolutely and could demand the services and wealth of his subjects ."
text4 = "When is the end of the month?"
text5 = "Why does the earth revolve?"
text6 = "What era is the right one?"
print(getPhrases(text6, "NP"))
print(getPhrases(text6, "NN"))
print(getPhrases(text6, "VP"))
print(getPhrases(text6, "PP"))
'''


