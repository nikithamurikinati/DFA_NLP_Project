import spacy
from spacy.pipeline import Sentencizer
from spacy.lang.en import English
from spacy import displacy
import en_core_web_sm
import en_core_web_md
import en_core_web_lg
        
def getNounPhrases(sentence):

    nlp = en_core_web_sm.load()
    doc = nlp(sentence)
    '''
    result = []
    for np in doc.noun_chunks:
        result.append(np.text)
    '''
    nlp = en_core_web_sm.load()
    doc = nlp(sentence)
    subtrees = []
    for token in doc:
        if token.pos_ == "NOUN" or token.pos_ == "PROPN" or token.pos_ == "NUM":
            subtrees.append(token.subtree)
    result = []
    for subtree in subtrees:
        phrase = " ".join([t.text for t in subtree])
        result.append(phrase)
    return list(result)

def getNounChunks(sentence):
    nlp = en_core_web_sm.load()
    doc = nlp(sentence)
    result = {np.text
    for nc in doc.noun_chunks
    for np in [
        nc, 
        doc[
        nc.root.left_edge.i
        :nc.root.right_edge.i+1]]}  
    return list(result)

def getVerbPhrases(sentence):
    nlp = en_core_web_sm.load()
    doc = nlp(sentence)
    subtrees = []
    for token in doc:
        if token.pos_ == "VERB" or (token.pos_ == "AUX" and (token.i + 1) < (len(doc)-1) and doc[token.i+1].pos_ != "VERB"):
            subtrees.append(token.subtree)
    result = []
    for subtree in subtrees:
        phrase = " ".join([t.text for t in subtree])
        result.append(phrase)
    for i in range(len(result)):
        for phrase in result:
            if result[i] != phrase: 
                result[i] = result[i].replace(phrase, "")
    return result
                

    #displacy.render(doc, style="dep")


def getNounVerbPhrasePairs(sentence):
    nouns = getNounPhrases(sentence)
    verbs = getVerbPhrases(sentence)
    result = []
    for noun in nouns:
        for verb in verbs:
            if noun in verb:
                result.append((noun, verb.replace(noun, "")))
                break
    return result

'''
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
'''
def splitWhatQuestion(question):
    words = question.split(" ")
    beginning = ""
    nlp = nlp = en_core_web_sm.load()
    doc = nlp(question)
    verb = ""
    verbFlag = False
    rest = ""
    for i in range(len(words)):
        if not verbFlag:
            if doc[i].pos_ == "VERB" or doc[i].pos_ == "AUX":
                verbFlag = True
                verb += (doc[i].text + " ")
            else:
                beginning += (doc[i].text + " ")
        else:
            if doc[i].pos_ != "VERB" and doc[i].pos_ != "AUX":
                rest = " ".join(words[i:])
                break
            verb += (doc[i].text + " ")
    rest = rest.replace("?", "")
    return beginning.strip()



#getNounVerbPhrasePairs("The Old Kingdom is the period in the third millennium (c. 2686-2181 BC) also known as the 'Age of the Pyramids' or 'Age of the Pyramid Builders' as it includes the great 4th Dynasty when King Sneferu perfected the art of pyramid building and the pyramids of Giza were constructed under the kings Khufu, Khafre, and Menkaure.")