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
    categ = beginning.split(" ")[1:]
    return beginning.strip(), " ".join(categ)



#getNounVerbPhrasePairs("The Old Kingdom is the period in the third millennium (c. 2686-2181 BC) also known as the 'Age of the Pyramids' or 'Age of the Pyramid Builders' as it includes the great 4th Dynasty when King Sneferu perfected the art of pyramid building and the pyramids of Giza were constructed under the kings Khufu, Khafre, and Menkaure.")