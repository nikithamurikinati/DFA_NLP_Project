import spacy
import string
from nltk.corpus import wordnet

def readFile(path):
    """
    Reads text, extracts valid sentences, returns single-line string of sentences
    """
    with open(path, "rt", encoding="utf8") as f:
        s =  f.read()
        text = ""
        for line in s.splitlines():
            line = line.strip()
            if line == "" or line[-1] != ".":
                continue
            text += " " + line
        return text

def cleanSentence(text):
    """
    Takes in 1 sentence as a string, removes parenthesis
    """
    ogText = text
    pairs = []
    shift = 0
    if text.count("(") != text.count(")"):
        return None
    while "(" in text:
        i = shift + text.find("(")
        j = shift + text.find(")")
        pairs.append((i,j))
        text = text[j+1:]
        shift = j + 1
    if pairs != []:
        text = ogText[:pairs[0][0]-1]
        for i in range(1, len(pairs)):
            prev = pairs[i-1][1]
            new = pairs[i][0]
            text += ogText[min(prev+2, len(ogText)):new]
        text += ogText[pairs[-1][1]+1:]
    return text

def extractSentences(nlp, text):
    """
    Takes a single-line string of sentences
    Makes a 1D list of cleaned sentences
    It also finds clauses that can stand as independent sentences
    """
    allSentences = []
    doc = nlp(text)
    for sent in doc.sents:
        thisSentence = ""
        marks = []
        for word in sent:
            thisSentence += word.text + " "
            if marks != []:
                for i in range(len(marks)):
                    marks[i] += word.text + " "
            if word.dep_ == "mark" and word.text in ["as", "that", "because", "since"]:
                marks.append("")
                if word.text != "that":
                    possibleSentence = (thisSentence.strip())[:-thisSentence.strip()[::-1].find(" ")] + "."
                    cleaned = cleanSentence(possibleSentence)
                    if cleaned != None:
                        allSentences.append(cleaned)
        thisSentence = thisSentence.strip()
        cleaned = cleanSentence(thisSentence)
        if cleaned != None:
            allSentences.append(cleaned)
        for s in marks:
            s = s.strip()
            s = s[0].upper() + s[1:]
            cleaned = cleanSentence(s)
            if cleaned != None:
                allSentences.append(cleaned)
    return allSentences

def invertWhy(nlp, question):
    #takes a sentence starting with "why", tries to invert to make valid
    originalSentence = question[4:]
    d = nlp(originalSentence)
    listSentence = [word.text for word in d]
    if listSentence == []:
        return None
    listSentence = [listSentence[0].lower()] + listSentence[1:]
    commaIndex = len(d)
    for i in range(len(d)):
        if d[i].text == ",":
            commaIndex = i
        if d[i].dep_ == "ROOT":
            if i > commaIndex:
                return None
            if "VB" not in d[i].tag_:
                return None
            prev = None
            for j in range(1, i):
                if "aux" in d[j].dep_:
                    prev = j
                    break
            if prev != None:
                newSentence = ["Why"] + [d[prev].text] + listSentence[:prev] + listSentence[prev+1:]
                if ":" in newSentence:
                    ind = newSentence.index(":")
                    return " ".join(newSentence[:ind])
                if ";" in newSentence:
                    ind = newSentence.index(";")
                    return " ".join(newSentence[:ind])
                return " ".join(newSentence)
            if d[i].lemma_ == "be":
                newSentence = ["Why"] + [d[i].text] + listSentence[:i] + listSentence[i+1:]
                if ":" in newSentence:
                    ind = newSentence.index(":")
                    return " ".join(newSentence[:ind])
                if ";" in newSentence:
                    ind = newSentence.index(";")
                    return " ".join(newSentence[:ind])
                return " ".join(newSentence)
            if "N" in d[i].tag_ or "D" in d[i].tag_:
                #past tense
                newSentence = ["Why did"] + listSentence[:i] + [d[i].lemma_] + listSentence[i+1:]
                if ":" in newSentence:
                    ind = newSentence.index(":")
                    return " ".join(newSentence[:ind])
                if ";" in newSentence:
                    ind = newSentence.index(";")
                    return " ".join(newSentence[:ind])
                return " ".join(newSentence)
            if d[i].tag_ == "VBP":
                word = "do"
            else:
                word = "does"
            newSentence = ["Why", word] + listSentence[:i] + [d[i].lemma_] + listSentence[i+1:]
            if ":" in newSentence:
                ind = newSentence.index(":")
                return " ".join(newSentence[:ind])
            if ";" in newSentence:
                ind = newSentence.index(";")
                return " ".join(newSentence[:ind])
            return " ".join(newSentence)
    return None

def binaryQuestions(nlp, question):
    """
    Returns yes/no questions
    """
    q = []
    question = "Why " + question
    inverted = invertWhy(nlp, question)
    if inverted == None:
        return []
    inverted = inverted[4:]
    final = (inverted[0].upper() + inverted[1:-1]).strip() + "?"
    d = nlp(final)
    punctIndex = len(d)
    for i in range(len(d)):
        if d[i].dep_ == "ROOT":
            if punctIndex < i:
                return []
            break
        if d[i].text == ",":
            punctIndex = i
    return [final]

def whyQuestions(nlp, sentence):
    """
    Returns a list of why questions
    Checks for marker
    """
    d = nlp(sentence)
    q = ""
    for i in range(len(d)):
        word = d[i]
        if word.dep_ == "mark" and word.text in ["as", "because", "since"]:
            if "VB" in d[i+1].tag_:
                return []
            if "CONJ" in d[i-1].pos_:
                q = q[:-len(d[i-1].text)-1]
            q = "Why " + q
            new = invertWhy(nlp, q)
            if new == None:
                return []
            new = new.strip()
            if new[-1] in string.punctuation and new[-1] != '"':
                new = new[:-2]
            w1 = new.split(" ")[-1]
            try:
                w1 = wordnet.synsets(w1)[0]
                for w2 in ["probably", "roughly", "mainly", "simply", "largely", "definitely", "likely", "surely", "only", "mostly", "usually"]:
                    w2 = wordnet.synsets(w2)[0]
                    if w1.wup_similarity(w2) > 0.9:
                        new = " ".join(new.split(" ")[:-1])
                        if new.split(" ")[-1] in ["and", "but", "or"]:
                            new = " ".join(new.split(" ")[:-1]) 
                        break
            except:
                pass
            new = new.strip()
            if new[-1] in string.punctuation and new[-1] != '"':
                new = new[:-1]
            new = new.strip()
            return [new + "?"]
        q += word.text + " "
    return []

def whomQuestions(nlp, sentence):
    """
    Returns a list of whom questions
    """
    d = nlp(sentence)
    q = []
    for chunk in d.noun_chunks:
        if chunk.root.head.text in ["at", "in"]:
            continue
        dep = chunk.root.dep_
        if "nsubj" in dep or "obj" in dep:
            if chunk.root.ent_type_ in ["PERSON", "NORP", "ORG"]:
                i = sentence.find(chunk.text + " ")
                j = sentence.find(" " + chunk.root.head.text + " ")
                if i >= j:
                    if i == 0: word = "Whom"
                    else: word = "whom"
                    q.append(sentence[:i] + word + sentence[i+len(chunk.text):-1] + "?")
    return q

def whenQuestions(nlp, sentence):
    d = nlp(sentence)
    q = []
    for chunk in d.noun_chunks:
        dep = chunk.root.dep_
        if "nsubj" in dep or "obj" in dep:
            if chunk.root.ent_type_ in ["DATE"]:
                i = sentence.find(" " + chunk.text + " ")
                if len(chunk.root.text) == 4 and chunk.root.text.isdigit():
                    q.append(sentence[:i] + "what year" + sentence[i+len(chunk.text):])
    return q

def getAllQuestions(path):
    nlp = spacy.load("en_core_web_md")
    text = readFile(path)
    docMain = nlp(text)
    allSentences = extractSentences(nlp, text)
    questions = []
    for sentence in allSentences:
        questions += whyQuestions(nlp, sentence)
        #questions += whomQuestions(nlp, sentence)
        #questions += binaryQuestions(nlp, sentence)
        #questions += whenQuestions(nlp, sentence)
    return questions

def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)

writeFile("tryingShit.txt", '\n'.join(getAllQuestions("Development_data/set1/a7.txt")))


