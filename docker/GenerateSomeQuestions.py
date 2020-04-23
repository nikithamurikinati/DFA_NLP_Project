import spacy
import string
from nltk.corpus import wordnet
import random

"""
GenerateSomeQuestions.py - Mihika Bairathi
Things to download: spacy, en_core_web_md (a spacy model), nltk
Instructions: Run getAllQuestions(path, numQuestions) to get top n questions
Output format: String of n questions, separated by newline
"""

def readFile(path):
    """
    Read text, extract valid sentences, return single-line string of sentences
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
    return result.strip()

def cleanSentence(text):
    """
    Takes a sentence and removes all parenthesis
    """
    ogText = text
    try:
        pattern = r'\(.*?\)'
        if (text.count("(") != text.count(")")) or \
            text.index("(") > text.index(")"):
            return None
        while "(" in text:
            i = text.find("(")
            i2 = text[i+1:].find("(") + i + 1
            j = text.find(")")
            if j == -1:
                return None
            if i2 < j and i2 != i:
                return None
            text = text[j+1:]
        return collapseWhitespace(re.sub(pattern, '', ogText))
    except:
        return ogText

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
            if word.dep_ == "mark" and word.text in \
                ["as", "that", "because", "since"]:
                marks.append("")
                if word.text != "that":
                    try:
                        ts = thisSentence
                        possibleSentence = ts.strip()[:-ts.strip()[::-1].find(" ")] + "."
                        cleaned = cleanSentence(possibleSentence)
                        if cleaned != None:
                            allSentences.append(cleaned)
                    except:
                        pass
        thisSentence = thisSentence.strip()
        cleaned = cleanSentence(thisSentence)
        if cleaned != None:
            allSentences.append(cleaned)
        try:
            for s in marks:
                s = s.strip()
                s = s[0].upper() + s[1:]
                cleaned = cleanSentence(s)
                if cleaned != None:
                    allSentences.append(cleaned)
        except:
            pass
    return allSentences

def invertWhy(nlp, question):
    """
    takes a sentence starting with "why", tries to invert to make valid
    """
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
                newSentence = ["Why"] + [d[prev].text] + listSentence[:prev] \
                    + listSentence[prev+1:]
                if ":" in newSentence:
                    ind = newSentence.index(":")
                    return " ".join(newSentence[:ind])
                if ";" in newSentence:
                    ind = newSentence.index(";")
                    return " ".join(newSentence[:ind])
                return " ".join(newSentence)
            if d[i].lemma_ == "be":
                newSentence = ["Why"] + [d[i].text] + listSentence[:i] \
                    + listSentence[i+1:]
                if ":" in newSentence:
                    ind = newSentence.index(":")
                    return " ".join(newSentence[:ind])
                if ";" in newSentence:
                    ind = newSentence.index(";")
                    return " ".join(newSentence[:ind])
                return " ".join(newSentence)
            if "N" in d[i].tag_ or "D" in d[i].tag_:
                #past tense
                newSentence = ["Why did"] + listSentence[:i] + [d[i].lemma_] \
                    + listSentence[i+1:]
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
            newSentence = ["Why", word] + listSentence[:i] + [d[i].lemma_] \
                + listSentence[i+1:]
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
            if i != 0:
                if "VB" in d[i].pos_:
                    num = random.randint(0, 1)
                    if num == 1:
                        cofinal = final + " "
                        j = cofinal.find(" " + d[i].text + " ")
                        final = final[:j] + " not" + final[j:]
            break
        if d[i].text == ",":
            punctIndex = i
    if final.count(" ") < 9 or final.count(" ") > 14:
        return []
    if final[0] in ['.', ',', '?', ';', '-']:
        final = final[1:]
        final = final.strip()
    if final[:-1].strip()[-1] in [".", ";"]:
        final = final[:-1].strip()[:-1].strip() + "?"
    dTemp = nlp(final)
    if "VB" in dTemp[1].tag_:
        return []
    return [final]

def whyQuestions(nlp, sentence, d):
    """
    Returns a list of why questions
    """
    q = ""
    for i in range(len(d)):
        word = d[i]
        if word.dep_ == "mark" and word.text in ["because", "since", "as"]:
            if i < len(d)-1 and "VB" in d[i+1].tag_:
                return []
            if i > 0 and "CONJ" in d[i-1].pos_:
                q = q[:-len(d[i-1].text)-1]
            q = "Why " + q
            new = invertWhy(nlp, q)
            if new == None:
                return []
            new = new.strip()
            if new[-1] in string.punctuation and new[-1] not in ['\"', ')']:
                new = new[:-1]
                new = new.strip()
            w1 = new.split(" ")[-1]
            try:
                w1 = wordnet.synsets(w1)[0]
                for w2 in ["probably", "roughly", "mainly", \
                           "simply", "largely", "definitely", \
                           "likely", "surely", "only", "mostly", \
                           "usually", "particularly"]:
                    w2 = wordnet.synsets(w2)[0]
                    if w1 == w2 or w1.wup_similarity(w2) > 0.9:
                        new = " ".join(new.split(" ")[:-1])
                        if new.split(" ")[-1] in ["and", "but", "or"]:
                            new = " ".join(new.split(" ")[:-1]) 
                        break
            except:
                pass
            new = new.strip()
            if new[-1] in string.punctuation and new[-1] not in  ['\"', ')']:
                new = new[:-1]
            new = new.strip()
            if new.count(" ") < 4 or new.count(" ") > 20:
                return []
            if new[0] in ['.', ',', '?', ';', '-']:
                new = new[1:]
                new = new.strip()
            return [new + "?"]
        q += word.text + " "
    return []

def whichTimeQuestions(nlp, sentence, d):
    """
    Finds words with "TIME" as NER tag
    """
    sentence = " " + sentence + " "
    q = []
    rootIndex = None
    for i in range(len(d)):
        if d[i].dep_ == "ROOT":
            rootIndex = i
    if rootIndex == None:
        return []
    for chunk in d.noun_chunks:
        dep = chunk.root.dep_
        if "nsubj" in dep or "obj" in dep:
            if chunk.root.ent_type_ in ["DATE"]: 
                i = sentence.find(" " + chunk.text + " ")
                if i == -1 or chunk.text.strip().count(" ") < 2:
                    continue
                try:
                    word = "which "
                    if chunk.root.text.lower() in ["decades", "years", \
                        "months", "days", "weeks", "weekends"]: 
                        word = "how many "
                    phrase = word + chunk.root.text
                    st = sentence
                    final = (st[:i+1] + phrase + st[i+len(chunk.text)+1:]).strip()
                    if final[-1] in ['.', ',', '?', ';', '-']:
                        final = final[:-1]
                        final = final.strip()
                        if final[-1] in ['.', ',', '?', ';', '-']:
                            final = final[:-1]
                            final = final.strip()
                    if final.count(" ") < 6 or final.count(" ") > 17:
                        pass
                    else:
                        if final[0] in ['.', ',', '?', ';', '-']:
                            final = final[1:]
                            final = final.strip()
                        final = final[0].upper() + final[1:]
                        q.append(final+"?")
                except:
                    pass
    return q

def getAllQuestions(path, numQuestions):
    """
    Takes a path to a file, and parses through it
    Comes up with why, time, and binary questions
    Returns the top n questions, n = numQuestions
    """
    nlp = spacy.load("en_core_web_md")
    text = readFile(path)
    allSentences = extractSentences(nlp, text)
    why = set()
    whom = set()
    whichTime = set()
    binary = set()
    for sentence in allSentences:
        #get questions
        d = nlp(sentence)
        whyQuestion = whyQuestions(nlp, sentence, d)
        timeQuestion = whichTimeQuestions(nlp, sentence, d)
        binaryQuestion = binaryQuestions(nlp, sentence)
        #why questions
        for q in whyQuestion:
            symbols = ['\"', "(", ")"]
            notGood = False
            for s in symbols:
                if q.count(s)%2 == 1:
                    notGood = True
            if q.count(" ") > 11 or notGood or q.count(" ") < 6 \
                or q[0] in string.punctuation:
                continue
            why.add(q)
        #time questions
        for q in timeQuestion:
            symbols = ['\"', "(", ")"]
            notGood = False
            for s in symbols:
                if q.count(s)%2 == 1:
                    notGood = True
            if q.count(" ") > 11 or notGood or q.count(" ") < 6 \
                or q[0] in string.punctuation:
                continue
            whichTime.add(q)
        #binary questions
        for q in binaryQuestion:
            symbols = ['\"', "(", ")"]
            notGood = False
            for s in symbols:
                if q.count(s)%2 == 1:
                    notGood = True
            if notGood or q.count(" ") < 6 or q[0] in string.punctuation:
                continue
            binary.add(q)
    #curate final lists
    finalWhy = list(why)
    finalTime = list(whichTime)
    finalWhyTime = []
    tempLen = min(len(finalWhy), len(finalTime))
    for i in range(tempLen):
        finalWhyTime += [finalWhy[i], finalTime[i]]
    finalWhyTime += finalWhy[tempLen:] + finalTime[tempLen:]
    finalBinary = list(binary)
    finalList = finalBinary[:1] + finalWhyTime + finalBinary[1:]
    finalList = finalList[:numQuestions]
    return '\n'.join(finalList)

def testing():
    #run this function to test the program
    for i in range(1, 5):
        for j in range(1, 11):
           print(getAllQuestions(f'data/set{i}/a{j}.txt', 6))
           print("---------------")