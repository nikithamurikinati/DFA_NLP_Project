#!/usr/bin/env python3
import argparse
import nltk
#import spacy
#from spacy import displacy
#from collections import Counter
import en_core_web_sm
import string
#import neuralcoref
#from nltk.corpus import wordnet
import corefRes
import os
import stanfordnlp
os.environ["CORENLP_HOME"] = "./corenlp"
from stanfordnlp.server import CoreNLPClient
import phrase_label
import tf_idf
import sys

class Sentence():
    def __init__(self, original, lemmatized):
        self.original = original
        self.lemmatized = lemmatized


class Parser():
    def __init__(self):
        self.weights = {
            "CC": 0.05, #not important
            "CD": 0.1, 
            "DT": 0.05, #not important
            "EX": 0.1,
            "FW": 0.1,
            "IN": 0.05, #not important
            "JJ": 0.3,
            "JJR": 0.3,
            "JJS": 0.3,
            "LS": 0.05, #not important
            "MD": 0.1,
            "NN": 0.6,
            "NNS": 0.6,
            "NNP": 0.75,
            "NNPS": 0.75,
            "PDT": 0.1,
            "POS": 0.15,
            "PRP": 0.2,
            "PRP$": 0.2,
            "RB": 0.3,
            "RBR": 0.3,
            "RBS": 0.3,
            "RP": 0.1,
            "TO": 0.05, #not important
            "UH": 0.05, #not important
            "VB": 0.6,
            "VBD": 0.4,
            "VBG": 0.4,
            "VBN": 0.4,
            "VBP": 0.4,
            "VBZ": 0.4,
            "WDT": 0.4,
            "WP": 0.4,
            "WP$": 0.4,
            "WRB": 0.4,
            ".": 0.05,
            "''": 0.05
        }

        self.ques_weights = {
            "CC": 0.05, #not important
            "CD": 0.1, 
            "DT": 0.05, #not important
            "EX": 0.1,
            "FW": 0.1,
            "IN": 0.05, #not important
            "JJ": 0.3,
            "JJR": 0.3,
            "JJS": 0.3,
            "LS": 0.05, #not important
            "MD": 0.1,
            "NN": 0.6,
            "NNS": 0.6,
            "NNP": 0.75,
            "NNPS": 0.75,
            "PDT": 0.1,
            "POS": 0.15,
            "PRP": 0.2,
            "PRP$": 0.2,
            "RB": 0.3,
            "RBR": 0.3,
            "RBS": 0.3,
            "RP": 0.1,
            "TO": 0.05, #not important
            "UH": 0.05, #not important
            "VB": 0.6,
            "VBD": 0.5,
            "VBG": 0.5,
            "VBN": 0.5,
            "VBP": 0.5,
            "VBZ": 0.5,
            "WDT": 0.4,
            "WP": 0.4,
            "WP$": 0.4,
            "WRB": 0.4,
            ".": 0.05,
            "''": 0.05
        }
        self.nlp = en_core_web_sm.load()
        self.nlpCore = CoreNLPClient(annotators=['tokenize','ssplit','pos','parse','coref', 'ner'],
        timeout=50000, endpoint='http://localhost:9001')
        self.nlpCore.start()
    
    def tokenize(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        tagged = nltk.pos_tag(tokens)
        topTen = len(tokens)//1.5
        mappedVals = []
        # Extracts more important words from text based on manual weight
        # dictionary
        for tag in tagged:
            mappedVals.append((tag[0], self.weights[tag[1]]))
        mostImp = []
        while len(mostImp) < topTen:
            bestWord = None
            bestScore = 0
            for word in mappedVals:
                if word[1] > bestScore:
                    bestWord = word
                    bestScore = word[1]
            mostImp.append(bestWord)
            mappedVals.remove(bestWord)
        topWords = []
        for word in mostImp:
            topWords.append(word[0])         


        # Using SpaCy for named entity recognition
        ner = self.nlp(sentence)
        lemmas = [(x.orth_,x.pos_, x.lemma_) for x in [y 
                                      for y
                                      in ner 
                                      if not y.is_stop and y.pos_ != 'PUNCT']]


        print([(X.text, X.label_) for X in ner.ents])
        print(lemmas)
        print(tagged)
        print(topWords)

    def question_parser(self, question):
        tokens = nltk.word_tokenize(question)
        tagged = nltk.pos_tag(tokens)
        topTen = len(tokens)//1.5
        mappedVals = []
        # Extracts more important words from text based on manual weight
        # dictionary
        for tag in tagged:
            mappedVals.append((tag[0], self.ques_weights[tag[1]]))
        mostImp = []
        while len(mostImp) < topTen:
            bestWord = None
            bestScore = 0
            for word in mappedVals:
                if word[1] > bestScore:
                    bestWord = word
                    bestScore = word[1]
            mostImp.append(bestWord)
            mappedVals.remove(bestWord)
        topWords = []
        for word in mostImp:
            topWords.append(word[0])
        
        # Using SpaCy for named entity recognition
        ner = self.nlp(question)

        #print([(X.text, X.label_) for X in ner.ents])
        #print(tagged)
        #print(topWords)
        return topWords
    
    def preprocess_text(self, text):
        # Replaces text with lemmatized form when found
        sentences = []
        start = 0
        #add in coreference resolution
        docCore = self.nlpCore.annotate(text)
        text = corefRes.modifyText(text, docCore)
        words = text.split()
        docCore = self.nlpCore.annotate(text)
        '''
        for i in range(len(words)):
            if words[i][-1] in string.punctuation and (i == len(words)-1 or words[i+1][0].isupper()):
                sentences.append(" ".join(words[start:i+1]))
                start = i + 1
        '''
        sentences_lemm = []
        for sentence in docCore.sentence:
            sentence = stanfordnlp.server.to_text(sentence)
            sentence_final = []
            for word in sentence.split(" "):
                if word == "-RRB-":
                    word = ")"
                elif word == "-LRB-":
                    word = "("
                sentence_final.append(word)
            sentence = " ".join(sentence_final)
            sentences_lemm.append(Sentence(sentence, self.lemmatize(sentence)))
        return sentences_lemm

    def lemmatize(self, sentence):
        lemmas = [(x.orth_,x.pos_, x.lemma_) for x in [y 
                                      for y
                                      in self.nlp(sentence) 
                                      if not y.is_stop and y.pos_ != 'PUNCT']]
        lemma_sent = []
        lemma_i = 0
        for word in sentence.split():
            if len(word) == 0: continue
            if word[-1] in string.punctuation:
                word = word[:-1]
            if len(word) == 0: continue
            if word[0] in string.punctuation:
                word = word[1:]
            if len(word) == 0: continue
            if lemmas != [] and (lemma_i >= len(lemmas) or word != lemmas[lemma_i][0]):
                lemma_sent.append(word)
            elif len(lemmas) != 0:
                lemma_sent.append(lemmas[lemma_i][2])
                lemma_i += 1
        return " ".join(lemma_sent)

    def extract_sentences_keyword(self, text, question):
        # Extracts sentences that most likely contain the answer
        # TODO: make question into class sentence before inputting
        question = self.lemmatize(question)
        keywords = self.question_parser(question)
        process_text = self.preprocess_text(text)
        result = []
        best = None
        bestRatio = 0
        for sentence in process_text:
            count = 0
            seen = set()
            for word in sentence.lemmatized.split():
                if word in keywords:
                    count += 1
                    if word not in seen:
                        seen.add(word)
                else:
                    # similarity checking
                    for kword in keywords:
                        try:
                            w1 = wordnet.synsets(word)[0]
                            w2 = wordnet.synsets(kword)[0]
                            if w1.wup_similarity(w2) > 0.9:
                                count += 1
                                seen.add(word)
                        except:
                            continue
            bigram_count = self.count_bigram(sentence, question)
            trigram_count = self.count_trigram(sentence, question)
            unique_count = len(seen)
            final_count = count*.3 + bigram_count*.4 + unique_count*.3 + trigram_count*.2
            ## Really need a new metric of "best"
            ## look into tf-idf
            if final_count >= (len(keywords)*.6+len(question.split(" "))*2*.4)*.6:
                result.append(sentence.original)
                ratio = count/(len(sentence.original.split(" ")))
                if ratio > bestRatio:
                    bestRatio = ratio
                    best = sentence.original
        # Comment out these two lines to revert to previous "best" measure
        result = list(set(result))
        best = tf_idf.compareToOriginal(question, result)
        
        return result, best, len(result)

    def count_bigram(self, sentence, question):
        # counts the number of bigram occurences from the question in the sentence
        count = 0
        question = "START " + question
        q_words = question.split(" ")
        s_words = sentence.lemmatized.split(" ")
        for i in range(1, len(q_words)):
            # Possibily add in similarity checking here too
            for j in range(len(s_words)):
                if j == 0:
                    count += question.count("START " + s_words[0])
                else:
                    count += question.count(s_words[j-1] + " " + s_words[j])
        return count

    def count_trigram(self, sentence, question):
        count = 0
        question = "START START " + question
        q_words = question.split(" ")
        s_words = sentence.lemmatized.split(" ")
        for i in range(2, len(q_words)):
            for j in range(len(s_words)):
                # Possibily add in similarity checking here too
                if j == 0:
                    prev1 = "START"
                    prev2 = "START"
                elif j == 1:
                    prev1 == "START"
                    prev2 = s_words[j-1]
                else:
                    prev1 = s_words[j-2]
                    prev2 = s_words[j-1]
                count += question.count(prev1 + " " + prev2 + " " + s_words[j])
        return count

    def get_possible_answers(self, text, question):
        keywords = self.question_parser(self.lemmatize(question))
        #print(keywords)
        return self.extract_sentences_keyword(text, question)

    def getWhereAnswer(self, question, bestAnswer):
        main,qverb = phrase_label.splitQuestion(question)
        phrases = phrase_label.getNounVerbPhrasePairs(bestAnswer)
        #print(phrases)
        words = question.split(" ")
        if len(phrases) == 1:
            return f'{main} {qverb} in {phrases[0][0]}'
        possible = []
        for phrase in phrases:
            flag = True
            noun, verb = phrase
            for w in noun.split(" "):
                #replace with stop word check instead
                if w in main and w not in {"the", "a", "an", "these", "that", "of"}:
                    flag = False
            if flag:
                possible.append(phrase)
        if len(possible) == 1:
            return f'{main} {qverb} in {possible[0][0]}'
        elif len(possible) == 0:
            possible = phrases
        np = self.bestNounPhrase(possible, question)
        print(main, qverb, np)
        return f'{main} {qverb} in {np}'

    def bestNounPhrase(self, possible, question):
        bestNoun = None
        bestVerb = None
        bestScore = 0
        for noun, verb in possible:
            count = 0
            question_lem = self.lemmatize(question)
            verb_lem = self.lemmatize(verb)
            keywords = self.question_parser(question_lem)
            for word in verb_lem.split(" "):
                if word in keywords:
                    count += 1
            ratio = count/len(verb_lem.split(" "))
            if ratio > bestScore:
                bestScore = ratio
                bestNoun, bestVerb = noun, verb
        return bestNoun

    def getWhoAnswer(self, question, bestAnswer):
        #print("in Who fn ", bestAnswer)
        phrases = phrase_label.getNounVerbPhrasePairs(bestAnswer)
        replace = phrase_label.splitWhatQuestion(question)
        #print(phrases)
        words = question.split(" ")
        #print('phrases ', phrases)
        if len(phrases) == 1:
            return question.replace(replace, phrases[0][0], 1)
        possible = []
        for phrase in phrases:
            flag = True
            noun, verb = phrase
            for w in noun.split(" "):
                #replace with stop word check instead
                if w in question and w not in {"the", "a", "an", "these", "that", "of"}:
                    flag = False
            if flag:
                possible.append(phrase)
        #print('possible', possible)
        if len(possible) == 1:
            return question.replace(replace, possible[0][0], 1)
        elif len(possible) == 0:
            possible = phrases
        np = self.bestNounPhrase(possible, question)
        return question.replace(replace, np, 1)

    def getWhichAnswer(self, question, bestAnswer):
        main, item, qverb = phrase_label.splitWhichQuestion(question)
        phrases = phrase_label.getNounVerbPhrasePairs(bestAnswer)
        #print(phrases)
        words = question.split(" ")
        if len(phrases) == 1:
            rest = " ".join(words[1:])
            if rest[-1] == "?":
                rest = rest[:-1] + "."
            return phrases[0][0] + " " + rest
        possible = []
        for phrase in phrases:
            flag = True
            noun, verb = phrase
            for w in noun.split(" "):
                #replace with stop word check instead
                if w in main and w not in {"the", "a", "an", "these", "that", "of"}:
                    flag = False
            if flag:
                possible.append(phrase)
        if len(possible) == 1:
            rest = " ".join(words[1:])
            if rest[-1] == "?":
                rest = rest[:-1] + "."
            return possible[0][0] + " " + rest
        elif len(possible) == 0:
            possible = phrases
        np = self.bestNounPhrase(possible, question)
        return f'{np} {qverb} {main}'

    def yesNoAnswer(self, question, bestAnswer):
        # Really basic negation checking right now
        q = "not" in question
        a = "not" in bestAnswer
        if (q and a) or (not q and not a):
            return "Yes."
        else:
            return "No."

    def getWhyAnswer(self, question, bestAnswer):
        if "because" in bestAnswer:
            words = bestAnswer.split(" ")
            i = words.index("because")
            return " ".join(words[i+1:])
        elif "since" in bestAnswer:
            words = bestAnswer.split(" ")
            i = words.index("since")
            return " ".join(words[i+1:])
        elif "due to" in bestAnswer:
            words = bestAnswer.split(" ")
            i = words.index("due")
            return " ".join(words[i+2:])
        else:
            return bestAnswer

    def getWhenAnswer(self, question, bestAnswer):
        pass

    def getAnswerBeginning(self, question, bestAnswer):
        words = question.split(" ")
        if words[0].lower() == "where":
            #return self.getWhereAnswer(question, bestAnswer)
            return bestAnswer
        elif words[0].lower() == "why":
            return self.getWhyAnswer(question, bestAnswer)
        elif words[0].lower() == "how":
            return bestAnswer
        elif words[0].lower() == "who":
            #print("WENT HERE", bestAnswer)
            return self.getWhoAnswer(question, bestAnswer)            
        elif words[0].lower() == "what":
            return self.getWhoAnswer(question, bestAnswer)
        elif words[0].lower() == "does":
            return self.yesNoAnswer(question, bestAnswer)
        elif words[0].lower() == "is":
            return self.yesNoAnswer(question, bestAnswer)
        elif words[0].lower() == "which":
            return self.getWhichAnswer(question, bestAnswer)
        elif words[0].lower() == "when":
            pass
        else:
            return bestAnswer

    def getBestAnswer(self, text, question):
        possible, best, length =  self.extract_sentences_keyword(text, question)
        #print(best )
        answer =  self.getAnswerBeginning(question, best).capitalize()
        if answer[-1] == "?":
            return answer[:-1] + "."
        return answer



parser = Parser()
#print(parser.tokenize("Bob went to Central Park on Wednesday."))
#print(parser.question_parser("Where did Bob go on Wednesday?"))

def readFile(path):
    with open(path, "rt") as f:
        return  f.read()

def parseFile(contents):
    lines = contents.splitlines()
    breakSpot = 0
    for i in range(len(lines)):
        line = lines[i]
        if len(line) > 1:
            if line[-1] != ".":
                breakSpot = i
                break
    lines = lines[breakSpot + 1:]
    return "\n".join(lines)

def getQuestions(path):
    contents = readFile(path)
    results = []
    for question in contents.splitlines():
        results.append(question)
    return results

def getAnswers(text, questions):
    text = parseFile(readFile(text))
    questions = getQuestions(questions)
    parse = Parser()
    result = ""
    for question in questions:
        result += parse.getBestAnswer(text, question)
        result += '\n'
    result.strip()
    sys.stdout.write(result)
                


#text = readFile("./Development_data/set1/a1.txt")

#print(parser.get_possible_answers(text, "Which era do historians regard as 'written in stone'?"))
#print(parser.getBestAnswer(text, "Who was not called the Pharaoh until the New Kingdom?"))
#print(parser.get_possible_answers(text, "Who asked for the wealth of his people?"))
#print(parser.getBestAnswer(text, "Who asked for the wealth of his people?"))
'''
print(parser.getBestAnswer(text, "What is the period in the third millennium ( c. 2686-2181 BC ) also known as the ' Age of the Pyramids ' or ' Age of the Pyramid Builders ' ?"))
print(parser.getBestAnswer(text, "What country attained its first continuous peak of civilization - the first of three so-called ' Kingdom ' periods ( followed by the Middle Kingdom and New Kingdom )?"))
print(parser.getBestAnswer(text, "Egypt attained its what of civilization - the first of three so-called ' Kingdom ' periods ( followed by the Middle Kingdom and New Kingdom ) ?"))
print(parser.getBestAnswer(text, "Not only was the last king of what , but the ' capital ' , the royal residence , remained at Ineb-Hedg , the Ancient Egyptian name for Memphis ?"))
print(parser.getBestAnswer(text, "What is the revolutionary change in architecture accompanied by the effects on Egyptian society and economy of large-scale building projects ?"))
print(parser.getBestAnswer(text, "The basic justification for a separation between the two periods is the revolutionary change in what ?"))
print(parser.getBestAnswer(text, "What is most commonly regarded as the period from the Third Dynasty through to the Sixth Dynasty ( 2686 - 2181 BC ) ?"))

print(parser.getBestAnswer(text, "Who also include the Memphite Seventh and Eighth Dynasties in the Old Kingdom as a continuation of the administration centralized at Memphis ?"))
print(parser.getBestAnswer(text, "Egyptologists also include what as a continuation of the administration centralized at Memphis ?"))
print(parser.getBestAnswer(text, "Egyptologists also include the Memphite Seventh and Eighth Dynasties in the Old Kingdom as what ?"))
print(parser.getBestAnswer(text, "What was followed by a period of disunity and relative cultural decline referred to by Egyptologists as the First Intermediate Period ?"))
print(parser.getBestAnswer(text, "The Old Kingdom was followed by what ?"))
print(parser.getBestAnswer(text, "The Old Kingdom was followed by a period of disunity and relative cultural decline referred to by Egyptologists as what ?"))
print(parser.getBestAnswer(text, "What was initiated at Saqqara under Djoser reign ?"))

print(parser.getBestAnswer(text, "A new era of building was initiated at what ?"))
print(parser.getBestAnswer(text, "After Khufu 's death , who may have quarrelled ?"))

print(parser.getBestAnswer(text, "To these ends , over a period of time , what adopted a limited repertoire of standard types and established a formal artistic canon?"))
'''
#print(parser.getBestAnswer("Bob went to the store.", "Where did Bob go?"))

#print(parser.getBestAnswer(text, "What was the population of the Indus Valley Civilization?"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('text', help='Path to corpus text')
    parser.add_argument('questions', help='Path to questions file')
    args = parser.parse_args()
    text = args.text
    questions = args.questions
    getAnswers(text, questions)

