'''
Currently this only works for very clear X did Y sentences, and even then it still needs a little tweeking
I didn't upload the parser data because it was too large
'''

import nltk
nltk.download('state_union')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
import re

from nltk.parse.stanford import StanfordDependencyParser

dependancy_parser = StanfordDependencyParser(
    path_to_jar = 'stanford-parser-full-2018-10-17/stanford-parser.jar',
    path_to_models_jar = 'stanford-english-corenlp-2018-02-27-models.jar')

def generate_questions(sentence):
    parse_tree = dependancy_parser.raw_parse(sentence)
    dep = next(parse_tree)
    simplified = list(dep.triples())
    get_subjects(simplified, sentence)
    get_objects(simplified, sentence)
    
def get_subjects(pt, sentence):
    all_subjects = []
    for entry in pt:
        if entry[1] == 'nsubj' or entry[1] == 'nmod':
            subject = [entry[2][0]]
            for entry_pass_2 in pt:
                if entry_pass_2[0][0] == entry[2][0] and entry_pass_2[1] == 'compound':
                    subject.append(entry_pass_2[2][0])
            return_subject = ''

            for word in re.sub(',','',sentence).split():
                if word in subject:
                    return_subject += word
                    return_subject += ' '
            if entry[0][1][:2] == 'VB':
                print('What did {} {}?'.format(return_subject, entry[0][0]))
            all_subjects.append(return_subject)
    return all_subjects

def get_objects(pt, sentence):
    all_objects = []
    for entry in pt:
        if entry[1] == 'dobj' or entry[1] == 'nsubjpass':
            object_ = [entry[2][0]]
            for entry_pass_2 in pt:
                if entry_pass_2[0][0] == entry[2][0] and entry_pass_2[1] == 'compound':
                    object_.append(entry_pass_2[2][0])
            return_object = ''

            for word in re.sub(',','',sentence).split():
                if word in object_:
                    return_object += word
                    return_object += ' '
            if entry[0][1][:2] == 'VB':
                print('Who {} {}?'.format(entry[0][0], return_object))
            all_objects.append(return_object)
    return all_objects

if __name__ == '__main__':

	generate_questions('Jim drove his car')
	generate_questions('Jim drove his car, adn jane rode her bike')

