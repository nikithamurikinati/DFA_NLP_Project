import nltk
nltk.download('state_union')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
import re
from nltk.parse.stanford import StanfordParser, StanfordDependencyParser
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.corpus import wordnet as wn
import os
from nltk.stem.wordnet import WordNetLemmatizer
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

MAPPING = {'ORG': 'Who', 'PERSON': 'Who', 'DATE': 'When', 'LOCATION': 'Where'}
#os.environ["CLASSPATH"]= "/usr/local/stanford-models/stanford-postagger-full-2016-10-31/:usr/local/stanford-models/stanford-ner-2016-10-31/:/usr/local/stanford-models/stanford-parser-full-2016-10-31/"
#os.environ["STANFORD_MODELS"]= "/usr/local/stanford-models/stanford-postagger-full-2016-10-31/models:/usr/local/stanford-models/stanford-ner-2016-10-31/classifiers"

dependancy_parser = StanfordDependencyParser(
    path_to_jar = 'stanford-parser-full-2018-10-17/stanford-parser.jar',
    path_to_models_jar = 'stanford-english-corenlp-2018-02-27-models.jar')

def search_for_dep_with_pos(tree, deps, collection, pos_list):
    for pos in pos_list:
        if pos in deps.keys():
            for j in deps[pos]:
                collection.append((tree.nodes[j]['word'], pos, compile_clause(tree, tree.nodes[j])))
    return collection

def clean_search_results(search_results):
    new_results = []
    for guess in search_results:
        valid = True
        for pos in guess.keys():
            if guess[pos] == []:
                valid = False
        if valid:
            new_results.append(guess)
    return new_results

##############################################################################
#DIFFERENT RELATIONS TO BE PLUGGED INTO TEMPLATES
##############################################################################

def get_verb_subj_dobj(tree):
    results = []
    for i in range(len(tree.nodes)):
        if tree.nodes[i]['tag'] is not None and tree.nodes[i]['tag'][:2] == 'VB':
            subject = []
            dir_object = []
            verb = []
            verb.append((tree.nodes[i]['word'], tree.nodes[i]['tag'], compile_clause(tree,tree.nodes[i])))
            deps = tree.nodes[i]['deps']
            subject = search_for_dep_with_pos(tree, deps, subject, ['nsubj', 'nsubjpass'])
            dir_object = search_for_dep_with_pos(tree, deps, dir_object, ['nmod', 'dobj'])
            results.append({"Subject": subject, "Direct Object": dir_object, "Verb": verb})
    return clean_search_results(results)

def get_noun_descriptor(tree):
    results = []
    for i in range(len(tree.nodes)):
        subject = []
        descriptor = []
        verb = []
        case = [('', 'ph')]
        deps = tree.nodes[i]['deps']
        descriptor.append((tree.nodes[i]['word']))
        subject = search_for_dep_with_pos(tree, deps, subject, ['nsubj'])
        verb = search_for_dep_with_pos(tree, deps, verb, ['cop'])
        case = search_for_dep_with_pos(tree, deps, case, ['case'])
        results.append({"Subject": subject, "Descriptor": descriptor, "Verb": verb, "Case": case})
    return clean_search_results(results)

#################################################################################

def get_parse_results(sentence, relation_finder):
    ans = []
    result = dependancy_parser.raw_parse(sentence)
    for node in result: ans.append(relation_finder(node))
    try:
        return ans[0][0]
    except IndexError:
        return None

##################################################################################
#QUESTION TEMPLATES
##################################################################################

def where_questions(sentence):
    parse_results = get_parse_results(sentence, get_noun_descriptor)
    if parse_results == None: return None
    assert(len(parse_results['Subject']) != 0)
    assert(len(parse_results['Verb']) != 0)
    assert(len(parse_results['Case']) != 0)
    is_where_q = False
    for case in parse_results['Case']:
        if case[1] == 'case':
            is_where_q = True
    subject = None
    for possible_subject in parse_results['Subject']:
        if possible_subject[1] == 'nsubj':
            subject = possible_subject[2]
    if is_where_q:
        print("Where is {}?".format(subject))
    else:
        print("What is {}?".format(subject))

    
def who_did_what_questions(sentence):
    parse_results = get_parse_results(sentence, get_verb_subj_dobj)
    if parse_results == None: return None

    assert(len(parse_results['Subject']) != 0)
    assert(len(parse_results['Verb']) != 0)
    assert(len(parse_results['Direct Object']) != 0)

    verb = parse_results['Verb'][0][2]

    entities = [(X.text, X.label_) for X in nlp(sentence).ents]

    subj_wword = 'What'
    for entity in entities:
        if parse_results['Subject'][0][0] in entity[0].split():
            try:
                subj_wword = MAPPING[entity[1]]
            except:
                continue

    do_wword = 'What'
    for entity in entities:
        if parse_results['Direct Object'][0][0] in entity[0].split():
            try:
                do_wword = MAPPING[entity[1]]
            except:
                continue

    if do_wword != 'When' and parse_results['Direct Object'][0][2].split()[0] in ['in', 'under', 'at']:
        do_wword = 'Where'
    
    print("{} {} {}?".format(subj_wword, verb, parse_results['Direct Object'][0][2]))
    if parse_results['Subject'][0][1] == 'nsubj':
        print("{} did {} {}?".format(do_wword, parse_results['Subject'][0][2], WordNetLemmatizer().lemmatize(verb,'v')))
    elif parse_results['Subject'][0][1] == 'nsubjpass':
        print("{} {} {}?".format(do_wword, re.sub(r'is |was ', '',verb), parse_results['Subject'][0][2]))
 
####################################################################################

sentence_split = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
def generate_questions(article):
    questions = []
    sentences = re.split(sentence_split, article)
    for sentence in sentences:
        try:
            new_questions = who_did_what_questions(sentence)
            questions += new_questions
        except:
            continue
        try:
            new_questions = where_questions(sentence)
            questions += new_questions
        except:
            continue
    return '\n'.join(questions)

def _get_clause(tree, node):
    words = [node['word']]
    indexes = [node['address']]
    if len(node['deps']) == 0:
        return words, indexes
    else:
        for dep in node['deps']:
            if dep in ['compound', 'det', 'case', 'auxpass', 'amod']:
                for ad in node['deps'][dep]:
                    words += [tree.nodes[ad]['word']]
                    indexes += [ad]
            
    return words, indexes

def compile_clause(tree, node):
    words, indexes = _get_clause(tree, node)
    ordered_words = [word for _,word in sorted(zip(indexes, words)) if word is not None]
    try:
        return ' '.join(ordered_words)
    except:
        return None
