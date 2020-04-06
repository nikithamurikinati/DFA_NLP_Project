import nltk
nltk.download('state_union')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer
import re
from nltk.parse.stanford import StanfordDependencyParser
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

dependancy_parser = StanfordDependencyParser(
    path_to_jar = 'stanford-parser-full-2018-10-17/stanford-parser.jar',
    path_to_models_jar = 'stanford-english-corenlp-2018-02-27-models.jar')

def search_for_dep_with_pos(tree, deps, collection, pos_list):
    for pos in pos_list:
        if pos in deps.keys():
            for j in deps[pos]:
                collection.append((tree.nodes[j]['word'], pos))
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
        if tree.nodes[i]['tag'][:2] == 'VB':
            subject = []
            dir_object = []
            verb = []
            verb.append(tree.nodes[i]['word'])
            deps = tree.nodes[i]['deps']
            subject = search_for_dep_with_pos(tree, deps, subject, ['nsubj', 'nsubjpass'])
            dir_object = search_for_dep_with_pos(tree, deps, dir_object, ['nmod', 'dobj'])
            verb = search_for_dep_with_pos(tree, deps, verb, ['auxpass'])
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
	questions = []
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
            subject = possible_subject[0]
    if is_where_q:
        questions.append("Where is {}?".format(subject))
    else:
        questions.append("What is {}?".format(subject))
    return questions

    
def who_did_what_questions(sentence):
	questions = []
    parse_results = get_parse_results(sentence, get_verb_subj_dobj)
    if parse_results == None: return None
    assert(len(parse_results['Subject']) != 0)
    assert(len(parse_results['Verb']) != 0)
    assert(len(parse_results['Direct Object']) != 0)   
    questions.append("Who {} the {}?".format(parse_results['Verb'][0], parse_results['Direct Object'][0][0]))
    questions.append("What did {} {}?".format(parse_results['Subject'][0][0], WordNetLemmatizer().lemmatize(parse_results['Verb'][0],'v')))
    return questions

####################################################################################

sentence_split = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'
def generate_questions(article):
	questions = []
    sentences = re.split(sentence_split, article)
    for sentence in sentences:
        questions += who_did_what_questions(sentence)
        questions += where_questions(sentence)
    return '\n'.join(questions)
