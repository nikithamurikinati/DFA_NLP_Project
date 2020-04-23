import nltk
#nltk.download('state_union')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
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
import random
import neuralcoref
import time
nlp = spacy.load('en_core_web_md')
neuralcoref.add_to_pipe(nlp)


MAPPING = {'ORG': 'Who', 'PERSON': 'Who', 'DATE': 'When', 'LOCATION': 'Where'}
BAD = ['it', 'him', 'her', 'he', 'she', 'them', 'one', 'that', 'this', 'they', 'ones']
SEEN = set()
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
        hcc("Where is {}? *".format(subject))
    else:
        hcc("What is {}? *".format(subject))

    
def who_did_what_questions(sentence):
    outputs = []
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
    if not parse_results['Direct Object'][0][2].lower() in BAD:
        if parse_results['Subject'][0][0] not in SEEN:
            outputs.append(hcc("{} {} {}?".format(subj_wword, add_about(verb, parse_results['Direct Object'][0][2]), parse_results['Direct Object'][0][2])))
        SEEN.add(parse_results['Subject'][0][0])
    if parse_results['Subject'][0][1] == 'nsubj' and not parse_results['Subject'][0][2].lower() in BAD:
        if parse_results['Direct Object'][0][0] not in SEEN:
            outputs.append(hcc("{} did {} {}?".format(do_wword, parse_results['Subject'][0][2], WordNetLemmatizer().lemmatize(verb,'v'))))
        SEEN.add(parse_results['Direct Object'][0][0])
    elif parse_results['Subject'][0][1] == 'nsubjpass' and not parse_results['Subject'][0][2].lower() in BAD:
        if parse_results['Direct Object'][0][0] not in SEEN:
            outputs.append(hcc("{} {} {}?".format(do_wword, re.sub(r'is |was ', '',verb), parse_results['Subject'][0][2])))
        SEEN.add(parse_results['Direct Object'][0][0])

    outputs = [output for output in outputs if output is not None]
    return outputs
 
####################################################################################

sentence_split = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'

def generate_questions(article, n, max_time = 120):
    article_parsed = nlp(article)
    article = str(article_parsed)
    questions = []
    sentences = re.split(sentence_split, article)
    start_time = time.time()
    for sentence in sentences:
        if time.time() - start_time > max_time:
            break
        try:
            new_questions = who_did_what_questions(sentence)
            questions += new_questions
        except:
            #new_questions = who_did_what_questions(sentence)
            continue

    questions = best_questions(questions, n)
    return '\n'.join(questions)

def _get_clause(tree, node):
    words = [node['word']]
    indexes = [node['address']]
    if len(node['deps']) == 0:
        return words, indexes
    else:
        for dep in node['deps']:
            if dep in ['compound', 'det', 'case', 'auxpass', 'amod', 'nmod:poss', 'advmod']:
                for ad in node['deps'][dep]:
                    words += [tree.nodes[ad]['word']]
                    indexes += [ad]
            elif dep in ['xcomp']:
                for ad in node['deps'][dep]:
                    if abs(ad - mode['address']) <= 2:
                        words += [tree.nodes[ad]['word']]
                        indexes += [ad]
                        for tier2_dep in tree.nodes[ad]['deps']:
                            if tier2_dep in ['mark', 'case', 'advmod', 'nsubj', 'nsubjpass']:
                                for ad2 in tree.nodes[ad]['deps'][tier2_dep]:
                                    words += [tree.nodes[ad2]['word']]
                                    indexes += [ad2]      
    return words, indexes


def compile_clause(tree, node):
    words, indexes = _get_clause(tree, node)
    ordered_words = [word for _ ,word in sorted(zip(indexes, words)) if word is not None]
    ordered_indexes = sorted(indexes)

    try:
        result =  ' '.join(ordered_words)
        return result
    except:
        return None

def add_about(s, sub):
    if (s[:3] == 'is ' and len(sub.split()) == 1) or (s[:4] == 'are ' and len(sub.split()) == 1):
        s += ' about'
    return s

def hcc(sentence):
    sentence = re.sub('been', 'was', sentence)

    nlpsentence = nlp(sentence)

    for i in range(len(nlpsentence)):
        try:
            if str(nlpsentence[i].pos_) == 'VERB' and str(nlpsentence[i - 1]) != 'was' and str(nlpsentence[i + 1]) == 'as':
                sentence = re.sub(str(nlpsentence[i]), 'was ' + str(nlpsentence[i]), sentence)

        except:
            continue
    if len(sentence.split()) == 3 and sentence.split()[0] == 'What': return
    if len(sentence.split()) == 4 and sentence.split()[0] == 'What' and sentence.split()[1] != 'did': return
    if sentence.split()[0] == 'Where' and nlpsentence[1].lemma_ not in ['be', 'do']: 
        return

    sentence = sentence.split()
    if str(nlpsentence[-2].pos_) == 'VERB': sentence[-1] = str(nlpsentence[-2].lemma_ + '?')
    sentence = ' '.join(sentence)

    sentence = re.sub(' when | where | who | how | How | When | Where | Who ', ' ', sentence)



    #sentence = fix_text(sentence)
    #print(sentence)
    return sentence

def best_questions(questions, n):

    if len(questions) <= n:
        return questions

    q = []
    n_where_questions = min(1, n)

    i = 0
    for question in questions:
        if i > n_where_questions:
            break
        if question.split()[0] == 'Where':
            qparse = nlp(question)
            no_date = True
            for token in qparse.ents:
                if token.label_ == 'DATE':
                    no_date == False
            if no_date:
                q.append(question)
                i += 1

    for question in questions:
        if question.split()[0] == 'Who':
            q.append(question)
            break

    while len(q) < n:
        q_candidate = questions[random.randint(0, len(questions))]
        if q_candidate not in q:
            q.append(q_candidate)

    return q
