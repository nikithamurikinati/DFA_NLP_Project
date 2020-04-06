import coreNLPQuestionGenerator as cqg
import template_question_generation as tqg
import sys

article = sys.argv[1]
nquestions = int(sys.argv[2])

line_split_article = cqg.readFile(article)
f_ = open(article)
string_article = f_.read()
f_.close()

s = ''

print('Starting')

s += '\n'.join(cqg.generateQuestions(line_split_article))

print('finished phase 1')

s += tqg.generate_questions(string_article)

sys.stdout.write(s)