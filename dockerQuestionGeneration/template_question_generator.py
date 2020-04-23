import elias_qg_library as helper



def generate_questions(article, n):
	return helper.generate_questions(article, n)

if __name__ == '__main__':

	import sys

	article = sys.argv[1]
	nquestions = int(sys.argv[2])

	f_ = open(article)
	string_article = f_.read()
	f_.close()

	s = ''

	s += generate_questions(string_article, nquestions)

	s = s.split('\n')
	s = s[:nquestions]
	s = '\n'.join(s)

	sys.stdout.write(s)
