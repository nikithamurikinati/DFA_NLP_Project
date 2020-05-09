# Team DFA NLP Project

Given a document (for example, the text of a document from Wikipedia), we generate a set of intelligent questions relevant to that document. Given a plain test document and a set of questions, we return an answer for each question.

## Language and Modules
This entire project was developed in Python 3. The different modules we used are:
1. SpaCy
2. NLTK
3. Neuralcoref

## Instructions for using the model
### Ask
The ask program has the following command-line interface:
./ask article.txt nquestions
Where article.txt is a path to an arbitrary plain text file (the document) and nquestions is an integer (the number of questions to be generated). The program will output to STDOUT a sequence of questions with each question terminated by a newline character.
### Answer
The answer program has the following command-line interface:
./answer article.txt questions.txt
Where article.txt is a path to an arbitrary plain text file (the document) and questions.txt is a path to an arbitrary file of questions separated by newlines. The program will output to STDOUT a sequence of answers with each answer terminated by a newline character.

## Files (code)
Here is a list of all the python files in the project:
1. "GenerateSomeQuestions.py": Parses and cleans the text, generates "Why", "How many", "Which", and Yes/No questions
2. "elias_gq_library.py": Parses and cleans the text, generates "who", "what", and "where" questions
3. "template_question_generator.py": Wrapper file for elias_gq_library.py
4. "phrase_label_spacy.py": Parses and cleans text, develops answers for different questions posed
5. "tf_idf2.py": Calculates the tf-idf score for sentences and questions to identify sentences that most likely have the answer

## Final Report
https://www.youtube.com/watch?v=zYHzdc3lZ_I

## Creators
Nikitha Murikinati, Mihika Bairathi, Elias Joseph, Sunjana Kulkarni
