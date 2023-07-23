import spacy

nlp = spacy.load('en_core_web_md')

sentence1 = "First sentence to compare."
sentence2 = "Second sentence to compare."

def SentenceSimilarityScore(sentence1, sentence2):
    doc1 = nlp(sentence1)
    doc2 = nlp(sentence2)

    similarity_score = doc1.similarity(doc2)
    return similarity_score
