from spacy.en import English
import preprocess.spacy_pipe as pp

nlp = English()
w = nlp("Trump was killing Democracy")
svos = pp.findSVOs(w)
pp.printDeps(w)