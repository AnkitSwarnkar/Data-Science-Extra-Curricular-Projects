from nltk.parse import stanford
import os
import operator
import sys
sys.path.append('//home/ankit/Documents/Ling/stanford-parser-full-2016-10-31/')
java_path = '/usr/lib/jvm/java-8-openjdk-amd64/jre'
os.environ['JAVAHOME'] = java_path
os.environ['STANFORD_PARSER'] = '/home/ankit/Documents/Ling/stanford-parser-full-2016-10-31/'
os.environ['CLASSPATH'] = '/home/ankit/Documents/Ling/stanford-parser-full-2016-10-31/'
os.environ['STANFORD_MODELS'] = '/home/ankit/Documents/Ling/stanford-parser-full-2016-10-31/'
from nltk.parse.stanford import StanfordParser
from nltk.tree import ParentedTree, Tree
parser = StanfordParser()
name=1
if name ==1:
    sent = 'A rare black squirrel has become a regular visitor to a suburban garden.'
    #sent='3 monkeys and a lion .'
    t = list(parser.raw_parse(sent))[0]
    t = ParentedTree.convert(t)
    t.pretty_print()
    # Breadth First Search the tree and take the first noun in the NP subtree.
    def find_subject(t):
        for s in t.subtrees(lambda t: t.label() == 'NP'):
            for n in s.subtrees(lambda n: n.label().startswith('NN')):
                return (n[0], find_attrs(n))
    # Depth First Search the tree and take the last verb in VP subtree.
    def find_predicate(t):
        v = None
        for s in t.subtrees(lambda t: t.label() == 'VP'):
            for n in s.subtrees(lambda n: n.label().startswith('VB')):
                v = n
        return (v[0], find_attrs(v))
    # Breadth First Search the siblings of VP subtree
    # and take the first noun or adjective
    def find_object(t):
        for s in t.subtrees(lambda t: t.label() == 'VP'):
            for n in s.subtrees(lambda n: n.label() in ['NP', 'PP', 'ADJP']):
                if n.label() in ['NP', 'PP']:
                    for c in n.subtrees(lambda c: c.label().startswith('NN')):
                        return (c[0], find_attrs(c))
                else:
                    for c in n.subtrees(lambda c: c.label().startswith('JJ')):
                        return (c[0], find_attrs(c))
    def find_attrs(node):
        attrs = []
        p = node.parent()
        # Search siblings of adjective for adverbs
        if node.label().startswith('JJ'):
            for s in p:
                if s.label() == 'RB':
                    attrs.append(s[0])
                elif node.label().startswith('NN'):
                    for s in p:
                        if s.label() in ['DT','PRP$','POS','JJ','CD','ADJP','QP','NP']:
                            attrs.append(s[0])
    # Search siblings of verbs for adverb phrase
        elif node.label().startswith('VB'):
            for s in p:
                if s.label() == 'ADVP':
                    attrs.append(' '.join(s.flatten()))
    # Search uncles
    # if the node is noun or adjective search for prepositional phrase
        if node.label().startswith('JJ') or node.label().startswith('NN'):
            for s in p.parent():
                if s != p and s.label() == 'PP':
                    attrs.append(' '.join(s.flatten()))
        elif node.label().startswith('VB'):
            for s in p.parent():
                if s != p and s.label().startswith('VB'):
                    attrs.append(' '.join(s.flatten()))
        return attrs
        print( find_subject(t))
        print (find_predicate(t))
        print (find_object(t))
        import json
        with open('t', 'w') as outfile:
            json.dump(t, outfile)