
EMB_INIT_RANGE = 1.0

# vocab
PAD_TOKEN = '<PAD>'
PAD_ID = 0
UNK_TOKEN = '<UNK>'
UNK_ID = 1
UNK_ID = 1
UNK_ID = 1

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids

SUBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'ORGANIZATION': 2, 'PERSON': 3}

OBJ_NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'PERSON': 2, 'ORGANIZATION': 3, 'DATE': 4, 'NUMBER': 5, 'TITLE': 6, 'COUNTRY': 7, 'LOCATION': 8, 'CITY': 9, 'MISC': 10, 'STATE_OR_PROVINCE': 11, 'DURATION': 12, 'NATIONALITY': 13, 'CAUSE_OF_DEATH': 14, 'CRIMINAL_CHARGE': 15, 'RELIGION': 16, 'URL': 17, 'IDEOLOGY': 18}

NER_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'O': 2, 'PERSON': 3, 'ORGANIZATION': 4, 'LOCATION': 5, 'DATE': 6, 'NUMBER': 7, 'MISC': 8, 'DURATION': 9, 'MONEY': 10, 'PERCENT': 11, 'ORDINAL': 12, 'TIME': 13, 'SET': 14}

POS_TO_ID = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'NNP': 2, 'NN': 3, 'IN': 4, 'DT': 5, ',': 6, 'JJ': 7, 'NNS': 8, 'VBD': 9, 'CD': 10, 'CC': 11, '.': 12, 'RB': 13, 'VBN': 14, 'PRP': 15, 'TO': 16, 'VB': 17, 'VBG': 18, 'VBZ': 19, 'PRP$': 20, ':': 21, 'POS': 22, '\'\'': 23, '``': 24, '-RRB-': 25, '-LRB-': 26, 'VBP': 27, 'MD': 28, 'NNPS': 29, 'WP': 30, 'WDT': 31, 'WRB': 32, 'RP': 33, 'JJR': 34, 'JJS': 35, '$': 36, 'FW': 37, 'RBR': 38, 'SYM': 39, 'EX': 40, 'RBS': 41, 'WP$': 42, 'PDT': 43, 'LS': 44, 'UH': 45, '#': 46, 'pad': 47}

#cpr
# DEPREL_TO_ID = {PAD_TOKEN: 0,'self': 1, 'nsubj': 2, 'ROOT': 3, 'det': 4, 'dobj': 5, 'case': 6, 'nmod': 7, 'amod': 8, 'compound': 9, 'punct': 10, 'cc': 11, 'conj': 12, 'mark': 13, 'acl:relcl': 14, 'advmod': 15, 'aux': 16, 'ccomp': 17, 'prep': 18, 'pobj': 19, 'nummod': 20, 'appos': 21, 'cop': 22, 'relcl': 23, 'nmod:poss': 24, 'attr': 25, 'xcomp': 26, 'dep': 27, 'nsubjpass': 28, 'auxpass': 29, 'nmod:npmod': 30, 'neg': 31, 'pcomp': 32, 'mwe': 33, 'advcl': 34, 'acl': 35, 'parataxis': 36, 'cc:preconj': 37, 'npadvmod': 38, 'acomp': 39, 'poss': 40, 'quantmod': 41, 'csubj': 42, 'prt': 43, 'agent': 44, 'expl': 45, 'det:predet': 46, 'oprd': 47, 'predet': 48}
# pgr
DEPREL_TO_ID = {PAD_TOKEN: 0,'self': 1, 'det': 2, 'nsubj': 3, 'punct': 4, 'acl:relcl': 5, 'amod': 6, 'compound': 7, 'dobj': 8, 'case': 9, 'nmod': 10, 'cop': 11, 'ROOT': 12, 'nmod:poss': 13, 'advmod': 14, 'mark': 15, 'nsubjpass': 16, 'aux': 17, 'auxpass': 18, 'ccomp': 19, 'acl': 20, 'nummod': 21, 'appos': 22, 'cc': 23, 'conj': 24, 'xcomp': 25, 'advcl': 26, 'prep': 27, 'pobj': 28, 'dep': 29, 'cc:preconj': 30, 'mwe': 31, 'nmod:npmod': 32, 'parataxis': 33, 'neg': 34, 'det:predet': 35, 'pcomp': 36, 'attr': 37, 'npadvmod': 38, 'expl': 39, 'csubjpass': 40, 'csubj': 41, 'acomp': 42, 'poss': 43, 'prt': 44, 'quantmod': 45, 'relcl': 46, 'agent': 47, 'oprd': 48, 'predet': 49}

NEGATIVE_LABEL = 'Other'

# #PGR
LABEL_TO_ID = {'False': 0, 'True': 1}

# CPR
# LABEL_TO_ID = {'None': 0, 'CPR:3': 1, 'CPR:4': 2, 'CPR:5': 3, 'CPR:6': 4, 'CPR:9':5}

INFINITY_NUMBER = 1e12

















