import os
import warnings

# Silence known non-critical warnings from benepar/transformers.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings(
    "ignore",
    message=r".*TreeCRF.*arg_constraints.*",
    category=UserWarning,
)

import spacy
import benepar

nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("benepar", config={"model": "benepar_en3"})

def cfg_tree_depth(sent): #Most complex metric: depth of the constituency parse tree
    tree_str = sent._.parse_string
    depth = max_depth = 0
    for char in tree_str:
        if char == '(':
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == ')':
            depth -= 1
    return max_depth

def subject_verb_distance(sent): # Medium complexity: distance between subject and main verb
    for token in sent:
        if token.dep_ in ("nsubj", "nsubjpass"):
            return abs(token.i - token.head.i)
    return 0

def clause_count(sent): # Simplest metric: count the number of verbs as a proxy for clauses
    return sum(1 for token in sent if token.pos_ in ("VERB", "AUX"))

def compute_complexity(text):
    doc = nlp(text)
    sent = list(doc.sents)[0]
    return {
        "clause_count": clause_count(sent),
        "subj_verb_dist": subject_verb_distance(sent),
        "cfg_depth": cfg_tree_depth(sent),
    }