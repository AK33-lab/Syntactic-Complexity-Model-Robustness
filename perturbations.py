import spacy
import pyinflect
import random
nlp = spacy.load("en_core_web_sm")

PERSON_NOUNS = {
    "man", "woman", "person", "boy", "girl", "child", "student",
    "teacher", "doctor", "investor", "player", "worker", "employee",
    "customer", "patient", "officer", "agent", "author", "researcher"
}

ANIMAL_NOUNS = {
    "fox", "dog", "cat", "bird", "horse", "fish", "bear", "wolf",
    "lion", "tiger", "elephant", "monkey", "rabbit", "deer"
}

def _group_adjectival_modifiers(doc):
    grouped = {}
    for token in doc:
        if token.dep_ == "amod" and token.head.pos_ in {"NOUN", "PROPN"}:
            grouped.setdefault(token.head.i, []).append(token)
    return grouped


def _capitalize_first_char(text):
    if not text:
        return text
    return text[0].upper() + text[1:]

# Perturbation 1: Adjectival modifiers to relative clauses
def adj_to_relative_clause(text):
    """
    Converts adjectival modifiers to relative clauses.

    Example:
    "The old wooden table broke." -> "The table, that is old and wooden, broke."
    """
    doc = nlp(text)
    grouped_amods = _group_adjectival_modifiers(doc)
    if not grouped_amods:
        return text

    drop_indices = set()
    for adj_tokens in grouped_amods.values():
        for adj_token in adj_tokens:
            next_token = doc[adj_token.i + 1] if adj_token.i + 1 < len(doc) else None
            prev_token = doc[adj_token.i - 1] if adj_token.i - 1 >= 0 else None
            if (
                "-" in adj_token.text
                or "-" in adj_token.head.text
                or (next_token and next_token.text.startswith("-"))
                or (prev_token and prev_token.text.startswith("-"))
            ):
                continue
            drop_indices.add(adj_token.i)

    # Remove adjectives from grouped_amods that weren't dropped
    cleaned_amods = {}
    for noun_i, adj_tokens in grouped_amods.items():
        kept_adjs = [adj for adj in adj_tokens if adj.i in drop_indices]
        if kept_adjs:
            cleaned_amods[noun_i] = kept_adjs
    grouped_amods = cleaned_amods

    rebuilt = []
    for token in doc:
        if token.i in drop_indices:
            continue

        piece = token.text
        if token.i in grouped_amods:
            adjs = [adj.text for adj in sorted(grouped_amods[token.i], key=lambda t: t.i)]
            copula = "are" if "Plur" in token.morph.get("Number") else "is"
            if len(adjs) == 1:
                adjective_phrase = adjs[0]
            else:
                adjective_phrase = ", ".join(adjs[:-1]) + " and " + adjs[-1]

            next_kept = next(
                (t for t in doc[token.i + 1 :] if t.i not in drop_indices),
                None,
            )
            trailing_comma = "" if (next_kept is not None and next_kept.is_punct) else ","

            # Prefer entity tags first; only use animal fallback when no entity is assigned.
            lemma = token.lemma_.lower()
            ent_type = token.ent_type_
            if ent_type == "PERSON" or lemma in PERSON_NOUNS:
                pronoun = "who"
            elif not ent_type and lemma in ANIMAL_NOUNS:
                pronoun = "who"
            else:
                pronoun = "which"
            piece = f"{piece}, {pronoun} {copula} {adjective_phrase}{trailing_comma}"

        rebuilt.append(piece + token.whitespace_)

    return "".join(rebuilt).strip()

# Perturbation 2: Active to passive voice
def passive_voice(text):
    """
    Converts active voice sentences to passive voice.
    Example:
    "The dog bit the man." -> "The man was bitten by the dog."
    """
    doc = nlp(text)
    for token in doc:
        # Find the main verb with a subject and object
        if token.pos_ == "VERB" and token.dep_ == "ROOT":
            subj = None
            obj = None
            for child in token.children:
                if child.dep_ == "nsubj":
                    subj = child
                if child.dep_ == "dobj":
                    obj = child

            if subj and obj:
                # Get full noun phrases
                subj_span = doc[subj.left_edge.i: subj.right_edge.i + 1]
                obj_span = doc[obj.left_edge.i: obj.right_edge.i + 1]

                # Get past participle (lemma as fallback)
                past_participle = token._.inflect("VBN") if token.has_extension("inflect") else None
                if not past_participle:
                    past_participle = token.lemma_

                obj_phrase = _capitalize_first_char(obj_span.text)

                subj_tokens = [t.text for t in subj_span]
                if subj_tokens and subj_span[0].dep_ == "det":
                    subj_tokens[0] = subj_tokens[0].lower()
                subj_phrase = " ".join(subj_tokens)

                ending_punct = doc[-1].text if doc and doc[-1].is_punct else "."
                return f"{obj_phrase} was {past_participle} by {subj_phrase}{ending_punct}"

    return text  # return unchanged if no transformation possible

# Perturbation 3
PP_INSERTIONS = [
    "at the time",
    "at that moment", 
    "in question",
    "in this case",
    "at that point",
    "on that occasion",
    "in the scenario",
    "at that instance",
    "which we are talking about"
]

_PP_RNG = random.Random(67)
_PP_POOL = []


def _next_pp_insertion():
    global _PP_POOL
    if not _PP_POOL:
        _PP_POOL = PP_INSERTIONS[:]
        _PP_RNG.shuffle(_PP_POOL)
    return _PP_POOL.pop()

def prepositional_phrase_insertion(text):
    """
    Inserts a generic prepositional phrase after the first noun phrase.
    Example:
    "The doctor examined the patient." -> "The doctor at that instance examined the patient."
    """
    doc = nlp(text)
    noun_chunks = list(doc.noun_chunks)

    if not noun_chunks:
        return text

    # Insert after the first noun chunk
    first_np = noun_chunks[0]
    pp = _next_pp_insertion()
    insert_at = first_np.end

    tokens = [token.text_with_ws for token in doc]
    # Insert PP after first noun chunk
    next_token = doc[insert_at] if insert_at < len(doc) else None
    if next_token and next_token.text.startswith("'"):
        return text
    tokens.insert(insert_at, pp + " ")

    return "".join(tokens).strip()
