"""
study.py - PDF study assistant. Classic NLP only, so every part can be explained in a viva:

  ask()      BM25 ranking of the PDF's sentences against the question   -> answers
  summary()  TextRank: sentences vote for each other (PageRank graph)    -> summaries
  quiz()     blank out the most informative keyword + pick distractors   -> MCQs
"""
import io
import json
import math
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import Counter, OrderedDict

import numpy as np

STOP = set("""a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her here
hers him his how i if in into is it its itself just me more most my no nor not of off on once only or other our out over
own same she should so some such than that the their them then there these they this those through to too under until
up very was we were what when where which while who whom why will with would you your yours explain define describe
tell give show say meaning means mean example examples""".split())

DOCS = OrderedDict()          # doc_id -> processed document (last 5 are kept in memory)


def stem(w):
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[:-len(suf)]
            break
    return w[:-1] if w.endswith("e") and len(w) > 4 else w      # reduce / reduced / reduces -> reduc


def words(s):
    return re.findall(r"[a-z][a-z\-]{2,}", s.lower())


def tokens(s):
    return [stem(w) for w in words(s) if w not in STOP]


def split_sentences(text):
    """Join wrapped lines of one paragraph, but keep slide titles and bullets as separate pieces."""
    paras, cur = [], ""
    lines = [l.strip() for l in text.splitlines()]
    for i, l in enumerate(lines):
        if not l:
            if cur:
                paras.append(cur)
            cur = ""
            continue
        wrapped = cur and not cur.endswith((".", "!", "?", ":")) and (l[0].islower() or len(cur) >= 75)
        if wrapped:
            cur += " " + l
        else:
            if cur:
                paras.append(cur)
            cur = l
    if cur:
        paras.append(cur)
    out = []
    for para in paras:
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"(\[\u2022])", re.sub(r"\s+", " ", para)):
            for part in re.split(r"\s*[\u2022\u25cf\u25aa]\s*", s):
                part = part.strip()
                if 12 <= len(part) <= 450:
                    out.append(part)
    return out


# ---------------------------------------------------------------- indexing --
def build_from_pages(page_texts):
    sents = [(p, s) for p, t in enumerate(page_texts, 1) for s in split_sentences(t)]
    if len(sents) < 3:
        raise ValueError("No readable text found - is this a scanned PDF? Try a text PDF.")
    toks = [tokens(s) for _, s in sents]
    N = len(sents)
    df = Counter(w for t in toks for w in set(t))
    idf = {w: math.log(1 + (N - n + .5) / (n + .5)) for w, n in df.items()}
    raw = Counter(w for _, s in sents for w in words(s) if w not in STOP)
    doc = dict(sents=sents, toks=toks, tf=[Counter(t) for t in toks], idf=idf, raw=raw,
               avgdl=(sum(len(t) for t in toks) / N) or 1, pages=len(page_texts), page_text=list(page_texts))
    doc_id = uuid.uuid4().hex[:12]
    DOCS[doc_id] = doc
    while len(DOCS) > 5:
        DOCS.popitem(last=False)
    return doc_id, doc


def build(file_bytes):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    texts = []
    for page in list(reader.pages)[:300]:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            texts.append("")
    return build_from_pages(texts)


# ------------------------------------------------------------ question answering --
def bm25(doc, question):
    q = set(tokens(question))
    k1, b = 1.5, 0.75
    scores = np.zeros(len(doc["sents"]))
    for i, tf in enumerate(doc["tf"]):
        dl = len(doc["toks"][i]) or 1
        for w in q:
            f = tf.get(w, 0)
            if f:
                scores[i] += doc["idf"][w] * f * (k1 + 1) / (f + k1 * (1 - b + b * dl / doc["avgdl"]))
    return scores


def ask(doc, question):
    sc = bm25(doc, question)
    if not len(sc) or sc.max() <= 0:
        return None
    key = " ".join(w for w in re.findall(r"[a-z\-]+", question.lower()) if w not in STOP)
    definition = bool(re.match(r"\s*(what is|what are|what's|define|explain|describe|meaning of)\b", question.lower()))
    if key:
        for i, (_, s) in enumerate(doc["sents"]):
            if sc[i] <= 0:
                continue
            low = s.lower()
            if key in low:                             # bonus when the exact phrase appears
                sc[i] *= 1.5
            if definition and re.match(rf"\W*(an?\s+|the\s+)?{re.escape(key)}\b", low) and \
                    re.search(rf"{re.escape(key)}\w*\s+(is|are|refers|happens|occurs|means|uses|creates|measures|applies)\b", low):
                sc[i] *= 2.0                           # "X is / happens when / refers to ..." = a definition
    order = [int(i) for i in np.argsort(-sc)[:3] if sc[i] > 0]
    if not order or sc[order[0]] < 1.0:
        return None
    top = order[0]
    page, text = doc["sents"][top]
    nxt = doc["sents"][top + 1] if top + 1 < len(doc["sents"]) else None
    answer = text + (" " + nxt[1] if nxt and nxt[0] == page and len(text) < 200 else "")
    return {"answer": answer,
            "sources": [{"page": doc["sents"][i][0], "text": doc["sents"][i][1], "score": round(float(sc[i]), 2)}
                        for i in order]}


# ---------------------------------------------------------------- summarising --
def textrank(doc, idxs, k):
    """Rank sentences with PageRank over a TF-IDF cosine-similarity graph; return the top k indices."""
    idxs = list(idxs)
    if len(idxs) <= k:
        return idxs
    if len(idxs) > 500:                                # keep memory small on big PDFs
        idxs = sorted(idxs, key=lambda i: -sum(doc["idf"].get(w, 0) for w in doc["toks"][i])
                      / (len(doc["toks"][i]) or 1))[:500]
    vocab = {}
    for i in idxs:
        for w in doc["tf"][i]:
            vocab.setdefault(w, len(vocab))
    M = np.zeros((len(idxs), max(1, len(vocab))), np.float32)
    for r, i in enumerate(idxs):
        for w, f in doc["tf"][i].items():
            M[r, vocab[w]] = f * doc["idf"][w]
    M /= np.linalg.norm(M, axis=1, keepdims=True) + 1e-9
    S = M @ M.T
    np.fill_diagonal(S, 0)
    P = S / (S.sum(axis=1, keepdims=True) + 1e-9)
    rank = np.ones(len(idxs)) / len(idxs)
    for _ in range(40):
        rank = 0.15 / len(idxs) + 0.85 * (P.T @ rank)
    return [idxs[j] for j in np.argsort(-rank)[:k]]


def summary(doc, page=None, n=4):
    idxs = [i for i, (p, _) in enumerate(doc["sents"]) if page is None or p == page]
    if not idxs:
        return None
    return [{"page": doc["sents"][i][0], "text": doc["sents"][i][1]} for i in sorted(textrank(doc, idxs, n))]


# ------------------------------------------------------------------- quiz --
def quiz(doc, n=5, seed=None):
    rnd = random.Random(seed)
    pool = [w for w, c in doc["raw"].most_common(150) if c >= 2 and len(w) >= 6]
    if len(pool) < 6:
        return None
    ranked = textrank(doc, range(len(doc["sents"])), min(len(doc["sents"]), n * 4))
    out = []
    for i in ranked:
        page, s = doc["sents"][i]
        if not 50 <= len(s) <= 220:
            continue
        cands = [w for w in set(words(s)) if doc["raw"].get(w, 0) >= 2 and len(w) >= 6]
        if not cands:
            continue
        ans = max(cands, key=lambda w: doc["idf"].get(stem(w), 0) * (1 + len(w) / 10))   # most informative word
        blank = re.sub(r"\b" + re.escape(ans) + r"\b", "_____", s, count=1, flags=re.I)
        if blank == s:
            continue
        inside = {stem(w) for w in words(s)}
        others = [w for w in pool if stem(w) not in inside]          # distractors must not appear in the question
        near = sorted(rnd.sample(others, min(len(others), 12)), key=lambda w: abs(len(w) - len(ans)))[:3]
        if len(near) < 3:
            continue
        opts = near + [ans]
        rnd.shuffle(opts)
        out.append({"q": blank, "options": opts, "answer": opts.index(ans) + 1, "page": page})
        if len(out) == n:
            break
    return out or None


# ------------------------------------------------------- "explain this" (pointing) --
def explain(doc, focus):
    """The user pointed at / highlighted some text.  Find the sentence in the PDF that DEFINES its key term;
    if there is none, fall back to the best-matching sentence."""
    toks = set(tokens(focus))
    if not toks:
        return None
    term = max(toks, key=lambda w: doc["idf"].get(w, 0))
    sc = bm25(doc, focus)
    pat = rf"\b{re.escape(term)}\w*\s+(is|are|refers|happens|occurs|means|uses|creates|measures|applies)\b"
    defs = [i for i, (_, s) in enumerate(doc["sents"]) if re.search(pat, s.lower())]
    cand = defs or ([int(np.argmax(sc))] if sc.max() > 0 else [])
    if not cand:
        return None
    i = max(cand, key=lambda j: sc[j])
    page, text = doc["sents"][i]
    shown = next((w for w in words(focus) if stem(w) == term), term)
    return {"answer": text, "term": shown,
            "sources": [{"page": page, "text": text, "score": round(float(sc[i]), 2)}]}


# ------------------------------------------------------------ word meanings --
# Free dictionary (no key needed, needs internet).  WordNet is used as a backup if it is installed.
_DEF_CACHE = {}
_API_DOWN_UNTIL = 0.0


def _lemmas(w):
    """leverages -> leverage, using -> use, studies -> study ..."""
    c = [w]
    if w.endswith("ies"):
        c.append(w[:-3] + "y")
    if w.endswith("es"):
        c += [w[:-1], w[:-2]]
    if w.endswith("s"):
        c.append(w[:-1])
    if w.endswith("ing"):
        c += [w[:-3] + "e", w[:-3]]
    if w.endswith("ed"):
        c += [w[:-1], w[:-2]]
    return list(dict.fromkeys(x for x in c if len(x) > 2))


def _lookup_api(word):
    global _API_DOWN_UNTIL
    if time.time() < _API_DOWN_UNTIL:
        return None
    try:
        req = urllib.request.Request("https://api.dictionaryapi.dev/api/v2/entries/en/" + urllib.parse.quote(word),
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=4) as r:
            data = json.load(r)
    except urllib.error.HTTPError:
        return None                                   # that word is not in the dictionary
    except Exception:
        _API_DOWN_UNTIL = time.time() + 60            # offline: do not wait again for a minute
        return None
    for entry in data if isinstance(data, list) else []:
        for m in entry.get("meanings", []):
            for d in m.get("definitions", []):
                if d.get("definition"):
                    return {"word": word, "pos": m.get("partOfSpeech", ""), "definition": d["definition"].rstrip(".")}
    return None


def _lookup_wordnet(word):
    try:
        from nltk.corpus import wordnet as wn  # type: ignore
        syn = wn.synsets(word)
        if syn:
            pos = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}.get(syn[0].pos(), "")
            return {"word": word, "pos": pos, "definition": syn[0].definition()}
    except Exception:
        pass
    return None


def define_word(word):
    w = word.lower()
    if w in _DEF_CACHE:
        return _DEF_CACHE[w]
    res = None
    for cand in _lemmas(w):
        res = _lookup_api(cand) or _lookup_wordnet(cand)
        if res:
            break
    if res or time.time() >= _API_DOWN_UNTIL:         # do not remember "not found" when we were simply offline
        _DEF_CACHE[w] = res
    return res


def _norm(s):
    return re.sub(r"\W+", " ", s.lower()).strip()


def key_terms(doc, text, k=4):
    """The most specific (rare in this PDF, long) words of a line - the ones worth looking up."""
    ws = {w for w in words(text) if w not in STOP and len(w) >= 5}
    return sorted(ws, key=lambda w: -(doc["idf"].get(stem(w), 0) * (1 + len(w) / 12)))[:k]


DEF_RE = r"\s*(please\s+)?(what is|what's|what are|what does|what do|define|meaning of|the meaning of|explain|tell me about|describe)\b"


def question_term(q):
    t = re.sub(r"[?.!,]", "", q.lower()).strip()
    t = re.sub(DEF_RE + r"\s+", "", t)
    t = re.sub(r"\s+(mean|means|meaning|stand for)$", "", t)
    return re.sub(r"^(a|an|the)\s+", "", t).strip()


def _meaning(doc, term):
    """Where does a term's meaning come from?  glossary -> dictionary / Wikipedia (order depends on how
    technical the word looks in THIS pdf) -> nothing."""
    g = _gloss(term)
    if g:
        return {"pos": "", "definition": g, "src": ""}
    ws = [w for w in words(term) if w not in STOP]
    hint = " ".join(w for w, _ in doc["raw"].most_common(3))        # topic words of the PDF: pick the right sense

    def dic():
        d = define_word(ws[0]) if len(ws) == 1 else None
        return dict(d, src="") if d else None

    def wik():
        w = wiki_summary(term, hint)
        return {"pos": "", "definition": w["text"], "src": "Wikipedia: " + w["title"]} if w else None

    technical = len(ws) != 1 or doc["raw"].get(ws[0], 0) >= 3        # repeated a lot in the PDF = a topic word
    return (wik() or dic()) if technical else (dic() or wik())


def define_flow(doc, term):
    """'what is leverages' -> the term's MEANING first, then what this PDF says about it."""
    ctx = explain(doc, term)
    if not ctx:
        return None                                    # the term is not in this PDF -> honest "not found"
    d = _meaning(doc, term)
    parts = []
    if d:
        head = f"{term} ({d['pos']})" if d["pos"] else term
        src = f"  [{d['src']}]" if d["src"] else ""
        parts.append(f"{head}: {d['definition'].rstrip('.')}.{src}")
    parts.append(("In this PDF: " if d else "") + ctx["answer"])
    return {"answer": "\n".join(parts), "term": term, "sources": ctx["sources"],
            "speak": f"{term} means {d['definition']}" if d else ctx["answer"]}


def explain_text(doc, focus):
    """'explain this' on a highlighted / pointed-at line: what the numbers mean, glossary terms, word meanings,
    and related context - never just the same line repeated back."""
    mets = [interpret_metric(f) for f in metric_findings(focus)]
    terms = glossary_terms(focus, 3)
    cands = [w for w in key_terms(doc, focus, 5) if not any(w in t for t, _ in terms)]
    bullets, spoken = [], ""
    for w in cands:
        d = define_word(w)
        if d:
            bullets.append(f"\u2022 {w} ({d['pos']}): {d['definition']}")
            spoken = spoken or f"{w} means {d['definition']}"
        if len(bullets) == 2:
            break
    fn = _norm(focus)
    sc = bm25(doc, " ".join([t for t, _ in terms] + cands) or focus)
    best = None
    for i in np.argsort(-sc):
        if sc[i] <= 0:
            break
        page, text = doc["sents"][i]
        ns = _norm(text)
        if fn and (fn in ns or ns in fn):              # skip the line the user already selected
            continue
        best = (page, text, float(sc[i]))
        break
    lines = []
    if mets:
        lines += ["In simple words:"] + [f"\u2022 {m}" for m in mets]
    if terms:
        lines += ["Terms:"] + [f"\u2022 {t}: {m}" for t, m in terms]
    if bullets:
        lines += ["Other words:"] + bullets
    if best:
        lines.append(f"What else the PDF says about it (p.{best[0]}): {best[1]}")
    if not lines:
        own = next((t for _, t in doc["sents"] if fn and fn in _norm(t)), None)
        if not own:
            return None
        lines = ["I could not look up word meanings (check your internet). The sentence says: " + own]
    first = re.sub(r"\s*\(.*?\)", "", mets[0]) if mets else (f"{terms[0][0]}: {terms[0][1]}" if terms else "")
    return {"answer": "\n".join(lines), "term": cands[0] if cands else "",
            "sources": [{"page": best[0], "text": best[1], "score": round(best[2], 2)}] if best else [],
            "speak": first or spoken or (best[1] if best else lines[0])}


def answer(doc, question):
    """Definition-style questions get a real explanation; everything else is retrieved from the PDF.
    The PDF decides WHETHER there is an answer; the AI (if enabled) only helps WORD it."""
    term = question_term(question)
    if term and len(term.split()) <= 3 and re.match(DEF_RE, question.lower()):
        r = define_flow(doc, term)                     # None if the term is not in the PDF
    else:
        r = ask(doc, question)
    if not r or not ai_available():
        return dict(r, via="rules") if r else r
    passages = "\n".join("- " + x["text"] for x in r["sources"][:3]) or "- " + r["answer"]
    out = llm(TUTOR, f"Question: {question}\n\nPassages from the student's PDF:\n{passages}\n\nBackground:\n{r['answer']}\n\n"
                     "Answer the question in your own words in 2-4 simple sentences, using the passages "
                     "(the background may be used to define terms). If the passages do not answer the question, "
                     "reply with exactly NOT_FOUND.", 350)
    if out and "NOT_FOUND" in out:
        return None
    return dict(r, answer=out, speak=_first_sentence(out), via="llm") if out else dict(r, via="rules")


def explain_focus(doc, focus, page=None):
    """The user highlighted / pointed at `focus`.  A word or short phrase -> define it; a longer line -> explain it."""
    n = len(words(focus))
    r = define_flow(doc, focus.strip()) if 0 < n <= 3 else None
    r = r or explain_text(doc, focus)
    if ai_available():
        ctx = doc["page_text"][page - 1] if page and 0 < int(page) <= len(doc["page_text"]) else ""
        extra = "\n".join("- " + x["text"] for x in (r or {}).get("sources", [])[:2])
        out = llm(TUTOR, f'The student highlighted this text: "{focus}"\n\nText of the slide/page it is on:\n{ctx[:1500]}\n\n'
                         f"Related text from the document:\n{extra}\n\nExplain what the highlighted text means in this context, "
                         "in 2-4 simple sentences. If it is a technical term, define it first.", 350)
        if out:
            return {"answer": out, "term": focus, "sources": [], "speak": _first_sentence(out), "via": "llm"}
    return dict(r, via="rules") if r else r


# ------------------------------------------------------- ML glossary (works offline) --
GLOSSARY = {
    "accuracy": "the share of all predictions that the model got right",
    "precision": "of all the cases the model labelled positive, how many really were positive",
    "recall": "of all the real positive cases, how many the model managed to find (also called sensitivity)",
    "sensitivity": "of all the real positive cases, how many the model managed to find (same as recall)",
    "specificity": "of all the real negative cases, how many the model correctly left as negative",
    "f1 score": "one number that balances precision and recall",
    "f1": "one number that balances precision and recall",
    "auc": "area under the ROC curve - how well the model separates the two classes (0.5 is guessing, 1.0 is perfect)",
    "roc curve": "a plot of true-positive rate against false-positive rate at every decision threshold",
    "confusion matrix": "a table showing, for each class, how many predictions were right and which classes got mixed up",
    "overfitting": "when a model memorises the training data and then does poorly on new data",
    "underfitting": "when a model is too simple to learn the pattern, so it does poorly even on the training data",
    "regularization": "techniques that keep a model from becoming too complex, so it overfits less",
    "dropout": "randomly switching off some neurons during training so the network does not depend on any single one",
    "epoch": "one full pass of the whole training dataset through the model",
    "batch size": "how many training samples are processed before the model's weights are updated",
    "learning rate": "the step size for updating the weights - too big is unstable, too small is slow",
    "loss": "a number measuring how wrong the predictions are; training tries to make it small",
    "gradient descent": "an optimisation method that repeatedly nudges the weights in the direction that reduces the loss",
    "backpropagation": "the way a neural network works out how much each weight contributed to the error",
    "cnn": "convolutional neural network - slides small filters over an image to detect edges, textures and shapes",
    "convolutional neural network": "a network that slides small filters over an image to detect edges, textures and shapes",
    "pooling": "shrinking a feature map by keeping only the strongest value in each small region",
    "activation function": "a function that decides how strongly a neuron fires and lets the network learn non-linear patterns",
    "relu": "an activation function that outputs zero for negative inputs and the input itself otherwise",
    "softmax": "turns a layer's outputs into probabilities that add up to 1",
    "feature": "an input variable the model uses to make a prediction",
    "label": "the correct answer attached to a training example",
    "training set": "the data the model learns from",
    "validation set": "data used during training to tune settings and spot overfitting",
    "test set": "data kept aside to measure the final performance on unseen examples",
    "cross validation": "splitting the data into folds and testing on each fold in turn for a more reliable score",
    "hyperparameter": "a setting chosen before training, such as the learning rate or number of layers",
    "data augmentation": "creating extra training samples by rotating, shifting or zooming the existing ones",
    "bias": "error from a model that is too simple to capture the real pattern",
    "variance": "error from a model that is too sensitive to the particular training data",
    "normalization": "rescaling values to a common range so features are comparable",
    "rmse": "root mean squared error - the typical size of the prediction error, in the same units as the target",
    "mae": "mean absolute error - the average size of the prediction error",
    "mse": "mean squared error - the average of the squared prediction errors",
    "r2": "R-squared - the share of the variation in the target that the model explains",
    "classification": "predicting which category an item belongs to",
    "regression": "predicting a number, such as a price or a temperature",
    "supervised learning": "learning from examples that already have the correct answers",
    "unsupervised learning": "finding structure in data that has no labels",
    "clustering": "grouping similar items together without being told the groups",
    "transfer learning": "starting from a model already trained on a large dataset and adapting it to a new task",
    "neural network": "layers of simple connected units that learn patterns from data",
}


def _gloss(term):
    t = re.sub(r"[\s-]+", " ", term.lower().strip())
    return GLOSSARY.get(t) or GLOSSARY.get(t.rstrip("s")) or GLOSSARY.get(t.replace(" ", ""))


def glossary_terms(text, k=3):
    low = " " + re.sub(r"[^a-z0-9 ]", " ", text.lower()) + " "
    found = []
    for term in sorted(GLOSSARY, key=len, reverse=True):          # longest first: "f1 score" before "f1"
        if re.search(rf"\W{re.escape(term)}s?\W", low) and not any(term in f for f, _ in found):
            found.append((term, GLOSSARY[term]))
    return found[:k]


# ------------------------------------------------- explaining the NUMBERS on a slide --
METRIC_RE = re.compile(
    r"(?i)(?:\b(train(?:ing)?|test(?:ing)?|validation|val)\s+)?"
    r"\b(accuracy|roc[\s-]?auc|auc|precision|recall|sensitivity|specificity|f1(?:[\s-]?score)?|rmse|mae|mse|r2|r-squared|loss)\b"
    r"(?:\s+score)?\s*(?:[:=]|\bis\b|\bof\b|\bwas\b|\bequals?\b(?:\s+to)?)?\s*(\d+(?:\.\d+)?)\s*(%?)")
RATIO = {"accuracy", "auc", "precision", "recall", "sensitivity", "specificity", "f1", "r2"}


def metric_findings(text):
    out, seen = [], set()
    for m in METRIC_RE.finditer(text):
        qual, name, num, pct = m.groups()
        key = re.sub(r"[\s-]", "", name.lower())
        key = {"rocauc": "auc", "f1score": "f1", "rsquared": "r2"}.get(key, key)
        v = float(num)
        if key in RATIO and (pct or 1 < v <= 100):
            v /= 100
        q = (qual or "").lower()
        q = {"training": "train", "testing": "test", "val": "validation"}.get(q, q)
        if (q, key) not in seen:
            seen.add((q, key))
            out.append({"qual": q, "metric": key, "v": v})
    return out


def _band(v, cuts=((0.9, "very good"), (0.8, "good"), (0.7, "fair"))):
    return next((w for c, w in cuts if v >= c), "needs improvement")


def interpret_metric(f):
    k, v = f["metric"], f["v"]
    q = {"train": "Training ", "test": "Test ", "validation": "Validation "}.get(f["qual"], "")
    N = lambda base: (q + base.lower()) if q else base
    R2 = "R\u00b2"
    n100 = round(v * 100)
    if k == "accuracy":
        return (f"{N('Accuracy')} {v:.1%}: the model got about {n100} of every 100 predictions right - {_band(v)}. "
                "(If one class is much bigger than the other, also check precision and recall.)")
    if k == "auc":
        band = _band(v, ((0.9, "excellent"), (0.8, "good"), (0.7, "fair"), (0.5, "weak - barely better than guessing")))
        return (f"{N('AUC')} {v:.2f}: how well the model separates the classes (0.5 = random guessing, 1.0 = perfect) - "
                + ("worse than random guessing" if v < 0.5 else band) + ".")
    if k == "precision":
        return f"{N('Precision')} {v:.1%}: when the model says 'positive' it is right about {n100} times out of 100."
    if k in ("recall", "sensitivity"):
        return f"{N('Recall')} {v:.1%}: the model finds about {n100} of every 100 real positive cases."
    if k == "specificity":
        return f"{N('Specificity')} {v:.1%}: it correctly rules out about {n100} of every 100 real negative cases."
    if k == "f1":
        return f"{N('F1')} {v:.2f}: one number balancing precision and recall - {_band(v)}."
    if k == "r2":
        return f"{N(R2)} {v:.2f}: the model explains about {n100}% of the variation in the data."
    if k in ("rmse", "mae", "mse"):
        return f"{N(k.upper())} {v:g}: the average size of the prediction error (lower is better; judge it against the scale of what you predict)."
    return f"{N('Loss')} {v:g}: how wrong the predictions are on average - lower is better; compare training and validation loss to spot overfitting."


def metric_notes(finds):
    notes = [interpret_metric(f) for f in finds]
    by = {(f["qual"], f["metric"]): f["v"] for f in finds}
    for k in {m for _, m in by}:
        if "train" in {q for q, m in by if m == k} and ({"test", "validation"} & {q for q, m in by if m == k}):
            tr = by[("train", k)]
            te = by.get(("test", k), by.get(("validation", k)))
            if k in RATIO and tr - te > 0.05:
                notes.append(f"Training {k} ({tr:.1%}) is clearly higher than test {k} ({te:.1%}): a sign of overfitting.")
            elif k in RATIO:
                notes.append(f"Training and test {k} are close ({tr:.1%} vs {te:.1%}): the model generalises well.")
    return notes


def _clean_lines(text, title):
    out = []
    for l in text.splitlines():
        l = re.sub(r"^[^\w(]+", "", l.strip())           # drop bullet / check-mark symbols
        if l and l != title:
            out.append(l)
    return out


def _as_sentence(line):
    """'Output: Rain/No Rain prediction.'  ->  'The output is Rain/No Rain prediction.'"""
    m = re.match(r"^([A-Za-z][A-Za-z0-9 /&()'-]{1,40}):\s*(.+)$", line)
    if m:
        return f"The {m.group(1).strip().lower()} is {m.group(2).strip().rstrip('.')}."
    return line if line.endswith((".", "!", "?")) else line + "."


def slide_summary(doc, page):
    """One slide/page: written by the AI if a key is set, otherwise topic + EVERY point + what the numbers mean."""
    if not 0 < page <= len(doc["page_text"]):
        return None
    text = doc["page_text"][page - 1]
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return None
    ai = llm(TUTOR, f"Slide text:\n{text[:2500]}\n\nSummarize this slide in your own words in 3-5 simple sentences. "
                    "Then explain anything technical, and what any numbers mean.", 400)
    if ai:
        return {"answer": ai, "speak": _first_sentence(ai), "via": "llm"}
    title = lines[0] if len(lines) > 1 and len(lines[0]) <= 80 and not lines[0].endswith(".") else None
    notes = metric_notes(metric_findings(text))
    body = [l for l in _clean_lines(text, title) if not (notes and METRIC_RE.search(l))]
    parts = []
    if title:
        parts.append(f"This slide is about: {title}.")
    if body and (len(body) >= 2 or notes) and sum(len(l) <= 110 for l in body) >= 0.6 * len(body):   # a bullet slide: cover every point
        parts.append("In short: " + " ".join(_as_sentence(l) for l in body[:8]))
        key = body
    else:
        idxs = [i for i, (p, t) in enumerate(doc["sents"]) if p == page and not METRIC_RE.search(t)]
        key = [doc["sents"][i][1] for i in sorted(textrank(doc, idxs, 2))] if idxs else []
        key = [k for k in key if k and k != title] or [_as_sentence(l) for l in body[:6]]
        if key:
            parts.append("What it says: " + " ".join(key))
    if notes:
        parts.append("In simple words:\n" + "\n".join("\u2022 " + n for n in notes))
    terms = glossary_terms(text, 3)
    if terms:
        parts.append("Terms on this slide:\n" + "\n".join(f"\u2022 {t}: {m}" for t, m in terms))
    if not parts:
        return None
    first = re.sub(r"\s*\(.*?\)", "", notes[0]) if notes else (_as_sentence(key[0]) if key else title or "")
    return {"answer": "\n".join(parts), "speak": first, "via": "rules"}


# ----------------------------------------------- more meanings: Wikipedia + your own glossary --
_WIKI_CACHE = {}
_WIKI_DOWN_UNTIL = 0.0


def _get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "AirDrawingSystem/1.0 (student project)"})
    with urllib.request.urlopen(req, timeout=4) as r:
        return json.load(r)


def wiki_summary(term, hint=""):
    """Short Wikipedia summary for a technical term.  'hint' = topic words of the PDF (so 'transformer'
    in an ML PDF finds the neural network, not the electrical device).  Needs internet."""
    global _WIKI_DOWN_UNTIL
    key = (term.lower(), hint)
    if key in _WIKI_CACHE:
        return _WIKI_CACHE[key]
    if time.time() < _WIKI_DOWN_UNTIL:
        return None
    res = None
    try:
        q = urllib.parse.quote(f"{term} {hint}".strip())
        hits = _get_json("https://en.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=1&srsearch=" + q) \
            .get("query", {}).get("search", [])
        if hits:
            title = hits[0]["title"]
            d = _get_json("https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_")))
            text = d.get("extract") or ""
            related = set(tokens(term)) & set(tokens(title + " " + text))         # safety: must actually be about the term
            if text and d.get("type") != "disambiguation" and related:
                res = {"title": d.get("title", title), "text": " ".join(re.split(r"(?<=[.!?])\s+", text)[:2])[:400]}
    except urllib.error.HTTPError:
        pass
    except Exception:
        _WIKI_DOWN_UNTIL = time.time() + 60
        return None
    _WIKI_CACHE[key] = res
    return res


GLOSSARY.update({
    "svm": "support vector machine - finds the boundary that separates two classes with the widest possible margin",
    "support vector machine": "finds the boundary that separates two classes with the widest possible margin",
    "random forest": "many decision trees trained on random parts of the data whose votes are combined for a steadier prediction",
    "decision tree": "a flowchart-like model that splits the data with yes/no questions until it can decide",
    "k means": "a clustering method that groups points around k centre points",
    "knn": "k-nearest neighbours - classifies a point by the majority class of the k closest training points",
    "naive bayes": "a simple classifier that uses probabilities and assumes the features are independent",
    "logistic regression": "a model that predicts the probability of a class (despite the name it is used for classification)",
    "linear regression": "fits a straight line (or plane) to predict a number from the inputs",
    "pca": "principal component analysis - reduces the number of features while keeping most of the variation",
    "transformer": "a neural network that uses attention to weigh how the parts of its input relate to each other",
    "attention": "a mechanism that lets a model focus on the most relevant parts of its input",
    "lstm": "a recurrent network with memory cells that can remember information over long sequences",
    "rnn": "recurrent neural network - processes a sequence step by step while carrying a memory forward",
    "gan": "generative adversarial network - one network creates fake samples while another tries to spot them",
    "embedding": "a list of numbers representing a word or item so that similar ones sit close together",
    "tokenization": "splitting text into small pieces (tokens) such as words or sub-words",
    "one hot encoding": "turning a category into a vector of zeros with a single 1 marking the category",
    "feature engineering": "creating or transforming input variables so the model can learn better",
    "ensemble": "combining several models so their joint prediction beats any single one",
    "bagging": "training several models on random samples of the data and averaging their predictions",
    "boosting": "training models one after another, each focusing on the mistakes of the previous one",
    "gradient boosting": "boosting where each new model is fitted to the error left by the earlier ones",
    "imbalanced data": "data in which one class is much more common than the others",
    "smote": "a technique that balances classes by creating synthetic examples of the rare class",
    "optimizer": "the algorithm that updates the weights to reduce the loss (for example SGD or Adam)",
    "adam": "a popular optimizer that adapts the learning rate for each weight",
    "sgd": "stochastic gradient descent - gradient descent using small random batches of data",
    "vanishing gradient": "when gradients shrink as they travel back through many layers, so early layers barely learn",
    "parameter": "a value the model learns during training, such as a weight",
    "weights": "the numbers a neural network adjusts during training to make better predictions",
    "threshold": "the cut-off probability above which the model predicts the positive class",
})


def _load_extra_glossary():
    """Add your own terms: create glossary_extra.txt next to study.py with lines like   term: meaning"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "glossary_extra.txt")
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if ":" in line and not line.lstrip().startswith("#"):
                    t, m = line.split(":", 1)
                    if t.strip() and m.strip():
                        GLOSSARY[re.sub(r"[\s-]+", " ", t.strip().lower())] = m.strip()
    except OSError:
        pass


_load_extra_glossary()


# ------------------------------------------------- optional: let an AI write the explanations --
TUTOR = ("You are a friendly study tutor helping a student understand their own slides / PDF. "
         "Use ONLY the text you are given for facts about the document. Write in simple, natural English, in your own "
         "words - never copy sentences word for word. Be short and clear. No markdown headings; use plain sentences.")
_LLM_DOWN_UNTIL = 0.0


def _api_key():
    k = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not k:
        try:
            with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "api_key.txt"), encoding="utf-8") as f:
                k = f.read().strip()
        except OSError:
            k = ""
    return k


_OLLAMA = {"t": 0.0, "model": None}


def _ollama_model():
    """Name of the installed local model (checked at most every 30 s), or None if Ollama is not running."""
    if time.time() - _OLLAMA["t"] < 30:
        return _OLLAMA["model"]
    _OLLAMA["t"], _OLLAMA["model"] = time.time(), None
    want = os.environ.get("OLLAMA_MODEL", "llama3.2")
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=0.8) as r:
            names = [m.get("name", "") for m in json.load(r).get("models", [])]
    except Exception:
        return None
    _OLLAMA["model"] = next((n for n in names if n == want or n.split(":")[0] == want.split(":")[0]), None)
    return _OLLAMA["model"]


def ai_mode():
    return "claude" if _api_key() else ("ollama" if _ollama_model() else "none")


def ai_available():
    return ai_mode() != "none"


def llm_status():
    m = ai_mode()
    if m == "claude":
        return f"ON - Claude ({os.environ.get('CLAUDE_MODEL', 'claude-haiku-5-5')})"
    if m == "ollama":
        return f"ON - free local model {_ollama_model()} (Ollama)"
    return "OFF (basic mode) - add api_key.txt, or install Ollama and run:  ollama pull llama3.2"


def _first_sentence(t):
    return re.split(r"(?<=[.!?])\s", t.strip())[0][:220]


def _llm_ollama(system, user, max_tokens):
    body = json.dumps({"model": _ollama_model(), "stream": False, "options": {"num_predict": max_tokens, "temperature": 0.3},
                       "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request("http://localhost:11434/api/chat", data=body, headers={"content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return (json.load(r).get("message", {}).get("content") or "").strip() or None
    except Exception as e:
        print("[AI] local model error:", type(e).__name__, flush=True)
        return None


def llm(system, user, max_tokens=350):
    """Write an explanation with the best AI available: Claude (if you have a key), else a free local model
    (Ollama), else None - so every feature falls back to basic mode instead of failing."""
    out = _llm_claude(system, user, max_tokens) if _api_key() else None
    if out:
        return out
    return _llm_ollama(system, user, max_tokens) if _ollama_model() else None


def _llm_claude(system, user, max_tokens=350):
    global _LLM_DOWN_UNTIL
    key = _api_key()
    if not key or time.time() < _LLM_DOWN_UNTIL:
        return None
    body = json.dumps({"model": os.environ.get("CLAUDE_MODEL", "claude-haiku-5-5"), "max_tokens": max_tokens,
                       "system": system, "messages": [{"role": "user", "content": user}]}).encode()
    req = urllib.request.Request("https://api.anthropic.com/v1/messages", data=body,
                                 headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            data = json.load(r)
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip() or None
    except urllib.error.HTTPError as e:                # wrong key, wrong model name, no credit ... shown in the terminal
        print("[AI] request failed:", e.code, e.read()[:300].decode("utf-8", "ignore"), flush=True)
        return None
    except Exception as e:
        print("[AI] unavailable:", type(e).__name__, flush=True)
        _LLM_DOWN_UNTIL = time.time() + 30
        return None


def llm_summary(sents):
    return llm(TUTOR, "Key sentences from a document:\n" + "\n".join("- " + x["text"] for x in sents) +
               "\n\nWrite a short summary of the whole document in your own words (4-6 sentences).", 400)