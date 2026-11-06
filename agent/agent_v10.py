"""DS-Harness agent : le LLM n'écrit que du code, le système fait le reste.

Pipeline par tâche :
  1. router      : pas de fichier -> "nofile", train.csv présent -> "fit", sinon "table"
  2. lecture     : CSV robustes (séparateur, virgule décimale, NA, unités, dates, doublons)
  3. prompt      : objectif + dictionnaire + vue compacte des tables + 1 exemple (few-shot dynamique)
  4. génération  : Qwen2.5-Coder-1.5B, 1 passe gloutonne puis échantillons par tâche (vote adaptatif)
  5. exécution   : code Python exécuté avec timeout, variable `answer`
  6. vote        : valeur majoritaire, arrondie comme demandé par l'énoncé
Chaque étape est protégée : solve() renvoie toujours un nombre, dans le budget de temps.
"""
import collections
import contextlib
import warnings
import csv
import datetime
import io
import math
import os
import re
import signal
import threading
import time
import unicodedata

import numpy as np
import pandas as pd

MODEL = os.environ.get("DSH_MODEL", "Qwen/Qwen2.5-Coder-1.5B-Instruct")   # test : DSH_MODEL=Qwen/Qwen3-1.7B
BUDGET = float(os.environ.get("DSH_BUDGET", 29.0))  # secondes utilisées sur les 40 allouées
DEBUG = bool(os.environ.get("DSH_DEBUG"))            # affiche les erreurs (local seulement)
EXEC_TIMEOUT = 1.5     # secondes max par exécution de code généré
MAX_NEW = 500          # PATCH 1: augmenté de 300 à 500 pour éviter les truncations
NSAMP = int(os.environ.get("DSH_SAMPLES", 4))   # échantillons initiaux par tâche pour le vote
# Tables : "v1" = chemin complet de la version à 88 % (bloc _v1_ en fin de fichier : lecture, règles, prompts, exécution) ;
# "v1_then_v2" = règles v1, puis règles v2 seulement si v1 s'abstient, puis LLM ; "v2" = tout en v2.
TABLE_MODE = os.environ.get("DSH_TABLE_MODE", "v1_v2wins")
# Repli LLM. SANITY : écarte les réponses du LLM qui violent la forme demandée (jamais celles des règles).
# LLM_HINTS : ajoute au prompt des tables ce que le parseur a compris. TABLE_SAMPLES : échantillons quand peu de tâches
# vont au LLM. RULES_OFF : mesure seulement ("table", "nofile", "fit", "all") — jamais en soumission.
SANITY = os.environ.get("DSH_SANITY", "1") != "0"
LLM_HINTS = os.environ.get("DSH_HINTS", "1") == "1"
TABLE_SAMPLES = int(os.environ.get("DSH_TABLE_SAMPLES", 8))
RULES_OFF = os.environ.get("DSH_RULES_OFF", "")
# Variantes (désactivées dans la version à 95,8 %) :
# HINT_WEIGHT : dans le vote, plus de poids aux programmes qui utilisent les colonnes repérées par le parseur.
# LIVE_MAX / EXTRA_ROUND : échantillons supplémentaires si peu de tâches vont au LLM et que le vote est serré.
# NOFILE_HINTS : indices aussi pour les tâches sans fichier qui partent au LLM.
HINT_WEIGHT = os.environ.get("DSH_HINT_WEIGHT", "0") == "1"
LIVE_MAX = int(os.environ.get("DSH_LIVE_MAX", 4))
EXTRA_ROUND = os.environ.get("DSH_EXTRA_ROUND", "0") == "1"
NOFILE_HINTS = os.environ.get("DSH_NOFILE_HINTS", "0") == "1"

NA_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "--", "?", "missing", ".", "n.a.", "#n/a", "unknown"}

# --------------------------------------------------------------------------- lecture

def _num_from_text(x):
    """'1 234,5 kg' -> 1234.5 ; '1,234.5' -> 1234.5 ; '3,14' -> 3.14."""
    if not isinstance(x, str):
        return np.nan
    m = re.search(r"[-+]?\d[\d.,\s]*", x.replace("\u202f", " ").replace("\xa0", " "))
    if not m:
        return np.nan
    t = m.group(0).strip().replace(" ", "")
    if "," in t and "." in t:
        dec = "," if t.rfind(",") > t.rfind(".") else "."
        thou = "." if dec == "," else ","
        t = t.replace(thou, "").replace(dec, ".")
    elif "," in t:
        if re.fullmatch(r"[-+]?\d{1,3}(,\d{3})+", t):
            t = t.replace(",", "")
        else:
            t = t.replace(",", ".")
    try:
        sign = -1.0 if x.strip().startswith("-") else 1.0
        return sign * abs(float(t))
    except ValueError:
        return np.nan


def clean_numeric(values):
    """Coerce mixed numeric strings, decimal commas, and simple units; unparseable entries become NaN."""
    def parse(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.number)):
            number = float(value)
            return number if math.isfinite(number) else np.nan
        return _num_from_text(str(value))
    if isinstance(values, pd.Series):
        return values.map(parse).astype(float)
    if np.isscalar(values):
        return parse(values)
    return pd.Series(values).map(parse).astype(float)


def clean_dates(values, dayfirst=False):
    """Parse mixed date strings while refusing to interpret arbitrary numbers as epoch timestamps."""
    def parse(value):
        if pd.isna(value):
            return pd.NaT
        if isinstance(value, (datetime.datetime, datetime.date, pd.Timestamp, np.datetime64)):
            return pd.to_datetime(value, errors="coerce")
        text = str(value).strip()
        compact = re.fullmatch(r"(\d{8})(?:\.0+)?", text)
        if compact:
            return pd.to_datetime(compact.group(1), format="%Y%m%d", errors="coerce")
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
            return pd.NaT
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return pd.to_datetime(text, dayfirst=dayfirst, errors="coerce")
    if isinstance(values, pd.Series):
        return values.map(parse)
    if np.isscalar(values):
        return parse(values)
    return pd.Series(values).map(parse)


def _numeric_metric(values, description="", force=False):
    """Return a finite numeric view for a measure, or None when text looks categorical/date-like.

    Numeric columns keep the existing path. Object columns are only coerced when their values
    look numeric and either nearly all parse or the data dictionary identifies a measurement.
    """
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    nonmissing = values.notna()
    if not nonmissing.any():
        return pd.Series(np.nan, index=values.index, dtype=float)
    desc = str(description).lower()
    measure_hint = bool(re.search(
        r"\b(amount|price|cost|income|sales|revenue|quantity|rate|ratio|percent(?:age)?|"
        r"age|height|weight|temperature|surface|area|distance|duration|count|number of|"
        r"days?|hours?|occupancy|population|volume|energy|score|billed|total|value|"
        r"mass|length|width|depth|probability|salary|wage|speed|pressure|capacity)\b", desc))
    categorical_hint = bool(re.search(
        r"\b(code|coded|identifier|key|reference|category|categorical|status|label)\b", desc))
    if categorical_hint and not measure_hint:
        return None
    if not measure_hint and not force:
        return None

    raw = values.loc[nonmissing]
    # Prevent date objects and common textual dates from being read as their first number.
    if pd.api.types.is_datetime64_any_dtype(values):
        return None
    date_objects = raw.map(lambda x: isinstance(x, (datetime.date, datetime.datetime, pd.Timestamp, np.datetime64)))
    texts = raw.astype(str).str.strip()
    date_text = texts.map(lambda x: bool(_DATE_RE.fullmatch(x) or re.fullmatch(r"\d{8}(?:\.0+)?", x)))
    if date_objects.mean() >= 0.5 or date_text.mean() >= 0.5:
        return None

    numeric_shape = texts.map(lambda x: bool(_NUM_RE.fullmatch(x) or re.fullmatch(r"[$€£]\s*" + _NUM_RE.pattern[1:-1], x)))
    units = []
    for text in texts[numeric_shape]:
        currency = re.match(r"^\s*([$€£])", text)
        core = text[currency.end():] if currency else text
        match = re.match(r"^[-+]?\s*\d[\d.,\s]*\s*(.*)$", core)
        units.append(((currency.group(1) if currency else "") + (match.group(1).strip().casefold() if match else "")))
    if len(set(units)) > 1:
        return None
    parsed = clean_numeric(values)
    parse_rate = parsed.loc[nonmissing].notna().mean()
    shape_rate = numeric_shape.mean()
    threshold = 0.70 if (measure_hint or force) else 0.95
    if parse_rate < threshold or shape_rate < threshold:
        return None
    return parsed.replace([np.inf, -np.inf], np.nan)


def safe_divide(numerator, denominator):
    """Divide while mapping zero denominators and non-finite outputs to NaN."""
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        if isinstance(numerator, pd.Series):
            n = clean_numeric(numerator)
            if isinstance(denominator, pd.Series):
                d = clean_numeric(denominator).reindex(n.index)
            elif np.isscalar(denominator):
                d = denominator
            else:
                d = clean_numeric(pd.Series(denominator, index=n.index))
            result = n.div(d)
        else:
            d = clean_numeric(denominator)
            n = clean_numeric(pd.Series(numerator, index=d.index))
            result = n.div(d)
        if isinstance(result, pd.Series):
            return result.replace([np.inf, -np.inf], np.nan).where(d != 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = np.divide(np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float))
    result = np.where(np.isfinite(result), result, np.nan)
    return float(result) if np.ndim(result) == 0 else result


_NUM_RE = re.compile(r"^[-+]?\s*\d[\d.,\s]*\s*(?:[A-Za-z%$€£°/²³µ.]+[23]?(?:[- ][A-Za-z%$€£°/²³µ.]+[23]?)*)?$")
_DATE_RE = re.compile(r"^\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}([ T]\d{1,2}:\d{2}(:\d{2})?)?$")


def _convert(s, dayfirst):
    nn = s.dropna()
    if nn.empty:
        return s
    if nn.str.match(_NUM_RE).mean() >= 0.9:
        suf = nn.str.extract(r"^[-+]?\s*\d[\d.,\s]*\s*(.*)$")[0].dropna().astype(str).str.strip()
        if len(set(suf) - {""}) <= 1:      # 0 ou 1 unité : vraie colonne numérique
            return s.map(_num_from_text).astype(float)
        return s                           # plusieurs suffixes (5A, 5B...) : un code, on le garde en texte
    if nn.str.match(_DATE_RE).mean() >= 0.9:
        parts = nn.str.extract(r"^(\d{1,2})[-/.](\d{1,2})[-/.]\d{4}")
        if parts[0].notna().mean() >= 0.9:        # jour/mois ou mois/jour : les valeurs > 12 tranchent
            if pd.to_numeric(parts[0]).max() > 12:
                dayfirst = True
            elif pd.to_numeric(parts[1]).max() > 12:
                dayfirst = False
        return pd.to_datetime(s, dayfirst=dayfirst, errors="coerce")
    return s


class Tables(dict):
    """dfs : {nom de fichier : DataFrame}, avec le dictionnaire de données (.dictionary) et {colonne : description} (.desc)."""
    dictionary = None
    desc = {}


def read_table(path, dayfirst=False, convert=True, na_extra=()):
    raw = None
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(path, "rb") as fh:
                raw = fh.read().decode(enc)
            break
        except Exception:
            continue
    head = raw[:5000]
    try:
        sep = csv.Sniffer().sniff(head, delimiters=",;\t|").delimiter
    except Exception:
        sep = max(",;\t|", key=head.count)
    df = pd.read_csv(io.StringIO(raw), sep=sep, dtype=str, keep_default_na=False,
                     skipinitialspace=True, on_bad_lines="skip", engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        s = s.where(~s.str.lower().isin(NA_TOKENS | set(na_extra)), np.nan)
        df[c] = _convert(s, dayfirst) if convert else s
    return df


def stem(path):
    return os.path.splitext(os.path.basename(path))[0]


def route(files):
    names = {os.path.basename(f).lower() for f in files or []}
    if not files:
        return "nofile"
    if "train.csv" in names:
        return "fit"
    return "table"


def load_task(objective, files):
    """-> (dfs, dictionary_text). Le dictionnaire est lu sans conversion."""
    low = objective.lower()
    dayfirst = bool(re.search(r"day[- ]?first|day/month|day-month|dd/mm|d/m/y|dd-mm|dd\.mm", low))
    dedupe = _dedupe_requested(low)
    # jetons de valeur manquante cités entre guillemets dans l'énoncé (« Missing values are written 'nd' »)
    na_extra = set()
    for sent in re.split(r"(?<=[.;?])\s+", objective):
        if re.search(r"missing|not recorded|unknown|blank|not available", sent, re.I):
            na_extra |= {q.strip().lower() for q in re.findall(r"['\"“‘]([^'\"”’]{1,12})['\"”’]", sent)}
    na_extra.discard("")
    dfs, dico = Tables(), ""
    for f in files or []:
        name = stem(f)
        try:
            if name == "data_dictionary":
                d = read_table(f, convert=False)
                dico = "\n".join(" | ".join(str(v) for v in row if str(v) != "nan") for row in d.values.tolist())
                dfs.dictionary = d
                dfs.desc = {c: ds for _, c, ds in _dico_rows(dico)}
                continue
            d = read_table(f, dayfirst=dayfirst, na_extra=na_extra)
            dfs[name] = d
        except Exception:
            if DEBUG:
                import traceback
                print(f"[debug] lecture impossible : {f}")
                traceback.print_exc()
            continue
    _apply_data_cleanup(dfs, objective, dico, dedupe)
    return dfs, dico


def _key_variants(v):
    """Formes plausibles d'une clé : telle quelle, minuscule, sans préfixe 'XXX-', chiffres seuls (sans zéros de tête)."""
    t = unicodedata.normalize("NFKC", str(v)).strip().casefold()
    folded = "".join(ch for ch in unicodedata.normalize("NFKD", t) if not unicodedata.combining(ch))
    out, pending = [], [t, folded]
    while pending:
        item = pending.pop()
        if not item or item in out:
            continue
        out.append(item)
        compact = re.sub(r"[^a-z0-9]+", "", item)
        if compact and compact != item:
            pending.append(compact)
        for sep in ("-", "_", ":"):
            if sep in item:
                pending.append(item.split(sep)[-1])
        decimal_id = re.fullmatch(r"([a-z]*\d+)\.0+", item)
        if decimal_id:
            pending.append(decimal_id.group(1))
        m = re.fullmatch(r"[a-z]*0*(\d+)", item)
        if m:
            pending.append(m.group(1))
    return out


def _composite_join_hint(dfs, dico):
    """Create a temporary verified join key when two semantically matched fields form a composite key."""
    import itertools
    if len(dfs) != 2:
        return None
    (a_name, A), (b_name, B) = list(dfs.items())
    a_desc = _table_descriptions(a_name, A.columns, dico)
    b_desc = _table_descriptions(b_name, B.columns, dico)
    generic = {"number", "code", "id", "key", "identifier", "value", "date", "year", "ref", "reference"}
    matches = []
    for ac in A.columns:
        at = _toks(str(ac) + " " + a_desc.get(ac, "")) - generic
        scores = []
        for bc in B.columns:
            bt = _toks(str(bc) + " " + b_desc.get(bc, "")) - generic
            common = at & bt
            if common:
                scores.append((len(common), bc))
        scores.sort(reverse=True)
        if scores and (len(scores) == 1 or scores[0][0] > scores[1][0]):
            matches.append((ac, scores[0][1], scores[0][0]))
    if len(matches) < 2:
        return None

    def tuple_variants(values):
        components = [_key_variants(value)[:6] for value in values]
        if any(not xs for xs in components):
            return set()
        return set(itertools.product(*components))

    candidates = []
    for first, second in itertools.combinations(matches, 2):
        if first[0] == second[0] or first[1] == second[1]:
            continue
        for fact_name, fact, fact_cols, dim_name, dim, dim_cols, semantic in (
            (a_name, A, (first[0], second[0]), b_name, B, (first[1], second[1]), first[2] + second[2]),
            (b_name, B, (first[1], second[1]), a_name, A, (first[0], second[0]), first[2] + second[2]),
        ):
            dim_rows = dim[list(dim_cols)].dropna()
            if len(dim_rows) < 2:
                continue
            dim_tuples = [tuple(row) for row in dim_rows.itertuples(index=False, name=None)]
            if len(set(dim_tuples)) < 0.95 * len(dim_tuples):
                continue
            fact_rows = fact[list(fact_cols)].dropna()
            fact_tuples = set(tuple(row) for row in fact_rows.itertuples(index=False, name=None))
            if len(fact_tuples) < 2 or len(fact_tuples) >= 0.95 * len(fact_rows):
                continue
            lookup, ambiguous = {}, set()
            for idx, values in zip(dim_rows.index, dim_tuples):
                for key in tuple_variants(values):
                    if key in lookup and lookup[key] != idx:
                        ambiguous.add(key)
                    else:
                        lookup.setdefault(key, idx)
            found = 0
            for values in fact_tuples:
                rows = {lookup[key] for key in tuple_variants(values) if key in lookup and key not in ambiguous}
                found += len(rows) == 1
            coverage = found / len(fact_tuples)
            if coverage >= 0.9:
                candidates.append((coverage, semantic, fact_name, fact_cols, dim_name, dim_cols,
                                   lookup, ambiguous, dim_rows.index.tolist()))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][:2] == best[:2] and (candidates[1][2:6] != best[2:6]):
        return None

    _, _, fact_name, fact_cols, dim_name, dim_cols, lookup, ambiguous, dim_indices = best
    fact, dim = dfs[fact_name], dfs[dim_name]
    key_col = "__dsh_join_key"
    suffix = 1
    while key_col in fact.columns or key_col in dim.columns:
        key_col = f"__dsh_join_key_{suffix}"
        suffix += 1
    ids = {idx: f"dsh{n}" for n, idx in enumerate(dim_indices)}
    dim[key_col] = [ids.get(idx) for idx in dim.index]
    fact_ids = []
    for values in fact[list(fact_cols)].itertuples(index=False, name=None):
        if any(pd.isna(value) for value in values):
            fact_ids.append(np.nan)
            continue
        rows = {lookup[key] for key in tuple_variants(values) if key in lookup and key not in ambiguous}
        fact_ids.append(ids[next(iter(rows))] if len(rows) == 1 else np.nan)
    fact[key_col] = fact_ids
    return (
        f"Join key: dfs['{fact_name}']['{key_col}'] matches dfs['{dim_name}']['{key_col}'] "
        f"(verified composite match: {fact_cols} ↔ {dim_cols}; temporary keys were added). "
        "Use this key instead of either component alone."
    )


def align_keys(dfs, dico=""):
    """Détecte, pour chaque couple de tables, la colonne-clé de la table de faits (valeurs répétées) dont les
    valeurs sont incluses dans celles d'une colonne unique de l'autre table, à la casse / aux préfixes / aux zéros
    près. Réécrit la clé de faits comme la clé de référence et renvoie des phrases d'indice pour le prompt."""
    cands = []
    entity_aliases = {
        "orders": "order", "products": "product", "items": "product", "item": "product", "sku": "product",
        "customers": "customer", "clients": "customer", "client": "customer",
        "suppliers": "supplier", "vendors": "supplier", "vendor": "supplier",
        "students": "student", "pupils": "student", "schools": "school",
        "loans": "loan", "shipments": "shipment", "shipment": "shipment", "parcels": "shipment", "deliveries": "shipment",
        "readings": "reading", "flats": "flat", "employees": "employee", "patients": "patient",
        "regions": "region", "countries": "country", "cities": "city", "carriers": "carrier",
        "invoices": "invoice", "payments": "payment", "transactions": "transaction",
    }
    descs = {(stem(f).lower(), c): desc for f, c, desc in _dico_rows(dico)}

    def entities(table, column):
        words = re.findall(r"[a-z]+", (str(column) + " " + descs.get((table, column), "")).lower())
        return {entity_aliases.get(w, entity_aliases.get(w[:-1], w[:-1] if w.endswith("s") else w))
                for w in words} & set(entity_aliases.values())

    def is_measure(table, column):
        text = (str(column) + " " + descs.get((table, column), "")).lower()
        identity = re.search(r"\b(id|identifier|key|reference|ref|code|number(?!\s+of))\b", text)
        measure = re.search(r"\b(quantity|amount|price|sales|revenue|weight|cost|rate|score|age|height|width|duration|total|count|units?|value|number of)\b", text)
        return bool(measure and not identity)

    names = list(dfs)
    for bn in names:
        B = dfs[bn]
        for bc in B.columns:
            bs = B[bc].dropna()
            if len(bs) < 2 or bs.nunique() < len(bs) * 0.95 or bs.dtype.kind == "M":
                continue
            lookup, ambiguous = {}, set()
            for orig in bs.unique():
                key = int(orig) if isinstance(orig, float) and float(orig).is_integer() else orig
                for var in _key_variants(key):
                    if var in lookup and lookup[var] != orig:
                        ambiguous.add(var)
                    else:
                        lookup.setdefault(var, orig)
            exact_lookup = {str(v).strip().lower(): v for v in bs.unique()}
            for an in names:
                if an == bn:
                    continue
                A = dfs[an]
                for ac in A.columns:
                    a_s = A[ac].dropna()
                    if len(a_s) < 2 or a_s.dtype.kind == "M":
                        continue
                    if is_measure(an, ac) or is_measure(bn, bc):
                        continue
                    a_entities, b_entities = entities(an, ac), entities(bn, bc)
                    # Avoid treating unrelated numeric-looking IDs as the same key (e.g. SKU P01 and order 001).
                    if a_entities and b_entities and not (a_entities & b_entities):
                        continue
                    uniq = list(a_s.unique())
                    if len(uniq) < 2 or len(uniq) > bs.nunique():
                        continue
                    mapped = {}
                    for u in uniq:
                        uu = int(u) if isinstance(u, float) and float(u).is_integer() else u
                        exact = exact_lookup.get(str(uu).strip().lower())
                        if exact is not None:
                            mapped[u] = exact
                            continue
                        for var in _key_variants(uu):
                            if var in lookup and var not in ambiguous:
                                mapped[u] = lookup[var]
                                break
                    cov = len(mapped) / len(uniq)
                    if cov < 0.9:
                        continue
                    both_text = A[ac].dtype.kind not in "fiu" and B[bc].dtype.kind not in "fiu"
                    repeat = 1 - len(uniq) / len(a_s)          # la clé de faits a des valeurs répétées
                    text_to_num = A[ac].dtype.kind not in "fiu" and B[bc].dtype.kind in "fiu"
                    semantic_match = bool(a_entities and b_entities and a_entities & b_entities)
                    cands.append((cov + 0.5 * both_text + 0.3 * repeat + 0.1 * text_to_num + 0.2 * semantic_match,
                                  an, ac, bn, bc, mapped))
    notes, done = [], set()
    cands.sort(key=lambda c: -c[0])
    for i, (score, an, ac, bn, bc, mapped) in enumerate(cands):
        pair = frozenset((an, bn))
        if pair in done:
            continue
        alternatives = [c for c in cands[i + 1:] if frozenset((c[1], c[3])) == pair]
        if alternatives and score - alternatives[0][0] < 0.05:
            # If two different column pairs are equally plausible, leave key selection to the model.
            done.add(pair)
            continue
        done.add(pair)
        if any(k != v for k, v in mapped.items()):
            dfs[an][ac] = dfs[an][ac].map(lambda x, m=mapped: m.get(x, x))
        notes.append(f"Join key: dfs['{an}']['{ac}'] matches dfs['{bn}']['{bc}'] "
                     "(already normalized: merge with left_on / right_on).")
    if len(dfs) == 2 and not notes:
        composite = _composite_join_hint(dfs, dico)
        if composite:
            notes.append(composite)
    return notes


def fit_hints(objective, dico, dfs):
    notes, desc2code = [], {}
    for line in dico.split("\n"):
        parts = [p.strip() for p in line.split(" | ")]
        if len(parts) >= 3:
            desc2code.setdefault(parts[2].lower(), set()).add(parts[1])
    for q in re.findall(r'"([^"]+)"', objective):
        codes = desc2code.get(q.strip().lower())
        if codes:
            notes.append("Do NOT use column(s) " + ", ".join(sorted(codes)) + f' (described as "{q}").')
    te = dfs.get("test")
    m = re.search(r"\b(\d+)\s+of test\.csv", objective)
    if m and te is not None and len(te.columns):
        idc = te.columns[0]
        notes.append(f"'row {m.group(1)} of test.csv' means the row where dfs['test']['{idc}'] == {m.group(1)}, "
                     "not the row at that position.")
    return notes


def entity_hints(objective, dico):
    """Le dictionnaire décrit des colonnes codées (« coded (S01 Bordeaux office, S02 Grenoble lab) ») ; l'énoncé cite
    le nom, pas le code. On dit au modèle quel code utiliser."""
    notes, low = [], objective.lower()
    for _, col, desc in _dico_rows(dico):
        m = re.search(r"coded \((.*)\)", desc)
        if not m:
            continue
        for item in re.split(r",\s*", m.group(1)):
            parts = item.strip().split(" ", 1)
            if len(parts) == 2 and len(parts[1]) > 2 and parts[1].lower() in low:
                notes.append(f"'{parts[1]}' corresponds to the code '{parts[0]}' in column {col} (the data stores the code, not the name).")
    return notes


def describe(dfs):
    out = []
    for name, d in dfs.items():
        cols = ", ".join(
            f"{c} ({str(d[c].dtype)[:8]}" + (f", NaN {d[c].isna().mean():.0%}" if d[c].isna().any() else "") + ")"
            for c in d.columns)
        out.append(f"dfs['{name}']: {len(d)} rows; columns: {cols}")
        out.append(d.head(3).to_string(max_cols=14, max_colwidth=14))
    return "\n".join(out)


def hints(objective):
    low = objective.lower()
    h = []
    m = re.search(r"(-\d+)[^.]{0,60}(not recorded|missing|unknown|no value)", low)
    if m or "not recorded" in low:
        h.append("Some column uses -1 for 'not recorded': replace it by NaN in that column before computing.")
    if "leave out" in low or "exclude" in low or "ignore" in low:
        h.append("Respect the instruction about a column or rows to leave out.")
    if "case" in low and "insensitive" in low:
        h.append("Compare text case-insensitively and strip spaces.")
    return h


# --------------------------------------------------------------------------- prompts

# PATCH 2: Ajout d'une consigne forte pour interdire les `.round()` dans le code
SYSTEM = (
    "You are a careful data scientist. Reply ONLY with a short Python program in one code block. "
    "The program must end by setting a variable `answer` to ONE number. No print, no explanation, no comments, "
    "no imports, at most 12 lines. Tables are already loaded in `dfs`: never call read_csv. "
    "Never compute by hand: let Python compute. "
    "DO NOT round the answer yourself (never use round() or .round()), the system will do the rounding."
)

TOOLS = {
    "nofile": "Available: math, numpy as np, sympy, scipy.stats as stats, datetime, safe_divide. "
              "Use safe_divide(a, b) when b may be zero; zero/non-finite results become NaN.",
    "table": "Available: pandas as pd, numpy as np, math, scipy.stats as stats. The tables are in dict `dfs` "
             "(keys = file names without .csv). Numeric/date types are inferred only when parsing is unambiguous; "
             "mixed columns may remain object, so inspect the dictionary, dtypes, and sample rows. Use clean_numeric(series) "
             "only for dictionary-described measures, never for IDs, codes, or categories; use clean_dates(series, "
             "dayfirst=...) for mixed date strings. Column codes are short: use ONLY the exact codes listed under each table. "
             "For joins, prefer any supplied Join key hint and check key uniqueness and match coverage with merge(indicator=True); "
             "preserve duplicate rows unless asked to remove them. Then filter and aggregate. For ratios with possible "
             "zero denominators use safe_divide(numerator, denominator).",
    "fit": "Available: pandas as pd, numpy as np, math, stats, LinearRegression, LogisticRegression, r2_score. "
           "Tables are in dict `dfs` ('train', 'test'), already parsed (floats, NaN). Column codes are short: "
           "find them in the dictionary. Drop rows with NaN in the used columns before fitting. "
           "LinearRegression and LogisticRegression automatically one-hot encode selected text columns; "
           "unknown test categories are ignored. Use safe_divide(a, b) for possible zero denominators.",
}

EXAMPLES = {
    "nofile": (
        "Objective: A price of 80 rises by 15% then falls by 10%. What is the final price? Round to 2 decimals.",
        "import math\nanswer = 80 * 1.15 * 0.90",
    ),
    "fit": (
        "Tables:\ndfs['train']: 80 rows; columns: a1 (float64), a2 (float64), a3 (float64)\n"
        "dfs['test']: 20 rows; columns: a2 (float64), a3 (float64)\n"
        "Dictionary:\na1 | weight in kg\na2 | height in cm\na3 | age in years\n"
        "Objective: Fit a linear regression of weight on height and age on train.csv. "
        "Predict the weight of the first row of test.csv. Round to 2 decimals.",
        "tr = dfs['train'].dropna(subset=['a1', 'a2', 'a3'])\n"
        "m = LinearRegression().fit(tr[['a2', 'a3']], tr['a1'])\n"
        "answer = m.predict(dfs['test'][['a2', 'a3']].iloc[[0]])[0]",
    ),
}

EXAMPLES["join"] = (
    "Tables:\ndfs['orders']: 8 rows; columns: o1 (float64), o2 (float64), o3 (float64)\n"
    "dfs['clients']: 5 rows; columns: k1 (float64), k2 (object), k3 (float64)\n"
    "Dictionary:\no1 | client id\no2 | order amount in euros\no3 | order year\n"
    "k1 | client id\nk2 | client city\nk3 | client age\n"
    "Objective: Total order amount of the clients living in 'Paris' (join orders with clients on client id). "
    "Round to 2 decimals.",
    "m = dfs['orders'].merge(dfs['clients'], left_on='o1', right_on='k1', how='inner')\n"
    "m = m[m['k2'].str.lower() == 'paris']\n"
    "answer = m['o2'].sum()",
)

# OPTIMISATION 1: Dictionnaire dynamique d'exemples pour les Tables
TABLE_EXAMPLES = [
    (r"standard deviation|std dev", (
        "Tables:\ndfs['data']: 10 rows; columns: c1 (float64), c2 (object)\nDictionary:\nc1 | weight\nc2 | category\nObjective: Standard deviation of weight for category 'A'.",
        "d = dfs['data']\nd = d[d['c2'].str.lower() == 'a']\nanswer = d['c1'].std()"
    )),
    (r"correlation", (
        "Tables:\ndfs['stats']: 15 rows; columns: x (float64), y (float64)\nDictionary:\nx | height\ny | age\nObjective: Correlation between height and age.",
        "d = dfs['stats'].dropna(subset=['x', 'y'])\nanswer = d['x'].corr(d['y'])"
    )),
    (r"second (largest|highest)|2nd (largest|highest)", (
        "Tables:\ndfs['scores']: 8 rows; columns: player (object), points (float64)\nDictionary:\nplayer | Name\npoints | Score\nObjective: What is the second highest score?",
        "s = dfs['scores']['points'].dropna().drop_duplicates()\nanswer = s.nlargest(2).iloc[-1]"
    )),
    (r"", (  # Fallback par défaut
        "Tables:\ndfs['sales']: 6 rows; columns: c1 (object), c2 (float64), c3 (float64)\nDictionary:\nc1 | product category\nc2 | unit price in euros\nc3 | quantity sold\nObjective: Average revenue (price x quantity) of the category 'toys'.",
        "d = dfs['sales']\nd = d[d['c1'].str.lower() == 'toys']\nanswer = (d['c2'] * d['c3']).mean()"
    ))
]

def get_table_example(objective):
    low_obj = objective.lower()
    for pattern, ex in TABLE_EXAMPLES:
        if pattern and re.search(pattern, low_obj):
            return ex
    return TABLE_EXAMPLES[-1][1]


def build_user(kind, objective, dfs, dico, notes=()):
    parts = []
    if kind != "nofile":
        parts.append("Tables:\n" + describe(dfs))
        if dico:
            parts.append("Dictionary:\n" + dico)
    parts += list(notes)
    parts.append("Objective: " + objective.strip())
    parts += hints(objective)
    return "\n".join(parts)


OLD_TABLE_TOOLS = ("Available: pandas as pd, numpy as np, math, scipy.stats as stats. The tables are in dict `dfs` "
                   "(keys = file names without .csv). They are already parsed: numbers are floats, missing values are NaN, "
                   "decimal commas and units are removed. Column codes are short: find them in the dictionary.")


def build_messages(kind, user, objective, ntables=1):
    if kind == "table" and ntables >= 2:
        ex_u, ex_a = EXAMPLES["join"]
    elif kind == "table":
        ex_u, ex_a = get_table_example(objective)
    else:
        ex_u, ex_a = EXAMPLES[kind]
        
    return [
        {"role": "system", "content": SYSTEM + "\n" + (OLD_TABLE_TOOLS if kind == "table" and os.environ.get("DSH_OLD_PROMPT") else TOOLS[kind])},
        {"role": "user", "content": ex_u},
        {"role": "assistant", "content": "```python\n" + ex_a + "\n```"},
        {"role": "user", "content": user},
    ]


# --------------------------------------------------------------------------- exécution


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _make_models(unreg=False):
    """LinearRegression / LogisticRegression tolérants aux NaN (sklearn plante sinon) :
    fit ignore les lignes incomplètes, predict impute par la moyenne d'apprentissage.
    Si l'énoncé dit « sans régularisation », la logistique est non pénalisée (le défaut sklearn, C=1, change les
    probabilités) ; elle est résolue sur variables standardisées puis ramenée à l'échelle d'origine.
    Les colonnes catégorielles textuelles sont encodées en one-hot, sans toucher au chemin numérique habituel."""
    from sklearn.linear_model import LinearRegression as _LR, LogisticRegression as _LogR
    from sklearn.preprocessing import OneHotEncoder as _OHE

    def _clean_fit(self, X, y, *a, **k):
        raw = pd.DataFrame(X).copy()
        yv = pd.Series(np.asarray(y).ravel(), index=raw.index)
        numeric, categorical = [], []
        converted = {}
        for col in raw.columns:
            values = pd.to_numeric(raw[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            nonmissing = raw[col].notna()
            rate = values[nonmissing].notna().mean() if nonmissing.any() else 0.0
            if pd.api.types.is_numeric_dtype(raw[col]) or rate >= 0.9:
                numeric.append(col)
                converted[col] = values
            else:
                categorical.append(col)

        if not categorical:
            Xd = pd.DataFrame({col: converted[col] for col in raw.columns}, index=raw.index)
            ok = Xd.notna().all(axis=1) & yv.notna()
            if pd.api.types.is_numeric_dtype(yv):
                y_num = pd.to_numeric(yv, errors="coerce")
                ok &= np.isfinite(y_num)
                yv = y_num
            self._mu = Xd.loc[ok].mean()
            self._dsh_numeric_cols = list(Xd.columns)
            self._dsh_numeric_coef_idx = {col: i for i, col in enumerate(Xd.columns)}
            self._dsh_cat_cols = []
            self._dsh_ohe = None
            self._dsh_input_cols = list(raw.columns)
            self._dsh_encoded_names = list(raw.columns)
            return Xd.loc[ok].values, yv.loc[ok].values

        Xd = pd.DataFrame(index=raw.index)
        for col in numeric:
            Xd[col] = converted[col]
        for col in categorical:
            Xd[col] = raw[col]
        ok = Xd.notna().all(axis=1) & yv.notna()
        if pd.api.types.is_numeric_dtype(yv):
            y_num = pd.to_numeric(yv, errors="coerce")
            ok &= np.isfinite(y_num)
            yv = y_num
        Xd, yv = Xd.loc[ok], yv.loc[ok]
        if Xd.empty:
            return np.empty((0, 0)), yv.values

        self._dsh_numeric_cols = list(numeric)
        self._dsh_cat_cols = list(categorical)
        self._dsh_input_cols = list(raw.columns)
        self._mu = Xd[numeric].mean() if numeric else pd.Series(dtype=float)
        if categorical:
            cat = Xd[categorical].astype(str)
            try:
                encoder = _OHE(handle_unknown="ignore", drop="first", sparse_output=False)
            except TypeError:  # scikit-learn < 1.2
                encoder = _OHE(handle_unknown="ignore", drop="first", sparse=False)
            cat_encoded = encoder.fit_transform(cat)
            self._dsh_ohe = encoder
            try:
                cat_names = list(encoder.get_feature_names_out(categorical))
            except Exception:
                cat_names = [f"{col}__onehot_{i}" for col in categorical for i in range(cat_encoded.shape[1])]
        else:
            cat_encoded = np.empty((len(Xd), 0))
            self._dsh_ohe = None
            cat_names = []
        num_encoded = Xd[numeric].to_numpy(dtype=float) if numeric else np.empty((len(Xd), 0))
        self._dsh_numeric_coef_idx = {col: i for i, col in enumerate(numeric)}
        self._dsh_encoded_names = list(numeric) + cat_names
        encoded = np.column_stack([num_encoded, cat_encoded])
        if encoded.shape[1] == 0:
            encoded = np.zeros((len(Xd), 1), dtype=float)
            self._dsh_encoded_names = ["__constant__"]
            self._dsh_numeric_coef_idx = {}
        return encoded, yv.values

    def _clean_X(self, X):
        raw = pd.DataFrame(X, columns=getattr(self, "_dsh_input_cols", None)).copy()
        if not getattr(self, "_dsh_cat_cols", []):
            Xd = raw.reindex(columns=self._dsh_numeric_cols).apply(pd.to_numeric, errors="coerce")
            Xd = Xd.replace([np.inf, -np.inf], np.nan).fillna(self._mu)
            return Xd.to_numpy(dtype=float)
        Xd = pd.DataFrame(index=raw.index)
        for col in self._dsh_numeric_cols:
            values = pd.to_numeric(raw[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            Xd[col] = values.fillna(self._mu.get(col, 0.0))
        num_encoded = Xd[self._dsh_numeric_cols].to_numpy(dtype=float) if self._dsh_numeric_cols else np.empty((len(raw), 0))
        cat = raw.reindex(columns=self._dsh_cat_cols).fillna("__missing__").astype(str)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            cat_encoded = self._dsh_ohe.transform(cat)
        return np.column_stack([num_encoded, cat_encoded])

    class LinearRegression(_LR):
        def fit(self, X, y, *a, **k):
            Xv, yv = _clean_fit(self, X, y)
            return super().fit(Xv, yv, *a, **k)

        def predict(self, X):
            return super().predict(_clean_X(self, X))

        def score(self, X, y, *a, **k):
            Xv = _clean_X(self, X)
            yv = pd.to_numeric(pd.Series(np.asarray(y).ravel()), errors="coerce")
            ok = yv.notna() & np.isfinite(yv)
            return super().score(Xv[ok.to_numpy()], yv[ok].to_numpy(), *a, **k)

    class LogisticRegression(_LogR):
        def __init__(self, penalty="l2", C=1.0, tol=1e-4, max_iter=5000, class_weight=None,
                     solver="lbfgs", fit_intercept=True, random_state=None, **ignored):
            if unreg:                      # sans régularisation : lbfgs seul le permet partout, tolérance serrée
                penalty, solver, tol = None, "lbfgs", 1e-9
            super().__init__(penalty=penalty, C=C, tol=tol, max_iter=max_iter, class_weight=class_weight,
                             solver=solver, fit_intercept=fit_intercept, random_state=random_state)

        def fit(self, X, y, *a, **k):
            Xv, yv = _clean_fit(self, X, y)
            if yv.dtype == object or yv.dtype.kind in "US":
                yv = pd.Series(yv).astype(str).str.strip().str.lower().isin(["1", "yes", "y", "true"]).astype(int).values
            if not unreg:
                return super().fit(Xv, yv, *a, **k)
            mu, sd = Xv.mean(axis=0), Xv.std(axis=0)
            sd[sd == 0] = 1.0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                super().fit((Xv - mu) / sd, yv, *a, **k)
            self.coef_ = self.coef_ / sd
            self.intercept_ = self.intercept_ - (self.coef_ * mu).sum(axis=1)
            return self

        def predict(self, X):
            return super().predict(_clean_X(self, X))

        def predict_proba(self, X):
            return super().predict_proba(_clean_X(self, X))

    return LinearRegression, LogisticRegression


def _resolve_col(key, columns, desc):
    cols = [str(c) for c in columns]
    k = key.strip()
    low = {c.lower(): c for c in cols}
    if k.lower() in low:
        return low[k.lower()]
    norm = lambda x: re.sub(r"[^a-z0-9]", "", x.lower())
    nm = {norm(c): c for c in cols}
    if norm(k) in nm:
        return nm[norm(k)]
    kt = _toks(re.sub(r"[_\W]+", " ", k))
    if kt:
        scores = []
        for c in cols:
            ct = _toks(desc.get(c, "")) | _toks(re.sub(r"[_\W]+", " ", c))
            sc = sum(1 for a in kt if any(a == b or (len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a)))
                                           for b in ct))
            scores.append((sc, c))
        scores.sort(reverse=True)
        if scores and scores[0][0] >= 1 and (len(scores) == 1 or scores[0][0] > scores[1][0]):
            return scores[0][1]
    import difflib
    m = difflib.get_close_matches(k, cols, n=1, cutoff=0.8)
    return m[0] if m else None


class SmartFrame(pd.DataFrame):
    """DataFrame dont df['colonne_inexistante'] cherche la colonne voulue au lieu de lever KeyError."""
    _desc = {}

    @property
    def _constructor(self):
        return SmartFrame

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            if isinstance(key, str):
                alt = _resolve_col(key, self.columns, SmartFrame._desc)
                if alt is not None:
                    return super().__getitem__(alt)
            elif isinstance(key, list) and key and all(isinstance(x, str) for x in key):
                alts = [x if x in self.columns else _resolve_col(x, self.columns, SmartFrame._desc) for x in key]
                if all(a is not None for a in alts):
                    return super().__getitem__(alts)
            raise


def _fake_read_csv(cdfs, extra=None):
    extra = extra or {}

    def read_csv(path, *a, **k):
        name = stem(str(path))
        if name in extra:
            return extra[name].copy()
        if name in cdfs:
            return cdfs[name].copy()
        for key, d in cdfs.items():
            if key in name or name in key:
                return d.copy()
                
        # PATCH 3: Fallback pour éviter les FileNotFoundError si le LLM invente 'school.csv'
        if cdfs:
            return list(cdfs.values())[0].copy()
            
        raise FileNotFoundError(path)
    return read_csv


LAST_ERR = ""
LAST_STATE = ""


def _runtime_state(ns, max_frames=4):
    """Compact DataFrame diagnostics captured after generated code fails."""
    candidates = []
    seen = set()
    for name, value in ns.items():
        if name == "dfs" or name.startswith("__"):
            continue
        if isinstance(value, pd.DataFrame):
            priority = 0 if re.search(r"merge|join|result|output|\bm\b", name, re.I) else 1
            candidates.append((priority, name, value))
    tables = ns.get("dfs")
    if isinstance(tables, dict):
        for name, value in tables.items():
            if isinstance(value, pd.DataFrame):
                candidates.append((2, f"dfs['{name}']", value))
    candidates.sort(key=lambda item: item[0])
    reports = []
    for _, name, frame in candidates:
        if id(frame) in seen:
            continue
        seen.add(id(frame))
        try:
            details = []
            for col in list(frame.columns)[:12]:
                series = frame[col]
                missing = series.isna().mean()
                details.append(f"{col}:{series.dtype},NaN={missing:.0%}")
            head = frame.head(2).to_string(index=False, max_cols=8, max_colwidth=28)
            reports.append(f"{name} shape={frame.shape}; " + ", ".join(details) + "\nhead(2): " + head[:900])
        except Exception as error:
            reports.append(f"{name}: diagnostic failed ({type(error).__name__})")
        if len(reports) >= max_frames:
            break
    return "\n\n".join(reports)[:5000]


def _run_with_timeout(fn, timeout):
    """Exécute fn dans un thread ; au-delà de `timeout` on y injecte une exception (boucle Python infinie) et on
    abandonne le thread s'il ne s'arrête pas. Fonctionne quel que soit le thread qui appelle solve()."""
    import ctypes
    box = {}

    def target():
        try:
            box["v"] = fn()
        except BaseException as e:
            box["e"] = e

    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(max(0.2, timeout))
    if th.is_alive():
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(th.ident), ctypes.py_object(_Timeout))
        th.join(0.3)
        raise _Timeout("exec timeout")
    if "e" in box:
        raise box["e"]
    return box.get("v")


def _exec_generated(code, ns):
    """Seul point d'exécution du code généré (partagé par les deux chemins)."""
    exec(code, ns)


def run_code(code, dfs, timeout, obj=""):
    """Exécute le code généré, renvoie un float ou None."""
    from scipy import stats
    from sklearn.metrics import r2_score
    unreg = bool(re.search(r"without regulari[sz]|no regulari[sz]|unregulari[sz]|unpenali|without penal|no penalty|penalty=none", obj.lower()))
    LinearRegression, LogisticRegression = _make_models(unreg)
    
    # PATCH 4: Remplacement des vieilles méthodes scipy.stats
    for _n in ("nanmean", "nanmedian", "nanstd", "nansum"):   # retirés de scipy.stats, le modèle les utilise encore
        if not hasattr(stats, _n):
            setattr(stats, _n, getattr(np, _n))
    if not hasattr(stats, "median"):
        setattr(stats, "median", np.median)
    if not hasattr(stats, "scoreatpercentile"):
        setattr(stats, "scoreatpercentile", np.percentile)
        
    import sympy

    try:
        SmartFrame._desc = dict(getattr(dfs, "desc", {}) or {})
        cdfs = {k: SmartFrame(v) for k, v in dfs.items()}
    except Exception:
        cdfs = {k: v.copy() for k, v in dfs.items()}
    dd = getattr(dfs, "dictionary", None)
    extra = {"data_dictionary": dd} if dd is not None else {}
    ns = {
        "pd": pd, "np": np, "math": math, "stats": stats, "sympy": sympy, "datetime": datetime,
        "clean_numeric": clean_numeric, "clean_dates": clean_dates, "safe_divide": safe_divide,
        "LinearRegression": LinearRegression, "LogisticRegression": LogisticRegression, "r2_score": r2_score,
        "dfs": cdfs,
    }
    orig_read_csv = pd.read_csv
    pd.read_csv = _fake_read_csv(cdfs, extra)   # le code généré recharge souvent les CSV lui-même

    global LAST_ERR, LAST_STATE
    LAST_ERR, LAST_STATE = "", ""

    def job():
        global LAST_STATE
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                _exec_generated(code, ns)
            value = to_float(ns.get("answer"))
            if value is None:
                raise ValueError("answer missing, non-numeric, or non-finite")
            return value
        except BaseException:
            LAST_STATE = _runtime_state(ns)
            raise

    try:
        return _run_with_timeout(job, timeout)
    except BaseException as e:
        LAST_ERR = f"{type(e).__name__}: {str(e)[:150]}"
        if not LAST_STATE:
            LAST_STATE = _runtime_state(ns)
        return None
    finally:
        pd.read_csv = orig_read_csv


def to_float(v):
    try:
        if isinstance(v, (pd.Series, pd.DataFrame)):
            v = np.asarray(v).ravel()[0]
        if isinstance(v, (list, tuple, np.ndarray)):
            v = np.asarray(v).ravel()[0]
        if isinstance(v, str):
            v = _num_from_text(v)
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None


def decimals(objective):
    low = objective.lower()
    m = re.search(r"(\d+)\s*(?:decimal|digit)", low) or re.search(r"decimals?\s*[:=]?\s*(\d+)", low)
    if m:
        return int(m.group(1))
    if re.search(r"integer|whole number|nearest (?:whole|integer)", low):
        return 0
    return None


_THOUSANDS_RE = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+)(?!\d|,\d)")


def _norm_thousands(text):
    """« 24,000 euros » -> « 24000 euros » ; les listes « 2.48, 1.43 » (virgule + espace) ne sont pas touchées."""
    return _THOUSANDS_RE.sub(lambda m: m.group(1).replace(",", ""), str(text))


def _asks_interest(o):
    return bool(re.search(r"\bhow much interest\b|\binterest (?:has been |was |is )?(?:earned|gained|received|paid)\b|\bamount of interest\b", o))


def calc_rules(objective):
    """Exact solvers for recurring, unambiguous calculation templates; None delegates to the LLM."""
    objective = _norm_thousands(objective)
    o = " ".join(objective.lower().split())

    # Sum a comma-separated list of explicitly enumerated receipts / values.
    m = re.search(
        r"\brecorded\s+(?:these|the following)\s+[^:]{0,100}:\s*"
        r"(\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?){1,})\s*[.;]?\s*"
        r"(?:[\s\S]{0,120}?)(?:what is the total|find the total|calculate the total|give the total|compute the total|determine the total)\b", o)
    if m:
        values = [float(x.strip()) for x in m.group(1).split(",")]
        return sum(values)

    # Payroll: each employee receives the weekly wage for every week plus one one-off bonus.
    m = re.search(
        r"employs?\s+(\d+)\s+people,?\s+each working\s+([\d.]+)\s+hours?\s+a week at\s+([\d.]+)\s+euros?\s+an hour"
        r"[\s\S]*?over\s+(\d+)\s+weeks?,?\s+each employee also receives a one-off bonus of\s+([\d.]+)\s+euros?",
        o)
    if m and re.search(r"total paid out", o):
        people, hours, rate, weeks, bonus = map(float, m.groups())
        return people * (hours * rate * weeks + bonus)

    # Ordered arrangements are permutations, n!/(n-k)!.
    m = re.search(r"ordered arrangements of\s+(\d+)\s+items?\s+can be made from\s+(\d+)\s+distinct items?", o)
    if m:
        k, n = map(int, m.groups())
        if 0 <= k <= n:
            return float(math.factorial(n) // math.factorial(n - k))

    # Two-variable linear system with integer coefficients.
    if "solve the system" in o:
        eqs = re.findall(r"([+-]?\s*\d*)\s*x\s*([+-])\s*(\d*)\s*y\s*=\s*([+-]?\d+(?:\.\d+)?)", o)
        if len(eqs) == 2:
            def coef(s):
                s = s.replace(" ", "")
                return -1.0 if s == "-" else (1.0 if s in ("", "+") else float(s))
            (a1, sign1, b1, c1), (a2, sign2, b2, c2) = eqs
            a1, a2 = coef(a1), coef(a2)
            b1 = coef(sign1 + b1 if b1 else sign1)
            b2 = coef(sign2 + b2 if b2 else sign2)
            c1, c2 = float(c1), float(c2)
            det = a1 * b2 - a2 * b1
            if det:
                x = (c1 * b2 - c2 * b1) / det
                y = (a1 * c2 - a2 * c1) / det
                target = re.search(r"value of ([xy])\b", o)
                if target:
                    return x if target.group(1) == "x" else y

    m = re.search(r"least common multiple of\s+(\d+)\s+and\s+(\d+)", o)
    if m:
        return float(math.lcm(*map(int, m.groups())))

    # Recurring departures: the next simultaneous departure is the LCM of the intervals.
    m = re.search(
        r"\bone every\s+([\d,]+)\s+minutes?[\s\S]{0,100}?\bone every\s+([\d,]+)\s+minutes?"
        r"[\s\S]{0,100}?\bafter how many minutes\b[\s\S]{0,60}?\b(?:next )?(?:leave|depart|run)\b"
        r"[\s\S]{0,20}?\btogether\b", o)
    if m:
        a, b = (int(x.replace(",", "")) for x in m.groups())
        return float(math.lcm(a, b))

    # Bayes' rule for a positive test, with prevalence, sensitivity, and false-positive rate.
    m = re.search(
        r"\b(\d+(?:\.\d+)?)%\s+of\s+(?:cases|transactions|people|patients|parts|items)\s+have\b"
        r"[\s\S]{0,180}?\btest detects it in\s+(\d+(?:\.\d+)?)%"
        r"[\s\S]{0,140}?\bwrongly flags\s+(\d+(?:\.\d+)?)%"
        r"[\s\S]{0,140}?\btests? positive\b", o)
    if m and re.search(r"\bprobability\b[\s\S]{0,100}\bpercent\b", o):
        prevalence, sensitivity, false_positive = (float(x) / 100 for x in m.groups())
        if all(0 <= x <= 1 for x in (prevalence, sensitivity, false_positive)):
            denominator = prevalence * sensitivity + (1 - prevalence) * false_positive
            if denominator > 0:
                return prevalence * sensitivity / denominator * 100

    # Repeated independent Bernoulli trials, including "at least k".
    m = re.search(r"probability\s+(\d+(?:\.\d+)?)[\s\S]{0,120}?\bamong\s+(\d+)\s+(?:parts?|trials?|items?)[\s\S]*?(?:at least|no less than|no fewer than)\s+(\d+)\s+(?:defective|success)", o)
    if m:
        p, n, k = float(m.group(1)), int(m.group(2)), int(m.group(3))
        if 0 <= p <= 1 and 0 <= k <= n:
            return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(k, n+1))

    # Sum of fair six-sided dice by dynamic programming.
    m = re.search(r"(\d+)\s+fair six-sided dice[\s\S]*?faces add up to\s+(\d+)", o)
    if m:
        n, target = map(int, m.groups())
        ways = [1]
        for _ in range(n):
            nxt = [0] * (len(ways) + 6)
            for total, count in enumerate(ways):
                for face in range(1, 7):
                    nxt[total + face] += count
            ways = nxt
        return ways[target] / (6 ** n) if target < len(ways) else 0.0

    # Successive percentage changes compound multiplicatively.
    if "overall percentage change" in o:
        changes = re.findall(r"\b(falls?|decreases?|drops?|declines?|rises?|increases?|grows?)\s+by\s+([\d.]+)\s*%", o)
        if len(changes) >= 2:
            factor = 1.0
            for direction, amount in changes:
                rate = float(amount) / 100.0
                factor *= (1 - rate) if direction.startswith(("fall", "decreas", "drop", "declin")) else (1 + rate)
            return (factor - 1) * 100

    # Compound annual growth rate.
    m = re.search(r"grew from\s+([\d.]+)\s+to\s+([\d.]+)\s+in\s+(\d+)\s+years?[\s\S]*?constant yearly growth rate,? in percent", o)
    if m:
        start, end, years = float(m.group(1)), float(m.group(2)), int(m.group(3))
        if start > 0 and years > 0:
            return ((end / start) ** (1 / years) - 1) * 100

    # Compound interest, with reinvested interest.
    m = re.search(r"starts? at\s+([\d.]+)\s+euros? and earns\s+([\d.]+)% per year,? interest reinvested[\s\S]*?after\s+(\d+)\s+years?", o)
    if m:
        principal, rate, years = float(m.group(1)), float(m.group(2)) / 100, int(m.group(3))
        balance = principal * (1 + rate) ** years
        return balance - principal if _asks_interest(o) else balance

    # First full year in which a compound-interest balance reaches a target threshold.
    m = re.search(
        r"\bamount of\s+([\d.]+)\s+euros? earns\s+([\d.]+)%\s+compound interest per year"
        r"[\s\S]{0,120}?\bafter how many full years\b[\s\S]{0,100}?\bfirst reach(?:es)? (?:at least|no less than)\s+([\d.]+)", o)
    if m:
        principal, rate, target = float(m.group(1)), float(m.group(2)) / 100, float(m.group(3))
        if principal > 0 and target <= principal:
            return 0.0
        if principal > 0 and rate > 0 and target > principal:
            years = math.log(target / principal) / math.log1p(rate)
            return float(math.ceil(years - 1e-12))

    # Find the original price when a quoted final price includes one increase.
    m = re.search(r"is\s+([\d.]+)\s+euros?\s+once a\s+([\d.]+)%\s+increase is applied[\s\S]*?original price", o)
    if m:
        final, rate = float(m.group(1)), float(m.group(2)) / 100
        return final / (1 + rate)

    # Two pumps fill while one leak empties the tank.
    m = re.search(r"one pump fills a tank in\s+([\d.]+)\s+hours? and a second pump fills it in\s+([\d.]+)\s+hours?[\s\S]*?leak empties the full tank in\s+([\d.]+)\s+hours?", o)
    if m:
        a, b, leak = map(float, m.groups())
        rate = 1/a + 1/b - 1/leak
        if rate > 0:
            return 1/rate
    try:
        return calc_rules_extra(objective)
    except Exception:
        return None



# --------------------------------------------------------------------------- calculs sans fichier (règles étendues)
# Gabarits stricts : chaque règle exige la forme complète de la question ; au moindre doute -> None (le LLM reprend).

import ast as _ast
import operator as _op

_MONTHS = {m: i for i, m in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august", "september",
     "october", "november", "december"), 1)}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_MONTHS["sept"] = 9

_NUM = r"[-+−]?\d+(?:\.\d+)?"
_DATE_PAT = (r"(?:\d{4}-\d{1,2}-\d{1,2}|\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?[a-z]{3,9}\.?,?\s+\d{4}"
             r"|[a-z]{3,9}\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})")


def _fnum(s):
    return float(str(s).replace("−", "-").replace(",", "").replace(" ", ""))


def _parse_date_text(s):
    s = s.strip().lower().rstrip(".,")
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", s)
    try:
        if m:
            return datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        m = re.fullmatch(r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([a-z]{3,9})\.?,?\s+(\d{4})", s)
        if m and m.group(2) in _MONTHS:
            return datetime.date(int(m.group(3)), _MONTHS[m.group(2)], int(m.group(1)))
        m = re.fullmatch(r"([a-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})", s)
        if m and m.group(1) in _MONTHS:
            return datetime.date(int(m.group(3)), _MONTHS[m.group(1)], int(m.group(2)))
    except ValueError:
        return None
    return None


_BINOPS = {_ast.Add: _op.add, _ast.Sub: _op.sub, _ast.Mult: _op.mul, _ast.Div: _op.truediv,
           _ast.Pow: _op.pow, _ast.Mod: _op.mod, _ast.FloorDiv: _op.floordiv}


def _safe_arith(expr):
    """Évalue une expression purement numérique (+ - × ÷ ^ racines, parenthèses) via l'AST, sans eval."""
    e = expr.strip().rstrip("?.").strip()
    e = (e.replace("×", "*").replace("·", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
          .replace("^", "**").replace("²", "**2").replace("³", "**3"))
    e = re.sub(r"(?<=\d)\s*x\s*(?=\d)", "*", e)
    e = re.sub(r"√\s*\(", "sqrt(", e)
    e = re.sub(r"√\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", e)
    e = re.sub(r"(?<=\d),(?=\d{3}\b)", "", e)
    if not re.fullmatch(r"[\d\s.+\-*/%()sqrt]+", e) or not re.search(r"\d", e):
        return None
    try:
        tree = _ast.parse(e, mode="eval")
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, _ast.Expression):
            return ev(n.body)
        if isinstance(n, _ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, _ast.UnaryOp) and isinstance(n.op, (_ast.USub, _ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, _ast.USub) else v
        if isinstance(n, _ast.BinOp) and type(n.op) in _BINOPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, _ast.Pow) and abs(b) > 64:
                raise ValueError("power too large")
            return _BINOPS[type(n.op)](a, b)
        if (isinstance(n, _ast.Call) and isinstance(n.func, _ast.Name) and n.func.id == "sqrt"
                and len(n.args) == 1 and not n.keywords):
            return math.sqrt(ev(n.args[0]))
        raise ValueError("unsupported")
    try:
        v = float(ev(tree))
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _num_list(text):
    """Plus longue liste « a, b, c » (séparateur virgule + espace) dans le texte."""
    best = []
    for m in re.finditer(r"[-−]?\d+(?:\.\d+)?(?:\s*,\s+(?:and\s+)?[-−]?\d+(?:\.\d+)?)+", text):
        vals = [_fnum(x) for x in re.findall(r"[-−]?\d+(?:\.\d+)?", m.group(0))]
        if len(vals) > len(best):
            best = vals
    return best


def _power_value(text):
    t = text.strip()
    m = re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", t)
    if m and int(m.group(2)) != 0:
        return int(m.group(1)) / int(m.group(2))
    try:
        return _fnum(t)
    except ValueError:
        return None




def calc_rules_extra3(objective):
    """Gabarits de calcul absents de dev (vague 3) ; None au moindre doute."""
    o = " ".join(_norm_thousands(objective).lower().replace("−", "-").replace("…", "...").split())
    N = _NUM
    G = r"[\s\S]{0,180}?"

    # ---- arithmétique
    m = re.search(rf"\bhow many values in (?:this|the following) list are between\s+({N})\s+and\s+({N})\s*\((?:both )?(included|inclusive|excluded|exclusive)\)\s*:", o)
    if m:
        a, b = _fnum(m.group(1)), _fnum(m.group(2))
        vals = _num_list(o[m.end():])
        if len(vals) >= 2:
            inc = m.group(3) in ("included", "inclusive")
            return float(sum((a <= v <= b) if inc else (a < v < b) for v in vals))
    m = re.search(r"\bwhat is the difference between the largest and the smallest of (?:these|the following) (?:numbers|values)\s*:", o)
    if m:
        vals = _num_list(o[m.end():])
        if len(vals) >= 2:
            return float(max(vals) - min(vals))
    m = re.search(rf"\bwhat is\s+(\d+)\s*/\s*(\d+)\s+of\s+({N})\s*\?", o)
    if m and int(m.group(2)):
        return int(m.group(1)) / int(m.group(2)) * _fnum(m.group(3))

    # ---- dates et heures
    m = re.search(rf"\bwhat is the date\s+(\d+)\s+months?\s+(after|before)\s+({_DATE_PAT})", o)
    if m and re.search(r"yyyymmdd", o):
        d0 = _parse_date_text(m.group(3))
        if d0:
            n = int(m.group(1)) * (1 if m.group(2) == "after" else -1)
            idx = d0.year * 12 + d0.month - 1 + n
            y, mth = idx // 12, idx % 12 + 1
            last = (datetime.date(y + (mth == 12), mth % 12 + 1, 1) - datetime.timedelta(days=1)).day
            d1 = datetime.date(y, mth, min(d0.day, last))
            return float(d1.year * 10000 + d1.month * 100 + d1.day)
    m = re.search(rf"\b(?:on )?which day of the week is the date\s+(\d+)\s+(days?|weeks?)\s+(after|before)\s+({_DATE_PAT})", o)
    if m and re.search(r"monday\s*=\s*1", o):
        d0 = _parse_date_text(m.group(4))
        if d0:
            k = int(m.group(1)) * (7 if m.group(2).startswith("week") else 1)
            return float((d0 + datetime.timedelta(days=k if m.group(3) == "after" else -k)).isoweekday())
    m = re.search(r"\b(?:leaves|departs|starts|begins) at\s+(\d{1,2}):(\d{2})\b" + G + r"\blasts?\s+(\d+)\s*h(?:ours?)?\s*(?:and\s+)?(\d+)?\s*(?:min(?:utes?)?)?" + G + r"\bat what time\b", o)
    if m and re.search(r"\bhhmm\b", o):
        t = int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) * 60 + int(m.group(4) or 0)
        t %= 1440
        return float((t // 60) * 100 + t % 60)
    m = re.search(r"\bhow many minutes are there from\s+(\d{1,2}):(\d{2})\s+to\s+(\d{1,2}):(\d{2})\b", o)
    if m and re.search(r"same day", o):
        a = int(m.group(1)) * 60 + int(m.group(2)); b = int(m.group(3)) * 60 + int(m.group(4))
        return float(b - a) if b >= a else None
    m = re.search(rf"\bhow many (?:full|whole|complete) months are there (?:from|between)\s+({_DATE_PAT})\s+(?:to|and)\s+({_DATE_PAT})", o)
    if m:
        a, b = _parse_date_text(m.group(1)), _parse_date_text(m.group(2))
        if a and b and b >= a:
            return float((b.year - a.year) * 12 + b.month - a.month - (b.day < a.day))

    # ---- fonctions
    m = re.search(rf"\bsolve\s+({N})\s*\^\s*x\s*=\s*({N})\b", o) or \
        re.search(rf"\bfor which x does\s+({N})\s+(?:raised )?to the power(?: of)? x equals?\s+({N})\b", o)
    if m:
        b, y = _fnum(m.group(1)), _fnum(m.group(2))
        return math.log(y) / math.log(b) if b > 0 and b != 1 and y > 0 else None

    # ---- croissance
    m = re.search(rf"\bbought for\s+({N})\s+euros? loses\s+({N})%\s+of its value (?:every|each|per) year{G}\bafter how many (?:full |whole )?years\b"
                  rf"{G}\b(?:worth )?(less than|below|under|at most|no more than)\s+({N})", o)
    if m:
        p, r_, how, tgt = _fnum(m.group(1)), _fnum(m.group(2)) / 100, m.group(3), _fnum(m.group(4))
        if 0 < r_ < 1 and 0 < tgt < p:
            y = math.log(tgt / p) / math.log(1 - r_)
            return float(math.floor(y + 1e-12) + 1) if how in ("less than", "below", "under") else float(math.ceil(y - 1e-12))
    m = re.search(rf"\b({N})\s+(?:bacteria|cells|users|followers|organisms|units)\b[^.?]{{0,40}}?\bdoubles? every\s+({N})\s+(hours?|days?|years?|minutes?)"
                  rf"{G}\bafter\s+({N})\s+(hours?|days?|years?|minutes?)", o)
    if m and m.group(3).rstrip("s") == m.group(5).rstrip("s"):
        v0, d, t = _fnum(m.group(1)), _fnum(m.group(2)), _fnum(m.group(4))
        return v0 * 2 ** (t / d) if d > 0 else None

    # ---- entiers
    m = re.search(r"\bsum of the integers from\s+(-?\d+)\s+to\s+(-?\d+)\b", o) or \
        re.search(r"\bcompute\s+(1)\s*\+\s*2\s*\+\s*(?:\.\.\.|…)\s*\+\s*(\d+)\b", o)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return float((a + b) * (b - a + 1) // 2) if b >= a else None
    m = re.search(r"\bhow many codes of\s+(\d+)\s+(?:characters?|symbols?|digits?|letters?)\s+can be formed from the (digits|letters)\b", o)
    if m and re.search(r"repetitions? (?:is |are )?allowed|with repetition", o):
        L = int(m.group(1)); D = 10 if m.group(2) == "digits" else 26
        if re.search(r"\(0 to 9\)|\(a to z\)|\b0-9\b|\ba-z\b", o) or True:
            return float(D ** L)
    m = re.search(r"\bremainder of\s+(\d+)\s*\^\s*(\d+)\s+(?:is )?divided by\s+(\d+)\b", o) or \
        re.search(r"\bremainder (?:of|when)\s+(\d+)\s+(?:raised )?to the power(?: of)?\s+(\d+)\s+(?:is )?divided by\s+(\d+)\b", o)
    if m and int(m.group(3)):
        return float(pow(int(m.group(1)), int(m.group(2)), int(m.group(3))))
    m = re.search(r"\bhow many distinct arrangements of the letters of the word\s+([a-z]+)\b", o)
    if m:
        w = m.group(1)
        v = math.factorial(len(w))
        for c in collections.Counter(w).values():
            v //= math.factorial(c)
        return float(v)
    m = re.search(r"\b(?:greatest common divisor|highest common factor|gcd) of\s+(\d+),\s*(\d+),?\s*and\s+(\d+)\b", o)
    if m:
        return float(math.gcd(*map(int, m.groups())))
    m = re.search(r"\bhow many integers between\s+(\d+)\s+and\s+(\d+)\s*\((?:both )?(included|inclusive)\)\s+are (?:divisible by|multiples of)\s+(\d+)\b", o)
    if m and int(m.group(4)):
        a, b, d = int(m.group(1)), int(m.group(2)), int(m.group(4))
        return float(b // d - (a - 1) // d) if b >= a else None

    # ---- pourcentages
    m = re.search(rf"^\s*({N})\s+is\s+({N})\s*%\s+of what number\b", o) or re.search(rf"\b({N})\s+is\s+({N})\s*%\s+of what number\b", o)
    if m and _fnum(m.group(2)):
        return _fnum(m.group(1)) / (_fnum(m.group(2)) / 100)
    m = re.search(rf"\bof\s+(\d+)\s+(students|pupils|people|employees|players|customers|patients|members|candidates),\s+(\d+)\s+([a-z ]+?)\.{G}\bwhat percentage of the \2\b", o)
    if m and int(m.group(1)):
        return int(m.group(3)) / int(m.group(1)) * 100
    m = re.search(rf"\bprice (?:drops|falls|goes down|is reduced) from\s+({N})\s+euros?\s+to\s+({N})\s+euros?{G}\bdiscount\b[^.?]{{0,20}}\bpercent", o)
    if m and _fnum(m.group(1)):
        a, b = _fnum(m.group(1)), _fnum(m.group(2))
        return (a - b) / a * 100
    m = re.search(rf"\bprice of\s+({N})\s+euros? is (increased|raised|reduced|decreased|lowered) by\s+({N})%{G}\bnew price\b", o)
    if m:
        s = 1 if m.group(2) in ("increased", "raised") else -1
        return _fnum(m.group(1)) * (1 + s * _fnum(m.group(3)) / 100)

    # ---- probabilités
    m = re.search(r"\bfair coin is (?:tossed|flipped|thrown)\s+(\d+)\s+times" + G + r"\b(exactly|at least|at most) (\d+) (heads|tails)\b", o)
    if m:
        n, how, k = int(m.group(1)), m.group(2), int(m.group(3))
        rng = {"exactly": [k], "at least": range(k, n + 1), "at most": range(0, k + 1)}[how]
        return sum(math.comb(n, i) for i in rng if 0 <= i <= n) / 2 ** n
    m = re.search(r"\bfair six-sided die is rolled\s+(\d+)\s+times" + G + r"\bat least one (six|one|two|three|four|five)\b", o)
    if m:
        return 1 - (5 / 6) ** int(m.group(1))
    m = re.search(r"\bcontains\s+(\d+)\s+([a-z]+),\s+(\d+)\s+([a-z]+)\s+and\s+(\d+)\s+([a-z]+)\s+(?:marbles|balls|tokens|cards|sweets)\b" + G +
                  r"\bone [a-z]+ is drawn at random" + G + r"\bprobability that it is ([a-z]+)\b", o)
    if m:
        counts = {m.group(2): int(m.group(1)), m.group(4): int(m.group(3)), m.group(6): int(m.group(5))}
        tot = sum(counts.values())
        return counts[m.group(7)] / tot if m.group(7) in counts and tot else None
    m = re.search(rf"\btwo independent events have probabilities\s+({N})\s+and\s+({N}){G}\bprobability that (both occur|at least one of them occurs|neither occurs|exactly one of them occurs)", o)
    if m:
        p, q = _fnum(m.group(1)), _fnum(m.group(2))
        return {"both occur": p * q, "at least one of them occurs": 1 - (1 - p) * (1 - q), "neither occurs": (1 - p) * (1 - q),
                "exactly one of them occurs": p * (1 - q) + q * (1 - p)}[m.group(3)]
    if re.search(r"\bfair six-sided die is rolled once\b" + G + r"\bexpected value\b", o):
        return 3.5

    # ---- statistiques
    m = re.search(r"\bwhat is (?:their|the) mean absolute deviation(?: from the mean)?\b", o)
    if m:
        vals = _num_list(o)
        if len(vals) >= 2:
            mu = sum(vals) / len(vals)
            return sum(abs(v - mu) for v in vals) / len(vals)
    m = re.search(rf"\ba new [a-z]+ (?:scores?|has|weighs?|measures?|earns?|gets?)\s+({N})\b{G}\b(?:what is|compute|find|give|determine|calculate) the mean of all\s+(\d+)\b", o)
    if m:
        vals = _num_list(o[:m.start()])
        if len(vals) >= 2 and int(m.group(2)) == len(vals) + 1:
            return (sum(vals) + _fnum(m.group(1))) / (len(vals) + 1)
    m = re.search(rf"\bmean of\s+({N})\s+and a standard deviation of\s+({N}){G}\bz-score of an? [a-z ]*?({N})\b", o)
    if m and _fnum(m.group(2)):
        return (_fnum(m.group(3)) - _fnum(m.group(1))) / _fnum(m.group(2))
    m = re.search(rf"\bthe mean of\s+(\d+)\s+values is\s+({N}){G}\bwhat is their sum\b", o)
    if m:
        return int(m.group(1)) * _fnum(m.group(2))

    # ---- problèmes
    m = re.search(rf"\bhow many (minutes|hours) does it take to travel\s+({N})\s+km at a (?:constant |average )?speed of\s+({N})\s+km/h", o)
    if m and _fnum(m.group(3)):
        h = _fnum(m.group(2)) / _fnum(m.group(3))
        return h * 60 if m.group(1) == "minutes" else h
    m = re.search(rf"\bone (?:pump|worker|machine|tap) (?:fills|does|completes|empties) (?:a|the) (?:tank|job|task|pool) in\s+({N})\s+hours? and a second "
                  rf"(?:pump|worker|machine|tap) (?:fills|does|completes|empties) it in\s+({N})\s+hours?", o)
    if m and not re.search(r"\bleak|empties the full|drains?\b", o) and re.search(r"\bboth\b|\btogether\b", o):
        a, b = _fnum(m.group(1)), _fnum(m.group(2))
        return 1 / (1 / a + 1 / b) if a > 0 and b > 0 else None
    m = re.search(rf"\b(?:pack|put|store|ship)\s+(\d+)\s+[a-z]+ into (?:boxes|crates|cartons|packs) of\s+(\d+)\b{G}\bhow many (?:boxes|crates|cartons|packs) are needed\b", o)
    if m and int(m.group(2)):
        return float(math.ceil(int(m.group(1)) / int(m.group(2))))
    m = re.search(rf"\b({N})\s+euros? (?:are|is) (?:shared|split|divided) in the ratio\s+(\d+)\s*:\s*(\d+){G}\b(larger|smaller|largest|smallest|bigger) share\b", o)
    if m:
        x, a, b = _fnum(m.group(1)), int(m.group(2)), int(m.group(3))
        if a + b:
            return x * (max(a, b) if m.group(4) in ("larger", "largest", "bigger") else min(a, b)) / (a + b)
    m = re.search(rf"\buses\s+({N})\s+litres? per 100 km{G}\bhow many litres\b[^.?]{{0,30}}?\bfor\s+({N})\s+km", o)
    if m:
        return _fnum(m.group(1)) * _fnum(m.group(2)) / 100
    m = re.search(rf"\bgoes to [a-z ]+? at\s+({N})\s+km/h and comes back[^.?]{{0,60}}?at\s+({N})\s+km/h{G}\baverage speed\b", o)
    if m:
        a, b = _fnum(m.group(1)), _fnum(m.group(2))
        return 2 * a * b / (a + b) if a + b else None
    return None


def calc_rules_extra2(objective):
    """Gabarits de calcul absents de dev (vague 2) ; None au moindre doute."""
    o = " ".join(objective.lower().replace("−", "-").split())
    N = _NUM
    BIG = r"(\d{1,3}(?:,\d{3})+|" + _NUM + r")"

    # ---- listes : parité, somme, extrêmes, mode, moyennes géométrique et harmonique
    m = re.search(r"\bhow many (even|odd) (?:numbers|values|integers) are there in (?:this|the following) list\s*:", o)
    if m:
        vals = _num_list(o[m.end():])
        if len(vals) >= 2 and all(float(v).is_integer() for v in vals):
            return float(sum((int(v) % 2 == 0) == (m.group(1) == "even") for v in vals))
    m = re.search(r"\bwhat is the (sum|largest|smallest|biggest|highest|lowest|maximum|minimum|product) of (?:these|the following) (?:numbers|values)\s*:", o)
    if m:
        vals = _num_list(o[m.end():])
        if len(vals) >= 2:
            w = m.group(1)
            if w == "sum":
                return float(sum(vals))
            if w == "product":
                return float(np.prod(vals))
            return float(max(vals) if w in ("largest", "biggest", "highest", "maximum") else min(vals))
    m = re.search(r"\bwhat is (?:the |their )?mode\b", o)
    if m:
        vals = _num_list(o)
        if len(vals) >= 3:
            c = collections.Counter(vals)
            best = max(c.values())
            tops = [v for v, n in c.items() if n == best]
            if len(tops) == 1 and best >= 2:
                return float(tops[0])
    m = re.search(r"\bwhat is the (geometric|harmonic) mean of\s+", o)
    if m:
        vals = _num_list(o[m.end():])
        if len(vals) >= 2 and all(v > 0 for v in vals):
            a = np.asarray(vals, dtype=float)
            return float(np.exp(np.log(a).mean())) if m.group(1) == "geometric" else float(len(a) / (1 / a).sum())

    # ---- dates
    m = re.search(r"\bhow many days (?:are there |does )?in (?:the year )?(\d{4})\b(?! [a-z]+ \d)", o) or \
        re.search(r"\bhow many days (?:does|are there in|has) (?:the year )?(\d{4}) (?:have|has|contain)\b", o)
    if m and not re.search(r"\bfrom\b|\bbetween\b|\bmonth\b|\bfebruary\b", o):
        y = int(m.group(1))
        return 366.0 if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0 else 365.0
    m = re.search(r"\bhow many days (?:does|are there in|has) february (\d{4})\b|\bhow many days (?:are there )?in february (?:of )?(\d{4})\b", o)
    if m:
        y = int(m.group(1) or m.group(2))
        return 29.0 if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0 else 28.0
    m = re.search(rf"\b(?:what is the )?day of the year (?:of|for|is)\s+({_DATE_PAT})", o)
    if m and re.search(r"january 1\s*=\s*1|1 january\s*=\s*1|jan(?:uary)? 1st\s*=\s*1", o):
        d = _parse_date_text(m.group(1))
        if d:
            return float(d.timetuple().tm_yday)
    m = re.search(rf"\bhow many (?:weekdays|working days|business days)(?: \(monday to friday\))? are there (?:from|between)\s+({_DATE_PAT})\s+(?:to|and)\s+({_DATE_PAT})", o)
    if m and re.search(r"both included|inclusive|including both", o):
        a, b = _parse_date_text(m.group(1)), _parse_date_text(m.group(2))
        if a and b and b >= a:
            return float(np.busday_count(a, b + datetime.timedelta(days=1)))
    m = re.search(rf"\bborn on\s+({_DATE_PAT})\b[\s\S]{{0,180}}?\bhow old\b[\s\S]{{0,180}}?\b(?:on|at)\s+({_DATE_PAT})", o)
    if m and re.search(r"full years|completed years|in years", o):
        birth, d = _parse_date_text(m.group(1)), _parse_date_text(m.group(2))
        if birth and d:
            return float(d.year - birth.year - ((d.month, d.day) < (birth.month, birth.day)))

    # ---- fonctions
    m = re.search(rf"\blog2\s*\(\s*({N})\s*\)", o)
    if m:
        x = _fnum(m.group(1))
        return math.log2(x) if x > 0 else None

    # ---- croissance
    m = re.search(rf"\b(?:amount of\s+)?{BIG}\s+euros? (?:earns?|earning|at)\s+({N})%\s+simple interest per year[\s\S]{{0,180}}?\bafter\s+(\d+)\s+years?", o)
    if m:
        p, rate, n = _fnum(m.group(1)), _fnum(m.group(2)) / 100, int(m.group(3))
        return p * rate * n if _asks_interest(o) else p * (1 + rate * n)
    m = re.search(rf"\b{BIG}\s+euros? (?:are|is) (?:placed|invested|deposited) at an annual rate of\s+({N})%\s+compounded (monthly|quarterly|daily|continuously)"
                  rf"[\s\S]{{0,180}}?\bafter\s+(\d+)\s+years?", o) or \
        re.search(rf"\b{BIG}\s+euros? grows? at\s+({N})%\s+per year,?\s+compounded (monthly|quarterly|daily|continuously)[\s\S]{{0,180}}?\bafter\s+(\d+)\s+years?", o)
    if m:
        p, rate, how, n = _fnum(m.group(1)), _fnum(m.group(2)) / 100, m.group(3), int(m.group(4))
        if how == "continuously":
            return p * (math.exp(rate * n) - 1) if _asks_interest(o) else p * math.exp(rate * n)
        k = {"monthly": 12, "quarterly": 4, "daily": 365}[how]
        return p * ((1 + rate / k) ** (k * n) - 1) if _asks_interest(o) else p * (1 + rate / k) ** (k * n)
    m = re.search(rf"\b(?:grows?|increases?|rises?)\s+by\s+({N})%\s+(?:per|a|each|every) year[\s\S]{{0,180}}?\bafter how many (?:full |whole )?years\b[\s\S]{{0,180}}?\b(?:at least )?(double|triple)", o)
    if m:
        r_ = _fnum(m.group(1)) / 100
        f = 2.0 if m.group(2) == "double" else 3.0
        if r_ > 0:
            return float(math.ceil(math.log(f) / math.log1p(r_) - 1e-12))
    m = re.search(rf"\b({N})\s+(grams?|g|kg|mg|milligrams?|units?)\b[^.?]{{0,40}}?\bhalf-life of\s+({N})\s+(days?|years?|hours?)[\s\S]{{0,180}}?\bafter\s+({N})\s+(days?|years?|hours?)", o)
    if m and m.group(4).rstrip("s") == m.group(6).rstrip("s"):
        m0, h, t = _fnum(m.group(1)), _fnum(m.group(3)), _fnum(m.group(5))
        return m0 * 0.5 ** (t / h) if h > 0 else None

    # ---- entiers
    m = re.search(r"\bhow many (?:positive )?divisors does\s+(\d+)\s+have\b", o)
    if m:
        n = int(m.group(1))
        if 0 < n <= 10 ** 7:
            c, d = 0, 1
            while d * d <= n:
                if n % d == 0:
                    c += 1 if d * d == n else 2
                d += 1
            return float(c)
    m = re.search(r"\bsum of the digits of\s+(\d+)\b", o)
    if m:
        return float(sum(map(int, m.group(1))))
    m = re.search(r"\bhow many prime numbers are there (below|less than|under|up to|at most|not exceeding)\s+(\d+)\b", o)
    if m:
        n = int(m.group(2)) + (1 if m.group(1) in ("up to", "at most", "not exceeding") else 0)
        if n <= 2 * 10 ** 6:
            sieve = np.ones(max(n, 2), dtype=bool); sieve[:2] = False
            for i in range(2, int(n ** 0.5) + 1):
                if sieve[i]:
                    sieve[i * i::i] = False
            return float(sieve[:n].sum())
    m = re.search(r"\bhow many multiples of\s+(\d+)\s+are there between\s+(\d+)\s+and\s+(\d+)\b", o)
    if m and re.search(r"both included|inclusive|including both", o):
        d, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return float(b // d - (a - 1) // d) if d and b >= a else None

    # ---- pourcentages
    m = re.search(rf"\bcosts?\s+({N})\s+euros?\.?[\s\S]{{0,120}}?\bit is (?:discounted|reduced) by\s+({N})%[\s\S]{{0,180}}\bnew price\b", o) or \
        re.search(rf"\bprice of an? ({N})-euros? item after an? ({N})% (?:discount|reduction)\b", o)
    if m:
        return _fnum(m.group(1)) * (1 - _fnum(m.group(2)) / 100)
    m = re.search(rf"\bwhat percentage of\s+({N})\s+is\s+({N})\s*\?", o)
    if m and _fnum(m.group(1)):
        return _fnum(m.group(2)) / _fnum(m.group(1)) * 100
    m = re.search(rf"\bgoes from\s+({N})\s*%\s+to\s+({N})\s*%[\s\S]{{0,180}}\bpercentage points?\b", o)
    if m:
        return _fnum(m.group(2)) - _fnum(m.group(1))
    m = re.search(rf"\b(?:goes|went|rises|rose|falls|fell|changes|changed|moves|moved) from\s+({N})\s+(?:euros?\s+)?to\s+({N})\b[\s\S]{{0,180}}\bpercentage (?:change|increase|decrease|variation)\b", o)
    if m and not re.search(r"percentage points?", o):
        a, b = _fnum(m.group(1)), _fnum(m.group(2))
        return (b - a) / a * 100 if a else None

    # ---- probabilités
    m = re.search(r"\b(\d+) (red|blue|white|black|green) balls? and (\d+) (red|blue|white|black|green) balls?\b[\s\S]{0,180}?(\d+) balls? (?:are|is) drawn without replacement"
                  r"[\s\S]{0,180}?\b(exactly|at least|at most) (\d+) (red|blue|white|black|green) balls?", o)
    if m:
        n1, c1, n2, c2, d, how, k, ck = m.groups()
        n1, n2, d, k = int(n1), int(n2), int(d), int(k)
        good, bad = (n1, n2) if ck == c1 else ((n2, n1) if ck == c2 else (None, None))
        if good is None or d > n1 + n2:
            return None
        tot = math.comb(n1 + n2, d)
        rng = {"exactly": [k], "at least": range(k, d + 1), "at most": range(0, k + 1)}[how]
        return sum(math.comb(good, i) * math.comb(bad, d - i) for i in rng if 0 <= i <= good and 0 <= d - i <= bad) / tot
    m = re.search(rf"\beach of\s+(\d+)\s+independent (?:trials|attempts|tests)\s+succeeds? with probability\s+({N})[\s\S]{{0,180}}\bat least one success\b", o)
    if m:
        n, p = int(m.group(1)), _fnum(m.group(2))
        return 1 - (1 - p) ** n if 0 <= p <= 1 else None

    # ---- problèmes
    m = re.search(rf"\bconvert\s+({N})\s+degrees? (celsius|fahrenheit)\s+(?:to|into)\s+degrees? (celsius|fahrenheit)\b", o)
    if m and m.group(2) != m.group(3):
        x = _fnum(m.group(1))
        return x * 9 / 5 + 32 if m.group(2) == "celsius" else (x - 32) * 5 / 9
    m = re.search(rf"\b({N})\s+(?:litres?|liters?|l|kg|kilograms?|ml)\s+of an? (?:solution|mixture|alloy|drink)\s+at\s+({N})%\s+(?:are|is) mixed with\s+({N})\s+"
                  rf"(?:litres?|liters?|l|kg|kilograms?|ml)\s+of an? (?:solution|mixture|alloy|drink)\s+at\s+({N})%", o)
    if m and re.search(r"\bconcentration\b", o):
        v1, c1, v2, c2 = (_fnum(x) for x in m.groups())
        return (v1 * c1 + v2 * c2) / (v1 + v2) if v1 + v2 else None
    return calc_rules_extra3(objective)


def calc_rules_extra(objective):
    o = " ".join(objective.lower().replace("−", "-").split())

    # ---- dates
    m = re.search(rf"\bhow many days (?:are there |elapse |pass )?(?:from|between)\s+({_DATE_PAT})\s+(?:to|and|until)\s+({_DATE_PAT})", o)
    if not m:
        m = re.search(rf"\bstarts? on\s+({_DATE_PAT})\s+and ends? on\s+({_DATE_PAT})[\s\S]{{0,140}}?\bhow many days\b", o)
    if m:
        a, b = _parse_date_text(m.group(1)), _parse_date_text(m.group(2))
        if a and b:
            n = (b - a).days
            if re.search(r"\b(?:count|counting|include|including)\s+both\b|\binclusive\b", o):
                n += 1 if n >= 0 else -1
            return float(n)
    m = re.search(rf"\bhow many (?:full )?weeks (?:are there )?(?:from|between)\s+({_DATE_PAT})\s+(?:to|and)\s+({_DATE_PAT})", o)
    if m:
        a, b = _parse_date_text(m.group(1)), _parse_date_text(m.group(2))
        if a and b:
            return float((b - a).days // 7)
    m = re.search(rf"\bwhat is the date\s+(\d+)\s+(days?|weeks?)\s+(after|before)\s+({_DATE_PAT})", o)
    if m and re.search(r"yyyymmdd", o):
        d0 = _parse_date_text(m.group(4))
        if d0:
            k = int(m.group(1)) * (7 if m.group(2).startswith("week") else 1)
            d1 = d0 + datetime.timedelta(days=k if m.group(3) == "after" else -k)
            return float(d1.year * 10000 + d1.month * 100 + d1.day)
    m = re.search(rf"\b(?:on )?which day of the week (?:does|did|is|was|will)\s+({_DATE_PAT})", o)
    if m and re.search(r"monday\s*=\s*1", o):
        d0 = _parse_date_text(m.group(1))
        if d0:
            return float(d0.isoweekday())
    if m and re.search(r"monday\s*=\s*0", o):
        d0 = _parse_date_text(m.group(1))
        if d0:
            return float(d0.weekday())
    if m and re.search(r"sunday\s*=\s*1", o):
        d0 = _parse_date_text(m.group(1))
        if d0:
            return float(d0.isoweekday() % 7 + 1)

    # ---- fonctions
    m = re.search(rf"\b(?:base-10|base 10|decimal|common) logarithm of\s+({_NUM})", o) or re.search(rf"\blog10\s*\(?\s*({_NUM})", o)
    if m:
        x = _fnum(m.group(1))
        return math.log10(x) if x > 0 else None
    m = re.search(rf"\b(?:natural logarithm|\bln)\s*(?:of\s+)?\(?\s*({_NUM})", o)
    if m:
        x = _fnum(m.group(1))
        return math.log(x) if x > 0 else None
    m = re.search(rf"\b(?:base-2|base 2|binary) logarithm of\s+({_NUM})", o)
    if m:
        x = _fnum(m.group(1))
        return math.log2(x) if x > 0 else None
    m = re.search(rf"\blogarithm (?:in |to the )?base\s+({_NUM})\s+of\s+({_NUM})", o) or \
        re.search(rf"\bbase-({_NUM}) logarithm of\s+({_NUM})", o)
    if m:
        b, x = _fnum(m.group(1)), _fnum(m.group(2))
        return math.log(x, b) if x > 0 and b > 0 and b != 1 else None
    m = re.search(rf"\b(?:compute|calculate|what is)\s+e\s+(?:raised )?to the power(?: of)?\s+({_NUM}(?:\s*/\s*\d+)?)", o)
    if m:
        p = _power_value(m.group(1))
        return math.exp(p) if p is not None and p < 700 else None
    m = re.search(rf"\b(?:compute|calculate|what is)\s+({_NUM})\s+(?:raised )?to the power(?: of)?\s+(-?\d+(?:\.\d+)?(?:\s*/\s*\d+)?)", o)
    if m:
        x, p = _fnum(m.group(1)), _power_value(m.group(2))
        if p is None or (x < 0 and not float(p).is_integer()):
            return None
        try:
            v = x ** p
        except (OverflowError, ZeroDivisionError):
            return None
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else None
    m = re.search(rf"\b(square|cube|fourth|fifth|(\d+)(?:st|nd|rd|th)) root of\s+({_NUM})", o)
    if m:
        k = {"square": 2, "cube": 3, "fourth": 4, "fifth": 5}.get(m.group(1)) or int(m.group(2))
        x = _fnum(m.group(3))
        if x < 0:
            return -((-x) ** (1 / k)) if k % 2 == 1 else None
        return x ** (1 / k)
    m = re.search(rf"\b(?:compute|calculate|what is)\s+√\s*\(?\s*({_NUM})\s*\)?\s*[?.]", o)
    if m:
        x = _fnum(m.group(1))
        return math.sqrt(x) if x >= 0 else None
    m = re.search(r"\b(?:compute|calculate|what is)\s+(\d+)\s*!|\bfactorial of\s+(\d+)", o)
    if m:
        n = int(m.group(1) or m.group(2))
        return float(math.factorial(n)) if n <= 170 else None

    # ---- entiers
    m = re.search(r"\bin how many ways can (?:a |an )?(?:team|group|committee|set|hand|selection|sample) of\s+(\d+)\s+[\s\S]{0,40}?\bbe (?:chosen|selected|formed|picked|drawn) from\s+(\d+)\b", o)
    if m and re.search(r"order does not matter|unordered|order is irrelevant|order doesn't matter", o):
        k, n = int(m.group(1)), int(m.group(2))
        return float(math.comb(n, k)) if 0 <= k <= n else None
    m = re.search(r"\bhow many (?:different |distinct )?(?:teams|groups|committees|combinations|subsets|selections) of\s+(\d+)\b[\s\S]{0,60}?\bfrom\s+(\d+)\b", o)
    if m and not re.search(r"\border(?:ed)? (?:matters|arrangements)\b", o):
        k, n = int(m.group(1)), int(m.group(2))
        return float(math.comb(n, k)) if 0 <= k <= n else None
    m = re.search(r"\bremainder (?:of|in|when) (?:the )?(?:division of\s+)?(\d+)\s+(?:is )?(?:divided )?by\s+(\d+)", o)
    if m and int(m.group(2)) != 0:
        return float(int(m.group(1)) % int(m.group(2)))
    m = re.search(r"\b(?:greatest common divisor|highest common factor|gcd)\s+of\s+(\d+)\s+and\s+(\d+)", o)
    if m:
        return float(math.gcd(int(m.group(1)), int(m.group(2))))
    m = re.search(r"\b(?:least|lowest|smallest) common multiple of\s+(\d+),\s*(\d+)\s*,?\s*and\s+(\d+)", o)
    if m:
        return float(math.lcm(*map(int, m.groups())))
    m = re.search(r"\bin how many (?:different )?(?:ways|orders) can\s+(\d+)\s+(?:distinct |different )?[a-z]+\s+be (?:arranged|ordered|lined up|seated)", o)
    if m:
        n = int(m.group(1))
        return float(math.factorial(n)) if n <= 170 else None

    # ---- pourcentages
    m = re.search(rf"\binvoice of\s+({_NUM})\s+euros? includes vat (?:at|of)\s+({_NUM})\s*%[\s\S]{{0,140}}?\bamount (?:before|excluding|without) vat\b", o)
    if m:
        return _fnum(m.group(1)) / (1 + _fnum(m.group(2)) / 100)
    m = re.search(rf"\binvoice of\s+({_NUM})\s+euros? includes vat (?:at|of)\s+({_NUM})\s*%[\s\S]{{0,140}}?\b(?:amount of vat|vat amount|how much vat)\b", o)
    if m:
        total, r = _fnum(m.group(1)), _fnum(m.group(2)) / 100
        return total - total / (1 + r)
    m = re.search(rf"\b(?:amount|price|invoice) of\s+({_NUM})\s+euros? (?:before|excluding|without) vat\b[\s\S]{{0,40}}\bvat (?:at|of)\s+({_NUM})\s*%[\s\S]{{0,60}}\b(?:including|with|after) vat\b", o)
    if m:
        return _fnum(m.group(1)) * (1 + _fnum(m.group(2)) / 100)
    m = re.search(rf"\bwent (?:up |down )?from\s+({_NUM})\s+to\s+({_NUM})\b[\s\S]{{0,60}}\bpercentage (?:change|increase|decrease|variation)\b", o)
    if m:
        a, b = _fnum(m.group(1)), _fnum(m.group(2))
        return (b - a) / a * 100 if a else None
    m = re.search(rf"\bwhat is\s+({_NUM})\s*%\s+of\s+({_NUM})\b", o)
    if m:
        return _fnum(m.group(1)) / 100 * _fnum(m.group(2))
    m = re.search(rf"\b({_NUM})\s+is what (?:percentage|percent) of\s+({_NUM})\b", o)
    if m and _fnum(m.group(2)):
        return _fnum(m.group(1)) / _fnum(m.group(2)) * 100
    m = re.search(rf"\bis\s+({_NUM})\s+euros? once a\s+({_NUM})%\s+(discount|reduction|decrease) is applied[\s\S]*?original price", o)
    if m:
        return _fnum(m.group(1)) / (1 - _fnum(m.group(2)) / 100)

    # ---- croissance
    m = re.search(rf"\b({_NUM}|\d{{1,3}}(?:,\d{{3}})+)\s+euros? (?:are|is) (?:placed|invested|deposited) at\s+({_NUM})%\s+compound interest per year[\s\S]{{0,80}}?\bafter\s+(\d+)\s+years?", o)
    if m:
        bal = _fnum(m.group(1)) * (1 + _fnum(m.group(2)) / 100) ** int(m.group(3))
        return bal - _fnum(m.group(1)) if _asks_interest(o) else bal
    m = re.search(rf"\bbought for\s+({_NUM}|\d{{1,3}}(?:,\d{{3}})+)\s+euros? loses\s+({_NUM})%\s+of its value (?:every|each|per) year[\s\S]{{0,80}}?\bafter\s+(\d+)\s+years?", o)
    if m:
        return _fnum(m.group(1)) * (1 - _fnum(m.group(2)) / 100) ** int(m.group(3))
    m = re.search(rf"\b(?:of|has|is|counts?|at)\s+({_NUM}|\d{{1,3}}(?:,\d{{3}})+)\b[^.?]{{0,60}}?\b(grows?|increases?|rises?|falls?|decreases?|declines?|shrinks?)\s+by\s+({_NUM})%\s+(?:per|a|each|every) year[\s\S]{{0,80}}?\bafter\s+(\d+)\s+years?", o)
    if m:
        s = 1 if m.group(2).startswith(("grow", "increas", "rise", "rises")) else -1
        if _asks_interest(o):
            return None
        return _fnum(m.group(1)) * (1 + s * _fnum(m.group(3)) / 100) ** int(m.group(4))
    m = re.search(rf"\bfrom\s+({_NUM}|\d{{1,3}}(?:,\d{{3}})+)\s+to\s+({_NUM}|\d{{1,3}}(?:,\d{{3}})+)\s+in\s+(\d+)\s+years?[\s\S]{{0,80}}?\b(?:constant|average|compound) (?:yearly|annual) growth rate", o)
    if m and "percent" in o:
        a, b, n = _fnum(m.group(1)), _fnum(m.group(2)), int(m.group(3))
        return ((b / a) ** (1 / n) - 1) * 100 if a > 0 and n > 0 else None
    m = re.search(rf"\bdoubl(?:e|ing)\b[\s\S]{{0,80}}\b({_NUM})%\s+(?:per|a|each) year", o)
    if m and re.search(r"\bhow many (?:full )?years\b", o):
        r = _fnum(m.group(1)) / 100
        if r > 0:
            return float(math.ceil(math.log(2) / math.log1p(r) - 1e-12))

    # ---- probabilités
    if re.search(r"\bexpected (?:gain|value|payoff|win|winnings)\b", o):
        pairs = re.findall(rf"\b(?:pays?|wins?|gives?|earns?|loses?)?\s*({_NUM})\s+euros?\s+with (?:a )?probability\s+({_NUM})\s*(%?)", o)
        pairs = pairs or re.findall(rf"({_NUM})\s+euros?\s+with (?:a )?probability\s+({_NUM})\s*(%?)", o)
        if pairs:
            ev_ = 0.0
            ps = 0.0
            for val, p, pct in pairs:
                pv = _fnum(p) / (100 if pct else 1)
                ev_ += _fnum(val) * pv
                ps += pv
            if re.search(r"\b(?:costs?|ticket|stake|entry fee|to play|bet|wager|lose|loses|nothing otherwise)\b", o):
                return None               # coût ou perte : gain net ambigu, le LLM décide
            if ps <= 1 + 1e-9:
                return ev_
    m = re.search(rf"\bprobability\s+({_NUM})[\s\S]{{0,120}}?\bamong\s+(\d+)\s+(?:parts?|trials?|items?|units?)[\s\S]*?\b(exactly|at most|at least|fewer than|more than|no more than|no less than|no fewer than)\s+(\d+)\s+(?:defective|success)", o)
    if m:
        p, n, how, k = _fnum(m.group(1)), int(m.group(2)), m.group(3), int(m.group(4))
        how = {"no more than": "at most", "no less than": "at least", "no fewer than": "at least"}.get(how, how)
        if 0 <= p <= 1 and 0 <= k <= n:
            rng = {"exactly": [k], "at most": range(0, k + 1), "at least": range(k, n + 1),
                   "fewer than": range(0, k), "more than": range(k + 1, n + 1)}[how]
            return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in rng)
    m = re.search(r"(\d+)\s+fair six-sided dice[\s\S]*?faces add up to (?:at least|more than|at most|less than)\s+(\d+)", o)
    if m:
        n, target = map(int, m.groups())
        ways = [1]
        for _ in range(n):
            nxt = [0] * (len(ways) + 6)
            for total, count in enumerate(ways):
                for face in range(1, 7):
                    nxt[total + face] += count
            ways = nxt
        how = re.search(r"add up to (at least|more than|at most|less than)", o).group(1)
        idx = {"at least": range(target, len(ways)), "more than": range(target + 1, len(ways)),
               "at most": range(0, min(target + 1, len(ways))), "less than": range(0, min(target, len(ways)))}[how]
        return sum(ways[i] for i in idx) / 6 ** n

    # ---- statistiques d'une liste donnée dans le texte
    m = re.search(r"\bweighted (?:mean|average)\b", o)
    if m:
        pairs = re.findall(rf"({_NUM})\s+in\s+[a-z][a-z ]*?\s*\((?:coefficient|weight)\s+({_NUM})\)", o)
        if len(pairs) >= 2:
            w = sum(_fnum(c) for _, c in pairs)
            return sum(_fnum(v) * _fnum(c) for v, c in pairs) / w if w else None
    count_m = re.search(rf"\b(?:count the values|how many (?:of these |of the )?(?:numbers|values))\s+(?:are\s+)?(strictly above|strictly below|strictly greater than|strictly less than|greater than|less than|above|below|at least|at most|equal to)\s+({_NUM})", o)
    if count_m:
        vals = _num_list(o[count_m.end():]) or _num_list(o)
        if len(vals) >= 2:
            how, thr = count_m.group(1), _fnum(count_m.group(2))
            f = {"strictly above": lambda v: v > thr, "strictly greater than": lambda v: v > thr,
                 "greater than": lambda v: v > thr, "above": lambda v: v > thr,
                 "strictly below": lambda v: v < thr, "strictly less than": lambda v: v < thr,
                 "less than": lambda v: v < thr, "below": lambda v: v < thr,
                 "at least": lambda v: v >= thr, "at most": lambda v: v <= thr,
                 "equal to": lambda v: v == thr}[how]
            return float(sum(1 for v in vals if f(v)))
    stat_q = re.search(r"\b(?:what (?:is|are)|find|compute|calculate|determine|give) (?:the |their |its )?(population standard deviation|sample standard deviation|"
                       r"standard deviation|population variance|sample variance|variance|mean|average|median|range|sum|total|"
                       r"maximum|minimum|largest value|smallest value)\b", o)
    question = o[stat_q.start():] if stat_q else ""
    if stat_q and re.search(r"\b(?:above|below|greater|less|more than|fewer|except|excluding|without|between|at least|at most|"
                            r"top|first|last|largest \d|smallest \d|highest \d|lowest \d|after removing|ignoring|outlier|"
                            r"if|when|only|positive|negative|even|odd|absolute deviation|deviation from|new|another|added|adding|"
                            r"all \d+|z-score|quartile|percentile|coefficient of variation|trimmed|geometric|harmonic)\b", question):
        stat_q = None                     # statistique d'une sous-liste : le LLM décide
    if stat_q and not re.search(r"\bweighted\b|\bprobabilit", o):
        vals = _num_list(o)
        if len(vals) >= 2:
            op = stat_q.group(1)
            a = np.asarray(vals, dtype=float)
            pop = bool(re.search(r"population|divide(?:d)? by n\b(?!\s*-)|divid\w* by n\)", o))
            samp = bool(re.search(r"sample|n\s*-\s*1|n−1", o))
            if op in ("mean", "average"):
                return float(a.mean())
            if op == "median":
                return float(np.median(a))
            if op in ("sum", "total"):
                return float(a.sum())
            if op in ("maximum", "largest value"):
                return float(a.max())
            if op in ("minimum", "smallest value"):
                return float(a.min())
            if op == "range":
                return float(a.max() - a.min())
            if "standard deviation" in op or "variance" in op:
                if op.startswith("population") or (pop and not samp):
                    ddof = 0
                elif op.startswith("sample") or (samp and not pop):
                    ddof = 1
                else:
                    return None
                if len(a) <= ddof:
                    return None
                return float(a.std(ddof=ddof) if "deviation" in op else a.var(ddof=ddof))

    # ---- conversions d'unités énoncées
    m = re.search(rf"\b1\s+([a-z]+)\s*=\s*({_NUM})\s*([a-z]+)\b[\s\S]{{0,40}}?\bconvert\s+({_NUM}|\d{{1,3}}(?:,\d{{3}})+)\s+([a-z]+)\s+(?:to|into)\s+([a-z]+)", o)
    if m:
        u1, f, u2, x, src, dst = m.group(1), _fnum(m.group(2)), m.group(3), _fnum(m.group(4)), m.group(5), m.group(6)
        def same(a, b):
            na, nb = _unit_norm(a), _unit_norm(b)
            if na and na == nb:
                return True
            a, b = a.rstrip("s"), b.rstrip("s")
            return a == b or {a, b} <= {"foot", "feet", "fet"}
        if same(src, u1) and same(dst, u2):
            return x * f
        if same(src, u2) and same(dst, u1) and f:
            return x / f

    # ---- vitesse / distance / durée
    m = re.search(rf"\b(?:travels?|drives?|covers?|runs?|cycles?)\s+({_NUM})\s+(km|kilometres?|kilometers?|miles?|m|metres?)\s+in\s+({_NUM})\s+(hours?|minutes?|h)\b", o)
    if m and re.search(r"\baverage speed\b", o):
        d, t = _fnum(m.group(1)), _fnum(m.group(3))
        if m.group(4).startswith("min"):
            if re.search(r"per hour|/h|km/h|an hour", o):
                t = t / 60
        return d / t if t else None

    # ---- arithmétique directe
    m = re.search(rf"\bwhat is\s+({_NUM})\s+(plus|minus|times|multiplied by|divided by)\s+({_NUM})\s*\?", o)
    if m:
        a, b = _fnum(m.group(1)), _fnum(m.group(3))
        return {"plus": a + b, "minus": a - b, "times": a * b, "multiplied by": a * b,
                "divided by": (a / b if b else None)}[m.group(2)]
    m = re.search(r"(?:^|[.?!]\s+)(?:compute|calculate|evaluate|what is)\s+(?:the value of\s+)?([0-9\s.,+\-−–×x*/÷^()√²³]+?)\s*(?:\?|\.(?:\s|$)|$)", o)
    if m and re.search(r"[+\-−–×*/÷^√²³]|\dx\d|\d x \d", m.group(1).strip()):
        return _safe_arith(m.group(1))
    return calc_rules_extra2(objective)


_V1 = []



# --------------------------------------------------------------------------- repli LLM : contrôle des réponses, indices, mesure

def _rules_off(kind):
    """Mode de mesure : coupe les règles d'une famille pour forcer le LLM (jamais activé par défaut)."""
    return bool(RULES_OFF) and (RULES_OFF == "all" or kind in RULES_OFF.split(","))


_UNIT_WORDS = {"hours", "hour", "minutes", "minute", "days", "day", "weeks", "week", "months", "month", "years", "year",
               "km", "kilometres", "kilometers", "metres", "meters", "litres", "liters", "kg", "grams", "euros", "percent",
               "times", "seconds", "points", "degrees", "cm", "miles", "inhabitants", "bacteria"}


def _answer_checks(objective):
    """Contraintes de forme déduites de la question : liste de (nom, prédicat, réparation ou None)."""
    o = " ".join(str(objective).lower().split())
    checks = []
    pct_change = re.search(r"\b(?:change|increase|decrease|growth|difference|higher|lower|points?|rate|interest|return|gain|loss)\b", o)
    if re.search(r"\bwhat fraction\b|\bfraction \(between 0 and 1\)|\bas a fraction\b|\bproportion\b", o) and not re.search(r"\bpercent", o):
        checks.append(("fraction", lambda v: -1e-9 <= v <= 1 + 1e-9, lambda v: v / 100 if 1 < v <= 100 else None))
    elif re.search(r"\bwhat is the (?:predicted )?probability\b", o) and not re.search(r"\bpercent|\bexpected\b|\bhow many\b", o):
        checks.append(("probability", lambda v: -1e-9 <= v <= 1 + 1e-9, lambda v: v / 100 if 1 < v <= 100 else None))
    if re.search(r"\bwhat percentage (?:of the [a-z]+ |of them )?(?:have|has|are|is|were|was|belong|comes?|of the)\b", o) and not pct_change:
        checks.append(("percentage", lambda v: -1e-9 <= v <= 100 + 1e-9, None))
    m = re.search(r"(?:^|[.?!]\s+)how many ([a-z]+)\b", o)
    if m and m.group(1) not in _UNIT_WORDS and not re.search(r"\bon average\b|\baverage\b|\bper\b|\bmean\b", o):
        checks.append(("count", lambda v: v >= -1e-9 and abs(v - round(v)) < 1e-6, None))
    if re.search(r"\byyyymmdd\b", o):
        def valid_date(v):
            try:
                n = int(round(v))
                return abs(v - n) < 1e-6 and datetime.date(n // 10000, (n // 100) % 100, n % 100) is not None
            except Exception:
                return False
        checks.append(("date", valid_date, None))
    if re.search(r"monday\s*=\s*1", o):
        checks.append(("weekday", lambda v: abs(v - round(v)) < 1e-6 and 1 <= round(v) <= 7, None))
    if re.search(r"\bwhat is the rank of\b", o):
        checks.append(("rank", lambda v: abs(v - round(v)) < 1e-6 and v >= 1, None))
    if re.search(r"\bwhat is (?:the |their |its )?(?:population |sample )?(?:standard deviation|variance|interquartile range)\b", o) \
            and not re.search(r"\bminus\b|\bdifference\b|\bz-score\b|\bchange\b", o):
        checks.append(("non-negative", lambda v: v >= -1e-9, None))
    return checks


def _sanitize(objective, cands):
    """Écarte les candidats du LLM qui violent une contrainte de forme ; répare si aucun n'est valide."""
    checks = _answer_checks(objective)
    if not checks or not cands:
        return cands
    ok = lambda v: all(pred(v) for _, pred, _ in checks)
    valid = [(v, w) for v, w in cands if v is not None and math.isfinite(v) and ok(v)]
    if valid:
        return valid
    repaired = []
    for v, w in cands:
        for _, pred, fix in checks:
            if fix is not None and not pred(v):
                nv = fix(v)
                if nv is not None and ok(nv):
                    repaired.append((nv, w))
                break
    return repaired or cands


def _fmt_cond(name, col, op, val):
    ref = f"dfs['{name}']['{col}']"
    sym = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "ne": "!=", "eq": "=="}
    if op in sym:
        if isinstance(val, str):
            return f"{ref} (as text, case-insensitive) {sym[op]} '{val}'"
        return f"{ref} {sym[op]} {fmt_num_hint(val)}"
    if op.startswith("date_"):
        return f"{ref} {sym[op[5:]]} pd.Timestamp('{pd.Timestamp(val).date()}')"
    if op == "year_eq":
        return f"{ref}.dt.year == {val}"
    if op == "month_eq":
        return f"({ref}.dt.month == {val[0]}) & ({ref}.dt.year == {val[1]})"
    if op == "between":
        return f"{ref}.between({fmt_num_hint(val[0])}, {fmt_num_hint(val[1])})"
    if op == "is_missing":
        return f"{ref}.isna()"
    if op == "not_missing":
        return f"{ref}.notna()"
    return None


def fmt_num_hint(x):
    try:
        x = float(x)
        return str(int(x)) if x.is_integer() else repr(x)
    except Exception:
        return repr(x)


def _hint_columns(objective, files):
    """Colonnes que le parseur a identifiées sans ambiguïté (mesure demandée, regroupement, filtres)."""
    try:
        dfs, dico = load_task(objective, files)
    except Exception:
        return set()
    o = " ".join(_norm_thousands(objective).lower().split())
    cols = set()
    if len(dfs) == 1:
        name = next(iter(dfs))
        m = re.search(r"\bwhat is the " + _AGG_WORDS + r" (?:of the )?(.+?) of the [a-z]+\b", o)
        if m:
            meas = _w2_measure(m.group(2), dfs, dico, name)
            if meas:
                cols.add(meas[0])
        g = re.search(r"\bfor each (.+?)[.,]", o)
        if g and not re.match(r"(?:the )?(?:year|month) of", g.group(1).strip()):
            col = _resolve_dico_column(g.group(1), dfs, dico)
            if col and col[0] == name:
                cols.add(col[1])
    try:
        for c in _parse_table_conditions_raw(o, dfs, dico):
            cols.add(c[1])
    except Exception:
        pass
    return {c for c in cols if isinstance(c, str) and len(c) >= 1}


def _code_uses(code, col):
    return bool(re.search(r"""['"]""" + re.escape(col) + r"""['"]|\.""" + re.escape(col) + r"\b", code))


def _nofile_hints(objective):
    """Indices pour un calcul sans fichier : nombres de l'énoncé et conventions repérées."""
    o = " ".join(_norm_thousands(objective).split())
    low = o.lower()
    body = re.split(r"(?i)\b(?:round|report it|give the answer|give a whole|answer with|the answer is)\b", o)[0]
    nums = re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w])", body)
    h = []
    if 2 <= len(nums) <= 40:
        h.append("Numbers in the statement, in order: " + ", ".join(nums) + ".")
    spread = re.search(r"\bstandard deviation\b|\bvariance\b", low)
    if spread and re.search(r"\bpopulation (?:standard deviation|variance)\b|divide by n\)", low):
        h.append("Population formula: divide by n (np.std(x, ddof=0), np.var(x, ddof=0)).")
    elif spread and re.search(r"\bsample (?:standard deviation|variance)\b|n - 1|n-1", low):
        h.append("Sample formula: divide by n - 1 (np.std(x, ddof=1), np.var(x, ddof=1)).")
    if re.search(r"\bhow many days\b", low) and re.search(r"\bboth (?:the first and the last day|included)\b|\binclusive\b|count both", low):
        h.append("Both the first and the last day are counted: add 1 to (d2 - d1).days.")
    if re.search(r"\bweekdays?\b|working days|business days", low):
        h.append("Monday to Friday between two dates (both included): np.busday_count(d1, d2 + datetime.timedelta(days=1)).")
    if re.search(r"\bin percent\b|\bwhat percentage\b|percentage change|in percentage", low):
        h.append("The answer is a percentage: multiply a ratio by 100.")
    if re.search(r"\bprobability\b", low) and not re.search(r"percent", low):
        h.append("A probability is a number between 0 and 1.")
    if re.search(r"\byyyymmdd\b", low):
        h.append("Build the date with datetime.date and answer = d.year * 10000 + d.month * 100 + d.day.")
    return h


def _llm_table_hints(objective, files):
    """Ce que le parseur a compris, même quand aucune règle n'a répondu : colonnes et filtres résolus sans ambiguïté."""
    try:
        dfs, dico = load_task(objective, files)
    except Exception:
        return []
    o = " ".join(_norm_thousands(objective).lower().split())
    lines = []
    if len(dfs) == 1:
        name = next(iter(dfs))
        m = re.search(r"\bwhat is the " + _AGG_WORDS + r" (?:of the )?(.+?) of the [a-z]+\b", o)
        if m:
            meas = _w2_measure(m.group(2), dfs, dico, name)
            if meas:
                lines.append(f"'{m.group(2).strip()}' is the column dfs['{name}']['{meas[0]}'].")
        g = re.search(r"\bfor each (.+?)[.,]", o)
        if g:
            gp = g.group(1).strip()
            ym = re.match(r"(?:the )?year of (?:the )?(.+)$", gp)
            if ym:
                dc = _w2_date(ym.group(1), dfs, dico, name)
                if dc:
                    lines.append(f"Group by the year of the date column: dfs['{name}']['{dc}'].dt.year.")
            else:
                col = _resolve_dico_column(gp, dfs, dico)
                if col and col[0] == name:
                    lines.append(f"'{gp}' is the column dfs['{name}']['{col[1]}'].")
    try:
        conds = _parse_table_conditions_raw(o, dfs, dico)
    except Exception:
        conds = []
    # Deux valeurs d'une même colonne (« A or B », deux côtés d'un « minus ») : pas d'indice, il serait trompeur.
    per_col = collections.Counter((c[0], c[1]) for c in conds)
    if re.search(r"\bminus\b|\bor\b|\bdifference between the mean\b|\bversus\b|\bcompared\b", o):
        conds = [c for c in conds if per_col[(c[0], c[1])] == 1]
    conds = [c for c in conds if per_col[(c[0], c[1])] == 1]
    prop = re.search(r"\bwhat (?:percentage|fraction|proportion|share)\b", o)
    seen = set()
    for name, col, op, val, pos in conds:
        txt = _fmt_cond(name, col, op, val)
        if not txt or txt in seen:
            continue
        seen.add(txt)
        if prop and pos > prop.start():
            lines.append("Counted rows (numerator of the proportion): " + txt + ".")
        elif prop:
            lines.append("Population (denominator of the proportion): rows where " + txt + ".")
        else:
            lines.append("Filter: keep the rows where " + txt + ".")
    if not lines:
        return []
    return ["Hints from a question parser (they may be incomplete; check them against the dictionary):"] + \
        ["- " + x for x in lines[:6]]


class _V1Path:
    """Chemin « tables » de la version précédente (code recopié plus bas avec le préfixe _v1_)."""

    def __init__(self):
        g = globals()
        for name in ("load_task", "table_stat_rule", "table_group_rule", "table_join_rule", "table_derive_rule",
                     "table_filter_rule", "align_keys", "entity_hints", "build_messages", "build_user", "run_code"):
            setattr(self, name, g["_v1_" + name])

    @property
    def LAST_ERR(self):
        return globals()["_v1_LAST_ERR"]

    @property
    def LAST_STATE(self):
        return globals()["_v1_LAST_STATE"]


def _load_v1():
    if not _V1:
        try:
            _V1.append(_V1Path())
        except Exception:
            _V1.append(None)
    return _V1[0]


def _v2_general_answer(obj, files):
    for fn in (table_stat_rule, table_group_rule, table_join_rule, table_derive_rule, table_filter_rule):
        try:
            d2, k2 = load_task(obj, files)
            rv = fn(obj, d2, k2)
        except Exception:
            rv = None
        if rv is not None and math.isfinite(rv):
            return rv
    return None


def _table_llm_entry(V1, t, obj, files, dfs, dico, **extra):
    notes = (V1.align_keys(dfs, dico) if len(dfs) >= 2 else []) + V1.entity_hints(obj, dico)
    if LLM_HINTS:
        try:
            notes = notes + _llm_table_hints(obj, files)
        except Exception:
            pass
    msgs = V1.build_messages("table", V1.build_user("table", obj, dfs, dico, notes), obj, len(dfs))
    hint_cols = set()
    if HINT_WEIGHT:
        try:
            hint_cols = _hint_columns(obj, files)
        except Exception:
            hint_cols = set()
    return dict(id=t["id"], obj=obj, dfs=dfs, msgs=msgs, c=[], ok=True, mod=V1, hint_cols=hint_cols, **extra)


def _prep_table_v1(V1, t, files, fallback_v2, arbitrate=False):
    """Prépare une tâche « table » exactement comme la version précédente."""
    obj = t["objective"]
    off = _rules_off("table")
    try:
        d2, k2 = load_task(obj, files)
        rv = None if off else table_extra_rule(obj, d2, k2)
    except Exception:
        rv = None
    dfs, dico = V1.load_task(obj, files)
    if rv is not None and math.isfinite(rv):
        return dict(id=t["id"], obj=obj, dfs=dfs, msgs=None, c=[(rv, 100.0)], ok=False)
    unsafe = off or _w2_unsafe(obj)           # construction mal lue par les règles historiques : LLM directement
    for fn in (() if unsafe else (V1.table_stat_rule, V1.table_group_rule, V1.table_join_rule, V1.table_derive_rule,
                                  V1.table_filter_rule)):
        try:
            rv = fn(obj, dfs, dico)
        except Exception:
            rv = None
        if rv is not None and math.isfinite(rv):
            if arbitrate and not off:
                rv2 = _v2_general_answer(obj, files)
                nd = decimals(obj)
                same = rv2 is None or round(rv2, nd if nd is not None else 6) == round(rv, nd if nd is not None else 6)
                if not same and arbitrate == "v2":
                    return dict(id=t["id"], obj=obj, dfs=dfs, msgs=None, c=[(rv2, 100.0)], ok=False)
                if not same:
                    # les deux jeux de règles divergent : le LLM départage (à égalité, la règle historique l'emporte)
                    return _table_llm_entry(V1, t, obj, files, dfs, dico, arb=(rv, rv2))
            return dict(id=t["id"], obj=obj, dfs=dfs, msgs=None, c=[(rv, 100.0)], ok=False)
    if fallback_v2 and not unsafe:
        for fn in (table_stat_rule, table_group_rule, table_join_rule, table_derive_rule, table_filter_rule):
            try:
                d2, k2 = load_task(obj, files)
                rv = fn(obj, d2, k2)
            except Exception:
                rv = None
            if rv is not None and math.isfinite(rv):
                return dict(id=t["id"], obj=obj, dfs=dfs, msgs=None, c=[(rv, 100.0)], ok=False)
    notes = (V1.align_keys(dfs, dico) if len(dfs) >= 2 else []) + V1.entity_hints(obj, dico)
    if LLM_HINTS:
        try:
            notes = notes + _llm_table_hints(obj, files)
        except Exception:
            pass
    msgs = V1.build_messages("table", V1.build_user("table", obj, dfs, dico, notes), obj, len(dfs))
    hint_cols = set()
    if HINT_WEIGHT:
        try:
            hint_cols = _hint_columns(obj, files)
        except Exception:
            hint_cols = set()
    return dict(id=t["id"], obj=obj, dfs=dfs, msgs=msgs, c=[], ok=True, mod=V1, hint_cols=hint_cols)


def extract_code(text):
    text = text.split("```")[0]
    return text.strip("\n")


def vote(cands, nd):
    """cands : liste de (valeur, poids). Renvoie la valeur majoritaire."""
    if not cands:
        return 0.0
    cnt = collections.defaultdict(float)
    rep = {}
    for v, w in cands:
        k = round(v, nd if nd is not None else 4)
        cnt[k] += w
        rep.setdefault(k, v)
    best = max(cnt, key=cnt.get)
    return rep[best]


# --------------------------------------------------------------------------- agent


# ---------------------------------------------------------------------------
# Solveur à règles pour les régressions (fits) : l'énoncé est très régulier. On lit la cible, les variables, le filtre
# d'années et la question dans l'énoncé ; le calcul est fait par code. Au moindre doute, on renvoie None et c'est le
# modèle de langage qui reprend la main.
# ---------------------------------------------------------------------------
_STOP = {"the", "of", "a", "an", "in", "on", "at", "per", "to", "for", "and", "with", "by", "is", "are", "last"}


def _toks(s):
    out = set()
    for w in re.findall(r"[a-z]+", s.lower()):
        if w in _STOP:
            continue
        if w in {"used", "uses", "using"}:
            w = "use"
        elif w in {"billed", "billing"}:
            w = "bill"
        elif w in {"born", "births", "birthday", "birthdate"}:
            w = "birth"
        elif w in {"shipped", "shipping", "shipment", "shipments"}:
            w = "ship"
        elif w in ("carrier", "carriers"):
            w = "transport"
        elif w == "occupancy":
            w = "people"
        elif w in {"pupil", "pupils", "enrollment", "enrolled"}:
            w = "size"
        elif w in {"absences", "absence"}:
            w = "absence"
        out.add(w[:-1] if w.endswith("s") and len(w) > 3 else w)
    return out


def _dico_rows(dico):
    rows = []
    for line in dico.split("\n"):
        parts = [p.strip() for p in line.split(" | ")]
        if len(parts) >= 3:
            rows.append((parts[0], parts[1], parts[2]))   # (fichier, code, description)
    return rows


def _dictionary_unit(name, column, dico):
    table = stem(name).lower()
    for line in dico.split("\n"):
        parts = [p.strip() for p in line.split(" | ")]
        if len(parts) >= 4 and stem(parts[0]).lower() == table and parts[1] == column:
            return parts[3]
    return ""


def _unit_key(unit):
    key = re.sub(r"[^a-z0-9]", "", str(unit).lower())
    return {
        "kg": "kg", "kilogram": "kg", "kilograms": "kg",
        "tonne": "tonne", "tonnes": "tonne", "metricton": "tonne", "metrictons": "tonne",
        "kwh": "kwh", "kilowatthour": "kwh", "kilowatthours": "kwh",
        "mj": "mj", "megajoule": "mj", "megajoules": "mj",
    }.get(key)


def _explicit_unit_factor(objective, source_unit, target_unit):
    """Return a conversion factor only when the objective states the exact unit equation."""
    source, target = _unit_key(source_unit), _unit_key(target_unit)
    if not source or not target:
        return None
    for m in re.finditer(r"\b1\s*([a-z][a-z0-9]*)\s*=\s*([\d.,]+)\s*([a-z][a-z0-9]*)\b", objective, re.I):
        left, right = _unit_key(m.group(1)), _unit_key(m.group(3))
        try:
            factor = float(m.group(2).replace(",", "."))
        except ValueError:
            continue
        if not math.isfinite(factor) or factor <= 0:
            continue
        if source == left and target == right:
            return factor
        if source == right and target == left:
            return 1.0 / factor
    return None


def _dedupe_requested(objective):
    """Recognise the common ways tasks ask us to keep each record only once."""
    return bool(re.search(
        r"\b(?:drop|remove|delete|ignore|exclude|deduplicat\w*|dedupe)\b[^.;?]{0,80}\bduplicat\w*\b|"
        r"\b(?:keep|retain)\b[^.;?]{0,80}\bone copy\b|"
        r"\bcount each .{1,50}\b(?:once|only once)\b",
        objective, re.I))


def _table_descriptions(name, columns, dico):
    """Descriptions from this table's dictionary rows, without collisions from other files."""
    table = stem(name).lower()
    return {c: desc for f, c, desc in _dico_rows(dico)
            if stem(f).lower() == table and c in columns}


def _identifier_column(name, columns, dico):
    descs = _table_descriptions(name, columns, dico)
    candidates = []
    for c in columns:
        text = (str(c) + " " + descs.get(c, "")).lower()
        score = 0
        if re.search(r"\b(identifier|unique id|record id|row id|shipment id|order id|loan id|pupil id|student id|flat id|reading id)\b", text):
            score += 5
        if re.search(r"\b(id|no|number|ref|reference)\b", str(c).lower()):
            score += 3
        if re.search(r"\b(identifier|identification number|record number|reading number|order number|loan number)\b", descs.get(c, ""), re.I):
            score += 3
        if score:
            candidates.append((score, c))
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        if len(candidates) == 1 or candidates[0][0] > candidates[1][0]:
            return candidates[0][1]
        # An ambiguous ID mapping is unsafe; whole-row deduplication remains the fallback.
        return None
    # These generated challenge tables place the record key first when no dictionary label says "ID".
    return columns[0] if len(columns) else None


def _distinct_target_column(objective, dfs, dico, table_name):
    """Resolve the field whose distinct values are requested; never guess a row ID."""
    o = " ".join(objective.lower().split())
    patterns = (
        r"\b(?:distinct|unique)\s+values?\s+(?:of|for|in)\s+(.+?)(?=\s+(?:are|is|in|among|for|from|where|whose|with|of the|across|over)\b|[?.;]|$)",
        r"\b(?:distinct|unique)\s+(.+?)(?=\s+(?:are|is|in|among|for|from|where|whose|with|of the|across|over|appear|appears|occur|occurs|can|placed|have|has)\b|[?.;]|$)",
        r"\bvalues?\s+of\s+(.+?)(?=\s+(?:are|is|in|among|for|from|where|whose|with|of the|across|over)\b|[?.;]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, o)
        if not match:
            continue
        phrase = match.group(1).strip(" ,.;:()")
        if re.fullmatch(r"(?:rows?|records?|entries|observations|people|persons)", phrase):
            return None
        resolved = _resolve_dico_column(phrase, dfs, dico)
        if resolved and resolved[0] == table_name:
            return resolved[1]
    return None


def _has_unparsed_filter_intent(objective, conditions):
    """Avoid deterministic aggregates when the question contains an unsupported filter."""
    o = " ".join(objective.lower().split())
    o = " ".join(s for s in re.split(r"(?<=[.?!])\s+", o)
                 if not (_META_SENTENCE.search(s) and not re.search(r"\b(?:what|how many|compute|consider|among|rank)\b", s)))
    # « … appear among the orders? » names the population, not a filter.
    o = re.sub(r"\bamong (?:all )?(?:the )?[a-z]+(?=\s*[?.;]|$)", " ", o)
    # Missing-value handling is often global task metadata, not a row filter.
    filter_text = re.sub(
        r"\b(?:ignore|leave|exclude|drop|values?)\b[^.;?]{0,160}\b(?:missing|not recorded|unrecorded)\b[^.;?]*(?:[.;?]|$)",
        " ", o)
    if not conditions and re.search(
        r"\b(?:where|whose|among|excluding|except|without|not equal|different from|"
        r"greater than|less than|at least|at most|no more than|no less than|"
        r"on or after|on or before|after|before|is not|is missing|is blank|is null|is recorded|"
        r"contains|starts with|ends with)\b", filter_text
    ):
        return True
    if re.search(r"\bbetween\s+-?\d", filter_text) and not any(c[2] == "between" for c in conditions):
        return True
    # Membership and disjunctions are not represented by the current condition parser.
    # Exclude the challenge's standard missing-value instruction; its "or" is not a filter.
    logical_text = re.sub(r"\bon or after\b", " ", filter_text)
    logical_text = re.sub(r"\b(?:missing|blank|null|unknown|not available)\s+or\s+(?:not recorded|missing|blank|null|unknown)\b", " ", logical_text)
    if re.search(r"\b(?:one of|any of|either|or)\b", logical_text):
        return True
    return False


def _is_identity_desc(desc):
    d = str(desc).strip().lower()
    return bool(re.search(r"\b(identifier|code|key|reference)\b", d) or re.search(r"\bnumber$", d)
                or re.search(r"^(?:\w+\s+)?(?:id|no\.?)$", d))


def _measure_allow(dfs, dico):
    descs = {(stem(f).lower(), c): d for f, c, d in _dico_rows(dico)}

    def allow(name, col):
        if _is_identity_desc(descs.get((name, col), "")):
            return False
        s = dfs[name][col]
        if s.dtype.kind in "fiub":
            return True
        if s.dtype.kind == "M":
            return False
        return _numeric_metric(s, descs.get((name, col), ""), force=True) is not None
    return allow


def _resolve_dico_column(phrase, dfs, dico, allow=None):
    """Resolve a natural-language field phrase only when one dictionary column clearly fits."""
    def field_toks(text):
        tokens = _toks(text)
        # `_toks` intentionally keeps its existing behavior for the fit/no-file paths.
        # Add the common English -ies -> -y plural form only for table-field matching.
        tokens.update(w[:-3] + "y" for w in re.findall(r"[a-z]+", str(text).lower()) if w.endswith("ies") and len(w) > 4)
        return tokens

    q = field_toks(phrase)
    if not q:
        return None
    q_order = [w for w in re.findall(r"[a-z]+", phrase.lower()) if w not in _STOP]
    q_last = _toks(q_order[-1]) if q_order else set()
    qnorm = re.sub(r"[^a-z0-9]", "", phrase.lower())
    scored = []
    for f, c, desc in _dico_rows(dico):
        name = stem(f).lower()
        if name not in dfs or c not in dfs[name].columns:
            continue
        if allow is not None and not allow(name, c):
            continue
        words = field_toks(desc) | field_toks(c)
        overlap = len(q & words)
        score = 3 * len(q & field_toks(desc)) + len(q & field_toks(c))
        if q_last & words:
            score += 2
        dnorm = re.sub(r"[^a-z0-9]", "", (desc + " " + c).lower())
        if qnorm and (qnorm in dnorm or dnorm in qnorm):
            score += 8
        if qnorm and dnorm.startswith(qnorm):
            score += 6
        if overlap:
            scored.append((score, name, c))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return scored[0][1], scored[0][2]


def _apply_data_cleanup(dfs, objective, dico, dedupe):
    """Apply explicit duplicate and missing-sentinel instructions before code generation."""
    if dedupe:
        for name, df in list(dfs.items()):
            key = _identifier_column(name, df.columns, dico)
            subset = [key] if key in df.columns else None
            dfs[name] = df.drop_duplicates(subset=subset).reset_index(drop=True)

    missing = r"(?:not recorded|missing|unknown|not available)"
    patterns = (
        rf"\bvalue\s+(-?\d+(?:[.,]\d+)?)\s+in the\s+([^.;?]+?)\s+column\s+means?\s+(?:it\s+)?(?:is\s+)?{missing}",
        rf"(?:^|[.;?]\s+)(?:a|an|the)\s+([^.;?]+?)\s+(?:of|equal(?:s)?|is)\s+(-?\d+(?:[.,]\d+)?)\s+means?\s+(?:it\s+)?(?:was\s+|is\s+)?{missing}",
    )
    for i, pattern in enumerate(patterns):
        for m in re.finditer(pattern, objective, re.I):
            sentinel, phrase = (m.group(1), m.group(2)) if i == 0 else (m.group(2), m.group(1))
            resolved = _resolve_dico_column(phrase, dfs, dico)
            if not resolved:
                continue
            name, col = resolved
            val = _num_from_text(sentinel)
            series = dfs[name][col]
            if pd.api.types.is_numeric_dtype(series):
                dfs[name][col] = series.mask(series == val, np.nan)
            else:
                dfs[name][col] = series.replace([sentinel, str(val)], np.nan)


def table_stat_rule(objective, dfs, dico):
    """Resolve simple whole-table statistics from the dictionary, otherwise leave them to the model."""
    if _w2_unsafe(objective):
        return None
    if len(dfs) != 1:
        return None
    o = " ".join(objective.lower().split())
    o = re.sub(r"\bhow many different\b", "how many distinct", o)
    # Leave row-level ratios and grouped/filtered questions to their dedicated rules.
    if re.search(r"\bdivided by\b|\bconsider the .+?\s+per\s+", o) or re.search(r"\bfor each\b|\brank\b|\bgrouped by\b", o):
        return None

    # Count values in the requested field, not rows via an arbitrary identifier.
    if re.search(r"\b(?:distinct|unique)\b", o) and re.search(r"\b(?:how many|number of|count of|count the number of)\b", o):
        table_name, df = next(iter(dfs.items()))
        conditions = [c for c in _parse_table_conditions(o, dfs, dico) if c[0] == table_name]
        if _has_unparsed_filter_intent(o, conditions):
            return None
        target = _distinct_target_column(o, dfs, dico, table_name)
        if not target and re.search(r"\b(?:distinct|unique)\s+(?:rows?|records?|entries|observations)\b", o):
            target = _identifier_column(table_name, df.columns, dico)
        if not target:
            return None
        mask = pd.Series(True, index=df.index)
        for _, col, op, val, _ in conditions:
            mask &= _condition_mask(df, col, op, val).fillna(False)
        return float(df.loc[mask, target].nunique(dropna=True))

    # The challenge's own example uses distinct order statistics for "second highest".
    ordinals = {"second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4,
                "fifth": 5, "5th": 5, "sixth": 6, "6th": 6, "seventh": 7, "7th": 7,
                "eighth": 8, "8th": 8, "ninth": 9, "9th": 9, "tenth": 10, "10th": 10}
    order_stat = re.search(
        r"\b(second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|seventh|7th|eighth|8th|ninth|9th|tenth|10th)"
        r"[- ]+(highest|largest|greatest|lowest|smallest|least)\b", o)
    if order_stat:
        tail = o[order_stat.end():].strip()
        tail = re.sub(r"^(?:value\s+of|the\s+value\s+of|the)\s+", "", tail)
        cut = re.search(
            r"\b(?:over all|across all|throughout all|among|where|whose|with|for|on or after|on or before|"
            r"strictly greater than|strictly less than|greater than|less than|at least|at most|between)\b|[?.;]", tail)
        phrase = tail[:cut.start()] if cut else tail
        col = _resolve_dico_column(phrase.strip(" ,.;:()"), dfs, dico)
        if not col or col[0] not in dfs:
            return None
        conditions = [c for c in _parse_table_conditions(o, dfs, dico) if c[0] == col[0]]
        if _has_unparsed_filter_intent(o, conditions):
            return None
        df = dfs[col[0]]
        descriptions = _table_descriptions(col[0], df.columns, dico)
        numeric_values = _numeric_metric(df[col[1]], descriptions.get(col[1], ""))
        if numeric_values is None:
            return None
        mask = pd.Series(True, index=df.index)
        for _, cond_col, cond_op, val, _ in conditions:
            mask &= _condition_mask(df, cond_col, cond_op, val).fillna(False)
        values = numeric_values.loc[mask].dropna().drop_duplicates()
        rank = ordinals[order_stat.group(1)]
        if len(values) < rank:
            return None
        ascending = order_stat.group(2) in ("lowest", "smallest", "least")
        return float(values.sort_values(ascending=ascending).iloc[rank - 1])

    # A pair statistic is unambiguous when both named fields resolve to the same table.
    corr = re.search(
        r"\b(?:pearson\s+)?correlation(?: coefficient)?\s+between\s+(.+?)\s+and\s+(.+?)"
        r"(?=\s+(?:for|among|over|across|within|where|whose|with)\b|[?.;]|$)", o)
    if not corr:
        corr = re.search(
            r"\b(?:pearson\s+)?correlation(?: coefficient)?\s+of\s+(.+?)\s+and\s+(.+?)"
            r"(?=\s+(?:for|among|over|across|within|where|whose|with)\b|[?.;]|$)", o)
    if corr:
        left = _resolve_dico_column(corr.group(1), dfs, dico)
        right = _resolve_dico_column(corr.group(2), dfs, dico)
        if not left or not right or left[0] != right[0]:
            return None
        conditions = [c for c in _parse_table_conditions(o, dfs, dico) if c[0] == left[0]]
        if _has_unparsed_filter_intent(o, conditions):
            return None
        df = dfs[left[0]]
        descriptions = _table_descriptions(left[0], df.columns, dico)
        left_values = _numeric_metric(df[left[1]], descriptions.get(left[1], ""))
        right_values = _numeric_metric(df[right[1]], descriptions.get(right[1], ""))
        if left_values is None or right_values is None:
            return None
        mask = pd.Series(True, index=df.index)
        for _, col, op, val, _ in conditions:
            mask &= _condition_mask(df, col, op, val).fillna(False)
        pair = pd.DataFrame({left[1]: left_values.loc[mask], right[1]: right_values.loc[mask]}).dropna()
        if len(pair) < 2:
            return None
        value = pair[left[1]].corr(pair[right[1]])
        return float(value) if pd.notna(value) and math.isfinite(float(value)) else None

    df_name, df = next(iter(dfs.items()))
    # A stated unit equation makes these converted whole-table means deterministic.
    unit_request = re.search(
        r"\bconsider\s+(?:the\s+)?(.+?)\s+in\s+([a-z][a-z0-9 -]*?)\s*\(\s*1\s+[a-z][a-z0-9]*\s*=\s*[\d.,]+\s*[a-z][a-z0-9]*\b",
        o)
    converted_mean = re.search(r"\b(?:its\s+(mean|average)|(mean|average)\s+of\s+it)\b", o)
    if unit_request and converted_mean:
        resolved = _resolve_dico_column(unit_request.group(1), dfs, dico)
        if not resolved or resolved[0] != df_name:
            return None
        conditions = _parse_table_conditions(objective, dfs, dico)
        clean_objective = re.sub(
            r"\b[\w.-]+\.csv contains duplicated rows\b[^.;?]*(?:[.;?]|$)",
            " ", o)
        if conditions or _has_unparsed_filter_intent(clean_objective, conditions):
            return None
        factor = _explicit_unit_factor(o, _dictionary_unit(resolved[0], resolved[1], dico), unit_request.group(2).strip())
        if factor is None:
            return None
        descriptions = _table_descriptions(resolved[0], dfs[resolved[0]].columns, dico)
        numeric_values = _numeric_metric(dfs[resolved[0]][resolved[1]], descriptions.get(resolved[1], ""))
        if numeric_values is None:
            return None
        values = numeric_values.dropna()
        return float(values.mean() * factor) if not values.empty else None

    # Count missing values only when the field and missingness are both explicit.
    missing_count = re.search(
        r"\bhow many\b[^.?;]*?\b(?:have|has)\s+no\s+(?:recorded\s+)?([^.?;]+?)(?=[?.;]|$)", o)
    if not missing_count:
        missing_count = re.search(
            r"\bhow many\b[^.?;]*?\bis\s+(?:the\s+)?([^.?;]+?)\s+(?:missing|not recorded|unrecorded)\b", o)
    if missing_count:
        resolved = _resolve_dico_column(missing_count.group(1), dfs, dico)
        if not resolved:
            return None
        conditions = [c for c in _parse_table_conditions(objective, dfs, dico) if c[0] == resolved[0]]
        clean_objective = re.sub(
            r"\b[\w.-]+\.csv contains duplicated rows\b[^.;?]*(?:[.;?]|$)",
            " ", o)
        if _has_unparsed_filter_intent(clean_objective, conditions):
            return None
        source = dfs[resolved[0]]
        mask = pd.Series(True, index=source.index)
        for _, cond_col, cond_op, value, _ in conditions:
            mask &= _condition_mask(source, cond_col, cond_op, value).fillna(False)
        return float(source.loc[mask, resolved[1]].isna().sum())

    if re.search(r"\bhow many\b[^.?;]*\b(?:have|has)\s+(?:a |an )?recorded\b", o):
        question = re.search(r"\bhow many\b[^.?;]*", o).group(0)
        if re.search(r"\b(?:among|whose|with|born|on or after|on or before)\b", question):
            return None
        m = re.search(r"\b(?:have|has)\s+(?:a |an )?(?:recorded )?([^.?;]+?)(?:\?|\.|$)", question)
        if not m:
            return None
        col = _resolve_dico_column(m.group(1), dfs, dico)
        if not col:
            return None
        return float(dfs[col[0]][col[1]].notna().sum())

    whole_scope = re.search(r"\b(?:(?:over|across|throughout|on|for)\s+(?:all|the entire|the whole)|overall)\b", o)
    scope_end = r"\s+(?:over all|across all|across the entire|across the whole|throughout all|for all|for the entire|for the whole|overall)\b"
    m = re.search(r"\b(\d+)(?:st|nd|rd|th) percentile of (?:the )?(.+?)(?=\s*(?:\(|\bover\b|\bamong\b|\bacross\b|[?.]|$))", o)
    if m:
        if not whole_scope:
            return None
        q, phrase = int(m.group(1)) / 100, m.group(2)
        operation = "quantile"
    else:
        # Require an explicit whole-table scope to avoid answering a filtered maximum/median incorrectly.
        if not whole_scope:
            return None
        m = re.search(
            rf"\b(standard deviation|std\.?\s*dev(?:iation)?|variance|range|spread|maximum|minimum|median|mean|average|sum|total)\s+(?:of\s+)?(?:the\s+)?(.+?)(?={scope_end}|\s*\(|[?.]|$)", o)
        if not m:
            return None
        op, phrase = m.group(1), m.group(2)
        operation = {
            "maximum": "max", "minimum": "min", "average": "mean", "std dev": "std", "total": "sum",
            "std. dev": "std", "std deviation": "std", "standard deviation": "std",
        }.get(op, op)

    col = _resolve_dico_column(phrase, dfs, dico)
    if not col:
        return None
    # "Over all" may still describe a filtered population. Do not return its unfiltered statistic.
    conditions = _parse_table_conditions(o, dfs, dico)
    if conditions or (operation in ("std", "variance", "range", "spread") and _has_unparsed_filter_intent(o, conditions)):
        return None
    raw_series = dfs[col[0]][col[1]]
    descriptions = _table_descriptions(col[0], dfs[col[0]].columns, dico)
    numeric_values = _numeric_metric(raw_series, descriptions.get(col[1], ""))
    if numeric_values is None:
        return None
    series = numeric_values.dropna()
    if series.empty:
        return None
    if operation == "quantile":
        return float(series.quantile(q, interpolation="linear"))
    if operation == "max":
        return float(series.max())
    if operation == "min":
        return float(series.min())
    if operation == "median":
        return float(series.median())
    if operation == "mean":
        return float(series.mean())
    if operation == "sum":
        return float(series.sum())
    if operation == "range" or operation == "spread":
        return float(series.max() - series.min())
    if operation in ("std", "variance"):
        ddof = 0 if re.search(r"\bpopulation\b", o) else 1
        if len(series) <= ddof:
            return None
        return float(series.std(ddof=ddof) if operation == "std" else series.var(ddof=ddof))
    return None


def _condition_context(prefix):
    """Keep the field phrase nearest to a filter operator/value (never beyond the current sentence)."""
    low = prefix.lower()
    cut = max(low.rfind(". "), low.rfind("? "), low.rfind("; "))
    if cut >= 0:
        low = low[cut + 2:]
    markers = (" and ", " with ", " whose ", " have ", " has ", " among ", " where ", " of ", " over ", " born ")
    cuts = [(low.rfind(mark), len(mark)) for mark in markers if low.rfind(mark) >= 0]
    for pos, size in sorted(cuts, reverse=True):
        value = low[pos + size:].strip()
        value = re.sub(r"^(?:the|a|an)\s+", "", value)
        value = re.sub(r"\s+(?:of|the|a|an)$", "", value).strip()
        if value:
            return value[-80:]
    return low[-80:].strip()


def _condition_contexts(prefix):
    """Candidate field phrases before an operator, nearest first, within the current clause only."""
    low = prefix.lower()
    clause = max(low.rfind(p) for p in (",", ";", "?", ". ", ":", "what percentage", "what fraction", "what share"))
    if clause >= 0:
        low = low[clause + 1:]
    markers = (" and ", " with ", " whose ", " have ", " has ", " among ", " where ", " of ", " over ", " born ", " for ")
    out = []
    for mark in markers:
        start = 0
        while True:
            pos = low.find(mark, start)
            if pos < 0:
                break
            value = low[pos + len(mark):].strip()
            value = re.sub(r"^(?:the|a|an)\s+", "", value)
            value = re.sub(r"\s+(?:of|the|a|an)$", "", value).strip()
            if value:
                out.append((pos, value[-80:]))
            start = pos + 1
    out.sort(key=lambda x: -x[0])
    seen, res = set(), []
    for _, v in out:
        if v not in seen:
            seen.add(v)
            res.append(v)
    return res


def _ctx_column(prefix, dfs, dico, date_like=False, numeric=False):
    """Resolve the field of a condition: the historical nearest context first, then wider phrases of the clause."""
    first = _condition_context(prefix)
    col = _condition_column(first, dfs, dico, date_like=date_like, numeric=numeric)
    if col:
        return col
    for ctx in _condition_contexts(prefix):
        if ctx == first:
            continue
        col = _condition_column(ctx, dfs, dico, date_like=False, numeric=numeric)
        if col:
            return col
    return None


_VERB_DATE = {"born": "date of birth", "shipped": "date of shipment", "delivered": "date of delivery",
              "listed": "date listed", "placed": "date order placed", "ordered": "date order placed",
              "granted": "date granted", "admitted": "date of admission", "taken": "day of the reading",
              "let": "date let", "joined": "date joined", "signed": "date signed", "opened": "date opened"}
_VERB_DATE_RE = r"\b(born|shipped|delivered|listed|placed|ordered|granted|admitted|taken|let|joined|signed|opened)"


def _verb_date_column(verb, dfs, dico):
    """Date column named by a participle (« shipped », « born ») ; None if not unique."""
    ctx = _VERB_DATE.get(verb)
    if not ctx:
        return None
    date_cols = [(n, c) for n, d in dfs.items() for c in d.columns if d[c].dtype.kind == "M"]
    if len(date_cols) == 1:
        return date_cols[0]
    key = {"born": "birth", "shipped": "ship", "delivered": "deliver", "admitted": "admission"}.get(verb, verb)
    hits = []
    for n, c in date_cols:
        desc = _table_descriptions(n, dfs[n].columns, dico).get(c, "")
        if key in _toks(desc) or key in desc.lower():
            hits.append((n, c))
    return hits[0] if len(hits) == 1 else None


def _condition_column(context, dfs, dico, date_like=False, numeric=False):
    if numeric:
        resolved = _resolve_dico_column(context, dfs, dico)
        if resolved and _measure_allow(dfs, dico)(*resolved):
            return resolved
        return _resolve_dico_column(context, dfs, dico, allow=_measure_allow(dfs, dico))
    if date_like:
        tokens = _toks(context)
        candidates = []
        for name, df in dfs.items():
            for col in df.columns:
                if df[col].dtype.kind != "M":
                    continue
                desc = _table_descriptions(name, df.columns, dico).get(col, "")
                if tokens & _toks(desc):
                    candidates.append((name, col, tokens & _toks(desc)))
        semantic = [c for c in candidates if c[2] & {"ship", "birth", "deliver"}]
        if len(semantic) == 1:
            return semantic[0][0], semantic[0][1]
    resolved = _resolve_dico_column(context, dfs, dico)
    if resolved:
        return resolved
    if date_like:
        date_cols = []
        for name, df in dfs.items():
            for col in df.columns:
                if df[col].dtype.kind == "M":
                    date_cols.append((name, col))
        if len(date_cols) == 1:
            return date_cols[0]
    return None


def _coded_conditions(objective, dfs, dico):
    low = objective.lower()
    out = []
    for f, col, desc in _dico_rows(dico):
        name = stem(f).lower()
        if name not in dfs or col not in dfs[name].columns:
            continue
        match = re.search(r"coded\s*\((.*?)\)", desc, re.I)
        if not match:
            continue
        for item in match.group(1).split(","):
            parts = item.strip().split(" ", 1)
            if len(parts) != 2:
                continue
            code, label = parts[0], parts[1].strip()
            pos = low.find(label.lower())
            if pos >= 0:
                out.append((name, col, "eq", code, pos))
    return out


class _UnparsedCondition(Exception):
    """A comparison in the question could not be tied to a column: a deterministic answer would be wrong."""


_META_SENTENCE = re.compile(
    r"missing values|not recorded|written|separat|decimal|duplicat|appear twice|carry their unit|"
    r"keep one copy|count each|lower case|upper case|prefix|same separator|round|answer|report it|whole number|"
    r"integer|leave out|left out|ignore|dates are|format", re.I)


def _parse_table_conditions(objective, dfs, dico):
    out = _parse_table_conditions_raw(objective, dfs, dico)
    o = " ".join(objective.lower().split())
    parsed = {c[4] for c in out}
    date_re = (r"(on or after|on or before|after|before)\s+(?:(?:january|february|march|april|may|june|july|august|september|"
               r"october|november|december)\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:january|february|march|april|may|june|july|"
               r"august|september|october|november|december)\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}-\d{2}-\d{2})")
    comparison_re = (r"(greater than or equal to|less than or equal to|higher than or equal to|lower than or equal to|"
                     r"strictly greater than|strictly less than|strictly more than|strictly above|strictly below|"
                     r"no more than|no less than|no fewer than|not more than|not less than|greater than|less than|more than|"
                     r"fewer than|higher than|lower than|smaller than|larger than|bigger than|exceeding|over|under|"
                     r"at least|at most|not equal to|different from|above|below|>=|<=|≥|≤)\s*-?\d")
    pos = 0
    for sent in re.split(r"(?<=[.?!])\s+", o):
        start = o.find(sent, pos)
        pos = start + len(sent)
        if not re.search(r"\b(?:what|how many|how much|compute|calculate|consider|among|rank|which|group|count|for how many)\b", sent):
            continue                      # phrase de contexte (« Deliveries are free above 50 euros. »)
        if _META_SENTENCE.search(sent) and not re.search(r"\b(?:what|how many|compute|consider only|among|rank)\b", sent):
            continue
        spans = []
        for m in re.finditer(_VERB_DATE_RE + r"\s+in\s+\d{4}\b", sent):
            if start + m.start() not in parsed and not re.match(r"opened a", sent[m.start():]):
                raise _UnparsedCondition(m.group(0))
        for m in re.finditer(date_re, sent):
            spans.append(m.span())
            if start + m.start() not in parsed:
                raise _UnparsedCondition(m.group(0))
        for m in re.finditer(comparison_re, sent):
            if any(a <= m.start() < b for a, b in spans):
                continue
            if re.match(r"(?:at least|at most)\s+\d+\s+(?:defective|success)", sent[m.start():]):
                continue
            if start + m.start() not in parsed:
                raise _UnparsedCondition(m.group(0))
    return out


def _parse_table_conditions_raw(objective, dfs, dico):
    o = " ".join(objective.lower().split())
    out, date_spans = [], []
    date_re = r"(on or after|on or before|after|before)\s+((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}-\d{2}-\d{2})"
    for m in re.finditer(date_re, o, re.I):
        ctx = _condition_context(o[:m.start()])
        if re.search(r"\bborn\b", o[max(0, m.start()-35):m.start()]):
            ctx = "date of birth"
        vm = re.search(_VERB_DATE_RE + r"\s+$", o[:m.start()])
        col = _verb_date_column(vm.group(1), dfs, dico) if vm else None
        col = col or _condition_column(ctx, dfs, dico, date_like=True)
        if col:
            val = pd.to_datetime(m.group(2), dayfirst=bool(re.search(r"day/month|dd/mm|d/m/y|day-first|day first", o)), errors="coerce")
            if not pd.isna(val):
                op = {"on or after": "date_ge", "on or before": "date_le", "after": "date_gt", "before": "date_lt"}[m.group(1)]
                out.append((col[0], col[1], op, val, m.start()))
                date_spans.append(m.span())

    comparison_re = (r"(greater than or equal to|less than or equal to|higher than or equal to|lower than or equal to|"
                     r"strictly greater than|strictly less than|strictly more than|strictly above|strictly below|"
                     r"no more than|no less than|no fewer than|not more than|not less than|greater than|less than|more than|"
                     r"fewer than|higher than|lower than|smaller than|larger than|bigger than|exceeding|over|under|"
                     r"at least|at most|not equal to|different from|equal to|equals?|above|below|>=|<=|≥|≤|>|<)\s*(-?\d+(?:\.\d+)?)")
    _cmp_op = {"greater than or equal to": "ge", "higher than or equal to": "ge", "no less than": "ge", "no fewer than": "ge",
               "not less than": "ge", "at least": "ge", ">=": "ge", "≥": "ge",
               "less than or equal to": "le", "lower than or equal to": "le", "no more than": "le", "not more than": "le",
               "at most": "le", "<=": "le", "≤": "le",
               "strictly greater than": "gt", "strictly more than": "gt", "strictly above": "gt", "greater than": "gt",
               "more than": "gt", "higher than": "gt", "larger than": "gt", "bigger than": "gt", "exceeding": "gt",
               "over": "gt", "above": "gt", ">": "gt",
               "strictly less than": "lt", "strictly below": "lt", "less than": "lt", "fewer than": "lt", "lower than": "lt",
               "smaller than": "lt", "under": "lt", "below": "lt", "<": "lt",
               "not equal to": "ne", "different from": "ne", "equal": "eq", "equals": "eq", "equal to": "eq"}
    for m in re.finditer(comparison_re, o, re.I):
        if m.group(1) in ("over", "under") and not re.match(r"\s", o[m.end(1):m.end(1) + 1]):
            continue
        if any(a <= m.start() < b for a, b in date_spans):
            continue
        col = _ctx_column(o[:m.start()], dfs, dico, numeric=True)
        if not col:
            continue
        op = _cmp_op[m.group(1).lower()]
        out.append((col[0], col[1], op, float(m.group(2)), m.start()))

    # Explicit missingness tests on a named field.
    missing_re = r"\b(is not missing|is not blank|is not null|is missing|is blank|is null|is not recorded|is recorded)\b"
    for m in re.finditer(missing_re, o, re.I):
        col = _ctx_column(o[:m.start()], dfs, dico)
        if col:
            op = "not_missing" if re.search(r"not|recorded", m.group(1)) and "not recorded" not in m.group(1) else "is_missing"
            if m.group(1) in ("is recorded",):
                op = "not_missing"
            out.append((col[0], col[1], op, None, m.start()))

    # Closed numeric ranges such as "with a sale amount between 10 and 20".
    for m in re.finditer(r"\bbetween\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)", o):
        if o[max(0, m.start() - 1):m.start()] == "(" or re.search(r"\b(?:fraction|proportion|share)\s*\($", o[:m.start()]):
            continue                      # « what fraction (between 0 and 1) » n'est pas un filtre
        col = _ctx_column(o[:m.start()], dfs, dico)
        if col and dfs[col[0]][col[1]].dtype.kind in "fiub":
            out.append((col[0], col[1], "between", (float(m.group(1)), float(m.group(2))), m.start()))

    # A date's calendar year, as in "pupils born in 2015".
    for m in re.finditer(r"\bborn in\s+(\d{4})\b", o):
        ctx = "date of birth"
        col = _condition_column(ctx, dfs, dico, date_like=True)
        if col:
            out.append((col[0], col[1], "year_eq", int(m.group(1)), m.start()))
    # Same for other participles: « shipped in 2022 », « listed in 2021 », « admitted in 2023 ».
    for m in re.finditer(_VERB_DATE_RE + r"\s+in\s+(\d{4})\b", o):
        if m.group(1) == "born":
            continue
        col = _verb_date_column(m.group(1), dfs, dico)
        if col:
            out.append((col[0], col[1], "year_eq", int(m.group(2)), m.start()))
    # « with a shipment date in 2022 »
    for m in re.finditer(r"\bwith an? ([a-z ]{3,40}?date)\s+in\s+(\d{4})\b", o):
        col = _condition_column(m.group(1), dfs, dico, date_like=True)
        if col and dfs[col[0]][col[1]].dtype.kind == "M":
            out.append((col[0], col[1], "year_eq", int(m.group(2)), m.start()))

    # Exact quoted categories, including a field named after the quote: 'N' in the scholarship field.
    for m in re.finditer(r"(?<![a-z0-9])['\u2018\u201c]([^'\u2019\u201d]{1,80})['\u2019\u201d](?![a-z0-9])", o):
        before, after = o[:m.start()], o[m.end():]
        op_match = re.search(r"\b(is\s+not|is|does\s+not\s+equal|not\s+equal\s+to|different\s+from|equals?|equal\s+to)\s*$", before)
        if op_match and len(before) - op_match.start() <= 50:
            ctx = _condition_context(before[:op_match.start()])
            category_op = "ne" if re.search(r"not|different", op_match.group(1)) else "eq"
        else:
            fm = re.match(r"\s+in\s+(?:the\s+)?(.+?)\s+field\b", after)
            if not fm:
                fm = re.match(r"\s+as\s+(?:the\s+)?(.+?)(?=[?.;,]|$)", after)
                if not fm:
                    continue
            ctx = fm.group(1)
            category_op = "eq"
        value = m.group(1).strip()
        if value.lower() in NA_TOKENS or re.search(r"(?:separat|missing value|not recorded|not available)[^.;]{0,60}$", before):
            continue
        col = _condition_column(ctx, dfs, dico)
        if col:
            out.append((col[0], col[1], category_op, value, m.start()))

    out.extend(_coded_conditions(o, dfs, dico))

    # Resolve a single unquoted category only when its value and field context agree.
    # Ambiguous values or multiple category mentions are left for code generation.
    category_candidates = []
    quoted_spans = [m.span() for m in re.finditer(r"(?<![a-z0-9])['\u2018\u201c]([^'\u2019\u201d]{1,80})['\u2019\u201d](?![a-z0-9])", o)]
    for name, df in dfs.items():
        for col in df.columns:
            series = df[col]
            if series.dtype.kind not in "OUS" or series.nunique(dropna=True) > 50:
                continue
            for raw_value in series.dropna().astype(str).str.strip().unique():
                value = raw_value.strip()
                if len(value) < 3 or value.lower() in NA_TOKENS or re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value):
                    continue
                value_re = re.compile(rf"(?<![a-z0-9]){re.escape(value.lower())}(?![a-z0-9])")
                for vm in value_re.finditer(o):
                    if any(a <= vm.start() < b for a, b in quoted_spans):
                        continue
                    prefix = o[max(0, vm.start()-100):vm.start()]
                    operator = re.search(r"\b(is|equals?|in|from|at)\s+(?:(?:the|a|an)\s+)?$", prefix)
                    if not operator:
                        continue
                    ctx = _condition_context(prefix[:operator.start()])
                    resolved = _condition_column(ctx, dfs, dico)
                    if resolved == (name, col):
                        category_candidates.append((name, col, "eq", raw_value, vm.start()))
    unique_categories = {(c[0], c[1], str(c[3]).lower()) for c in category_candidates}
    if len(unique_categories) == 1:
        out.extend(category_candidates)

    # Month/year filters, often phrased as "delivery date in August 2023".
    months = {name.lower(): i for i, name in enumerate(("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"), 1)}
    for m in re.finditer(r"\bin\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b", o, re.I):
        ctx = _condition_context(o[:m.start()])
        vm = re.search(_VERB_DATE_RE + r"\s+$", o[:m.start()])
        col = _verb_date_column(vm.group(1), dfs, dico) if vm else None
        col = col or _condition_column(ctx, dfs, dico, date_like=True)
        if col:
            out.append((col[0], col[1], "month_eq", (months[m.group(1).lower()], int(m.group(2))), m.start()))
    return out


def _condition_mask(df, col, op, value):
    series = df[col]
    if op == "is_missing":
        return series.isna()
    if op == "not_missing":
        return series.notna()
    if op in ("eq", "ne") and pd.api.types.is_numeric_dtype(series):
        number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(number):
            return pd.Series(False, index=df.index) if op == "eq" else series.notna()
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric == number if op == "eq" else (series.notna() & (numeric != number))
    if op == "eq":
        return series.astype(str).str.strip().str.lower() == str(value).strip().lower()
    if op == "ne":
        return series.notna() & (series.astype(str).str.strip().str.lower() != str(value).strip().lower())
    if op in ("date_ge", "date_le", "date_gt", "date_lt"):
        dates = pd.to_datetime(series, errors="coerce")
        return {"date_ge": dates >= value, "date_le": dates <= value,
                "date_gt": dates > value, "date_lt": dates < value}[op]
    if op == "year_eq":
        if series.dtype.kind == "M":
            return series.dt.year == value
        return pd.to_numeric(series, errors="coerce") == value
    if op == "month_eq":
        if series.dtype.kind != "M":
            return pd.Series(False, index=df.index)
        month, year = value
        return (series.dt.month == month) & (series.dt.year == year)
    numeric = _numeric_metric(series, force=True)
    if numeric is None:
        numeric = pd.to_numeric(series, errors="coerce")
    if op == "between":
        low, high = value
        return numeric.between(low, high, inclusive="both")
    return {"gt": numeric > value, "ge": numeric >= value, "lt": numeric < value,
            "le": numeric <= value, "ne": numeric != value, "eq": numeric == value}[op]


def table_filter_rule(objective, dfs, dico):
    """Handle common single-table filters and filtered aggregates when their fields resolve cleanly."""
    if _w2_unsafe(objective):
        return None
    if len(dfs) != 1:
        return None
    o = " ".join(objective.lower().split())
    o = re.sub(r"\bhow many different\b", "how many distinct", o)
    if re.search(r"\brank\b|\bfor each\b", o):
        return None
    name, df = next(iter(dfs.items()))
    conditions = [c for c in _parse_table_conditions(o, dfs, dico) if c[0] == name]
    if not conditions:
        return None
    proportion = re.search(r"what (percentage|fraction)", o)
    if proportion and re.search(r"what (?:percentage|fraction|share|proportion) of the (?:total|sum|overall)\b", o):
        return None
    if proportion:
        marker = proportion.start()
        base = [c for c in conditions if c[4] < marker]
        target = [c for c in conditions if c[4] > marker]
        if not target:
            return None
        base_mask = pd.Series(True, index=df.index)
        for _, col, op, val, _ in base:
            base_mask &= _condition_mask(df, col, op, val).fillna(False)
        eligible = df[base_mask].copy()
        explicit = re.search(r"(?:count only|only the)\s+(?:pupils|students|rows|records|shipments)?\s*(?:whose|with)?\s*(.+?)\s+is recorded", o)
        if explicit:
            resolved = _resolve_dico_column(explicit.group(1), dfs, dico)
            if resolved:
                eligible = eligible[eligible[resolved[1]].notna()]
        elif re.search(r"\b(?:missing|not recorded|not available)\b", o):
            for col in dict.fromkeys(c[1] for c in target):
                eligible = eligible[eligible[col].notna()]
        if eligible.empty:
            return None
        numerator = pd.Series(True, index=eligible.index)
        for _, col, op, val, _ in target:
            numerator &= _condition_mask(eligible, col, op, val).fillna(False)
        ratio = float(numerator.sum()) / len(eligible)
        return ratio * 100 if proportion.group(1) == "percentage" else ratio

    count_request = re.search(
        r"(?:^|[.?!]\s+)(?:how many\b|what(?: is|'s) the (?:number|count) of\b|count the number of\b)|\bcount of\b", o)
    if count_request:
        mask = pd.Series(True, index=df.index)
        for _, col, op, val, _ in conditions:
            mask &= _condition_mask(df, col, op, val).fillna(False)
        if re.search(r"\b(?:distinct|unique)\b", o):
            if _has_unparsed_filter_intent(o, conditions):
                return None
            target = _distinct_target_column(o, dfs, dico, name)
            if not target:
                # Row-level uniqueness is supported only when the question says so explicitly.
                if not re.search(r"\b(?:distinct|unique)\s+(?:rows?|records?|entries|observations)\b", o):
                    return None
                target = _identifier_column(name, df.columns, dico)
            return float(df.loc[mask, target].nunique(dropna=True)) if target in df.columns else None
        return float(mask.sum())

    # Common scalar aggregates over a filtered subset.
    metric = re.search(
        r"\b(standard deviation|std\.?\s*dev(?:iation)?|variance|range|spread|"
        r"sum|total|average|mean|median|maximum|minimum|max|min)\b", o)
    if not metric:
        return None
    op = metric.group(1)
    op = {
        "total": "sum", "average": "mean", "maximum": "max", "minimum": "min",
        "std dev": "std", "std. dev": "std", "std deviation": "std",
        "std. deviation": "std", "standard deviation": "std",
    }.get(op, op)
    if op in ("std", "variance", "range", "spread") and _has_unparsed_filter_intent(o, conditions):
        return None
    tail = o[metric.end():].strip()
    tail = re.sub(r"^\([^)]*\)\s*", "", tail)
    tail = re.sub(r"^(?:of\s+)?the\s+", "", tail)
    # Stop at the table population or its filter; preserve field names such as "number of items".
    bounds = [m.start() for m in re.finditer(r"\bof the (?:orders?|shipments?|pupils?|students?|flats?|readings?|loans?|stays?|rows?|records?)\b|\bover all\b|\bamong\b|\bwith\b|\bwhose\b|\bon or after\b|\bon or before\b|\bat least\b|\bstrictly (?:greater|less) than\b", tail)]
    # A parsed categorical condition may be phrased as "for region 'North'" or "where status is 'open'".
    for _, _, _, _, pos in conditions:
        if pos > metric.start():
            between = o[metric.end():pos]
            cues = list(re.finditer(r"\b(?:for|where|with|among|whose)\b", between))
            if cues:
                bounds.append(cues[-1].start())
    phrase = tail[:min(bounds)] if bounds else tail
    phrase = phrase.strip(" ,.;:()")
    resolved = _resolve_dico_column(phrase, dfs, dico)
    if not resolved or resolved[0] != name:
        return None
    if op in ("std", "variance", "range", "spread") and df[resolved[1]].dtype.kind not in "fiub":
        return None
    mask = pd.Series(True, index=df.index)
    for _, col, cond_op, val, _ in conditions:
        mask &= _condition_mask(df, col, cond_op, val).fillna(False)
    descriptions = _table_descriptions(name, df.columns, dico)
    numeric_values = _numeric_metric(df[resolved[1]], descriptions.get(resolved[1], ""))
    if numeric_values is None:
        return None
    values = numeric_values.loc[mask].dropna()
    if values.empty:
        return None
    if op == "sum":
        return float(values.sum())
    if op == "mean":
        return float(values.mean())
    if op == "median":
        return float(values.median())
    if op == "max":
        return float(values.max())
    if op == "min":
        return float(values.min())
    if op in ("range", "spread"):
        return float(values.max() - values.min())
    if op in ("std", "variance"):
        ddof = 0 if re.search(r"\bpopulation\b", o) else 1
        if len(values) <= ddof:
            return None
        return float(values.std(ddof=ddof) if op == "std" else values.var(ddof=ddof))
    return None


def _mask_from(conditions, df, name):
    mask = pd.Series(True, index=df.index)
    for c_name, col, op, val, _ in conditions:
        if c_name == name:
            mask &= _condition_mask(df, col, op, val).fillna(False)
    return mask


def table_extra_rule(objective, dfs, dico):
    """Narrow templates absent from dev: share of a total, days between two dates, A-minus-B group gap,
    percentage of missing values, plus the wave-2 templates. Each returns None unless the whole question is understood."""
    try:
        w2 = table_wave2_rule(objective, dfs, dico)
    except Exception:
        w2 = None
    if w2 is not None and math.isfinite(w2):
        return w2
    if len(dfs) != 1:
        return None
    o = " ".join(objective.lower().split())
    name, df = next(iter(dfs.items()))
    descs = _table_descriptions(name, df.columns, dico)

    # share of a total: « What percentage of the total load weight comes from the shipments of the Rouen depot? »
    m = re.search(r"\bwhat (percentage|share|fraction|proportion) of the (?:total|overall|sum of the) (.+?)\s+"
                  r"(?:comes from|come from|is accounted for by|belongs to|is due to|is generated by|corresponds to|is made by|is in)\s+"
                  r"(?:the\s+)?(.+?)\?", o)
    if m:
        col = _resolve_dico_column(m.group(2), dfs, dico, allow=_measure_allow(dfs, dico))
        if not col or col[0] != name:
            return None
        conditions = [c for c in _parse_table_conditions(o, dfs, dico) if c[0] == name]
        if not conditions or any(c[4] < m.start(3) for c in conditions):
            return None
        values = _numeric_metric(df[col[1]], descs.get(col[1], ""))
        if values is None:
            return None
        total = values.dropna().sum()
        if not total:
            return None
        part = values[_mask_from(conditions, df, name)].dropna().sum()
        ratio = float(part / total)
        pct = m.group(1) == "percentage" or re.search(r"\bin percent\b|\(%\)|as a percentage", o)
        frac = m.group(1) == "fraction" or re.search(r"between 0 and 1|as a fraction|as a decimal", o)
        if not pct and not frac:
            return None                   # « share » / « proportion » sans unité : ambigu
        return ratio * 100 if pct else ratio

    # days between two dates: « Consider the delivery time (the number of days between the shipment date and the delivery date) »
    m = re.search(r"\bconsider (?:the )?(?:[a-z -]+?)\s*\(the number of (days|weeks) (?:between|from) (?:the )?(.+?) (?:and|to) (?:the )?(.+?)\)", o)
    if m:
        def date_col(phrase):
            cands = [(n, c) for n, d in dfs.items() for c in d.columns if d[c].dtype.kind == "M"]
            scored = []
            for n, c in cands:
                desc = _table_descriptions(n, dfs[n].columns, dico).get(c, "")
                scored.append((len(_toks(phrase) & _toks(desc + " " + c)), (n, c)))
            scored.sort(key=lambda x: -x[0])
            if scored and scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                return scored[0][1]
            vm = re.search(_VERB_DATE_RE, phrase)
            return _verb_date_column(vm.group(1), dfs, dico) if vm else None
        a, b = date_col(m.group(2)), date_col(m.group(3))
        if not a or not b or a == b:
            return None
        delta = (df[b[1]] - df[a[1]]).dt.days
        if m.group(1) == "weeks":
            delta = delta / 7.0
        rest = o[m.end():]
        q = re.search(r"\bwhat is (?:its|the)\s+" + _AGG_WORDS + r"\b", rest)
        if not q:
            return None
        conditions = [c for c in _parse_table_conditions(o, dfs, dico) if c[0] == name]
        if _has_unparsed_filter_intent(rest, conditions):
            return None
        return _apply_agg(delta[_mask_from(conditions, df, name)].astype(float), q.group(1), o)

    # gap between two coded groups: « What is the mean X of the stays of the cardiology department minus the mean X of … »
    m = re.search(r"\bwhat is the (mean|average|median|total) (.+?) of the [a-z]+ (of the .+?) minus the (mean|average|median|total) "
                  r"(.+?) of the [a-z]+ (of the .+?)(?:\s+\(in [^)]*\))?\?", o)
    if m:
        agg1, ph1, side1, agg2, ph2, side2 = m.groups()
        if agg1 != agg2 and {agg1, agg2} != {"mean", "average"}:
            return None
        col1 = _resolve_dico_column(ph1, dfs, dico, allow=_measure_allow(dfs, dico))
        col2 = _resolve_dico_column(ph2, dfs, dico, allow=_measure_allow(dfs, dico))
        if not col1 or col1 != col2 or col1[0] != name:
            return None
        c1 = [c for c in _coded_conditions(side1, dfs, dico) if c[0] == name]
        c2 = [c for c in _coded_conditions(side2, dfs, dico) if c[0] == name]
        if len({c[3] for c in c1}) != 1 or len({c[3] for c in c2}) != 1:
            return None
        values = _numeric_metric(df[col1[1]], descs.get(col1[1], ""))
        if values is None:
            return None
        how = {"average": "mean", "total": "sum"}.get(agg1, agg1)
        v1 = values[_mask_from(c1[:1], df, name)].dropna()
        v2 = values[_mask_from(c2[:1], df, name)].dropna()
        if v1.empty or v2.empty:
            return None
        return float(getattr(v1, how)() - getattr(v2, how)())

    # share of missing values: « What percentage of the flats have no recorded number of rooms? »
    m = re.search(r"\bwhat (percentage|fraction|proportion) of the [a-z]+ (?:have no recorded|have no|are missing (?:a |an |the )?|"
                  r"have a missing) (.+?)\?", o)
    if m:
        col = _resolve_dico_column(m.group(2), dfs, dico)
        if not col or col[0] != name:
            return None
        conditions = [c for c in _parse_table_conditions(o, dfs, dico) if c[0] == name]
        if conditions:
            return None
        ratio = float(df[col[1]].isna().mean())
        return ratio * 100 if m.group(1) == "percentage" else ratio
    return None


# --------------------------------------------------------------------------- tables : types absents de dev (vague 2)
# Règles étroites, testées sur des variantes générées ; chacune s'abstient (None) dès qu'un élément n'est pas compris.

_W2_DATE = (r"(?:(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}"
            r"|\d{1,2}\s+(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}"
            r"|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})")
_W2_GROUPS = r"(?:site|department|store|school|branch|depot|district|carrier|city|channel|tariff|segment|sector|class)"
_W2_WDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

# Constructions que les règles historiques lisent mal (réponse fausse au lieu d'une abstention).
_W2_UNSAFE = re.compile(
    r"interquartile|number of (?:distinct|different|unique) [a-z ]+ (?:for each|per|by)\b|\bnot of the\b|\boutside the\b|"
    r"\bother than\b|\bexcluding the\b|\bexcept\b|\bapart from\b|'[^']{1,40}' or '|\bof the [a-z ]+ or (?:the )?[a-z ]+ " + _W2_GROUPS +
    r"\b|\bweighted\b|\bweekends?\b|\bweekdays?\b|\bon an? (?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b|"
    r"\bquarter of \d{4}|\bq[1-4] \d{4}|\bin which (?:calendar )?month\b|\bpercentage change in\b|\bby how many percent\b|"
    r"\bhow many [a-z]+ have (?:more than|at least|fewer than|at most|less than|exactly) \d+ [a-z]+\?|"
    r"\bwith the (?:most|fewest) [a-z]+\b|\bon average, how many\b|\baverage number of [a-z]+ per\b|"
    r"\b(?:earliest|first) and the (?:latest|last)\b|\b(?:above|below|greater than|less than|higher than|lower than) the "
    r"(?:mean|average|median)\b|\bdivided by the total\b|\bratio of the total\b|\b\d+ [a-z]+ with the (?:largest|highest|smallest|lowest)\b|"
    r"\bmost (?:frequent|common)\b|\bbetween " + _W2_DATE + r" and\b", re.I)


def _w2_unsafe(objective):
    """Le veto ne regarde que les phrases qui posent la question (pas « The figures were exported on a Monday. »)."""
    o = " ".join(objective.lower().split())
    for sent in re.split(r"(?<=[.?!])\s+", o):
        if re.search(r"\b(?:what|how many|how much|compute|calculate|consider|among|rank|which|group|count|for how many|by how)\b", sent) \
                and _W2_UNSAFE.search(sent):
            return True
    return False


def _w2_measure(phrase, dfs, dico, name):
    col = _resolve_dico_column(phrase.strip(" ,.;:()"), dfs, dico, allow=_measure_allow(dfs, dico))
    if not col or col[0] != name:
        return None
    df = dfs[name]
    vals = _numeric_metric(df[col[1]], _table_descriptions(name, df.columns, dico).get(col[1], ""))
    return None if vals is None else (col[1], vals)


def _w2_date(phrase, dfs, dico, name):
    df = dfs[name]
    dates = [c for c in df.columns if df[c].dtype.kind == "M"]
    if not dates:
        return None
    phrase = (phrase or "").strip()
    vm = re.search(_VERB_DATE_RE, phrase)
    if vm:
        col = _verb_date_column(vm.group(1), dfs, dico)
        return col[1] if col and col[0] == name else None
    if phrase:
        descs = _table_descriptions(name, df.columns, dico)
        scored = sorted(((len(_toks(phrase) & _toks(descs.get(c, "") + " " + c)), c) for c in dates), reverse=True)
        if scored[0][0] > 0 and (len(scored) == 1 or scored[0][0] > scored[1][0]):
            return scored[0][1]
    return dates[0] if len(dates) == 1 else None


def _w2_group(phrase, dfs, dico, name):
    """Série de groupement : « year of the shipment date », « depot », « district »…"""
    df = dfs[name]
    phrase = phrase.strip(" ,.;:()")
    ym = re.match(r"(?:the )?year of (?:the )?(.+)$", phrase)
    if ym:
        dc = _w2_date(ym.group(1), dfs, dico, name)
        return df[dc].dt.year if dc else None
    mm = re.match(r"(?:the )?month of (?:the )?(.+)$", phrase)
    if mm:
        dc = _w2_date(mm.group(1), dfs, dico, name)
        return df[dc].dt.month if dc else None
    allow_cat = lambda n, c: dfs[n][c].dtype.kind not in "fM" and not _is_identity_desc(
        _table_descriptions(n, dfs[n].columns, dico).get(c, "")) or "coded" in _table_descriptions(n, dfs[n].columns, dico).get(c, "")
    col = _resolve_dico_column(phrase, dfs, dico, allow=allow_cat)
    if not col or col[0] != name:
        return None
    s = df[col[1]]
    return s if s.nunique(dropna=True) <= 60 else None


def _w2_labels(df, name, dico):
    """{libellé en minuscules: (colonne, valeur)} pour les colonnes codées et les petites colonnes texte."""
    out = {}
    for f, c, desc in _dico_rows(dico):
        if stem(f).lower() != name or c not in df.columns:
            continue
        m = re.search(r"coded\s*\((.*?)\)", desc, re.I)
        if m:
            for item in m.group(1).split(","):
                parts = item.strip().split(" ", 1)
                if len(parts) == 2:
                    out.setdefault(parts[1].strip().lower(), (c, parts[0]))
        elif df[c].dtype.kind == "O" and df[c].nunique(dropna=True) <= 30:
            for v in df[c].dropna().astype(str).str.strip().unique():
                if len(v) >= 3:
                    out.setdefault(v.lower(), (c, v))
    return out


def _w2_eq(df, col, val):
    return df[col].astype(str).str.strip().str.lower() == str(val).strip().lower()


def _w2_other_conditions(o, dfs, dico, name, drop_cols=(), drop_span=None):
    conds = [c for c in _parse_table_conditions(o, dfs, dico) if c[0] == name and c[1] not in drop_cols]
    if drop_span:
        conds = [c for c in conds if not (drop_span[0] <= c[4] < drop_span[1])]
    return conds


def _w2_is_measure_noun(word, dfs, dico, name):
    """« items » désigne-t-il une mesure (Number of items in the order) plutôt que des lignes ?"""
    w = _toks(word) - {"number", "count", "total"}
    if not w:
        return False
    allow = _measure_allow(dfs, dico)
    for c, desc in _table_descriptions(name, dfs[name].columns, dico).items():
        counted = re.match(r"\s*(?:number|count|quantity) of (\w+(?:[- ]\w+)?)", desc.lower())
        if allow(name, c) and counted and w & _toks(counted.group(1)):
            return True
    return False


def _w2_finish(o, df, name, dfs, dico, mask):
    """Termine une question sur une population : comptage, pourcentage ou agrégat d'une mesure."""
    q = re.search(r"\b(how many|what percentage|what fraction)\b", o)
    agg_q = re.search(r"\bwhat is the " + _AGG_WORDS + r" (?:of the )?(.+?) of the [a-z]+\b", o)
    if q and (not agg_q or q.start() < agg_q.start()):
        if q.group(1) == "how many":
            return float(mask.sum())
        frac = float(mask.mean()) if len(mask) else None
        return None if frac is None else (frac * 100 if q.group(1) == "what percentage" else frac)
    if not agg_q:
        return None
    meas = _w2_measure(agg_q.group(2), dfs, dico, name)
    if not meas:
        return None
    return _apply_agg(meas[1][mask], agg_q.group(1), o)


def _w2_outer(o, grouped):
    grouped = pd.to_numeric(grouped, errors="coerce").dropna()
    if grouped.empty:
        return None
    if re.search(r"difference between the largest and (?:the )?smallest", o):
        return float(grouped.max() - grouped.min())
    m = re.search(r"\bwhat is the (largest|highest|maximum|smallest|lowest|minimum|mean|average|median|sum|total) of these values\b", o)
    if not m:
        return None
    w = m.group(1)
    if w in ("largest", "highest", "maximum"):
        return float(grouped.max())
    if w in ("smallest", "lowest", "minimum"):
        return float(grouped.min())
    return float(getattr(grouped, {"average": "mean", "total": "sum"}.get(w, w))())


def table_wave2_rule(objective, dfs, dico):
    o = " ".join(objective.lower().split())
    if len(dfs) == 2:
        return _w2_join_distinct(o, dfs, dico)
    if len(dfs) != 1:
        return None
    name, df = next(iter(dfs.items()))

    # écart interquartile
    m = re.search(r"\binterquartile range (?:\([^)]*\) )?of (?:the )?(.+?)(?=\s+(?:over|among|for|across)\b|\s+of the [a-z]+ (?:with|whose|of)\b|\s*\(in|\?)", o)
    if m:
        meas = _w2_measure(m.group(1), dfs, dico, name)
        if not meas:
            return None
        mask = _mask_from(_w2_other_conditions(o, dfs, dico, name), df, name)
        s = meas[1][mask].dropna()
        return float(s.quantile(0.75) - s.quantile(0.25)) if len(s) else None

    # top-k
    m = re.search(r"\bwhat is the (total|sum|mean|average|median|minimum|maximum) (?:of the )?(.+?) of the (\d+) [a-z]+ with the "
                  r"(largest|highest|biggest|greatest|smallest|lowest) (.+?)(?:\s*\(in [^)]*\))?\?", o)
    m2 = re.search(r"\bwhat is the (sum|total|mean|average) of the (\d+) (largest|highest|biggest|smallest|lowest) (.+?) values\b", o)
    if m or m2:
        if m:
            word, ph_val, k, d, ph_rank = m.groups()
        else:
            word, k, d, ph_rank = m2.groups()
            ph_val = ph_rank
        val, rk = _w2_measure(ph_val, dfs, dico, name), _w2_measure(ph_rank, dfs, dico, name)
        if not val or not rk or _w2_other_conditions(o, dfs, dico, name):
            return None
        frame = pd.DataFrame({"v": val[1], "r": rk[1]}).dropna(subset=["r"])
        top = frame.sort_values("r", ascending=d in ("smallest", "lowest"), kind="mergesort").head(int(k))
        if len(top) < int(k):
            return None
        return _apply_agg(top["v"], word, o)

    # effectif du groupe le plus / le moins fréquent
    m = (re.search(r"\bhow many [a-z]+ (?:belong to|are in|are there in|does) the most (?:frequent|common) ([a-z ]+?)(?: have)?\?", o)
         or re.search(r"\bhow many [a-z]+ does the ([a-z ]+?) with the (most|fewest) [a-z]+ have\?", o))
    if m:
        key = _w2_group(m.group(1), dfs, dico, name)
        if key is None or _w2_other_conditions(o, dfs, dico, name):
            return None
        vc = key.value_counts()
        fewest = len(m.groups()) > 1 and m.group(2) == "fewest"
        return float(vc.min() if fewest else vc.max())

    # agrégat du groupe le plus fréquent
    m = re.search(r"\bwhat is the (total|sum|mean|average|median|maximum|minimum) (.+?) of the ([a-z ]+?) with the (most|fewest) [a-z]+(?:\s*\(in [^)]*\))?\?", o)
    if m:
        key = _w2_group(m.group(3), dfs, dico, name)
        meas = _w2_measure(m.group(2), dfs, dico, name)
        if key is None or not meas:
            return None
        vc = key.value_counts()
        best = vc.max() if m.group(4) == "most" else vc.min()
        if (vc == best).sum() != 1:
            return None
        g = vc[vc == best].index[0]
        return _apply_agg(meas[1][key == g], m.group(1), o)

    # nombre de groupes au-dessus d'un effectif
    m = re.search(r"\bhow many ([a-z]+?)s? have (more than|at least|fewer than|at most|less than|exactly) (\d+) ([a-z]+)\?", o)
    if m:
        key = _w2_group(m.group(1), dfs, dico, name)
        if key is None or _w2_is_measure_noun(m.group(4), dfs, dico, name):
            return None
        vc = key.value_counts(); x = int(m.group(3))
        hit = {"more than": vc > x, "at least": vc >= x, "fewer than": vc < x, "less than": vc < x, "at most": vc <= x,
               "exactly": vc == x}[m.group(2)]
        return float(hit.sum())

    # effectif moyen par groupe
    m = re.search(r"\bon average, how many ([a-z]+) does each ([a-z ]+?) have\?", o) or \
        re.search(r"\bwhat is the average number of ([a-z]+) per ([a-z ]+?)\?", o)
    if m:
        if _w2_is_measure_noun(m.group(1), dfs, dico, name):
            return None                      # « average number of items per order » : moyenne d'une mesure
        key = _w2_group(m.group(2), dfs, dico, name)
        if key is None or _w2_other_conditions(o, dfs, dico, name):
            return None
        k = key.dropna()
        return float(len(k) / k.nunique()) if k.nunique() else None

    # écart-type ou nombre de distincts par groupe
    m = re.search(r"\bcompute the (?:standard deviation|std)(?: \(([^)]*)\))? of (?:the )?(.+?) for each (.+?)\.", o)
    if m:
        pop = re.search(r"population|dividing by n\b(?!\s*-)", m.group(1) or o)
        samp = re.search(r"sample|n\s*-\s*1", m.group(1) or o)
        if bool(pop) == bool(samp):
            return None
        key, meas = _w2_group(m.group(3), dfs, dico, name), _w2_measure(m.group(2), dfs, dico, name)
        if key is None or not meas:
            return None
        mask = _mask_from(_w2_other_conditions(o, dfs, dico, name), df, name)
        g = meas[1][mask].groupby(key[mask]).std(ddof=0 if pop else 1)
        return _w2_outer(o, g)
    m = re.search(r"\bcompute the number of (?:distinct|different|unique) (.+?) (?:for each|per|by) (.+?)\.", o)
    if m:
        key = _w2_group(m.group(2), dfs, dico, name)
        target = _distinct_target_column("distinct " + m.group(1) + " are", dfs, dico, name)
        if key is None or not target:
            return None
        mask = _mask_from(_w2_other_conditions(o, dfs, dico, name), df, name)
        return _w2_outer(o, df.loc[mask, target].groupby(key[mask]).nunique())

    # étendue des dates
    m = re.search(r"\bhow many days (?:are there |separate |elapse |lie )?(?:between )?the (?:earliest|first) and the (?:latest|last) (.+?)\?", o)
    if m:
        dc = _w2_date(m.group(1), dfs, dico, name)
        if not dc or _w2_other_conditions(o, dfs, dico, name):
            return None
        d = df[dc].dropna()
        return float((d.max() - d.min()).days) if len(d) else None

    # mois de l'année le plus chargé
    m = re.search(r"\bin which (?:calendar )?month\b", o)
    if m:
        dm = re.search(r"\bmonth (?:\([^)]*\) )?of the (.+?) (?:is|are|was|were|has|have)\b", o)
        dc = _w2_date(dm.group(1) if dm else "", dfs, dico, name)
        if not dc or _w2_other_conditions(o, dfs, dico, name):
            return None
        months = df[dc].dt.month
        tm = re.search(r"\bthe (total|mean|average|median) (.+?) the (largest|highest|smallest|lowest)\b", o)
        if tm:
            meas = _w2_measure(tm.group(2), dfs, dico, name)
            if not meas:
                return None
            vals = meas[1].groupby(months).agg({"total": "sum", "average": "mean"}.get(tm.group(1), tm.group(1))).dropna()
            best = vals.max() if tm.group(3) in ("largest", "highest") else vals.min()
        else:
            cm = re.search(r"\bare there the (most|fewest)\b", o)
            if not cm:
                return None
            vals = months.dropna().value_counts()
            best = vals.max() if cm.group(1) == "most" else vals.min()
        hits = vals[vals == best]
        return float(hits.index[0]) if len(hits) == 1 else None

    # moyenne pondérée
    m = re.search(r"\b(?:mean|average) (?:of the )?(.+?) weighted by (?:the )?(.+?)(?=,|\?| over| for| among| \()", o)
    if m:
        x, w = _w2_measure(m.group(1), dfs, dico, name), _w2_measure(m.group(2), dfs, dico, name)
        if not x or not w:
            return None
        mask = _mask_from(_w2_other_conditions(o, dfs, dico, name), df, name)
        fr = pd.DataFrame({"x": x[1], "w": w[1]})[mask].dropna()
        return float((fr["x"] * fr["w"]).sum() / fr["w"].sum()) if fr["w"].sum() else None

    # rapport de deux sommes
    m = re.search(r"\b(?:what is the total (.+?) divided by the total (.+?)|ratio of the total (.+?) to the total (.+?))(?=,|\?|\.| over| computed| \()", o)
    if m:
        a = _w2_measure(m.group(1) or m.group(3), dfs, dico, name)
        b = _w2_measure(m.group(2) or m.group(4), dfs, dico, name)
        if not a or not b:
            return None
        mask = _mask_from(_w2_other_conditions(o, dfs, dico, name), df, name)
        fr = pd.DataFrame({"a": a[1], "b": b[1]})[mask]
        if re.search(r"both (?:are|values are) recorded|where both", o) or not fr.isna().any().any():
            fr = fr.dropna()
            return float(fr["a"].sum() / fr["b"].sum()) if fr["b"].sum() else None
        return None

    # évolution entre deux années
    m = re.search(r"\b(?:percentage change in|by how many percent did) the (total|number of|mean|average|median) (.+?) (?:change )?"
                  r"(?:from|between) (\d{4}) (?:to|and) (\d{4})", o)
    if m:
        ym = re.search(r"\(year of the (.+?)\)", o)
        dc = _w2_date(ym.group(1) if ym else "", dfs, dico, name)
        if not dc or _w2_other_conditions(o, dfs, dico, name):
            return None
        yr = df[dc].dt.year
        y1, y2 = int(m.group(3)), int(m.group(4))
        if m.group(1) == "number of":
            s1, s2 = float((yr == y1).sum()), float((yr == y2).sum())
        else:
            meas = _w2_measure(m.group(2), dfs, dico, name)
            if not meas:
                return None
            how = {"total": "sum", "average": "mean"}.get(m.group(1), m.group(1))
            s1 = getattr(meas[1][yr == y1].dropna(), how)()
            s2 = getattr(meas[1][yr == y2].dropna(), how)()
        return float((s2 - s1) / s1 * 100) if s1 else None

    # au-dessus / en dessous de la moyenne
    m = re.search(r"\bhave an? (.+?) (above|below|greater than|less than|higher than|lower than) the (mean|average|median) (.+?)\?", o)
    if m:
        a, b = _w2_measure(m.group(1), dfs, dico, name), _w2_measure(m.group(4), dfs, dico, name)
        if not a or not b or a[0] != b[0]:
            return None
        s = a[1].dropna()
        ref = s.median() if m.group(3) == "median" else s.mean()
        hit = (s > ref) if m.group(2) in ("above", "greater than", "higher than") else (s < ref)
        if re.search(r"\bwhat percentage\b", o):
            return float(hit.mean() * 100)
        if re.search(r"\bwhat fraction\b", o):
            return float(hit.mean())
        return float(hit.sum()) if re.search(r"\bhow many\b", o) else None

    labels = _w2_labels(df, name, dico)

    # négation d'un groupe codé : « not of the Lyon store », « outside the Grenoble lab site »
    m = re.search(r"\b(?:not of|outside|other than(?: those of)?|excluding|except(?: those of| for)?|apart from) the ([^\W\d_][\w' -]*?) "
                  + _W2_GROUPS + r"\b", o)
    if m:
        lab = labels.get(m.group(1).split(" of the ")[-1].strip())
        if not lab:
            return None
        col, val = lab
        mask = df[col].notna() & ~_w2_eq(df, col, val)
        mask &= _mask_from(_w2_other_conditions(o, dfs, dico, name, drop_cols=(col,)), df, name)
        return _w2_finish(o, df, name, dfs, dico, mask)

    # disjonction : « of the Lyon or Paris store », « whose carrier is 'A' or 'B' »
    m = re.search(r"\bof the ([^\W\d_][\w' -]*?) (?:" + _W2_GROUPS + r" )?or (?:the )?([^\W\d_][\w' -]*?) " + _W2_GROUPS + r"\b", o)
    mq = re.search(r"\bwhose ([a-z ]+?) is '([^']+)' or '([^']+)'", o)
    if m or mq:
        if m:
            l1 = labels.get(m.group(1).split(" of the ")[-1].strip())
            l2 = labels.get(m.group(2).strip())
            if not l1 or not l2 or l1[0] != l2[0]:
                return None
            col, vals = l1[0], [l1[1], l2[1]]
        else:
            colr = _resolve_dico_column(mq.group(1), dfs, dico)
            if not colr or colr[0] != name:
                return None
            col, vals = colr[1], [mq.group(2), mq.group(3)]
        mask = _w2_eq(df, col, vals[0]) | _w2_eq(df, col, vals[1])
        mask &= _mask_from(_w2_other_conditions(o, dfs, dico, name, drop_cols=(col,)), df, name)
        return _w2_finish(o, df, name, dfs, dico, mask)

    # jour de la semaine, trimestre, plage de dates
    verbs = r"(?:born|shipped|delivered|listed|placed|ordered|granted|admitted|taken|let|joined|signed)"
    m = re.search(r"\b(?:(" + verbs + r")|(?:have|with) an? ([a-z ]+?)(?: falling)?) on (?:a |the )?"
                  r"(weekends?|weekdays?|monday|tuesday|wednesday|thursday|friday|saturday|sunday)s?\b", o)
    if m:
        dc = _w2_date(m.group(1) or m.group(2), dfs, dico, name)
        if not dc:
            return None
        wd = df[dc].dt.weekday
        day = m.group(3).rstrip("s")
        mask = (wd >= 5) if day == "weekend" else ((wd < 5) if day == "weekday" else (wd == _W2_WDAYS.index(day)))
        mask = mask & df[dc].notna()
        mask &= _mask_from(_w2_other_conditions(o, dfs, dico, name), df, name)
        return _w2_finish(o, df, name, dfs, dico, mask)
    m = re.search(r"\b(?:(" + verbs + r")|with an? ([a-z ]+?)) in (?:q([1-4])|the (first|second|third|fourth) quarter of) (\d{4})\b", o)
    if m:
        dc = _w2_date(m.group(1) or m.group(2), dfs, dico, name)
        if not dc:
            return None
        qn = int(m.group(3)) if m.group(3) else ["first", "second", "third", "fourth"].index(m.group(4)) + 1
        mask = (df[dc].dt.year == int(m.group(5))) & (df[dc].dt.quarter == qn)
        mask &= _mask_from(_w2_other_conditions(o, dfs, dico, name), df, name)
        return _w2_finish(o, df, name, dfs, dico, mask)
    m = re.search(r"\b(?:(" + verbs + r")|(?:have|with) an? ([a-z ]+?)) between (" + _W2_DATE + r") and (" + _W2_DATE + r")", o)
    if m:
        dc = _w2_date(m.group(1) or m.group(2), dfs, dico, name)
        dayfirst = bool(re.search(r"day/month|dd/mm|d/m/y|day-first|day first", o))
        a = pd.to_datetime(m.group(3), dayfirst=dayfirst, errors="coerce")
        b = pd.to_datetime(m.group(4), dayfirst=dayfirst, errors="coerce")
        if not dc or pd.isna(a) or pd.isna(b):
            return None
        excl = re.search(r"both excluded|excluding both|exclusive", o)
        mask = ((df[dc] > a) & (df[dc] < b)) if excl else ((df[dc] >= a) & (df[dc] <= b))
        mask &= _mask_from(_w2_other_conditions(o, dfs, dico, name, drop_span=(m.start(), m.end())), df, name)
        return _w2_finish(o, df, name, dfs, dico, mask)

    # différence entre deux sous-populations : « mean X of the orders with 'web' as sales channel minus the mean X of … »
    m = re.search(r"\bwhat is the (mean|average|median|total) (.+?) of the [a-z]+ (.+?) minus the (mean|average|median|total) "
                  r"(.+?) of the [a-z]+ (.+?)(?:\s+\(in [^)]*\))?\?", o)
    if m:
        agg1, ph1, side1, agg2, ph2, side2 = m.groups()
        if {"average": "mean"}.get(agg1, agg1) != {"average": "mean"}.get(agg2, agg2):
            return None
        a, b = _w2_measure(ph1, dfs, dico, name), _w2_measure(ph2, dfs, dico, name)
        if not a or not b or a[0] != b[0]:
            return None
        sides = []
        for side in (side1, side2):
            c = [x for x in _parse_table_conditions_raw(side, dfs, dico) if x[0] == name]
            if not c:
                return None
            sides.append(_mask_from(c, df, name))
        v1, v2 = a[1][sides[0]].dropna(), a[1][sides[1]].dropna()
        if v1.empty or v2.empty:
            return None
        how = {"average": "mean", "total": "sum"}.get(agg1, agg1)
        return float(getattr(v1, how)() - getattr(v2, how)())
    return None


def _w2_join_distinct(o, dfs, dico):
    """« How many distinct customers whose customer segment is 'pro' placed at least one of the orders … ? »"""
    m = re.search(r"\bhow many (?:distinct|different) ([a-z]+) whose (.+?) is '([^']+)' (?:placed|made|have|has|had|are linked to|appear)", o)
    if not m:
        return None
    work = {k: v.copy() for k, v in dfs.items()}
    notes = align_keys(work, dico)
    key_note = next((n for n in notes if "Join key:" in n), None)
    if not key_note:
        return None
    km = re.search(r"dfs\['([^']+)'\]\['([^']+)'\] matches dfs\['([^']+)'\]\['([^']+)'\]", key_note)
    fact, fkey, dim, dkey = km.groups()
    col = _resolve_dico_column(m.group(2), work, dico)
    if not col or col[0] != dim:
        return None
    dmask = _w2_eq(work[dim], col[1], m.group(3))
    keys = set(work[dim].loc[dmask, dkey].dropna())
    conds = [c for c in _parse_table_conditions(o, work, dico) if c[0] == fact]
    fmask = _mask_from(conds, work[fact], fact)
    hits = work[fact].loc[fmask, fkey]
    return float(hits[hits.isin(keys)].nunique())


_UNIT_SYN = {
    "tonne": "t", "ton": "t", "t": "t", "metric ton": "t", "kg": "kg", "kilogram": "kg", "g": "g", "gram": "g",
    "kwh": "kwh", "kilowatt hour": "kwh", "kilowatt-hour": "kwh", "mwh": "mwh", "megawatt hour": "mwh", "wh": "wh",
    "mj": "mj", "megajoule": "mj", "gj": "gj", "gigajoule": "gj",
    "km": "km", "kilometre": "km", "kilometer": "km", "mile": "mile", "m": "m", "metre": "m", "meter": "m",
    "thousand euro": "keur", "k€": "keur", "keur": "keur", "euro": "eur", "€": "eur", "eur": "eur", "cent": "cent",
    "day": "day", "week": "week", "month": "month", "year": "year", "hour": "hour", "h": "hour", "minute": "minute",
    "min": "minute", "second": "second", "square metre": "m2", "square meter": "m2", "m²": "m2", "m2": "m2",
    "square foot": "ft2", "square feet": "ft2", "sq ft": "ft2", "foot": "ft", "feet": "ft", "percent": "pct", "%": "pct",
    "point": "pt", "pts": "pt", "half-day": "halfday", "people": "person", "person": "person", "litre": "l", "liter": "l",
    "cm": "cm", "centimetre": "cm", "centimeter": "cm", "mm": "mm", "millimetre": "mm", "millimeter": "mm",
    "inch": "in", "in": "in", "pound": "lb", "lb": "lb", "lbs": "lb", "ounce": "oz", "oz": "oz", "gallon": "gal", "gal": "gal",
    "ml": "ml", "millilitre": "ml", "milliliter": "ml", "yard": "yd", "yd": "yd", "ft": "ft", "mi": "mile",
    "kilometre per hour": "kmh", "km/h": "kmh", "mile per hour": "mph", "mph": "mph",
}


def _unit_norm(u):
    t = re.sub(r"\s+", " ", str(u).strip().lower().rstrip("."))
    if t in _UNIT_SYN:
        return _UNIT_SYN[t]
    if t.endswith("s") and t[:-1] in _UNIT_SYN:
        return _UNIT_SYN[t[:-1]]
    if t.endswith("es") and t[:-2] in _UNIT_SYN:
        return _UNIT_SYN[t[:-2]]
    if t in ("feet",):
        return "ft"
    t2 = re.sub(r"s\b", "", t)
    return _UNIT_SYN.get(t2, t2 or None)


_AGG_WORDS = (r"(standard deviation|variance|mean|average|median|maximum|minimum|largest value|smallest value|"
              r"highest value|lowest value|largest|smallest|total|sum|range)")


def _apply_agg(values, word, o):
    s = values.replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None
    w = {"average": "mean", "largest value": "max", "highest value": "max", "largest": "max", "maximum": "max",
         "smallest value": "min", "lowest value": "min", "smallest": "min", "minimum": "min", "total": "sum"}.get(word, word)
    if w in ("standard deviation", "variance"):
        if re.search(r"population|dividing by n\b(?!\s*-)", o):
            ddof = 0
        elif re.search(r"sample|n\s*-\s*1", o):
            ddof = 1
        else:
            return None
        if len(s) <= ddof:
            return None
        return float(s.std(ddof=ddof) if w == "standard deviation" else s.var(ddof=ddof))
    if w == "range":
        return float(s.max() - s.min())
    return float(getattr(s, w)())


def table_convert_rule(objective, dfs, dico):
    """« Consider the load weight in tonnes (1 tonne = 1000 kg). What is its median over the shipments … ? »"""
    if len(dfs) != 1:
        return None
    o = " ".join(objective.lower().split())
    m = re.search(r"\bconsider (?:the )?(.+?) in ([a-z€°²% -]+?)\s*\(\s*1\s+([a-z€°²% -]+?)\s*=\s*([\d.,]+)\s*([a-z€°²% -]+?)\s*\)", o)
    if not m:
        return None
    phrase, target, left, factor, right = m.groups()
    try:
        factor = float(factor.replace(",", "."))
    except ValueError:
        return None
    if not math.isfinite(factor) or factor <= 0:
        return None
    resolved = _resolve_dico_column(phrase, dfs, dico, allow=_measure_allow(dfs, dico))
    if not resolved:
        return None
    name, col = resolved
    src_unit = _unit_norm(_dictionary_unit(name, col, dico))
    tgt, lu, ru = _unit_norm(target), _unit_norm(left), _unit_norm(right)
    if src_unit == ru and tgt == lu:
        mult = 1.0 / factor
    elif src_unit == lu and tgt == ru:
        mult = factor
    else:
        return None
    rest = o[m.end():]
    q = re.search(r"\bwhat is (?:its|the)\s+" + _AGG_WORDS + r"\b", rest)
    if not q:
        return None
    df = dfs[name]
    conditions = [c for c in _parse_table_conditions(o, dfs, dico) if c[0] == name]
    if _has_unparsed_filter_intent(rest, conditions):
        return None
    mask = pd.Series(True, index=df.index)
    for _, ccol, op, val, _ in conditions:
        mask &= _condition_mask(df, ccol, op, val).fillna(False)
    values = _numeric_metric(df[col], _table_descriptions(name, df.columns, dico).get(col, ""))
    if values is None:
        return None
    return _apply_agg(values[mask] * mult, q.group(1), o)


def table_derive_rule(objective, dfs, dico):
    """Evaluate a named ratio (numerator divided by denominator) and aggregate the derived values."""
    if _w2_unsafe(objective):
        return None
    if len(dfs) != 1:
        return None
    o = " ".join(objective.lower().split())
    name, df = next(iter(dfs.items()))
    converted = table_convert_rule(objective, dfs, dico)
    if converted is not None:
        return converted
    expr = re.search(r"\((?:the\s+)?(.+?)\s+divided by\s+(?:the\s+)?(.+?)\)", o)
    if expr:
        num_phrase, den_phrase = expr.group(1), expr.group(2)
    else:
        expr = re.search(r"\bconsider the\s+(.+?)\s+per\s+(.+?)(?:\.|\(|;|,)", o)
        if not expr:
            return None
        num_phrase, den_phrase = expr.group(1), expr.group(2)
        if den_phrase in ("day of stay", "day spent", "day of the stay"):
            den_phrase = "length of stay"

    num_col = _resolve_dico_column(num_phrase, dfs, dico)
    den_col = _resolve_dico_column(den_phrase, dfs, dico)
    if not num_col or not den_col or num_col[0] != name or den_col[0] != name:
        return None
    request = re.search(r"\bwhat\s+(?:is|was)\b", o)
    agg_text = o[request.end():] if request else o
    agg = re.search(r"\b(median|minimum|maximum|mean|average|largest|smallest|min|max)\b", agg_text)
    if not agg:
        return None
    operation = {"minimum": "min", "largest": "max", "maximum": "max", "smallest": "min", "average": "mean"}.get(agg.group(1), agg.group(1))
    conditions = [c for c in _parse_table_conditions(o, dfs, dico) if c[0] == name]
    has_filter = bool(re.search(r"strictly greater than|strictly less than|at least|at most|on or after|on or before|\bafter\b|\bbefore\b|\bin [A-Z][a-z]+ \d{4}", objective))
    if has_filter and not conditions:
        return None
    mask = pd.Series(True, index=df.index)
    for _, col, op, val, _ in conditions:
        mask &= _condition_mask(df, col, op, val).fillna(False)
    descriptions = _table_descriptions(name, df.columns, dico)
    numerator = _numeric_metric(df[num_col[1]], descriptions.get(num_col[1], ""))
    denominator = _numeric_metric(df[den_col[1]], descriptions.get(den_col[1], ""))
    if numerator is None or denominator is None:
        return None
    ratio = safe_divide(numerator, denominator)
    if re.search(r"amount billed is in thousand euros", o) and re.search(r"\bin euros per day\b", o):
        ratio = ratio * 1000
    values = ratio[mask].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    if operation == "median":
        return float(values.median())
    if operation == "min":
        return float(values.min())
    if operation == "max":
        return float(values.max())
    if operation == "mean":
        return float(values.mean())
    return None


def table_group_rule(objective, dfs, dico):
    """Aggregate by one dictionary field, then summarize the resulting group values."""
    if _w2_unsafe(objective):
        return None
    if len(dfs) != 1:
        return None
    o = " ".join(objective.lower().split())
    name, df = next(iter(dfs.items()))
    ranking = re.search(r"rank the (.+?) values by the (mean|average|sum|total|median) (.+?)(?:,|\?|\.| from highest| from lowest)", o)
    row_count = False
    summary_operation = None
    if ranking:
        group_phrase, agg, metric_phrase = ranking.groups()
        agg = {"average": "mean", "total": "sum"}.get(agg, agg)
    else:
        nested = re.search(
            r"\b(mean|average|median|sum|total|maximum|minimum|max|min|largest|smallest)\s+of\s+(?:the\s+)?"
            r"(sum|total|mean|average|median|maximum|minimum|max|min)\s+(?:of\s+)?(.+?)\s+"
            r"(?:for each|for every|within each|in each|per|by)\s+(.+?)(?:\.|;|,|\?|\s+consider only\b)", o)
        if nested:
            summary_operation, agg, metric_phrase, group_phrase = nested.groups()
            summary_operation = {"average": "mean", "total": "sum", "maximum": "max",
                                 "minimum": "min", "largest": "max", "smallest": "min"}.get(
                                     summary_operation, summary_operation)
            agg = {"total": "sum", "average": "mean", "maximum": "max", "minimum": "min"}.get(agg, agg)
        else:
            grouped = re.search(
                r"\b(sum|total|mean|average|median|maximum|minimum|max|min)\s+(?:of\s+)?(.+?)\s+"
                r"(?:for each|for every|within each|in each|per|by)\s+(.+?)(?:\.|;|,|\?|\s+consider only\b)", o)
            if not grouped:
                counted = re.search(
                    r"\b(?:compute|calculate|find|determine)\s+(?:the\s+)?number of\s+(.+?)\s+"
                    r"(?:for each|for every|within each|in each|per|by)\s+(.+?)(?:\.|;|,|\?|\s+consider only\b)", o)
                if not counted:
                    return None
                metric_phrase, group_phrase = counted.groups()
                agg, row_count = "count", True
            else:
                agg, metric_phrase, group_phrase = grouped.groups()
                agg = {"total": "sum", "average": "mean", "maximum": "max", "minimum": "min"}.get(agg, agg)

    metric = None if row_count else _resolve_dico_column(metric_phrase, dfs, dico)
    group = _resolve_dico_column(group_phrase, dfs, dico)
    if not group:
        year_group = re.match(r"\s*year\s+of\s+(.+?)\s*", group_phrase)
        if year_group:
            candidate = _resolve_dico_column(year_group.group(1), dfs, dico)
            if candidate and candidate[0] == name:
                desc = _table_descriptions(name, df.columns, dico).get(candidate[1], "")
                if df[candidate[1]].dtype.kind == "M" or re.search(r"\bdate\b", desc, re.I):
                    group = candidate
            if not group:
                date_cols = [c for c in df.columns if df[c].dtype.kind == "M"]
                if len(date_cols) == 1:
                    group = (name, date_cols[0])
    if (not row_count and not metric) or not group or (metric and metric[0] != name) or group[0] != name or (metric and metric[1] == group[1]):
        return None
    o_cond = re.sub(r"\bhow many [a-z]+ have (?:a value|an? (?:mean|average|total|median))[^.?]*[.?]", " ", o)
    conditions = [c for c in _parse_table_conditions(o_cond, dfs, dico) if c[0] == name]
    rank_request = re.search(r"\bwhat is the rank of\b", o_cond)
    if rank_request:
        conditions = [c for c in conditions if c[4] < rank_request.start()]
    mask = pd.Series(True, index=df.index)
    for _, col, op, val, _ in conditions:
        mask &= _condition_mask(df, col, op, val).fillna(False)
    groups = df.loc[mask, [group[1]]].copy()
    key = groups[group[1]]
    if re.search(r"\byear\b", group_phrase):
        if key.dtype.kind == "M":
            key = key.dt.year
        else:
            numeric = pd.to_numeric(key, errors="coerce")
            if numeric.notna().any():
                key = numeric
    groups["__group"] = key
    if row_count:
        grouped_values = groups.dropna(subset=["__group"]).groupby("__group").size()
    else:
        descriptions = _table_descriptions(name, df.columns, dico)
        vals = _numeric_metric(df[metric[1]], descriptions.get(metric[1], ""))
        if vals is None:
            return None
        groups["__value"] = vals.loc[mask]
        grouped_values = groups.dropna(subset=["__group", "__value"]).groupby("__group")["__value"].agg(agg)
    if grouped_values.empty:
        return None

    if rank_request:
        target_m = re.search(r"\bwhat is the rank of\s+(?:the\s+)?(.+?)(?:\?|\.|$)", o)
        if not target_m:
            return None
        target_text = target_m.group(1).strip()
        target_text = re.sub(r"\s+(?:district|department|school|store|site|branch|depot|carrier|city|channel|tariff)$", "", target_text)
        target = None
        for c_name, c_col, c_op, c_val, c_pos in _coded_conditions(o, dfs, dico):
            if c_name == name and c_col == group[1] and c_pos >= rank_request.start():
                target = c_val
                break
        if target is None:
            target = target_text
        labels = grouped_values.index
        if pd.api.types.is_numeric_dtype(labels):
            try:
                target_value = float(target)
                target_value = int(target_value) if target_value.is_integer() else target_value
            except Exception:
                return None
        else:
            target_value = str(target).strip().lower()
            matching = [x for x in labels if str(x).strip().lower() == target_value]
            if not matching:
                return None
            target_value = matching[0]
        if target_value not in labels:
            return None
        descending = not re.search(r"\bfrom (?:the )?(?:lowest|smallest)\b", o)
        return float(grouped_values.rank(method="min", ascending=not descending).loc[target_value])

    if summary_operation:
        vals = pd.to_numeric(grouped_values, errors="coerce").dropna()
        if vals.empty:
            return None
        return float({"mean": vals.mean, "median": vals.median, "sum": vals.sum,
                      "max": vals.max, "min": vals.min}[summary_operation]())

    which = re.search(r"\bwhich [a-z]+ has the (largest|highest|greatest|smallest|lowest)\b", o)
    if which:
        if not pd.api.types.is_numeric_dtype(grouped_values.index):
            return None
        best = grouped_values.max() if which.group(1) in ("largest", "highest", "greatest") else grouped_values.min()
        hits = grouped_values[grouped_values == best]
        return float(hits.index[0]) if len(hits) == 1 else None
    n_groups = re.search(r"\bhow many [a-z]+ have (?:a value|an? (?:mean|average|total|median)[^.?]*?)\s+"
                         r"(of at least|at least|of at most|at most|strictly greater than|greater than|strictly less than|"
                         r"less than|above|below)\s+(-?\d+(?:\.\d+)?)", o)
    if n_groups:
        op, x = n_groups.group(1).replace("of ", ""), float(n_groups.group(2))
        f = {"at least": grouped_values >= x, "at most": grouped_values <= x, "strictly greater than": grouped_values > x,
             "greater than": grouped_values > x, "above": grouped_values > x, "strictly less than": grouped_values < x,
             "less than": grouped_values < x, "below": grouped_values < x}[op]
        return float(f.sum())
    of_these = re.search(r"\bwhat is the (mean|average|median|sum|total) of these values\b", o)
    if of_these:
        how = {"average": "mean", "total": "sum"}.get(of_these.group(1), of_these.group(1))
        return float(getattr(pd.to_numeric(grouped_values, errors="coerce").dropna(), how)())
    if re.search(r"difference between the largest and (?:the )?smallest", o):
        return float(grouped_values.max() - grouped_values.min())
    if re.search(r"\b(?:smallest|lowest|minimum) of these values\b", o):
        return float(grouped_values.min())
    if re.search(r"\b(?:largest|highest|maximum) of these values\b", o):
        return float(grouped_values.max())
    if re.search(r"\b(?:range|spread)\b.{0,30}\b(?:these|the)?\s*(?:group )?values\b", o):
        return float(grouped_values.max() - grouped_values.min())
    return None


def table_join_rule(objective, dfs, dico):
    """Join one fact table to one lookup table, filter, and aggregate a value from the fact table."""
    if _w2_unsafe(objective):
        return None
    if len(dfs) != 2:
        return None
    o = " ".join(objective.lower().split())
    group_request = re.search(
        r"\bgroup\s+(?:the\s+)?(.+?)\s+by\s+(?:the\s+)?(.+?)\s+and\s+compute\s+(?:the\s+)?"
        r"(mean|average|median|sum|total|maximum|minimum|max|min)\s+(.+?)\s+per group\b", o)
    if not group_request and not re.search(r"\bbelong to\b|\bwhose\b", o):
        return None
    notes = align_keys(dfs, dico)
    key_note = next((n for n in notes if "Join key:" in n), None)
    if not key_note:
        return None
    m = re.search(r"dfs\['([^']+)'\]\['([^']+)'\] matches dfs\['([^']+)'\]\['([^']+)'\]", key_note)
    if not m:
        return None
    fact, fact_key, dim, dim_key = m.groups()
    if fact == dim:
        return None
    conditions = _parse_table_conditions(o, dfs, dico)
    frames = {name: df.copy() for name, df in dfs.items()}
    for name, col, op, val, _ in conditions:
        if name in frames:
            mask = _condition_mask(frames[name], col, op, val).fillna(False)
            frames[name] = frames[name][mask]
    if frames[fact].empty or frames[dim].empty:
        return None

    if group_request:
        clean_objective = re.sub(
            r"\b[\w.-]+\.csv contains duplicated rows\b[^.;?]*(?:[.;?]|$)",
            " ", o)
        if _has_unparsed_filter_intent(clean_objective, conditions):
            return None
        _, group_phrase, agg, metric_phrase = group_request.groups()
        group_phrase = re.sub(r"\s+of\s+(?:their|its|the)\s+[a-z]+$", "", group_phrase.strip())
        group_col = _resolve_dico_column(group_phrase, dfs, dico)
        metric_col = _resolve_dico_column(metric_phrase, dfs, dico)
        if not group_col or group_col[0] != dim or not metric_col or metric_col[0] != fact:
            return None
        outer = re.search(r"\b(largest|highest|maximum|max|smallest|lowest|minimum|min)\s+value\b", o)
        if not outer:
            return None
        left = frames[fact].copy()
        left["__join"] = left[fact_key]
        descriptions = _table_descriptions(fact, frames[fact].columns, dico)
        metric_values = _numeric_metric(left[metric_col[1]], descriptions.get(metric_col[1], ""))
        if metric_values is None:
            return None
        left["__metric"] = metric_values
        right = frames[dim][[dim_key, group_col[1]]].copy()
        right = right.rename(columns={dim_key: "__join", group_col[1]: "__group"})
        joined = left.merge(right, on="__join", how="inner")
        values = joined.dropna(subset=["__group", "__metric"])
        if values.empty:
            return None
        agg_name = {"average": "mean", "total": "sum", "maximum": "max", "minimum": "min"}.get(agg, agg)
        grouped_values = values.groupby("__group")["__metric"].agg(agg_name)
        if grouped_values.empty:
            return None
        return float(grouped_values.max() if outer.group(1) in ("largest", "highest", "maximum", "max")
                     else grouped_values.min())

    if re.search(r"\bhow many orders\b", o):
        order_col = _resolve_dico_column("order number", dfs, dico)
        if not order_col or order_col[0] != fact:
            return None
        left = frames[fact].rename(columns={fact_key: "__join", order_col[1]: "__order"})
        right = frames[dim].rename(columns={dim_key: "__join"})
        joined = left.merge(right, on="__join", how="inner")
        return float(joined["__order"].nunique()) if not joined.empty else 0.0

    count_request = re.search(r"\bhow many\s+(.+?)\s+(?:belong to|whose)\b", o)
    if count_request:
        entity_tokens = _toks(count_request.group(1))
        id_col = _identifier_column(fact, frames[fact].columns, dico)
        id_desc = _table_descriptions(fact, frames[fact].columns, dico).get(id_col, "") if id_col else ""
        if not id_col:
            return None
        if not (entity_tokens & (_toks(fact) | _toks(id_desc))):
            # « How many flats belong to … » with listings.csv : the counted entity must not be the lookup table's.
            if entity_tokens & (_toks(dim) | _toks(" ".join(_table_descriptions(dim, frames[dim].columns, dico).values()))):
                return None
        left = frames[fact].rename(columns={fact_key: "__join", id_col: "__entity"})
        right = frames[dim].rename(columns={dim_key: "__join"})
        joined = left.merge(right, on="__join", how="inner")
        return float(joined["__entity"].nunique(dropna=True)) if not joined.empty else 0.0

    op_m = re.search(r"\b(mean|average|median|maximum|minimum|max|min|sum|total)\b", o)
    if not op_m:
        return None
    operation = {"average": "mean", "maximum": "max", "minimum": "min", "total": "sum"}.get(op_m.group(1), op_m.group(1))
    tail = o[op_m.end():].strip()
    row_phrase = re.search(r"\bof the (?:orders?|order_lines?|pupils?|students?|flats?|readings?|shipments?|loans?|stays?)\b", tail)
    metric_phrase = tail[:row_phrase.start()] if row_phrase else tail
    metric_phrase = metric_phrase.strip(" ,.;:()")
    metric = _resolve_dico_column(metric_phrase, dfs, dico)
    if not metric or metric[0] != fact:
        # Try the uncut tail when a description uses "of" as part of its field name.
        metric = _resolve_dico_column(tail, dfs, dico)
    if not metric or metric[0] != fact:
        return None
    left = frames[fact].rename(columns={fact_key: "__join", metric[1]: "__metric"})
    right = frames[dim].rename(columns={dim_key: "__join"})
    joined = left.merge(right, on="__join", how="inner")
    descriptions = _table_descriptions(fact, frames[fact].columns, dico)
    values = _numeric_metric(joined["__metric"], descriptions.get(metric[1], ""))
    if values is None:
        return None
    values = values.dropna()
    if values.empty:
        return None
    if operation == "mean":
        return float(values.mean())
    if operation == "median":
        return float(values.median())
    if operation == "max":
        return float(values.max())
    if operation == "min":
        return float(values.min())
    if operation == "sum":
        return float(values.sum())
    return None


def _row_filter(df, col, year, how):
    return df[df[col] >= year] if how == "later" else df[df[col] <= year]


def fit_rules(objective, dfs, dico):
    """Renvoie un nombre ou None (None = pas sûr : on laisse le modèle faire)."""
    import itertools
    o = " ".join(objective.split())
    if "train" not in dfs or "test" not in dfs:
        return None
    tr, te = dfs["train"], dfs["test"]
    kind = "logistic" if re.search(r"logistic regression", o, re.I) else (
        "linear" if re.search(r"least[- ]squares|\bOLS\b|linear regression", o, re.I) else None)
    if kind is None:
        return None
    desc = {}
    for f, c, d in _dico_rows(dico):
        if f.lower().startswith("train"):
            desc[c] = d
    target = [c for c in tr.columns if c not in te.columns]
    if len(target) != 1:
        return None
    target = target[0]
    idc = te.columns[0]
    # colonne exclue (description entre guillemets)
    excl = set()
    for q in re.findall(r'"([^"]+)"', o):
        excl |= {c for c, d in desc.items() if d.lower() == q.strip().lower()}
    # colonne année
    ycols = [c for c in tr.columns if "year" in desc.get(c, "").lower() and tr[c].dtype.kind in "fi"]
    # phrase des variables
    if kind == "linear":
        m = re.search(r"intercept,? of the (.+?) on the (.+?), using", o)
    else:
        m = re.search(r"intercept, of (.+?) \(.*?\) on the (.+?), using", o)
    if not m:
        return None
    items = [x.strip() for x in re.split(r",\s*|\s+and\s+", m.group(2)) if x.strip()]
    cands = [c for c in tr.columns if c in te.columns and c != idc and c not in excl and c != target
             and c not in ycols and (tr[c].dtype.kind in "fiub" or
                                    (tr[c].dtype.kind in "OUS" and tr[c].nunique(dropna=True) <= 20))]
    if not items or len(items) > len(cands):
        return None
    best, best_s = None, -1
    for perm in itertools.permutations(range(len(cands)), len(items)):
        sc = sum(len(_toks(items[i]) & _toks(desc.get(cands[p], ""))) for i, p in enumerate(perm))
        if sc > best_s:
            best, best_s = perm, sc
    # sûreté : si on doit choisir un sous-ensemble, chaque variable doit avoir au moins un mot en commun
    if len(items) < len(cands):
        if any(not (_toks(items[i]) & _toks(desc.get(cands[p], ""))) for i, p in enumerate(best)):
            return None
    feats = [cands[p] for p in best]
    # filtre d'années sur l'apprentissage
    d = tr.copy()
    mf = re.search(r"train\.csv with an? (.+?) of (\d{4}) or (earlier|later)", o)
    if mf:
        if len(ycols) != 1:
            return None
        d = _row_filter(d, ycols[0], int(mf.group(2)), mf.group(3))
    d = d.dropna(subset=feats + [target])
    if len(d) < len(feats) + 2:
        return None
    y = d[target]
    if kind == "logistic":
        if y.dtype.kind not in "fi":
            y = y.astype(str).str.strip().str.lower().isin(["y", "yes", "1", "true"]).astype(float)
        y = y.astype(float)
    LR, LogR = _make_models(True)
    model = (LR() if kind == "linear" else LogR()).fit(d[feats], y)

    def predict(frame):
        X = frame[feats]
        return model.predict(X) if kind == "linear" else model.predict_proba(X)[:, 1]

    # ---- la question
    mq = re.search(r"predicted [a-z ]+? for the \w+ (\d+) of test\.csv", o)
    if mq:
        row = te[te[idc] == float(mq.group(1))]
        if len(row) != 1:
            return None
        return float(predict(row)[0])
    if re.search(r"intercept of the fitted model", o):
        return float(np.ravel(model.intercept_)[0])
    if re.search(r"R². of the fit|R2 of the fit|R² of the fit", o) and kind == "linear":
        return float(model.score(d[feats], y))
    def feature_of(item):
        item = item.strip().lower()
        named = [i for i, it in enumerate(items) if it.lower() == item]
        if len(named) == 1:
            return feats[named[0]]
        sc = [(len(_toks(item) & _toks(desc.get(f, ""))), f) for f in feats]
        sc.sort(reverse=True)
        if not sc or sc[0][0] == 0 or (len(sc) > 1 and sc[0][0] == sc[1][0]):
            return None
        return sc[0][1]

    mc = re.search(r"(?:coefficient|slope) (?:of|for) the (.+?)(?: \(|\?)", o)
    odds = re.search(r"odds ratio (?:of|for|associated with) the (.+?)(?: \(|\?)", o)
    if mc or (odds and kind == "logistic"):
        f = feature_of((mc or odds).group(1))
        if f is None:
            return None
        coef_idx = getattr(model, "_dsh_numeric_coef_idx", {}).get(f)
        if coef_idx is None:
            return None  # a categorical field has several one-hot coefficients, not one scalar slope
        coef = float(np.ravel(model.coef_)[coef_idx])
        return math.exp(coef) if (odds and not mc) else coef
    mn = re.search(r"How many [a-z ]+ of test\.csv have an? predicted [a-z ]+? (?:of )?(greater than or equal to|less than or equal to|"
                   r"at least|at most|above|below|more than|less than|"
                   r"strictly greater than|strictly less than|greater than|no more than|no less than) (-?[\d.]+)", o)
    if mn:
        t = te.dropna(subset=feats)
        p, x = predict(t), float(mn.group(2))
        op = mn.group(1)
        hit = {"at least": p >= x, "no less than": p >= x, "greater than or equal to": p >= x, "less than or equal to": p <= x, "at most": p <= x, "no more than": p <= x, "above": p > x,
               "more than": p > x, "strictly greater than": p > x, "greater than": p > x, "below": p < x, "less than": p < x,
               "strictly less than": p < x}[op]
        return float(np.sum(hit))
    mm = re.search(r"mean predicted [a-z ]+ over (?:the|all the|all) \w+ of test\.csv (?:with an? (.+?) of (\d{4}) or (earlier|later)|whose (.+?) is '([^']+)')", o)
    if not mm and re.search(r"mean predicted [a-z ]+? over (?:the|all the|all) \w+ of test\.csv\s*(?:\(|\?|\.)", o):
        t = te.dropna(subset=feats)
        return float(np.mean(predict(t))) if len(t) else None
    if mm:
        t = te.dropna(subset=feats)
        if mm.group(2):
            if len(ycols) != 1:
                return None
            t = _row_filter(t, ycols[0], int(mm.group(2)), mm.group(3))
        else:
            key = mm.group(4)
            cols = [c for c in te.columns if key.lower() in desc.get(c, "").lower() and te[c].dtype.kind not in "fiM"]
            if len(cols) != 1:
                return None
            t = t[t[cols[0]].astype(str).str.strip().str.lower() == mm.group(5).strip().lower()]
        if len(t) == 0:
            return None
        return float(predict(t).mean())
    return None



# ===========================================================================
# Version précédente (88 % sur les tables), recopiée telle quelle avec le préfixe _v1_ sur ses noms.
# Utilisée pour les tâches « table » quand TABLE_MODE vaut "v1" ou "v1_then_v2".
# ===========================================================================
_v1_MODEL = os.environ.get('DSH_MODEL', 'Qwen/Qwen2.5-Coder-1.5B-Instruct')
_v1_BUDGET = float(os.environ.get('DSH_BUDGET', 29.0))
_v1_DEBUG = bool(os.environ.get('DSH_DEBUG'))
_v1_EXEC_TIMEOUT = 1.5
_v1_MAX_NEW = 500
_v1_NSAMP = int(os.environ.get('DSH_SAMPLES', 4))
_v1_NA_TOKENS = {'', 'na', 'n/a', 'nan', 'null', 'none', '-', '--', '?', 'missing', '.', 'n.a.', '#n/a', 'unknown'}

def _v1__num_from_text(x):
    """'1 234,5 kg' -> 1234.5 ; '1,234.5' -> 1234.5 ; '3,14' -> 3.14."""
    if not isinstance(x, str):
        return np.nan
    m = re.search('[-+]?\\d[\\d.,\\s]*', x.replace('\u202f', ' ').replace('\xa0', ' '))
    if not m:
        return np.nan
    t = m.group(0).strip().replace(' ', '')
    if ',' in t and '.' in t:
        dec = ',' if t.rfind(',') > t.rfind('.') else '.'
        thou = '.' if dec == ',' else ','
        t = t.replace(thou, '').replace(dec, '.')
    elif ',' in t:
        if re.fullmatch('[-+]?\\d{1,3}(,\\d{3})+', t):
            t = t.replace(',', '')
        else:
            t = t.replace(',', '.')
    try:
        sign = -1.0 if x.strip().startswith('-') else 1.0
        return sign * abs(float(t))
    except ValueError:
        return np.nan

def _v1_clean_numeric(values):
    """Coerce mixed numeric strings, decimal commas, and simple units; unparseable entries become NaN."""

    def parse(value):
        if pd.isna(value):
            return np.nan
        if isinstance(value, (int, float, np.number)):
            number = float(value)
            return number if math.isfinite(number) else np.nan
        return _v1__num_from_text(str(value))
    if isinstance(values, pd.Series):
        return values.map(parse).astype(float)
    if np.isscalar(values):
        return parse(values)
    return pd.Series(values).map(parse).astype(float)

def _v1_clean_dates(values, dayfirst=False):
    """Parse mixed date strings while refusing to interpret arbitrary numbers as epoch timestamps."""

    def parse(value):
        if pd.isna(value):
            return pd.NaT
        if isinstance(value, (datetime.datetime, datetime.date, pd.Timestamp, np.datetime64)):
            return pd.to_datetime(value, errors='coerce')
        text = str(value).strip()
        compact = re.fullmatch('(\\d{8})(?:\\.0+)?', text)
        if compact:
            return pd.to_datetime(compact.group(1), format='%Y%m%d', errors='coerce')
        if re.fullmatch('[-+]?\\d+(?:\\.\\d+)?', text):
            return pd.NaT
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            return pd.to_datetime(text, dayfirst=dayfirst, errors='coerce')
    if isinstance(values, pd.Series):
        return values.map(parse)
    if np.isscalar(values):
        return parse(values)
    return pd.Series(values).map(parse)

def _v1__numeric_metric(values, description='', force=False):
    """Return a finite numeric view for a measure, or None when text looks categorical/date-like.

    Numeric columns keep the existing path. Object columns are only coerced when their values
    look numeric and either nearly all parse or the data dictionary identifies a measurement.
    """
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan)
    nonmissing = values.notna()
    if not nonmissing.any():
        return pd.Series(np.nan, index=values.index, dtype=float)
    desc = str(description).lower()
    measure_hint = bool(re.search('\\b(amount|price|cost|income|sales|revenue|quantity|rate|ratio|percent(?:age)?|age|height|weight|temperature|surface|area|distance|duration|count|number of|days?|hours?|occupancy|population|volume|energy|score|billed|total|value|mass|length|width|depth|probability|salary|wage|speed|pressure|capacity)\\b', desc))
    categorical_hint = bool(re.search('\\b(code|coded|identifier|key|reference|category|categorical|status|label)\\b', desc))
    if categorical_hint and (not measure_hint):
        return None
    if not measure_hint and (not force):
        return None
    raw = values.loc[nonmissing]
    if pd.api.types.is_datetime64_any_dtype(values):
        return None
    date_objects = raw.map(lambda x: isinstance(x, (datetime.date, datetime.datetime, pd.Timestamp, np.datetime64)))
    texts = raw.astype(str).str.strip()
    date_text = texts.map(lambda x: bool(_v1__DATE_RE.fullmatch(x) or re.fullmatch('\\d{8}(?:\\.0+)?', x)))
    if date_objects.mean() >= 0.5 or date_text.mean() >= 0.5:
        return None
    numeric_shape = texts.map(lambda x: bool(_v1__NUM_RE.fullmatch(x) or re.fullmatch('[$€£]\\s*' + _v1__NUM_RE.pattern[1:-1], x)))
    units = []
    for text in texts[numeric_shape]:
        currency = re.match('^\\s*([$€£])', text)
        core = text[currency.end():] if currency else text
        match = re.match('^[-+]?\\s*\\d[\\d.,\\s]*\\s*(.*)$', core)
        units.append((currency.group(1) if currency else '') + (match.group(1).strip().casefold() if match else ''))
    if len(set(units)) > 1:
        return None
    parsed = _v1_clean_numeric(values)
    parse_rate = parsed.loc[nonmissing].notna().mean()
    shape_rate = numeric_shape.mean()
    threshold = 0.7 if measure_hint or force else 0.95
    if parse_rate < threshold or shape_rate < threshold:
        return None
    return parsed.replace([np.inf, -np.inf], np.nan)

def _v1_safe_divide(numerator, denominator):
    """Divide while mapping zero denominators and non-finite outputs to NaN."""
    if isinstance(numerator, pd.Series) or isinstance(denominator, pd.Series):
        if isinstance(numerator, pd.Series):
            n = _v1_clean_numeric(numerator)
            if isinstance(denominator, pd.Series):
                d = _v1_clean_numeric(denominator).reindex(n.index)
            elif np.isscalar(denominator):
                d = denominator
            else:
                d = _v1_clean_numeric(pd.Series(denominator, index=n.index))
            result = n.div(d)
        else:
            d = _v1_clean_numeric(denominator)
            n = _v1_clean_numeric(pd.Series(numerator, index=d.index))
            result = n.div(d)
        if isinstance(result, pd.Series):
            return result.replace([np.inf, -np.inf], np.nan).where(d != 0)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        result = np.divide(np.asarray(numerator, dtype=float), np.asarray(denominator, dtype=float))
    result = np.where(np.isfinite(result), result, np.nan)
    return float(result) if np.ndim(result) == 0 else result
_v1__NUM_RE = re.compile('^[-+]?\\s*\\d[\\d.,\\s]*\\s*[A-Za-z%$€£°/²³µ.]*$')
_v1__DATE_RE = re.compile('^\\d{1,4}[-/.]\\d{1,2}[-/.]\\d{1,4}([ T]\\d{1,2}:\\d{2}(:\\d{2})?)?$')

def _v1__convert(s, dayfirst):
    nn = s.dropna()
    if nn.empty:
        return s
    if nn.str.match(_v1__NUM_RE).mean() >= 0.9:
        suf = nn.str.extract('^[-+]?\\s*\\d[\\d.,\\s]*\\s*(.*)$')[0].dropna().astype(str).str.strip()
        if len(set(suf) - {''}) <= 1:
            return s.map(_v1__num_from_text).astype(float)
        return s
    if nn.str.match(_v1__DATE_RE).mean() >= 0.9:
        parts = nn.str.extract('^(\\d{1,2})[-/.](\\d{1,2})[-/.]\\d{4}')
        if parts[0].notna().mean() >= 0.9:
            if pd.to_numeric(parts[0]).max() > 12:
                dayfirst = True
            elif pd.to_numeric(parts[1]).max() > 12:
                dayfirst = False
        return pd.to_datetime(s, dayfirst=dayfirst, errors='coerce')
    return s

class _v1_Tables(dict):
    """dfs : {nom de fichier : DataFrame}, avec le dictionnaire de données (.dictionary) et {colonne : description} (.desc)."""
    dictionary = None
    desc = {}

def _v1_read_table(path, dayfirst=False, convert=True, na_extra=()):
    raw = None
    for enc in ('utf-8-sig', 'latin-1'):
        try:
            with open(path, 'rb') as fh:
                raw = fh.read().decode(enc)
            break
        except Exception:
            continue
    head = raw[:5000]
    try:
        sep = csv.Sniffer().sniff(head, delimiters=',;\t|').delimiter
    except Exception:
        sep = max(',;\t|', key=head.count)
    df = pd.read_csv(io.StringIO(raw), sep=sep, dtype=str, keep_default_na=False, skipinitialspace=True, on_bad_lines='skip', engine='python')
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        s = df[c].astype(str).str.strip()
        s = s.where(~s.str.lower().isin(_v1_NA_TOKENS | set(na_extra)), np.nan)
        df[c] = _v1__convert(s, dayfirst) if convert else s
    return df

def _v1_stem(path):
    return os.path.splitext(os.path.basename(path))[0]

def _v1_route(files):
    names = {os.path.basename(f).lower() for f in files or []}
    if not files:
        return 'nofile'
    if 'train.csv' in names:
        return 'fit'
    return 'table'

def _v1_load_task(objective, files):
    """-> (dfs, dictionary_text). Le dictionnaire est lu sans conversion."""
    low = objective.lower()
    dayfirst = bool(re.search('day[- ]?first|day/month|day-month|dd/mm|d/m/y|dd-mm|dd\\.mm', low))
    dedupe = _v1__dedupe_requested(low)
    na_extra = set()
    for sent in re.split('(?<=[.;?])\\s+', objective):
        if re.search('missing|not recorded|unknown|blank|not available', sent, re.I):
            na_extra |= {q.strip().lower() for q in re.findall('[\'\\"“‘]([^\'\\"”’]{1,12})[\'\\"”’]', sent)}
    na_extra.discard('')
    dfs, dico = (_v1_Tables(), '')
    for f in files or []:
        name = _v1_stem(f)
        try:
            if name == 'data_dictionary':
                d = _v1_read_table(f, convert=False)
                dico = '\n'.join((' | '.join((str(v) for v in row if str(v) != 'nan')) for row in d.values.tolist()))
                dfs.dictionary = d
                dfs.desc = {c: ds for _, c, ds in _v1__dico_rows(dico)}
                continue
            d = _v1_read_table(f, dayfirst=dayfirst, na_extra=na_extra)
            dfs[name] = d
        except Exception:
            if _v1_DEBUG:
                import traceback
                print(f'[debug] lecture impossible : {f}')
                traceback.print_exc()
            continue
    _v1__apply_data_cleanup(dfs, objective, dico, dedupe)
    return (dfs, dico)

def _v1__key_variants(v):
    """Formes plausibles d'une clé : telle quelle, minuscule, sans préfixe 'XXX-', chiffres seuls (sans zéros de tête)."""
    t = unicodedata.normalize('NFKC', str(v)).strip().casefold()
    folded = ''.join((ch for ch in unicodedata.normalize('NFKD', t) if not unicodedata.combining(ch)))
    out, pending = ([], [t, folded])
    while pending:
        item = pending.pop()
        if not item or item in out:
            continue
        out.append(item)
        compact = re.sub('[^a-z0-9]+', '', item)
        if compact and compact != item:
            pending.append(compact)
        for sep in ('-', '_', ':'):
            if sep in item:
                pending.append(item.split(sep)[-1])
        decimal_id = re.fullmatch('([a-z]*\\d+)\\.0+', item)
        if decimal_id:
            pending.append(decimal_id.group(1))
        m = re.fullmatch('[a-z]*0*(\\d+)', item)
        if m:
            pending.append(m.group(1))
    return out

def _v1__composite_join_hint(dfs, dico):
    """Create a temporary verified join key when two semantically matched fields form a composite key."""
    import itertools
    if len(dfs) != 2:
        return None
    (a_name, A), (b_name, B) = list(dfs.items())
    a_desc = _v1__table_descriptions(a_name, A.columns, dico)
    b_desc = _v1__table_descriptions(b_name, B.columns, dico)
    generic = {'number', 'code', 'id', 'key', 'identifier', 'value', 'date', 'year', 'ref', 'reference'}
    matches = []
    for ac in A.columns:
        at = _v1__toks(str(ac) + ' ' + a_desc.get(ac, '')) - generic
        scores = []
        for bc in B.columns:
            bt = _v1__toks(str(bc) + ' ' + b_desc.get(bc, '')) - generic
            common = at & bt
            if common:
                scores.append((len(common), bc))
        scores.sort(reverse=True)
        if scores and (len(scores) == 1 or scores[0][0] > scores[1][0]):
            matches.append((ac, scores[0][1], scores[0][0]))
    if len(matches) < 2:
        return None

    def tuple_variants(values):
        components = [_v1__key_variants(value)[:6] for value in values]
        if any((not xs for xs in components)):
            return set()
        return set(itertools.product(*components))
    candidates = []
    for first, second in itertools.combinations(matches, 2):
        if first[0] == second[0] or first[1] == second[1]:
            continue
        for fact_name, fact, fact_cols, dim_name, dim, dim_cols, semantic in ((a_name, A, (first[0], second[0]), b_name, B, (first[1], second[1]), first[2] + second[2]), (b_name, B, (first[1], second[1]), a_name, A, (first[0], second[0]), first[2] + second[2])):
            dim_rows = dim[list(dim_cols)].dropna()
            if len(dim_rows) < 2:
                continue
            dim_tuples = [tuple(row) for row in dim_rows.itertuples(index=False, name=None)]
            if len(set(dim_tuples)) < 0.95 * len(dim_tuples):
                continue
            fact_rows = fact[list(fact_cols)].dropna()
            fact_tuples = set((tuple(row) for row in fact_rows.itertuples(index=False, name=None)))
            if len(fact_tuples) < 2 or len(fact_tuples) >= 0.95 * len(fact_rows):
                continue
            lookup, ambiguous = ({}, set())
            for idx, values in zip(dim_rows.index, dim_tuples):
                for key in tuple_variants(values):
                    if key in lookup and lookup[key] != idx:
                        ambiguous.add(key)
                    else:
                        lookup.setdefault(key, idx)
            found = 0
            for values in fact_tuples:
                rows = {lookup[key] for key in tuple_variants(values) if key in lookup and key not in ambiguous}
                found += len(rows) == 1
            coverage = found / len(fact_tuples)
            if coverage >= 0.9:
                candidates.append((coverage, semantic, fact_name, fact_cols, dim_name, dim_cols, lookup, ambiguous, dim_rows.index.tolist()))
    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    best = candidates[0]
    if len(candidates) > 1 and candidates[1][:2] == best[:2] and (candidates[1][2:6] != best[2:6]):
        return None
    _, _, fact_name, fact_cols, dim_name, dim_cols, lookup, ambiguous, dim_indices = best
    fact, dim = (dfs[fact_name], dfs[dim_name])
    key_col = '__dsh_join_key'
    suffix = 1
    while key_col in fact.columns or key_col in dim.columns:
        key_col = f'__dsh_join_key_{suffix}'
        suffix += 1
    ids = {idx: f'dsh{n}' for n, idx in enumerate(dim_indices)}
    dim[key_col] = [ids.get(idx) for idx in dim.index]
    fact_ids = []
    for values in fact[list(fact_cols)].itertuples(index=False, name=None):
        if any((pd.isna(value) for value in values)):
            fact_ids.append(np.nan)
            continue
        rows = {lookup[key] for key in tuple_variants(values) if key in lookup and key not in ambiguous}
        fact_ids.append(ids[next(iter(rows))] if len(rows) == 1 else np.nan)
    fact[key_col] = fact_ids
    return f"Join key: dfs['{fact_name}']['{key_col}'] matches dfs['{dim_name}']['{key_col}'] (verified composite match: {fact_cols} ↔ {dim_cols}; temporary keys were added). Use this key instead of either component alone."

def _v1_align_keys(dfs, dico=''):
    """Détecte, pour chaque couple de tables, la colonne-clé de la table de faits (valeurs répétées) dont les
    valeurs sont incluses dans celles d'une colonne unique de l'autre table, à la casse / aux préfixes / aux zéros
    près. Réécrit la clé de faits comme la clé de référence et renvoie des phrases d'indice pour le prompt."""
    cands = []
    entity_aliases = {'orders': 'order', 'products': 'product', 'items': 'product', 'item': 'product', 'sku': 'product', 'customers': 'customer', 'clients': 'customer', 'client': 'customer', 'suppliers': 'supplier', 'vendors': 'supplier', 'vendor': 'supplier', 'students': 'student', 'pupils': 'student', 'schools': 'school', 'loans': 'loan', 'shipments': 'shipment', 'shipment': 'shipment', 'parcels': 'shipment', 'deliveries': 'shipment', 'readings': 'reading', 'flats': 'flat', 'employees': 'employee', 'patients': 'patient', 'regions': 'region', 'countries': 'country', 'cities': 'city', 'carriers': 'carrier', 'invoices': 'invoice', 'payments': 'payment', 'transactions': 'transaction'}
    descs = {(_v1_stem(f).lower(), c): desc for f, c, desc in _v1__dico_rows(dico)}

    def entities(table, column):
        words = re.findall('[a-z]+', (str(column) + ' ' + descs.get((table, column), '')).lower())
        return {entity_aliases.get(w, entity_aliases.get(w[:-1], w[:-1] if w.endswith('s') else w)) for w in words} & set(entity_aliases.values())

    def is_measure(table, column):
        text = (str(column) + ' ' + descs.get((table, column), '')).lower()
        identity = re.search('\\b(id|identifier|key|reference|ref|code|number(?!\\s+of))\\b', text)
        measure = re.search('\\b(quantity|amount|price|sales|revenue|weight|cost|rate|score|age|height|width|duration|total|count|units?|value|number of)\\b', text)
        return bool(measure and (not identity))
    names = list(dfs)
    for bn in names:
        B = dfs[bn]
        for bc in B.columns:
            bs = B[bc].dropna()
            if len(bs) < 2 or bs.nunique() < len(bs) * 0.95 or bs.dtype.kind == 'M':
                continue
            lookup, ambiguous = ({}, set())
            for orig in bs.unique():
                key = int(orig) if isinstance(orig, float) and float(orig).is_integer() else orig
                for var in _v1__key_variants(key):
                    if var in lookup and lookup[var] != orig:
                        ambiguous.add(var)
                    else:
                        lookup.setdefault(var, orig)
            exact_lookup = {str(v).strip().lower(): v for v in bs.unique()}
            for an in names:
                if an == bn:
                    continue
                A = dfs[an]
                for ac in A.columns:
                    a_s = A[ac].dropna()
                    if len(a_s) < 2 or a_s.dtype.kind == 'M':
                        continue
                    if is_measure(an, ac) or is_measure(bn, bc):
                        continue
                    a_entities, b_entities = (entities(an, ac), entities(bn, bc))
                    if a_entities and b_entities and (not a_entities & b_entities):
                        continue
                    uniq = list(a_s.unique())
                    if len(uniq) < 2 or len(uniq) > bs.nunique():
                        continue
                    mapped = {}
                    for u in uniq:
                        uu = int(u) if isinstance(u, float) and float(u).is_integer() else u
                        exact = exact_lookup.get(str(uu).strip().lower())
                        if exact is not None:
                            mapped[u] = exact
                            continue
                        for var in _v1__key_variants(uu):
                            if var in lookup and var not in ambiguous:
                                mapped[u] = lookup[var]
                                break
                    cov = len(mapped) / len(uniq)
                    if cov < 0.9:
                        continue
                    both_text = A[ac].dtype.kind not in 'fiu' and B[bc].dtype.kind not in 'fiu'
                    repeat = 1 - len(uniq) / len(a_s)
                    text_to_num = A[ac].dtype.kind not in 'fiu' and B[bc].dtype.kind in 'fiu'
                    semantic_match = bool(a_entities and b_entities and a_entities & b_entities)
                    cands.append((cov + 0.5 * both_text + 0.3 * repeat + 0.1 * text_to_num + 0.2 * semantic_match, an, ac, bn, bc, mapped))
    notes, done = ([], set())
    cands.sort(key=lambda c: -c[0])
    for i, (score, an, ac, bn, bc, mapped) in enumerate(cands):
        pair = frozenset((an, bn))
        if pair in done:
            continue
        alternatives = [c for c in cands[i + 1:] if frozenset((c[1], c[3])) == pair]
        if alternatives and score - alternatives[0][0] < 0.05:
            done.add(pair)
            continue
        done.add(pair)
        if any((k != v for k, v in mapped.items())):
            dfs[an][ac] = dfs[an][ac].map(lambda x, m=mapped: m.get(x, x))
        notes.append(f"Join key: dfs['{an}']['{ac}'] matches dfs['{bn}']['{bc}'] (already normalized: merge with left_on / right_on).")
    if len(dfs) == 2 and (not notes):
        composite = _v1__composite_join_hint(dfs, dico)
        if composite:
            notes.append(composite)
    return notes

def _v1_fit_hints(objective, dico, dfs):
    notes, desc2code = ([], {})
    for line in dico.split('\n'):
        parts = [p.strip() for p in line.split(' | ')]
        if len(parts) >= 3:
            desc2code.setdefault(parts[2].lower(), set()).add(parts[1])
    for q in re.findall('"([^"]+)"', objective):
        codes = desc2code.get(q.strip().lower())
        if codes:
            notes.append('Do NOT use column(s) ' + ', '.join(sorted(codes)) + f' (described as "{q}").')
    te = dfs.get('test')
    m = re.search('\\b(\\d+)\\s+of test\\.csv', objective)
    if m and te is not None and len(te.columns):
        idc = te.columns[0]
        notes.append(f"'row {m.group(1)} of test.csv' means the row where dfs['test']['{idc}'] == {m.group(1)}, not the row at that position.")
    return notes

def _v1_entity_hints(objective, dico):
    """Le dictionnaire décrit des colonnes codées (« coded (S01 Bordeaux office, S02 Grenoble lab) ») ; l'énoncé cite
    le nom, pas le code. On dit au modèle quel code utiliser."""
    notes, low = ([], objective.lower())
    for _, col, desc in _v1__dico_rows(dico):
        m = re.search('coded \\((.*)\\)', desc)
        if not m:
            continue
        for item in re.split(',\\s*', m.group(1)):
            parts = item.strip().split(' ', 1)
            if len(parts) == 2 and len(parts[1]) > 2 and (parts[1].lower() in low):
                notes.append(f"'{parts[1]}' corresponds to the code '{parts[0]}' in column {col} (the data stores the code, not the name).")
    return notes

def _v1_describe(dfs):
    out = []
    for name, d in dfs.items():
        cols = ', '.join((f'{c} ({str(d[c].dtype)[:8]}' + (f', NaN {d[c].isna().mean():.0%}' if d[c].isna().any() else '') + ')' for c in d.columns))
        out.append(f"dfs['{name}']: {len(d)} rows; columns: {cols}")
        out.append(d.head(3).to_string(max_cols=14, max_colwidth=14))
    return '\n'.join(out)

def _v1_hints(objective):
    low = objective.lower()
    h = []
    m = re.search('(-\\d+)[^.]{0,60}(not recorded|missing|unknown|no value)', low)
    if m or 'not recorded' in low:
        h.append("Some column uses -1 for 'not recorded': replace it by NaN in that column before computing.")
    if 'leave out' in low or 'exclude' in low or 'ignore' in low:
        h.append('Respect the instruction about a column or rows to leave out.')
    if 'case' in low and 'insensitive' in low:
        h.append('Compare text case-insensitively and strip spaces.')
    return h
_v1_SYSTEM = 'You are a careful data scientist. Reply ONLY with a short Python program in one code block. The program must end by setting a variable `answer` to ONE number. No print, no explanation, no comments, no imports, at most 12 lines. Tables are already loaded in `dfs`: never call read_csv. Never compute by hand: let Python compute. DO NOT round the answer yourself (never use round() or .round()), the system will do the rounding.'
_v1_TOOLS = {'nofile': 'Available: math, numpy as np, sympy, scipy.stats as stats, datetime, safe_divide. Use safe_divide(a, b) when b may be zero; zero/non-finite results become NaN.', 'table': 'Available: pandas as pd, numpy as np, math, scipy.stats as stats. The tables are in dict `dfs` (keys = file names without .csv). Numeric/date types are inferred only when parsing is unambiguous; mixed columns may remain object, so inspect the dictionary, dtypes, and sample rows. Use clean_numeric(series) only for dictionary-described measures, never for IDs, codes, or categories; use clean_dates(series, dayfirst=...) for mixed date strings. Column codes are short: use ONLY the exact codes listed under each table. For joins, prefer any supplied Join key hint and check key uniqueness and match coverage with merge(indicator=True); preserve duplicate rows unless asked to remove them. Then filter and aggregate. For ratios with possible zero denominators use safe_divide(numerator, denominator).', 'fit': "Available: pandas as pd, numpy as np, math, stats, LinearRegression, LogisticRegression, r2_score. Tables are in dict `dfs` ('train', 'test'), already parsed (floats, NaN). Column codes are short: find them in the dictionary. Drop rows with NaN in the used columns before fitting. LinearRegression and LogisticRegression automatically one-hot encode selected text columns; unknown test categories are ignored. Use safe_divide(a, b) for possible zero denominators."}
_v1_EXAMPLES = {'nofile': ('Objective: A price of 80 rises by 15% then falls by 10%. What is the final price? Round to 2 decimals.', 'import math\nanswer = 80 * 1.15 * 0.90'), 'fit': ("Tables:\ndfs['train']: 80 rows; columns: a1 (float64), a2 (float64), a3 (float64)\ndfs['test']: 20 rows; columns: a2 (float64), a3 (float64)\nDictionary:\na1 | weight in kg\na2 | height in cm\na3 | age in years\nObjective: Fit a linear regression of weight on height and age on train.csv. Predict the weight of the first row of test.csv. Round to 2 decimals.", "tr = dfs['train'].dropna(subset=['a1', 'a2', 'a3'])\nm = LinearRegression().fit(tr[['a2', 'a3']], tr['a1'])\nanswer = m.predict(dfs['test'][['a2', 'a3']].iloc[[0]])[0]")}
_v1_EXAMPLES['join'] = ("Tables:\ndfs['orders']: 8 rows; columns: o1 (float64), o2 (float64), o3 (float64)\ndfs['clients']: 5 rows; columns: k1 (float64), k2 (object), k3 (float64)\nDictionary:\no1 | client id\no2 | order amount in euros\no3 | order year\nk1 | client id\nk2 | client city\nk3 | client age\nObjective: Total order amount of the clients living in 'Paris' (join orders with clients on client id). Round to 2 decimals.", "m = dfs['orders'].merge(dfs['clients'], left_on='o1', right_on='k1', how='inner')\nm = m[m['k2'].str.lower() == 'paris']\nanswer = m['o2'].sum()")
_v1_TABLE_EXAMPLES = [('standard deviation|std dev', ("Tables:\ndfs['data']: 10 rows; columns: c1 (float64), c2 (object)\nDictionary:\nc1 | weight\nc2 | category\nObjective: Standard deviation of weight for category 'A'.", "d = dfs['data']\nd = d[d['c2'].str.lower() == 'a']\nanswer = d['c1'].std()")), ('correlation', ("Tables:\ndfs['stats']: 15 rows; columns: x (float64), y (float64)\nDictionary:\nx | height\ny | age\nObjective: Correlation between height and age.", "d = dfs['stats'].dropna(subset=['x', 'y'])\nanswer = d['x'].corr(d['y'])")), ('second (largest|highest)|2nd (largest|highest)', ("Tables:\ndfs['scores']: 8 rows; columns: player (object), points (float64)\nDictionary:\nplayer | Name\npoints | Score\nObjective: What is the second highest score?", "s = dfs['scores']['points'].dropna().drop_duplicates()\nanswer = s.nlargest(2).iloc[-1]")), ('', ("Tables:\ndfs['sales']: 6 rows; columns: c1 (object), c2 (float64), c3 (float64)\nDictionary:\nc1 | product category\nc2 | unit price in euros\nc3 | quantity sold\nObjective: Average revenue (price x quantity) of the category 'toys'.", "d = dfs['sales']\nd = d[d['c1'].str.lower() == 'toys']\nanswer = (d['c2'] * d['c3']).mean()"))]

def _v1_get_table_example(objective):
    low_obj = objective.lower()
    for pattern, ex in _v1_TABLE_EXAMPLES:
        if pattern and re.search(pattern, low_obj):
            return ex
    return _v1_TABLE_EXAMPLES[-1][1]

def _v1_build_user(kind, objective, dfs, dico, notes=()):
    parts = []
    if kind != 'nofile':
        parts.append('Tables:\n' + _v1_describe(dfs))
        if dico:
            parts.append('Dictionary:\n' + dico)
    parts += list(notes)
    parts.append('Objective: ' + objective.strip())
    parts += _v1_hints(objective)
    return '\n'.join(parts)
_v1_OLD_TABLE_TOOLS = 'Available: pandas as pd, numpy as np, math, scipy.stats as stats. The tables are in dict `dfs` (keys = file names without .csv). They are already parsed: numbers are floats, missing values are NaN, decimal commas and units are removed. Column codes are short: find them in the dictionary.'

def _v1_build_messages(kind, user, objective, ntables=1):
    if kind == 'table' and ntables >= 2:
        ex_u, ex_a = _v1_EXAMPLES['join']
    elif kind == 'table':
        ex_u, ex_a = _v1_get_table_example(objective)
    else:
        ex_u, ex_a = _v1_EXAMPLES[kind]
    return [{'role': 'system', 'content': _v1_SYSTEM + '\n' + (_v1_OLD_TABLE_TOOLS if kind == 'table' and os.environ.get('DSH_OLD_PROMPT') else _v1_TOOLS[kind])}, {'role': 'user', 'content': ex_u}, {'role': 'assistant', 'content': '```python\n' + ex_a + '\n```'}, {'role': 'user', 'content': user}]

class _v1__Timeout(Exception):
    pass

def _v1__alarm(signum, frame):
    raise _v1__Timeout()

def _v1__make_models(unreg=False):
    """LinearRegression / LogisticRegression tolérants aux NaN (sklearn plante sinon) :
    fit ignore les lignes incomplètes, predict impute par la moyenne d'apprentissage.
    Si l'énoncé dit « sans régularisation », la logistique est non pénalisée (le défaut sklearn, C=1, change les
    probabilités) ; elle est résolue sur variables standardisées puis ramenée à l'échelle d'origine.
    Les colonnes catégorielles textuelles sont encodées en one-hot, sans toucher au chemin numérique habituel."""
    from sklearn.linear_model import LinearRegression as _LR, LogisticRegression as _LogR
    from sklearn.preprocessing import OneHotEncoder as _OHE

    def _clean_fit(self, X, y, *a, **k):
        raw = pd.DataFrame(X).copy()
        yv = pd.Series(np.asarray(y).ravel(), index=raw.index)
        numeric, categorical = ([], [])
        converted = {}
        for col in raw.columns:
            values = pd.to_numeric(raw[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
            nonmissing = raw[col].notna()
            rate = values[nonmissing].notna().mean() if nonmissing.any() else 0.0
            if pd.api.types.is_numeric_dtype(raw[col]) or rate >= 0.9:
                numeric.append(col)
                converted[col] = values
            else:
                categorical.append(col)
        if not categorical:
            Xd = pd.DataFrame({col: converted[col] for col in raw.columns}, index=raw.index)
            ok = Xd.notna().all(axis=1) & yv.notna()
            if pd.api.types.is_numeric_dtype(yv):
                y_num = pd.to_numeric(yv, errors='coerce')
                ok &= np.isfinite(y_num)
                yv = y_num
            self._mu = Xd.loc[ok].mean()
            self._dsh_numeric_cols = list(Xd.columns)
            self._dsh_numeric_coef_idx = {col: i for i, col in enumerate(Xd.columns)}
            self._dsh_cat_cols = []
            self._dsh_ohe = None
            self._dsh_input_cols = list(raw.columns)
            self._dsh_encoded_names = list(raw.columns)
            return (Xd.loc[ok].values, yv.loc[ok].values)
        Xd = pd.DataFrame(index=raw.index)
        for col in numeric:
            Xd[col] = converted[col]
        for col in categorical:
            Xd[col] = raw[col]
        ok = Xd.notna().all(axis=1) & yv.notna()
        if pd.api.types.is_numeric_dtype(yv):
            y_num = pd.to_numeric(yv, errors='coerce')
            ok &= np.isfinite(y_num)
            yv = y_num
        Xd, yv = (Xd.loc[ok], yv.loc[ok])
        if Xd.empty:
            return (np.empty((0, 0)), yv.values)
        self._dsh_numeric_cols = list(numeric)
        self._dsh_cat_cols = list(categorical)
        self._dsh_input_cols = list(raw.columns)
        self._mu = Xd[numeric].mean() if numeric else pd.Series(dtype=float)
        if categorical:
            cat = Xd[categorical].astype(str)
            try:
                encoder = _OHE(handle_unknown='ignore', drop='first', sparse_output=False)
            except TypeError:
                encoder = _OHE(handle_unknown='ignore', drop='first', sparse=False)
            cat_encoded = encoder.fit_transform(cat)
            self._dsh_ohe = encoder
            try:
                cat_names = list(encoder.get_feature_names_out(categorical))
            except Exception:
                cat_names = [f'{col}__onehot_{i}' for col in categorical for i in range(cat_encoded.shape[1])]
        else:
            cat_encoded = np.empty((len(Xd), 0))
            self._dsh_ohe = None
            cat_names = []
        num_encoded = Xd[numeric].to_numpy(dtype=float) if numeric else np.empty((len(Xd), 0))
        self._dsh_numeric_coef_idx = {col: i for i, col in enumerate(numeric)}
        self._dsh_encoded_names = list(numeric) + cat_names
        encoded = np.column_stack([num_encoded, cat_encoded])
        if encoded.shape[1] == 0:
            encoded = np.zeros((len(Xd), 1), dtype=float)
            self._dsh_encoded_names = ['__constant__']
            self._dsh_numeric_coef_idx = {}
        return (encoded, yv.values)

    def _clean_X(self, X):
        raw = pd.DataFrame(X, columns=getattr(self, '_dsh_input_cols', None)).copy()
        if not getattr(self, '_dsh_cat_cols', []):
            Xd = raw.reindex(columns=self._dsh_numeric_cols).apply(pd.to_numeric, errors='coerce')
            Xd = Xd.replace([np.inf, -np.inf], np.nan).fillna(self._mu)
            return Xd.to_numpy(dtype=float)
        Xd = pd.DataFrame(index=raw.index)
        for col in self._dsh_numeric_cols:
            values = pd.to_numeric(raw[col], errors='coerce').replace([np.inf, -np.inf], np.nan)
            Xd[col] = values.fillna(self._mu.get(col, 0.0))
        num_encoded = Xd[self._dsh_numeric_cols].to_numpy(dtype=float) if self._dsh_numeric_cols else np.empty((len(raw), 0))
        cat = raw.reindex(columns=self._dsh_cat_cols).fillna('__missing__').astype(str)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            cat_encoded = self._dsh_ohe.transform(cat)
        return np.column_stack([num_encoded, cat_encoded])

    class LinearRegression(_LR):

        def fit(self, X, y, *a, **k):
            Xv, yv = _clean_fit(self, X, y)
            return super().fit(Xv, yv, *a, **k)

        def predict(self, X):
            return super().predict(_clean_X(self, X))

        def score(self, X, y, *a, **k):
            Xv = _clean_X(self, X)
            yv = pd.to_numeric(pd.Series(np.asarray(y).ravel()), errors='coerce')
            ok = yv.notna() & np.isfinite(yv)
            return super().score(Xv[ok.to_numpy()], yv[ok].to_numpy(), *a, **k)

    class LogisticRegression(_LogR):

        def __init__(self, penalty='l2', C=1.0, tol=0.0001, max_iter=5000, class_weight=None, solver='lbfgs', fit_intercept=True, random_state=None, **ignored):
            if unreg:
                penalty, solver, tol = (None, 'lbfgs', 1e-09)
            super().__init__(penalty=penalty, C=C, tol=tol, max_iter=max_iter, class_weight=class_weight, solver=solver, fit_intercept=fit_intercept, random_state=random_state)

        def fit(self, X, y, *a, **k):
            Xv, yv = _clean_fit(self, X, y)
            if yv.dtype == object or yv.dtype.kind in 'US':
                yv = pd.Series(yv).astype(str).str.strip().str.lower().isin(['1', 'yes', 'y', 'true']).astype(int).values
            if not unreg:
                return super().fit(Xv, yv, *a, **k)
            mu, sd = (Xv.mean(axis=0), Xv.std(axis=0))
            sd[sd == 0] = 1.0
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                super().fit((Xv - mu) / sd, yv, *a, **k)
            self.coef_ = self.coef_ / sd
            self.intercept_ = self.intercept_ - (self.coef_ * mu).sum(axis=1)
            return self

        def predict(self, X):
            return super().predict(_clean_X(self, X))

        def predict_proba(self, X):
            return super().predict_proba(_clean_X(self, X))
    return (LinearRegression, LogisticRegression)

def _v1__resolve_col(key, columns, desc):
    cols = [str(c) for c in columns]
    k = key.strip()
    low = {c.lower(): c for c in cols}
    if k.lower() in low:
        return low[k.lower()]
    norm = lambda x: re.sub('[^a-z0-9]', '', x.lower())
    nm = {norm(c): c for c in cols}
    if norm(k) in nm:
        return nm[norm(k)]
    kt = _v1__toks(re.sub('[_\\W]+', ' ', k))
    if kt:
        scores = []
        for c in cols:
            ct = _v1__toks(desc.get(c, '')) | _v1__toks(re.sub('[_\\W]+', ' ', c))
            sc = sum((1 for a in kt if any((a == b or (len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a))) for b in ct))))
            scores.append((sc, c))
        scores.sort(reverse=True)
        if scores and scores[0][0] >= 1 and (len(scores) == 1 or scores[0][0] > scores[1][0]):
            return scores[0][1]
    import difflib
    m = difflib.get_close_matches(k, cols, n=1, cutoff=0.8)
    return m[0] if m else None

class _v1_SmartFrame(pd.DataFrame):
    """DataFrame dont df['colonne_inexistante'] cherche la colonne voulue au lieu de lever KeyError."""
    _desc = {}

    @property
    def _constructor(self):
        return _v1_SmartFrame

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            if isinstance(key, str):
                alt = _v1__resolve_col(key, self.columns, _v1_SmartFrame._desc)
                if alt is not None:
                    return super().__getitem__(alt)
            elif isinstance(key, list) and key and all((isinstance(x, str) for x in key)):
                alts = [x if x in self.columns else _v1__resolve_col(x, self.columns, _v1_SmartFrame._desc) for x in key]
                if all((a is not None for a in alts)):
                    return super().__getitem__(alts)
            raise

def _v1__fake_read_csv(cdfs, extra=None):
    extra = extra or {}

    def read_csv(path, *a, **k):
        name = _v1_stem(str(path))
        if name in extra:
            return extra[name].copy()
        if name in cdfs:
            return cdfs[name].copy()
        for key, d in cdfs.items():
            if key in name or name in key:
                return d.copy()
        if cdfs:
            return list(cdfs.values())[0].copy()
        raise FileNotFoundError(path)
    return read_csv
_v1_LAST_ERR = ''
_v1_LAST_STATE = ''

def _v1__runtime_state(ns, max_frames=4):
    """Compact DataFrame diagnostics captured after generated code fails."""
    candidates = []
    seen = set()
    for name, value in ns.items():
        if name == 'dfs' or name.startswith('__'):
            continue
        if isinstance(value, pd.DataFrame):
            priority = 0 if re.search('merge|join|result|output|\\bm\\b', name, re.I) else 1
            candidates.append((priority, name, value))
    tables = ns.get('dfs')
    if isinstance(tables, dict):
        for name, value in tables.items():
            if isinstance(value, pd.DataFrame):
                candidates.append((2, f"dfs['{name}']", value))
    candidates.sort(key=lambda item: item[0])
    reports = []
    for _, name, frame in candidates:
        if id(frame) in seen:
            continue
        seen.add(id(frame))
        try:
            details = []
            for col in list(frame.columns)[:12]:
                series = frame[col]
                missing = series.isna().mean()
                details.append(f'{col}:{series.dtype},NaN={missing:.0%}')
            head = frame.head(2).to_string(index=False, max_cols=8, max_colwidth=28)
            reports.append(f'{name} shape={frame.shape}; ' + ', '.join(details) + '\nhead(2): ' + head[:900])
        except Exception as error:
            reports.append(f'{name}: diagnostic failed ({type(error).__name__})')
        if len(reports) >= max_frames:
            break
    return '\n\n'.join(reports)[:5000]

def _v1__run_with_timeout(fn, timeout):
    """Exécute fn dans un thread ; au-delà de `timeout` on y injecte une exception (boucle Python infinie) et on
    abandonne le thread s'il ne s'arrête pas. Fonctionne quel que soit le thread qui appelle solve()."""
    import ctypes
    box = {}

    def target():
        try:
            box['v'] = fn()
        except BaseException as e:
            box['e'] = e
    th = threading.Thread(target=target, daemon=True)
    th.start()
    th.join(max(0.2, timeout))
    if th.is_alive():
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(th.ident), ctypes.py_object(_v1__Timeout))
        th.join(0.3)
        raise _v1__Timeout('exec timeout')
    if 'e' in box:
        raise box['e']
    return box.get('v')

def _v1_run_code(code, dfs, timeout, obj=''):
    """Exécute le code généré, renvoie un float ou None."""
    from scipy import stats
    from sklearn.metrics import r2_score
    unreg = bool(re.search('without regularis|no regularis|unregularis|unpenali|without penal', obj.lower()))
    LinearRegression, LogisticRegression = _v1__make_models(unreg)
    for _n in ('nanmean', 'nanmedian', 'nanstd', 'nansum'):
        if not hasattr(stats, _n):
            setattr(stats, _n, getattr(np, _n))
    if not hasattr(stats, 'median'):
        setattr(stats, 'median', np.median)
    if not hasattr(stats, 'scoreatpercentile'):
        setattr(stats, 'scoreatpercentile', np.percentile)
    import sympy
    try:
        _v1_SmartFrame._desc = dict(getattr(dfs, 'desc', {}) or {})
        cdfs = {k: _v1_SmartFrame(v) for k, v in dfs.items()}
    except Exception:
        cdfs = {k: v.copy() for k, v in dfs.items()}
    dd = getattr(dfs, 'dictionary', None)
    extra = {'data_dictionary': dd} if dd is not None else {}
    ns = {'pd': pd, 'np': np, 'math': math, 'stats': stats, 'sympy': sympy, 'datetime': datetime, 'clean_numeric': _v1_clean_numeric, 'clean_dates': _v1_clean_dates, 'safe_divide': _v1_safe_divide, 'LinearRegression': LinearRegression, 'LogisticRegression': LogisticRegression, 'r2_score': r2_score, 'dfs': cdfs}
    orig_read_csv = pd.read_csv
    pd.read_csv = _v1__fake_read_csv(cdfs, extra)
    global _v1_LAST_ERR, _v1_LAST_STATE
    _v1_LAST_ERR, _v1_LAST_STATE = ('', '')

    def job():
        global _v1_LAST_STATE
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                _exec_generated(code, ns)
            value = _v1_to_float(ns.get('answer'))
            if value is None:
                raise ValueError('answer missing, non-numeric, or non-finite')
            return value
        except BaseException:
            _v1_LAST_STATE = _v1__runtime_state(ns)
            raise
    try:
        return _v1__run_with_timeout(job, timeout)
    except BaseException as e:
        _v1_LAST_ERR = f'{type(e).__name__}: {str(e)[:150]}'
        if not _v1_LAST_STATE:
            _v1_LAST_STATE = _v1__runtime_state(ns)
        return None
    finally:
        pd.read_csv = orig_read_csv

def _v1_to_float(v):
    try:
        if isinstance(v, (pd.Series, pd.DataFrame)):
            v = np.asarray(v).ravel()[0]
        if isinstance(v, (list, tuple, np.ndarray)):
            v = np.asarray(v).ravel()[0]
        if isinstance(v, str):
            v = _v1__num_from_text(v)
        f = float(v)
        return f if math.isfinite(f) else None
    except Exception:
        return None

def _v1_decimals(objective):
    low = objective.lower()
    m = re.search('(\\d+)\\s*(?:decimal|digit)', low) or re.search('decimals?\\s*[:=]?\\s*(\\d+)', low)
    if m:
        return int(m.group(1))
    if re.search('integer|whole number|nearest (?:whole|integer)', low):
        return 0
    return None

def _v1_calc_rules(objective):
    """Exact solvers for recurring, unambiguous calculation templates; None delegates to the LLM."""
    o = ' '.join(objective.lower().split())
    m = re.search('\\brecorded\\s+(?:these|the following)\\s+[^:]{0,100}:\\s*(\\d+(?:\\.\\d+)?(?:\\s*,\\s*\\d+(?:\\.\\d+)?){1,})\\s*[.;]?\\s*(?:what is the total|find the total|calculate the total)\\b', o)
    if m:
        values = [float(x.strip()) for x in m.group(1).split(',')]
        return sum(values)
    m = re.search('employs?\\s+(\\d+)\\s+people,?\\s+each working\\s+([\\d.]+)\\s+hours?\\s+a week at\\s+([\\d.]+)\\s+euros?\\s+an hour[\\s\\S]*?over\\s+(\\d+)\\s+weeks?,?\\s+each employee also receives a one-off bonus of\\s+([\\d.]+)\\s+euros?', o)
    if m and re.search('total paid out', o):
        people, hours, rate, weeks, bonus = map(float, m.groups())
        return people * (hours * rate * weeks + bonus)
    m = re.search('ordered arrangements of\\s+(\\d+)\\s+items?\\s+can be made from\\s+(\\d+)\\s+distinct items?', o)
    if m:
        k, n = map(int, m.groups())
        if 0 <= k <= n:
            return float(math.factorial(n) // math.factorial(n - k))
    if 'solve the system' in o:
        eqs = re.findall('([+-]?\\s*\\d*)\\s*x\\s*([+-])\\s*(\\d*)\\s*y\\s*=\\s*([+-]?\\d+(?:\\.\\d+)?)', o)
        if len(eqs) == 2:

            def coef(s):
                s = s.replace(' ', '')
                return -1.0 if s == '-' else 1.0 if s in ('', '+') else float(s)
            (a1, sign1, b1, c1), (a2, sign2, b2, c2) = eqs
            a1, a2 = (coef(a1), coef(a2))
            b1 = coef(sign1 + b1 if b1 else sign1)
            b2 = coef(sign2 + b2 if b2 else sign2)
            c1, c2 = (float(c1), float(c2))
            det = a1 * b2 - a2 * b1
            if det:
                x = (c1 * b2 - c2 * b1) / det
                y = (a1 * c2 - a2 * c1) / det
                target = re.search('value of ([xy])\\b', o)
                if target:
                    return x if target.group(1) == 'x' else y
    m = re.search('least common multiple of\\s+(\\d+)\\s+and\\s+(\\d+)', o)
    if m:
        return float(math.lcm(*map(int, m.groups())))
    m = re.search('\\bone every\\s+([\\d,]+)\\s+minutes?[\\s\\S]{0,100}?\\bone every\\s+([\\d,]+)\\s+minutes?[\\s\\S]{0,100}?\\bafter how many minutes\\b[\\s\\S]{0,60}?\\b(?:next )?(?:leave|depart|run)\\b[\\s\\S]{0,20}?\\btogether\\b', o)
    if m:
        a, b = (int(x.replace(',', '')) for x in m.groups())
        return float(math.lcm(a, b))
    m = re.search('\\b(\\d+(?:\\.\\d+)?)%\\s+of\\s+(?:cases|transactions|people|patients|parts|items)\\s+have\\b[\\s\\S]{0,180}?\\btest detects it in\\s+(\\d+(?:\\.\\d+)?)%[\\s\\S]{0,140}?\\bwrongly flags\\s+(\\d+(?:\\.\\d+)?)%[\\s\\S]{0,140}?\\btests? positive\\b', o)
    if m and re.search('\\bprobability\\b[\\s\\S]{0,100}\\bpercent\\b', o):
        prevalence, sensitivity, false_positive = (float(x) / 100 for x in m.groups())
        if all((0 <= x <= 1 for x in (prevalence, sensitivity, false_positive))):
            denominator = prevalence * sensitivity + (1 - prevalence) * false_positive
            if denominator > 0:
                return prevalence * sensitivity / denominator * 100
    m = re.search('probability\\s+(\\d+(?:\\.\\d+)?)\\s*[.;,]?\\s*among\\s+(\\d+)\\s+(?:parts?|trials?|items?)[\\s\\S]*?at least\\s+(\\d+)\\s+(?:defective|success)', o)
    if m:
        p, n, k = (float(m.group(1)), int(m.group(2)), int(m.group(3)))
        if 0 <= p <= 1 and 0 <= k <= n:
            return sum((math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))
    m = re.search('(\\d+)\\s+fair six-sided dice[\\s\\S]*?faces add up to\\s+(\\d+)', o)
    if m:
        n, target = map(int, m.groups())
        ways = [1]
        for _ in range(n):
            nxt = [0] * (len(ways) + 6)
            for total, count in enumerate(ways):
                for face in range(1, 7):
                    nxt[total + face] += count
            ways = nxt
        return ways[target] / 6 ** n if target < len(ways) else 0.0
    if 'overall percentage change' in o:
        changes = re.findall('\\b(falls?|decreases?|drops?|declines?|rises?|increases?|grows?)\\s+by\\s+([\\d.]+)\\s*%', o)
        if len(changes) >= 2:
            factor = 1.0
            for direction, amount in changes:
                rate = float(amount) / 100.0
                factor *= 1 - rate if direction.startswith(('fall', 'decreas', 'drop', 'declin')) else 1 + rate
            return (factor - 1) * 100
    m = re.search('grew from\\s+([\\d.]+)\\s+to\\s+([\\d.]+)\\s+in\\s+(\\d+)\\s+years?[\\s\\S]*?constant yearly growth rate,? in percent', o)
    if m:
        start, end, years = (float(m.group(1)), float(m.group(2)), int(m.group(3)))
        if start > 0 and years > 0:
            return ((end / start) ** (1 / years) - 1) * 100
    m = re.search('starts? at\\s+([\\d.]+)\\s+euros? and earns\\s+([\\d.]+)% per year,? interest reinvested[\\s\\S]*?after\\s+(\\d+)\\s+years?', o)
    if m:
        principal, rate, years = (float(m.group(1)), float(m.group(2)) / 100, int(m.group(3)))
        return principal * (1 + rate) ** years
    m = re.search('\\bamount of\\s+([\\d.]+)\\s+euros? earns\\s+([\\d.]+)%\\s+compound interest per year[\\s\\S]{0,120}?\\bafter how many full years\\b[\\s\\S]{0,100}?\\bfirst reach(?:es)? at least\\s+([\\d.]+)', o)
    if m:
        principal, rate, target = (float(m.group(1)), float(m.group(2)) / 100, float(m.group(3)))
        if principal > 0 and target <= principal:
            return 0.0
        if principal > 0 and rate > 0 and (target > principal):
            years = math.log(target / principal) / math.log1p(rate)
            return float(math.ceil(years - 1e-12))
    m = re.search('is\\s+([\\d.]+)\\s+euros?\\s+once a\\s+([\\d.]+)%\\s+increase is applied[\\s\\S]*?original price', o)
    if m:
        final, rate = (float(m.group(1)), float(m.group(2)) / 100)
        return final / (1 + rate)
    m = re.search('one pump fills a tank in\\s+([\\d.]+)\\s+hours? and a second pump fills it in\\s+([\\d.]+)\\s+hours?[\\s\\S]*?leak empties the full tank in\\s+([\\d.]+)\\s+hours?', o)
    if m:
        a, b, leak = map(float, m.groups())
        rate = 1 / a + 1 / b - 1 / leak
        if rate > 0:
            return 1 / rate
    return None

def _v1_extract_code(text):
    text = text.split('```')[0]
    return text.strip('\n')

def _v1_vote(cands, nd):
    """cands : liste de (valeur, poids). Renvoie la valeur majoritaire."""
    if not cands:
        return 0.0
    cnt = collections.defaultdict(float)
    rep = {}
    for v, w in cands:
        k = round(v, nd if nd is not None else 4)
        cnt[k] += w
        rep.setdefault(k, v)
    best = max(cnt, key=cnt.get)
    return rep[best]
_v1__STOP = {'the', 'of', 'a', 'an', 'in', 'on', 'at', 'per', 'to', 'for', 'and', 'with', 'by', 'is', 'are', 'last'}

def _v1__toks(s):
    out = set()
    for w in re.findall('[a-z]+', s.lower()):
        if w in _v1__STOP:
            continue
        if w in {'used', 'uses', 'using'}:
            w = 'use'
        elif w in {'billed', 'billing'}:
            w = 'bill'
        elif w in {'born', 'births', 'birthday', 'birthdate'}:
            w = 'birth'
        elif w in {'shipped', 'shipping', 'shipment', 'shipments'}:
            w = 'ship'
        elif w == 'carrier':
            w = 'transport'
        elif w == 'occupancy':
            w = 'people'
        elif w in {'pupil', 'pupils', 'enrollment', 'enrolled'}:
            w = 'size'
        elif w in {'absences', 'absence'}:
            w = 'absence'
        out.add(w[:-1] if w.endswith('s') and len(w) > 3 else w)
    return out

def _v1__dico_rows(dico):
    rows = []
    for line in dico.split('\n'):
        parts = [p.strip() for p in line.split(' | ')]
        if len(parts) >= 3:
            rows.append((parts[0], parts[1], parts[2]))
    return rows

def _v1__dictionary_unit(name, column, dico):
    table = _v1_stem(name).lower()
    for line in dico.split('\n'):
        parts = [p.strip() for p in line.split(' | ')]
        if len(parts) >= 4 and _v1_stem(parts[0]).lower() == table and (parts[1] == column):
            return parts[3]
    return ''

def _v1__unit_key(unit):
    key = re.sub('[^a-z0-9]', '', str(unit).lower())
    return {'kg': 'kg', 'kilogram': 'kg', 'kilograms': 'kg', 'tonne': 'tonne', 'tonnes': 'tonne', 'metricton': 'tonne', 'metrictons': 'tonne', 'kwh': 'kwh', 'kilowatthour': 'kwh', 'kilowatthours': 'kwh', 'mj': 'mj', 'megajoule': 'mj', 'megajoules': 'mj'}.get(key)

def _v1__explicit_unit_factor(objective, source_unit, target_unit):
    """Return a conversion factor only when the objective states the exact unit equation."""
    source, target = (_v1__unit_key(source_unit), _v1__unit_key(target_unit))
    if not source or not target:
        return None
    for m in re.finditer('\\b1\\s*([a-z][a-z0-9]*)\\s*=\\s*([\\d.,]+)\\s*([a-z][a-z0-9]*)\\b', objective, re.I):
        left, right = (_v1__unit_key(m.group(1)), _v1__unit_key(m.group(3)))
        try:
            factor = float(m.group(2).replace(',', '.'))
        except ValueError:
            continue
        if not math.isfinite(factor) or factor <= 0:
            continue
        if source == left and target == right:
            return factor
        if source == right and target == left:
            return 1.0 / factor
    return None

def _v1__dedupe_requested(objective):
    """Recognise the common ways tasks ask us to keep each record only once."""
    return bool(re.search('\\b(?:drop|remove|delete|ignore|exclude|deduplicat\\w*|dedupe)\\b[^.;?]{0,80}\\bduplicat\\w*\\b|\\b(?:keep|retain)\\b[^.;?]{0,80}\\bone copy\\b|\\bcount each .{1,50}\\b(?:once|only once)\\b', objective, re.I))

def _v1__table_descriptions(name, columns, dico):
    """Descriptions from this table's dictionary rows, without collisions from other files."""
    table = _v1_stem(name).lower()
    return {c: desc for f, c, desc in _v1__dico_rows(dico) if _v1_stem(f).lower() == table and c in columns}

def _v1__identifier_column(name, columns, dico):
    descs = _v1__table_descriptions(name, columns, dico)
    candidates = []
    for c in columns:
        text = (str(c) + ' ' + descs.get(c, '')).lower()
        score = 0
        if re.search('\\b(identifier|unique id|record id|row id|shipment id|order id|loan id|pupil id|student id|flat id|reading id)\\b', text):
            score += 5
        if re.search('\\b(id|no|number|ref|reference)\\b', str(c).lower()):
            score += 3
        if re.search('\\b(identifier|identification number|record number|reading number|order number|loan number)\\b', descs.get(c, ''), re.I):
            score += 3
        if score:
            candidates.append((score, c))
    if candidates:
        candidates.sort(key=lambda x: -x[0])
        if len(candidates) == 1 or candidates[0][0] > candidates[1][0]:
            return candidates[0][1]
        return None
    return columns[0] if len(columns) else None

def _v1__distinct_target_column(objective, dfs, dico, table_name):
    """Resolve the field whose distinct values are requested; never guess a row ID."""
    o = ' '.join(objective.lower().split())
    patterns = ('\\b(?:distinct|unique)\\s+values?\\s+(?:of|for|in)\\s+(.+?)(?=\\s+(?:are|is|in|among|for|from|where|whose|with|of the|across|over)\\b|[?.;]|$)', '\\b(?:distinct|unique)\\s+(.+?)(?=\\s+(?:are|is|in|among|for|from|where|whose|with|of the|across|over)\\b|[?.;]|$)', '\\bvalues?\\s+of\\s+(.+?)(?=\\s+(?:are|is|in|among|for|from|where|whose|with|of the|across|over)\\b|[?.;]|$)')
    for pattern in patterns:
        match = re.search(pattern, o)
        if not match:
            continue
        phrase = match.group(1).strip(' ,.;:()')
        if re.fullmatch('(?:rows?|records?|entries|observations|people|persons)', phrase):
            return None
        resolved = _v1__resolve_dico_column(phrase, dfs, dico)
        if resolved and resolved[0] == table_name:
            return resolved[1]
    return None

def _v1__has_unparsed_filter_intent(objective, conditions):
    """Avoid deterministic aggregates when the question contains an unsupported filter."""
    o = ' '.join(objective.lower().split())
    filter_text = re.sub('\\b(?:ignore|leave|exclude|drop|values?)\\b[^.;?]{0,160}\\b(?:missing|not recorded|unrecorded)\\b[^.;?]*(?:[.;?]|$)', ' ', o)
    if not conditions and re.search('\\b(?:where|whose|among|excluding|except|without|not equal|different from|greater than|less than|at least|at most|no more than|no less than|on or after|on or before|after|before|is not|is missing|is blank|is null|is recorded|contains|starts with|ends with)\\b', filter_text):
        return True
    if re.search('\\bbetween\\s+-?\\d', filter_text) and (not any((c[2] == 'between' for c in conditions))):
        return True
    logical_text = re.sub('\\bon or after\\b', ' ', filter_text)
    logical_text = re.sub('\\b(?:missing|blank|null|unknown|not available)\\s+or\\s+(?:not recorded|missing|blank|null|unknown)\\b', ' ', logical_text)
    if re.search('\\b(?:one of|any of|either|or)\\b', logical_text):
        return True
    return False

def _v1__resolve_dico_column(phrase, dfs, dico):
    """Resolve a natural-language field phrase only when one dictionary column clearly fits."""

    def field_toks(text):
        tokens = _v1__toks(text)
        tokens.update((w[:-3] + 'y' for w in re.findall('[a-z]+', str(text).lower()) if w.endswith('ies') and len(w) > 4))
        return tokens
    q = field_toks(phrase)
    if not q:
        return None
    q_order = [w for w in re.findall('[a-z]+', phrase.lower()) if w not in _v1__STOP]
    q_last = _v1__toks(q_order[-1]) if q_order else set()
    qnorm = re.sub('[^a-z0-9]', '', phrase.lower())
    scored = []
    for f, c, desc in _v1__dico_rows(dico):
        name = _v1_stem(f).lower()
        if name not in dfs or c not in dfs[name].columns:
            continue
        words = field_toks(desc) | field_toks(c)
        overlap = len(q & words)
        score = 3 * len(q & field_toks(desc)) + len(q & field_toks(c))
        if q_last & words:
            score += 2
        dnorm = re.sub('[^a-z0-9]', '', (desc + ' ' + c).lower())
        if qnorm and (qnorm in dnorm or dnorm in qnorm):
            score += 8
        if qnorm and dnorm.startswith(qnorm):
            score += 6
        if overlap:
            scored.append((score, name, c))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None
    return (scored[0][1], scored[0][2])

def _v1__apply_data_cleanup(dfs, objective, dico, dedupe):
    """Apply explicit duplicate and missing-sentinel instructions before code generation."""
    if dedupe:
        for name, df in list(dfs.items()):
            key = _v1__identifier_column(name, df.columns, dico)
            subset = [key] if key in df.columns else None
            dfs[name] = df.drop_duplicates(subset=subset).reset_index(drop=True)
    missing = '(?:not recorded|missing|unknown|not available)'
    patterns = (f'\\bvalue\\s+(-?\\d+(?:[.,]\\d+)?)\\s+in the\\s+(.+?)\\s+column\\s+means?\\s+(?:it\\s+)?(?:is\\s+)?{missing}', f'\\b(?:a|an|the)\\s+(.+?)\\s+(?:of|equal(?:s)?|is)\\s+(-?\\d+(?:[.,]\\d+)?)\\s+means?\\s+(?:it\\s+)?(?:was\\s+|is\\s+)?{missing}')
    for i, pattern in enumerate(patterns):
        for m in re.finditer(pattern, objective, re.I):
            sentinel, phrase = (m.group(1), m.group(2)) if i == 0 else (m.group(2), m.group(1))
            resolved = _v1__resolve_dico_column(phrase, dfs, dico)
            if not resolved:
                continue
            name, col = resolved
            val = _v1__num_from_text(sentinel)
            series = dfs[name][col]
            if pd.api.types.is_numeric_dtype(series):
                dfs[name][col] = series.mask(series == val, np.nan)
            else:
                dfs[name][col] = series.replace([sentinel, str(val)], np.nan)

def _v1_table_stat_rule(objective, dfs, dico):
    """Resolve simple whole-table statistics from the dictionary, otherwise leave them to the model."""
    if len(dfs) != 1:
        return None
    o = ' '.join(objective.lower().split())
    if re.search('\\bdivided by\\b|\\bconsider the .+?\\s+per\\s+', o) or re.search('\\bfor each\\b|\\brank\\b|\\bgrouped by\\b', o):
        return None
    if re.search('\\b(?:distinct|unique)\\b', o) and re.search('\\b(?:how many|number of|count of|count the number of)\\b', o):
        table_name, df = next(iter(dfs.items()))
        conditions = [c for c in _v1__parse_table_conditions(o, dfs, dico) if c[0] == table_name]
        if _v1__has_unparsed_filter_intent(o, conditions):
            return None
        target = _v1__distinct_target_column(o, dfs, dico, table_name)
        if not target and re.search('\\b(?:distinct|unique)\\s+(?:rows?|records?|entries|observations)\\b', o):
            target = _v1__identifier_column(table_name, df.columns, dico)
        if not target:
            return None
        mask = pd.Series(True, index=df.index)
        for _, col, op, val, _ in conditions:
            mask &= _v1__condition_mask(df, col, op, val).fillna(False)
        return float(df.loc[mask, target].nunique(dropna=True))
    ordinals = {'second': 2, '2nd': 2, 'third': 3, '3rd': 3, 'fourth': 4, '4th': 4, 'fifth': 5, '5th': 5, 'sixth': 6, '6th': 6, 'seventh': 7, '7th': 7, 'eighth': 8, '8th': 8, 'ninth': 9, '9th': 9, 'tenth': 10, '10th': 10}
    order_stat = re.search('\\b(second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th|seventh|7th|eighth|8th|ninth|9th|tenth|10th)[- ]+(highest|largest|greatest|lowest|smallest|least)\\b', o)
    if order_stat:
        tail = o[order_stat.end():].strip()
        tail = re.sub('^(?:value\\s+of|the\\s+value\\s+of|the)\\s+', '', tail)
        cut = re.search('\\b(?:over all|across all|throughout all|among|where|whose|with|for|on or after|on or before|strictly greater than|strictly less than|greater than|less than|at least|at most|between)\\b|[?.;]', tail)
        phrase = tail[:cut.start()] if cut else tail
        col = _v1__resolve_dico_column(phrase.strip(' ,.;:()'), dfs, dico)
        if not col or col[0] not in dfs:
            return None
        conditions = [c for c in _v1__parse_table_conditions(o, dfs, dico) if c[0] == col[0]]
        if _v1__has_unparsed_filter_intent(o, conditions):
            return None
        df = dfs[col[0]]
        descriptions = _v1__table_descriptions(col[0], df.columns, dico)
        numeric_values = _v1__numeric_metric(df[col[1]], descriptions.get(col[1], ''))
        if numeric_values is None:
            return None
        mask = pd.Series(True, index=df.index)
        for _, cond_col, cond_op, val, _ in conditions:
            mask &= _v1__condition_mask(df, cond_col, cond_op, val).fillna(False)
        values = numeric_values.loc[mask].dropna().drop_duplicates()
        rank = ordinals[order_stat.group(1)]
        if len(values) < rank:
            return None
        ascending = order_stat.group(2) in ('lowest', 'smallest', 'least')
        return float(values.sort_values(ascending=ascending).iloc[rank - 1])
    corr = re.search('\\b(?:pearson\\s+)?correlation(?: coefficient)?\\s+between\\s+(.+?)\\s+and\\s+(.+?)(?=\\s+(?:for|among|over|across|within|where|whose|with)\\b|[?.;]|$)', o)
    if not corr:
        corr = re.search('\\b(?:pearson\\s+)?correlation(?: coefficient)?\\s+of\\s+(.+?)\\s+and\\s+(.+?)(?=\\s+(?:for|among|over|across|within|where|whose|with)\\b|[?.;]|$)', o)
    if corr:
        left = _v1__resolve_dico_column(corr.group(1), dfs, dico)
        right = _v1__resolve_dico_column(corr.group(2), dfs, dico)
        if not left or not right or left[0] != right[0]:
            return None
        conditions = [c for c in _v1__parse_table_conditions(o, dfs, dico) if c[0] == left[0]]
        if _v1__has_unparsed_filter_intent(o, conditions):
            return None
        df = dfs[left[0]]
        descriptions = _v1__table_descriptions(left[0], df.columns, dico)
        left_values = _v1__numeric_metric(df[left[1]], descriptions.get(left[1], ''))
        right_values = _v1__numeric_metric(df[right[1]], descriptions.get(right[1], ''))
        if left_values is None or right_values is None:
            return None
        mask = pd.Series(True, index=df.index)
        for _, col, op, val, _ in conditions:
            mask &= _v1__condition_mask(df, col, op, val).fillna(False)
        pair = pd.DataFrame({left[1]: left_values.loc[mask], right[1]: right_values.loc[mask]}).dropna()
        if len(pair) < 2:
            return None
        value = pair[left[1]].corr(pair[right[1]])
        return float(value) if pd.notna(value) and math.isfinite(float(value)) else None
    df_name, df = next(iter(dfs.items()))
    unit_request = re.search('\\bconsider\\s+(?:the\\s+)?(.+?)\\s+in\\s+([a-z][a-z0-9 -]*?)\\s*\\(\\s*1\\s+[a-z][a-z0-9]*\\s*=\\s*[\\d.,]+\\s*[a-z][a-z0-9]*\\b', o)
    converted_mean = re.search('\\b(?:its\\s+(mean|average)|(mean|average)\\s+of\\s+it)\\b', o)
    if unit_request and converted_mean:
        resolved = _v1__resolve_dico_column(unit_request.group(1), dfs, dico)
        if not resolved or resolved[0] != df_name:
            return None
        conditions = _v1__parse_table_conditions(objective, dfs, dico)
        clean_objective = re.sub('\\b[\\w.-]+\\.csv contains duplicated rows\\b[^.;?]*(?:[.;?]|$)', ' ', o)
        if conditions or _v1__has_unparsed_filter_intent(clean_objective, conditions):
            return None
        factor = _v1__explicit_unit_factor(o, _v1__dictionary_unit(resolved[0], resolved[1], dico), unit_request.group(2).strip())
        if factor is None:
            return None
        descriptions = _v1__table_descriptions(resolved[0], dfs[resolved[0]].columns, dico)
        numeric_values = _v1__numeric_metric(dfs[resolved[0]][resolved[1]], descriptions.get(resolved[1], ''))
        if numeric_values is None:
            return None
        values = numeric_values.dropna()
        return float(values.mean() * factor) if not values.empty else None
    missing_count = re.search('\\bhow many\\b.*?\\b(?:have|has)\\s+no\\s+(?:recorded\\s+)?(.+?)(?=[?.;]|$)', o)
    if not missing_count:
        missing_count = re.search('\\bhow many\\b.*?\\bis\\s+(?:the\\s+)?(.+?)\\s+(?:missing|not recorded|unrecorded)\\b', o)
    if missing_count:
        resolved = _v1__resolve_dico_column(missing_count.group(1), dfs, dico)
        if not resolved:
            return None
        conditions = [c for c in _v1__parse_table_conditions(objective, dfs, dico) if c[0] == resolved[0]]
        clean_objective = re.sub('\\b[\\w.-]+\\.csv contains duplicated rows\\b[^.;?]*(?:[.;?]|$)', ' ', o)
        if _v1__has_unparsed_filter_intent(clean_objective, conditions):
            return None
        source = dfs[resolved[0]]
        mask = pd.Series(True, index=source.index)
        for _, cond_col, cond_op, value, _ in conditions:
            mask &= _v1__condition_mask(source, cond_col, cond_op, value).fillna(False)
        return float(source.loc[mask, resolved[1]].isna().sum())
    if re.search('\\bhow many\\b', o) and re.search('\\b(?:have|has)\\s+(?:a |an )?(?:recorded )', o):
        if re.search('\\b(?:among|whose|with|born|on or after|on or before)\\b', o):
            return None
        m = re.search('\\b(?:have|has)\\s+(?:a |an )?(?:recorded )?(.+?)(?:\\?|\\.|$)', o)
        if not m:
            return None
        col = _v1__resolve_dico_column(m.group(1), dfs, dico)
        if not col:
            return None
        return float(dfs[col[0]][col[1]].notna().sum())
    whole_scope = re.search('\\b(?:(?:over|across|throughout|on|for)\\s+(?:all|the entire|the whole)|overall)\\b', o)
    scope_end = '\\s+(?:over all|across all|across the entire|across the whole|throughout all|for all|for the entire|for the whole|overall)\\b'
    m = re.search('\\b(\\d+)(?:st|nd|rd|th) percentile of (?:the )?(.+?)(?=\\s*(?:\\(|\\bover\\b|\\bamong\\b|\\bacross\\b|[?.]|$))', o)
    if m:
        if not whole_scope:
            return None
        q, phrase = (int(m.group(1)) / 100, m.group(2))
        operation = 'quantile'
    else:
        if not whole_scope:
            return None
        m = re.search(f'\\b(standard deviation|std\\.?\\s*dev(?:iation)?|variance|range|spread|maximum|minimum|median|mean|average|sum)\\s+(?:of\\s+)?(?:the\\s+)?(.+?)(?={scope_end}|\\s*\\(|[?.]|$)', o)
        if not m:
            return None
        op, phrase = (m.group(1), m.group(2))
        operation = {'maximum': 'max', 'minimum': 'min', 'average': 'mean', 'std dev': 'std', 'std. dev': 'std', 'std deviation': 'std', 'standard deviation': 'std'}.get(op, op)
    col = _v1__resolve_dico_column(phrase, dfs, dico)
    if not col:
        return None
    conditions = _v1__parse_table_conditions(o, dfs, dico)
    if conditions or (operation in ('std', 'variance', 'range', 'spread') and _v1__has_unparsed_filter_intent(o, conditions)):
        return None
    raw_series = dfs[col[0]][col[1]]
    descriptions = _v1__table_descriptions(col[0], dfs[col[0]].columns, dico)
    numeric_values = _v1__numeric_metric(raw_series, descriptions.get(col[1], ''))
    if numeric_values is None:
        return None
    series = numeric_values.dropna()
    if series.empty:
        return None
    if operation == 'quantile':
        return float(series.quantile(q, interpolation='linear'))
    if operation == 'max':
        return float(series.max())
    if operation == 'min':
        return float(series.min())
    if operation == 'median':
        return float(series.median())
    if operation == 'mean':
        return float(series.mean())
    if operation == 'sum':
        return float(series.sum())
    if operation == 'range' or operation == 'spread':
        return float(series.max() - series.min())
    if operation in ('std', 'variance'):
        ddof = 0 if re.search('\\bpopulation\\b', o) else 1
        if len(series) <= ddof:
            return None
        return float(series.std(ddof=ddof) if operation == 'std' else series.var(ddof=ddof))
    return None

def _v1__condition_context(prefix):
    """Keep the field phrase nearest to a filter operator/value."""
    low = prefix.lower()
    markers = (' and ', ' with ', ' whose ', ' have ', ' has ', ' among ', ' where ', ' of ', ' over ', ' born ')
    cuts = [(low.rfind(mark), len(mark)) for mark in markers if low.rfind(mark) >= 0]
    for pos, size in sorted(cuts, reverse=True):
        value = low[pos + size:].strip()
        value = re.sub('^(?:the|a|an)\\s+', '', value)
        value = re.sub('\\s+(?:of|the|a|an)$', '', value).strip()
        if value:
            return value[-80:]
    return low[-80:].strip()

def _v1__condition_column(context, dfs, dico, date_like=False):
    if date_like:
        tokens = _v1__toks(context)
        candidates = []
        for name, df in dfs.items():
            for col in df.columns:
                if df[col].dtype.kind != 'M':
                    continue
                desc = _v1__table_descriptions(name, df.columns, dico).get(col, '')
                if tokens & _v1__toks(desc):
                    candidates.append((name, col, tokens & _v1__toks(desc)))
        semantic = [c for c in candidates if c[2] & {'ship', 'birth', 'deliver'}]
        if len(semantic) == 1:
            return (semantic[0][0], semantic[0][1])
    resolved = _v1__resolve_dico_column(context, dfs, dico)
    if resolved:
        return resolved
    if date_like:
        date_cols = []
        for name, df in dfs.items():
            for col in df.columns:
                if df[col].dtype.kind == 'M':
                    date_cols.append((name, col))
        if len(date_cols) == 1:
            return date_cols[0]
    return None

def _v1__coded_conditions(objective, dfs, dico):
    low = objective.lower()
    out = []
    for f, col, desc in _v1__dico_rows(dico):
        name = _v1_stem(f).lower()
        if name not in dfs or col not in dfs[name].columns:
            continue
        match = re.search('coded\\s*\\((.*?)\\)', desc, re.I)
        if not match:
            continue
        for item in match.group(1).split(','):
            parts = item.strip().split(' ', 1)
            if len(parts) != 2:
                continue
            code, label = (parts[0], parts[1].strip())
            pos = low.find(label.lower())
            if pos >= 0:
                out.append((name, col, 'eq', code, pos))
    return out

def _v1__parse_table_conditions(objective, dfs, dico):
    o = ' '.join(objective.lower().split())
    out, date_spans = ([], [])
    date_re = '(on or after|on or before|after|before)\\s+((?:january|february|march|april|may|june|july|august|september|october|november|december)\\s+\\d{1,2},?\\s+\\d{4}|\\d{1,2}[/-]\\d{1,2}[/-]\\d{4}|\\d{4}-\\d{2}-\\d{2})'
    for m in re.finditer(date_re, o, re.I):
        ctx = _v1__condition_context(o[:m.start()])
        if re.search('\\bborn\\b', o[max(0, m.start() - 35):m.start()]):
            ctx = 'date of birth'
        col = _v1__condition_column(ctx, dfs, dico, date_like=True)
        if col:
            val = pd.to_datetime(m.group(2), dayfirst=bool(re.search('day/month|dd/mm|d/m/y|day-first|day first', o)), errors='coerce')
            if not pd.isna(val):
                op = {'on or after': 'date_ge', 'on or before': 'date_le', 'after': 'date_gt', 'before': 'date_lt'}[m.group(1)]
                out.append((col[0], col[1], op, val, m.start()))
                date_spans.append(m.span())
    comparison_re = '(strictly greater than|strictly less than|greater than|less than|at least|at most|no more than|no less than|not equal to|different from|equal to|equals?|above|below)\\s+(-?\\d+(?:\\.\\d+)?)'
    for m in re.finditer(comparison_re, o, re.I):
        if any((a <= m.start() < b for a, b in date_spans)):
            continue
        ctx = _v1__condition_context(o[:m.start()])
        col = _v1__condition_column(ctx, dfs, dico)
        if not col:
            continue
        op = m.group(1)
        if op in ('not equal to', 'different from'):
            op = 'ne'
        elif op in ('equal', 'equals', 'equal to'):
            op = 'eq'
        else:
            op = 'gt' if 'greater than' in op or op == 'above' else 'lt' if 'less than' in op or op == 'below' else 'ge' if op in ('at least', 'no less than') else 'le'
        out.append((col[0], col[1], op, float(m.group(2)), m.start()))
    missing_re = '\\b(is not missing|is not blank|is not null|is missing|is blank|is null|is not recorded|is recorded)\\b'
    for m in re.finditer(missing_re, o, re.I):
        ctx = _v1__condition_context(o[:m.start()])
        col = _v1__condition_column(ctx, dfs, dico)
        if col:
            op = 'not_missing' if re.search('not|recorded', m.group(1)) and 'not recorded' not in m.group(1) else 'is_missing'
            if m.group(1) in ('is recorded',):
                op = 'not_missing'
            out.append((col[0], col[1], op, None, m.start()))
    for m in re.finditer('\\bbetween\\s+(-?\\d+(?:\\.\\d+)?)\\s+and\\s+(-?\\d+(?:\\.\\d+)?)', o):
        ctx = _v1__condition_context(o[:m.start()])
        col = _v1__condition_column(ctx, dfs, dico)
        if col and dfs[col[0]][col[1]].dtype.kind in 'fiub':
            out.append((col[0], col[1], 'between', (float(m.group(1)), float(m.group(2))), m.start()))
    for m in re.finditer('\\bborn in\\s+(\\d{4})\\b', o):
        ctx = 'date of birth'
        col = _v1__condition_column(ctx, dfs, dico, date_like=True)
        if col:
            out.append((col[0], col[1], 'year_eq', int(m.group(1)), m.start()))
    for m in re.finditer("['\\u2018\\u201c]([^'\\u2019\\u201d]{1,80})['\\u2019\\u201d]", o):
        before, after = (o[:m.start()], o[m.end():])
        op_match = re.search('\\b(is\\s+not|is|does\\s+not\\s+equal|not\\s+equal\\s+to|different\\s+from|equals?|equal\\s+to)\\s*$', before)
        if op_match and len(before) - op_match.start() <= 50:
            ctx = _v1__condition_context(before[:op_match.start()])
            category_op = 'ne' if re.search('not|different', op_match.group(1)) else 'eq'
        else:
            fm = re.match('\\s+in\\s+(?:the\\s+)?(.+?)\\s+field\\b', after)
            if not fm:
                fm = re.match('\\s+as\\s+(?:the\\s+)?(.+?)(?=[?.;,]|$)', after)
                if not fm:
                    continue
            ctx = fm.group(1)
            category_op = 'eq'
        value = m.group(1).strip()
        if value.lower() in _v1_NA_TOKENS or re.search('(?:separat|missing value|not recorded|not available)[^.;]{0,60}$', before):
            continue
        col = _v1__condition_column(ctx, dfs, dico)
        if col:
            out.append((col[0], col[1], category_op, value, m.start()))
    out.extend(_v1__coded_conditions(o, dfs, dico))
    category_candidates = []
    quoted_spans = [m.span() for m in re.finditer("['\\u2018\\u201c]([^'\\u2019\\u201d]{1,80})['\\u2019\\u201d]", o)]
    for name, df in dfs.items():
        for col in df.columns:
            series = df[col]
            if series.dtype.kind not in 'OUS' or series.nunique(dropna=True) > 50:
                continue
            for raw_value in series.dropna().astype(str).str.strip().unique():
                value = raw_value.strip()
                if len(value) < 3 or value.lower() in _v1_NA_TOKENS or re.fullmatch('[-+]?\\d+(?:\\.\\d+)?', value):
                    continue
                value_re = re.compile(f'(?<![a-z0-9]){re.escape(value.lower())}(?![a-z0-9])')
                for vm in value_re.finditer(o):
                    if any((a <= vm.start() < b for a, b in quoted_spans)):
                        continue
                    prefix = o[max(0, vm.start() - 100):vm.start()]
                    operator = re.search('\\b(is|equals?|in|from|at)\\s+(?:(?:the|a|an)\\s+)?$', prefix)
                    if not operator:
                        continue
                    ctx = _v1__condition_context(prefix[:operator.start()])
                    resolved = _v1__condition_column(ctx, dfs, dico)
                    if resolved == (name, col):
                        category_candidates.append((name, col, 'eq', raw_value, vm.start()))
    unique_categories = {(c[0], c[1], str(c[3]).lower()) for c in category_candidates}
    if len(unique_categories) == 1:
        out.extend(category_candidates)
    months = {name.lower(): i for i, name in enumerate(('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'), 1)}
    for m in re.finditer('\\bin\\s+(January|February|March|April|May|June|July|August|September|October|November|December)\\s+(\\d{4})\\b', o, re.I):
        ctx = _v1__condition_context(o[:m.start()])
        col = _v1__condition_column(ctx, dfs, dico, date_like=True)
        if col:
            out.append((col[0], col[1], 'month_eq', (months[m.group(1).lower()], int(m.group(2))), m.start()))
    return out

def _v1__condition_mask(df, col, op, value):
    series = df[col]
    if op == 'is_missing':
        return series.isna()
    if op == 'not_missing':
        return series.notna()
    if op in ('eq', 'ne') and pd.api.types.is_numeric_dtype(series):
        number = pd.to_numeric(pd.Series([value]), errors='coerce').iloc[0]
        if pd.isna(number):
            return pd.Series(False, index=df.index) if op == 'eq' else series.notna()
        numeric = pd.to_numeric(series, errors='coerce')
        return numeric == number if op == 'eq' else series.notna() & (numeric != number)
    if op == 'eq':
        return series.astype(str).str.strip().str.lower() == str(value).strip().lower()
    if op == 'ne':
        return series.notna() & (series.astype(str).str.strip().str.lower() != str(value).strip().lower())
    if op in ('date_ge', 'date_le', 'date_gt', 'date_lt'):
        dates = pd.to_datetime(series, errors='coerce')
        return {'date_ge': dates >= value, 'date_le': dates <= value, 'date_gt': dates > value, 'date_lt': dates < value}[op]
    if op == 'year_eq':
        if series.dtype.kind == 'M':
            return series.dt.year == value
        return pd.to_numeric(series, errors='coerce') == value
    if op == 'month_eq':
        if series.dtype.kind != 'M':
            return pd.Series(False, index=df.index)
        month, year = value
        return (series.dt.month == month) & (series.dt.year == year)
    numeric = _v1__numeric_metric(series, force=True)
    if numeric is None:
        numeric = pd.to_numeric(series, errors='coerce')
    if op == 'between':
        low, high = value
        return numeric.between(low, high, inclusive='both')
    return {'gt': numeric > value, 'ge': numeric >= value, 'lt': numeric < value, 'le': numeric <= value, 'ne': numeric != value, 'eq': numeric == value}[op]

def _v1_table_filter_rule(objective, dfs, dico):
    """Handle common single-table filters and filtered aggregates when their fields resolve cleanly."""
    if len(dfs) != 1:
        return None
    o = ' '.join(objective.lower().split())
    if re.search('\\brank\\b|\\bfor each\\b', o):
        return None
    name, df = next(iter(dfs.items()))
    conditions = [c for c in _v1__parse_table_conditions(o, dfs, dico) if c[0] == name]
    if not conditions:
        return None
    proportion = re.search('what (percentage|fraction)', o)
    if proportion:
        marker = proportion.start()
        base = [c for c in conditions if c[4] < marker]
        target = [c for c in conditions if c[4] > marker]
        if not target:
            return None
        base_mask = pd.Series(True, index=df.index)
        for _, col, op, val, _ in base:
            base_mask &= _v1__condition_mask(df, col, op, val).fillna(False)
        eligible = df[base_mask].copy()
        explicit = re.search('(?:count only|only the)\\s+(?:pupils|students|rows|records|shipments)?\\s*(?:whose|with)?\\s*(.+?)\\s+is recorded', o)
        if explicit:
            resolved = _v1__resolve_dico_column(explicit.group(1), dfs, dico)
            if resolved:
                eligible = eligible[eligible[resolved[1]].notna()]
        elif re.search('\\b(?:missing|not recorded|not available)\\b', o):
            for col in dict.fromkeys((c[1] for c in target)):
                eligible = eligible[eligible[col].notna()]
        if eligible.empty:
            return None
        numerator = pd.Series(True, index=eligible.index)
        for _, col, op, val, _ in target:
            numerator &= _v1__condition_mask(eligible, col, op, val).fillna(False)
        ratio = float(numerator.sum()) / len(eligible)
        return ratio * 100 if proportion.group(1) == 'percentage' else ratio
    count_request = re.search("^\\s*(?:how many\\b|what(?: is|'s) the (?:number|count) of\\b|count the number of\\b)|\\bcount of\\b", o)
    if count_request:
        mask = pd.Series(True, index=df.index)
        for _, col, op, val, _ in conditions:
            mask &= _v1__condition_mask(df, col, op, val).fillna(False)
        if re.search('\\b(?:distinct|unique)\\b', o):
            if _v1__has_unparsed_filter_intent(o, conditions):
                return None
            target = _v1__distinct_target_column(o, dfs, dico, name)
            if not target:
                if not re.search('\\b(?:distinct|unique)\\s+(?:rows?|records?|entries|observations)\\b', o):
                    return None
                target = _v1__identifier_column(name, df.columns, dico)
            return float(df.loc[mask, target].nunique(dropna=True)) if target in df.columns else None
        return float(mask.sum())
    metric = re.search('\\b(standard deviation|std\\.?\\s*dev(?:iation)?|variance|range|spread|sum|total|average|mean|median|maximum|minimum|max|min)\\b', o)
    if not metric:
        return None
    op = metric.group(1)
    op = {'total': 'sum', 'average': 'mean', 'maximum': 'max', 'minimum': 'min', 'std dev': 'std', 'std. dev': 'std', 'std deviation': 'std', 'std. deviation': 'std', 'standard deviation': 'std'}.get(op, op)
    if op in ('std', 'variance', 'range', 'spread') and _v1__has_unparsed_filter_intent(o, conditions):
        return None
    tail = o[metric.end():].strip()
    tail = re.sub('^(?:of\\s+)?the\\s+', '', tail)
    bounds = [m.start() for m in re.finditer('\\bof the (?:orders?|shipments?|pupils?|students?|flats?|readings?|loans?|stays?|rows?|records?)\\b|\\bover all\\b|\\bamong\\b|\\bwith\\b|\\bwhose\\b|\\bon or after\\b|\\bon or before\\b|\\bat least\\b|\\bstrictly (?:greater|less) than\\b', tail)]
    for _, _, _, _, pos in conditions:
        if pos > metric.start():
            between = o[metric.end():pos]
            cues = list(re.finditer('\\b(?:for|where|with|among|whose)\\b', between))
            if cues:
                bounds.append(cues[-1].start())
    phrase = tail[:min(bounds)] if bounds else tail
    phrase = phrase.strip(' ,.;:()')
    resolved = _v1__resolve_dico_column(phrase, dfs, dico)
    if not resolved or resolved[0] != name:
        return None
    if op in ('std', 'variance', 'range', 'spread') and df[resolved[1]].dtype.kind not in 'fiub':
        return None
    mask = pd.Series(True, index=df.index)
    for _, col, cond_op, val, _ in conditions:
        mask &= _v1__condition_mask(df, col, cond_op, val).fillna(False)
    descriptions = _v1__table_descriptions(name, df.columns, dico)
    numeric_values = _v1__numeric_metric(df[resolved[1]], descriptions.get(resolved[1], ''))
    if numeric_values is None:
        return None
    values = numeric_values.loc[mask].dropna()
    if values.empty:
        return None
    if op == 'sum':
        return float(values.sum())
    if op == 'mean':
        return float(values.mean())
    if op == 'median':
        return float(values.median())
    if op == 'max':
        return float(values.max())
    if op == 'min':
        return float(values.min())
    if op in ('range', 'spread'):
        return float(values.max() - values.min())
    if op in ('std', 'variance'):
        ddof = 0 if re.search('\\bpopulation\\b', o) else 1
        if len(values) <= ddof:
            return None
        return float(values.std(ddof=ddof) if op == 'std' else values.var(ddof=ddof))
    return None

def _v1_table_derive_rule(objective, dfs, dico):
    """Evaluate a named ratio (numerator divided by denominator) and aggregate the derived values."""
    if len(dfs) != 1:
        return None
    o = ' '.join(objective.lower().split())
    name, df = next(iter(dfs.items()))
    expr = re.search('\\((?:the\\s+)?(.+?)\\s+divided by\\s+(?:the\\s+)?(.+?)\\)', o)
    if expr:
        num_phrase, den_phrase = (expr.group(1), expr.group(2))
    else:
        expr = re.search('\\bconsider the\\s+(.+?)\\s+per\\s+(.+?)(?:\\.|\\(|;|,)', o)
        if not expr:
            return None
        num_phrase, den_phrase = (expr.group(1), expr.group(2))
        if den_phrase in ('day of stay', 'day spent', 'day of the stay'):
            den_phrase = 'length of stay'
    num_col = _v1__resolve_dico_column(num_phrase, dfs, dico)
    den_col = _v1__resolve_dico_column(den_phrase, dfs, dico)
    if not num_col or not den_col or num_col[0] != name or (den_col[0] != name):
        return None
    request = re.search('\\bwhat\\s+(?:is|was)\\b', o)
    agg_text = o[request.end():] if request else o
    agg = re.search('\\b(median|minimum|maximum|mean|average|largest|smallest|min|max)\\b', agg_text)
    if not agg:
        return None
    operation = {'minimum': 'min', 'largest': 'max', 'maximum': 'max', 'smallest': 'min', 'average': 'mean'}.get(agg.group(1), agg.group(1))
    conditions = [c for c in _v1__parse_table_conditions(o, dfs, dico) if c[0] == name]
    has_filter = bool(re.search('strictly greater than|strictly less than|at least|at most|on or after|on or before|\\bafter\\b|\\bbefore\\b|\\bin [A-Z][a-z]+ \\d{4}', objective))
    if has_filter and (not conditions):
        return None
    mask = pd.Series(True, index=df.index)
    for _, col, op, val, _ in conditions:
        mask &= _v1__condition_mask(df, col, op, val).fillna(False)
    descriptions = _v1__table_descriptions(name, df.columns, dico)
    numerator = _v1__numeric_metric(df[num_col[1]], descriptions.get(num_col[1], ''))
    denominator = _v1__numeric_metric(df[den_col[1]], descriptions.get(den_col[1], ''))
    if numerator is None or denominator is None:
        return None
    ratio = _v1_safe_divide(numerator, denominator)
    if re.search('amount billed is in thousand euros', o) and re.search('\\bin euros per day\\b', o):
        ratio = ratio * 1000
    values = ratio[mask].replace([np.inf, -np.inf], np.nan).dropna()
    if values.empty:
        return None
    if operation == 'median':
        return float(values.median())
    if operation == 'min':
        return float(values.min())
    if operation == 'max':
        return float(values.max())
    if operation == 'mean':
        return float(values.mean())
    return None

def _v1_table_group_rule(objective, dfs, dico):
    """Aggregate by one dictionary field, then summarize the resulting group values."""
    if len(dfs) != 1:
        return None
    o = ' '.join(objective.lower().split())
    name, df = next(iter(dfs.items()))
    ranking = re.search('rank the (.+?) values by the (mean|average|sum|total|median) (.+?)(?:,|\\?|\\.| from highest| from lowest)', o)
    row_count = False
    summary_operation = None
    if ranking:
        group_phrase, agg, metric_phrase = ranking.groups()
        agg = {'average': 'mean', 'total': 'sum'}.get(agg, agg)
    else:
        nested = re.search('\\b(mean|average|median|sum|total|maximum|minimum|max|min|largest|smallest)\\s+of\\s+(?:the\\s+)?(sum|total|mean|average|median|maximum|minimum|max|min)\\s+(?:of\\s+)?(.+?)\\s+(?:for each|for every|within each|in each|per|by)\\s+(.+?)(?:\\.|;|,|\\?|\\s+consider only\\b)', o)
        if nested:
            summary_operation, agg, metric_phrase, group_phrase = nested.groups()
            summary_operation = {'average': 'mean', 'total': 'sum', 'maximum': 'max', 'minimum': 'min', 'largest': 'max', 'smallest': 'min'}.get(summary_operation, summary_operation)
            agg = {'total': 'sum', 'average': 'mean', 'maximum': 'max', 'minimum': 'min'}.get(agg, agg)
        else:
            grouped = re.search('\\b(sum|total|mean|average|median|maximum|minimum|max|min)\\s+(?:of\\s+)?(.+?)\\s+(?:for each|for every|within each|in each|per|by)\\s+(.+?)(?:\\.|;|,|\\?|\\s+consider only\\b)', o)
            if not grouped:
                counted = re.search('\\b(?:compute|calculate|find|determine)\\s+(?:the\\s+)?number of\\s+(.+?)\\s+(?:for each|for every|within each|in each|per|by)\\s+(.+?)(?:\\.|;|,|\\?|\\s+consider only\\b)', o)
                if not counted:
                    return None
                metric_phrase, group_phrase = counted.groups()
                agg, row_count = ('count', True)
            else:
                agg, metric_phrase, group_phrase = grouped.groups()
                agg = {'total': 'sum', 'average': 'mean', 'maximum': 'max', 'minimum': 'min'}.get(agg, agg)
    metric = None if row_count else _v1__resolve_dico_column(metric_phrase, dfs, dico)
    group = _v1__resolve_dico_column(group_phrase, dfs, dico)
    if not group:
        year_group = re.match('\\s*year\\s+of\\s+(.+?)\\s*', group_phrase)
        if year_group:
            candidate = _v1__resolve_dico_column(year_group.group(1), dfs, dico)
            if candidate and candidate[0] == name:
                desc = _v1__table_descriptions(name, df.columns, dico).get(candidate[1], '')
                if df[candidate[1]].dtype.kind == 'M' or re.search('\\bdate\\b', desc, re.I):
                    group = candidate
            if not group:
                date_cols = [c for c in df.columns if df[c].dtype.kind == 'M']
                if len(date_cols) == 1:
                    group = (name, date_cols[0])
    if not row_count and (not metric) or not group or (metric and metric[0] != name) or (group[0] != name) or (metric and metric[1] == group[1]):
        return None
    conditions = [c for c in _v1__parse_table_conditions(o, dfs, dico) if c[0] == name]
    rank_request = re.search('\\bwhat is the rank of\\b', o)
    if rank_request:
        conditions = [c for c in conditions if c[4] < rank_request.start()]
    mask = pd.Series(True, index=df.index)
    for _, col, op, val, _ in conditions:
        mask &= _v1__condition_mask(df, col, op, val).fillna(False)
    groups = df.loc[mask, [group[1]]].copy()
    key = groups[group[1]]
    if re.search('\\byear\\b', group_phrase):
        if key.dtype.kind == 'M':
            key = key.dt.year
        else:
            numeric = pd.to_numeric(key, errors='coerce')
            if numeric.notna().any():
                key = numeric
    groups['__group'] = key
    if row_count:
        grouped_values = groups.dropna(subset=['__group']).groupby('__group').size()
    else:
        descriptions = _v1__table_descriptions(name, df.columns, dico)
        vals = _v1__numeric_metric(df[metric[1]], descriptions.get(metric[1], ''))
        if vals is None:
            return None
        groups['__value'] = vals.loc[mask]
        grouped_values = groups.dropna(subset=['__group', '__value']).groupby('__group')['__value'].agg(agg)
    if grouped_values.empty:
        return None
    if rank_request:
        target_m = re.search('\\bwhat is the rank of\\s+(?:the\\s+)?(.+?)(?:\\?|\\.|$)', o)
        if not target_m:
            return None
        target_text = target_m.group(1).strip()
        target = None
        for c_name, c_col, c_op, c_val, c_pos in _v1__coded_conditions(o, dfs, dico):
            if c_name == name and c_col == group[1] and (c_pos >= rank_request.start()):
                target = c_val
                break
        if target is None:
            target = target_text
        labels = grouped_values.index
        if pd.api.types.is_numeric_dtype(labels):
            try:
                target_value = float(target)
                target_value = int(target_value) if target_value.is_integer() else target_value
            except Exception:
                return None
        else:
            target_value = str(target).strip().lower()
            matching = [x for x in labels if str(x).strip().lower() == target_value]
            if not matching:
                return None
            target_value = matching[0]
        if target_value not in labels:
            return None
        descending = 'highest' in o or 'highest (rank 1)' in o
        return float(grouped_values.rank(method='min', ascending=not descending).loc[target_value])
    if summary_operation:
        vals = pd.to_numeric(grouped_values, errors='coerce').dropna()
        if vals.empty:
            return None
        return float({'mean': vals.mean, 'median': vals.median, 'sum': vals.sum, 'max': vals.max, 'min': vals.min}[summary_operation]())
    if re.search('difference between the largest and (?:the )?smallest', o):
        return float(grouped_values.max() - grouped_values.min())
    if re.search('\\b(?:smallest|lowest|minimum) of these values\\b', o):
        return float(grouped_values.min())
    if re.search('\\b(?:largest|highest|maximum) of these values\\b', o):
        return float(grouped_values.max())
    if re.search('\\b(?:range|spread)\\b.{0,30}\\b(?:these|the)?\\s*(?:group )?values\\b', o):
        return float(grouped_values.max() - grouped_values.min())
    return None

def _v1_table_join_rule(objective, dfs, dico):
    """Join one fact table to one lookup table, filter, and aggregate a value from the fact table."""
    if len(dfs) != 2:
        return None
    o = ' '.join(objective.lower().split())
    group_request = re.search('\\bgroup\\s+(?:the\\s+)?(.+?)\\s+by\\s+(?:the\\s+)?(.+?)\\s+and\\s+compute\\s+(?:the\\s+)?(mean|average|median|sum|total|maximum|minimum|max|min)\\s+(.+?)\\s+per group\\b', o)
    if not group_request and (not re.search('\\bbelong to\\b|\\bwhose\\b', o)):
        return None
    notes = _v1_align_keys(dfs, dico)
    key_note = next((n for n in notes if 'Join key:' in n), None)
    if not key_note:
        return None
    m = re.search("dfs\\['([^']+)'\\]\\['([^']+)'\\] matches dfs\\['([^']+)'\\]\\['([^']+)'\\]", key_note)
    if not m:
        return None
    fact, fact_key, dim, dim_key = m.groups()
    if fact == dim:
        return None
    conditions = _v1__parse_table_conditions(o, dfs, dico)
    frames = {name: df.copy() for name, df in dfs.items()}
    for name, col, op, val, _ in conditions:
        if name in frames:
            mask = _v1__condition_mask(frames[name], col, op, val).fillna(False)
            frames[name] = frames[name][mask]
    if frames[fact].empty or frames[dim].empty:
        return None
    if group_request:
        clean_objective = re.sub('\\b[\\w.-]+\\.csv contains duplicated rows\\b[^.;?]*(?:[.;?]|$)', ' ', o)
        if _v1__has_unparsed_filter_intent(clean_objective, conditions):
            return None
        _, group_phrase, agg, metric_phrase = group_request.groups()
        group_col = _v1__resolve_dico_column(group_phrase, dfs, dico)
        metric_col = _v1__resolve_dico_column(metric_phrase, dfs, dico)
        if not group_col or group_col[0] != dim or (not metric_col) or (metric_col[0] != fact):
            return None
        outer = re.search('\\b(largest|highest|maximum|max|smallest|lowest|minimum|min)\\s+value\\b', o)
        if not outer:
            return None
        left = frames[fact].copy()
        left['__join'] = left[fact_key]
        descriptions = _v1__table_descriptions(fact, frames[fact].columns, dico)
        metric_values = _v1__numeric_metric(left[metric_col[1]], descriptions.get(metric_col[1], ''))
        if metric_values is None:
            return None
        left['__metric'] = metric_values
        right = frames[dim][[dim_key, group_col[1]]].copy()
        right = right.rename(columns={dim_key: '__join', group_col[1]: '__group'})
        joined = left.merge(right, on='__join', how='inner')
        values = joined.dropna(subset=['__group', '__metric'])
        if values.empty:
            return None
        agg_name = {'average': 'mean', 'total': 'sum', 'maximum': 'max', 'minimum': 'min'}.get(agg, agg)
        grouped_values = values.groupby('__group')['__metric'].agg(agg_name)
        if grouped_values.empty:
            return None
        return float(grouped_values.max() if outer.group(1) in ('largest', 'highest', 'maximum', 'max') else grouped_values.min())
    if re.search('\\bhow many orders\\b', o):
        order_col = _v1__resolve_dico_column('order number', dfs, dico)
        if not order_col or order_col[0] != fact:
            return None
        left = frames[fact].rename(columns={fact_key: '__join', order_col[1]: '__order'})
        right = frames[dim].rename(columns={dim_key: '__join'})
        joined = left.merge(right, on='__join', how='inner')
        return float(joined['__order'].nunique()) if not joined.empty else 0.0
    count_request = re.search('\\bhow many\\s+(.+?)\\s+(?:belong to|whose)\\b', o)
    if count_request:
        entity_tokens = _v1__toks(count_request.group(1))
        id_col = _v1__identifier_column(fact, frames[fact].columns, dico)
        id_desc = _v1__table_descriptions(fact, frames[fact].columns, dico).get(id_col, '') if id_col else ''
        if not id_col or not entity_tokens & (_v1__toks(fact) | _v1__toks(id_desc)):
            return None
        left = frames[fact].rename(columns={fact_key: '__join', id_col: '__entity'})
        right = frames[dim].rename(columns={dim_key: '__join'})
        joined = left.merge(right, on='__join', how='inner')
        return float(joined['__entity'].nunique(dropna=True)) if not joined.empty else 0.0
    op_m = re.search('\\b(mean|average|median|maximum|minimum|max|min|sum|total)\\b', o)
    if not op_m:
        return None
    operation = {'average': 'mean', 'maximum': 'max', 'minimum': 'min', 'total': 'sum'}.get(op_m.group(1), op_m.group(1))
    tail = o[op_m.end():].strip()
    row_phrase = re.search('\\bof the (?:orders?|order_lines?|pupils?|students?|flats?|readings?|shipments?|loans?|stays?)\\b', tail)
    metric_phrase = tail[:row_phrase.start()] if row_phrase else tail
    metric_phrase = metric_phrase.strip(' ,.;:()')
    metric = _v1__resolve_dico_column(metric_phrase, dfs, dico)
    if not metric or metric[0] != fact:
        metric = _v1__resolve_dico_column(tail, dfs, dico)
    if not metric or metric[0] != fact:
        return None
    left = frames[fact].rename(columns={fact_key: '__join', metric[1]: '__metric'})
    right = frames[dim].rename(columns={dim_key: '__join'})
    joined = left.merge(right, on='__join', how='inner')
    descriptions = _v1__table_descriptions(fact, frames[fact].columns, dico)
    values = _v1__numeric_metric(joined['__metric'], descriptions.get(metric[1], ''))
    if values is None:
        return None
    values = values.dropna()
    if values.empty:
        return None
    if operation == 'mean':
        return float(values.mean())
    if operation == 'median':
        return float(values.median())
    if operation == 'max':
        return float(values.max())
    if operation == 'min':
        return float(values.min())
    if operation == 'sum':
        return float(values.sum())
    return None

def _v1__row_filter(df, col, year, how):
    return df[df[col] >= year] if how == 'later' else df[df[col] <= year]

def _v1_fit_rules(objective, dfs, dico):
    """Renvoie un nombre ou None (None = pas sûr : on laisse le modèle faire)."""
    import itertools
    o = ' '.join(objective.split())
    if 'train' not in dfs or 'test' not in dfs:
        return None
    tr, te = (dfs['train'], dfs['test'])
    kind = 'logistic' if re.search('logistic regression', o) else 'linear' if re.search('least-squares', o) else None
    if kind is None:
        return None
    desc = {}
    for f, c, d in _v1__dico_rows(dico):
        if f.lower().startswith('train'):
            desc[c] = d
    target = [c for c in tr.columns if c not in te.columns]
    if len(target) != 1:
        return None
    target = target[0]
    idc = te.columns[0]
    excl = set()
    for q in re.findall('"([^"]+)"', o):
        excl |= {c for c, d in desc.items() if d.lower() == q.strip().lower()}
    ycols = [c for c in tr.columns if 'year' in desc.get(c, '').lower() and tr[c].dtype.kind in 'fi']
    if kind == 'linear':
        m = re.search('intercept of the (.+?) on the (.+?), using', o)
    else:
        m = re.search('intercept, of (.+?) \\(.*?\\) on the (.+?), using', o)
    if not m:
        return None
    items = [x.strip() for x in re.split(',\\s*|\\s+and\\s+', m.group(2)) if x.strip()]
    cands = [c for c in tr.columns if c in te.columns and c != idc and (c not in excl) and (c != target) and (c not in ycols) and (tr[c].dtype.kind in 'fiub' or (tr[c].dtype.kind in 'OUS' and tr[c].nunique(dropna=True) <= 20))]
    if not items or len(items) > len(cands):
        return None
    best, best_s = (None, -1)
    for perm in itertools.permutations(range(len(cands)), len(items)):
        sc = sum((len(_v1__toks(items[i]) & _v1__toks(desc.get(cands[p], ''))) for i, p in enumerate(perm)))
        if sc > best_s:
            best, best_s = (perm, sc)
    if len(items) < len(cands):
        if any((not _v1__toks(items[i]) & _v1__toks(desc.get(cands[p], '')) for i, p in enumerate(best))):
            return None
    feats = [cands[p] for p in best]
    d = tr.copy()
    mf = re.search('train\\.csv with an? (.+?) of (\\d{4}) or (earlier|later)', o)
    if mf:
        if len(ycols) != 1:
            return None
        d = _v1__row_filter(d, ycols[0], int(mf.group(2)), mf.group(3))
    d = d.dropna(subset=feats + [target])
    if len(d) < len(feats) + 2:
        return None
    y = d[target]
    if kind == 'logistic':
        if y.dtype.kind not in 'fi':
            y = y.astype(str).str.strip().str.lower().isin(['y', 'yes', '1', 'true']).astype(float)
        y = y.astype(float)
    LR, LogR = _v1__make_models(True)
    model = (LR() if kind == 'linear' else LogR()).fit(d[feats], y)

    def predict(frame):
        X = frame[feats]
        return model.predict(X) if kind == 'linear' else model.predict_proba(X)[:, 1]
    mq = re.search('predicted [a-z ]+? for the \\w+ (\\d+) of test\\.csv', o)
    if mq:
        row = te[te[idc] == float(mq.group(1))]
        if len(row) != 1:
            return None
        return float(predict(row)[0])
    if re.search('intercept of the fitted model', o):
        return float(np.ravel(model.intercept_)[0])
    if re.search('R². of the fit|R2 of the fit|R² of the fit', o) and kind == 'linear':
        return float(model.score(d[feats], y))
    mc = re.search('coefficient of the (.+?) \\(', o)
    if mc:
        item = mc.group(1).strip()
        sc = [(len(_v1__toks(item) & _v1__toks(desc.get(f, ''))), f) for f in feats]
        sc.sort(reverse=True)
        if not sc or sc[0][0] == 0 or (len(sc) > 1 and sc[0][0] == sc[1][0]):
            return None
        coef_idx = getattr(model, '_dsh_numeric_coef_idx', {}).get(sc[0][1])
        if coef_idx is None:
            return None
        return float(np.ravel(model.coef_)[coef_idx])
    mn = re.search('How many [a-z ]+ of test\\.csv have a predicted probability of [a-z ]+ of at least ([\\d.]+)', o)
    if mn and kind == 'logistic':
        t = te.dropna(subset=feats)
        return float((predict(t) >= float(mn.group(1))).sum())
    mm = re.search("mean predicted [a-z ]+ over the \\w+ of test\\.csv (?:with an? (.+?) of (\\d{4}) or (earlier|later)|whose (.+?) is '([^']+)')", o)
    if mm and kind == 'linear':
        t = te.dropna(subset=feats)
        if mm.group(2):
            if len(ycols) != 1:
                return None
            t = _v1__row_filter(t, ycols[0], int(mm.group(2)), mm.group(3))
        else:
            key = mm.group(4)
            cols = [c for c in te.columns if key.lower() in desc.get(c, '').lower() and te[c].dtype.kind not in 'fiM']
            if len(cols) != 1:
                return None
            t = t[t[cols[0]].astype(str).str.strip().str.lower() == mm.group(5).strip().lower()]
        if len(t) == 0:
            return None
        return float(predict(t).mean())
    return None


class Agent:
    def __init__(self):
        import torch

        self.torch = torch
        import dsh                                  # uploadé avec agent.py ; aucun autre chargement de modèle
        llm = dsh.load_llm(MODEL)                   # LLM(model, tok, device) : poids du cache de la plateforme, révision épinglée
        self.model, self.tok = llm.model, llm.tok
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model.eval()
        # imports lourds ici (60 s autorisées) et non au premier solve() (40 s) : scipy, sklearn et sympy coûtent plusieurs secondes
        for mod in ("scipy.stats", "sklearn.linear_model", "sklearn.metrics", "sympy"):
            try:
                __import__(mod)
            except Exception:
                pass
        if TABLE_MODE in ("v1", "v1_then_v2", "v1_arbitrate", "v1_v2wins", "v1_then_v2_v2wins"):
            print("[init] tables :", TABLE_MODE if _load_v1() is not None else "v2 (bloc v1 indisponible)")
        self.generate(["warmup"], 1, 0.0, max_new=4, raw=True)
        print(f"[init] modèle {MODEL} sur {getattr(self.model, 'device', '?')}, échantillons {NSAMP}, budget {BUDGET}s")

    # -- génération par lots : retourne, pour chaque prompt, n textes
    def generate(self, prompts, n, temp, max_new=MAX_NEW, raw=False, max_time=None):
        torch = self.torch
        # Qwen3 réfléchit par défaut (des milliers de tokens) : on coupe le mode « thinking » comme le fait dsh.LLM.chat
        tkw = {"enable_thinking": False} if "enable_thinking" in (getattr(self.tok, "chat_template", None) or "") else {}
        texts = prompts if raw else [
            self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **tkw) + "```python\n"
            for m in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True).to(self.model.device)
        kw = dict(max_new_tokens=max_new, pad_token_id=self.tok.pad_token_id,
                  stop_strings=["```"], tokenizer=self.tok)
        if max_time is not None:
            kw["max_time"] = max(1.0, float(max_time))
        if temp > 0:
            kw.update(do_sample=True, temperature=temp, top_p=0.95, num_return_sequences=n)
        else:
            kw.update(do_sample=False)
        with torch.no_grad():
            out = self.model.generate(**enc, **kw)
        dec = self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        k = n if temp > 0 else 1
        return [dec[i * k:(i + 1) * k] for i in range(len(texts))]

    def solve(self, tasks):
        try:
            self.torch.manual_seed(0)   # échantillonnage reproductible
        except Exception:
            pass
        t0 = time.time()
        left = lambda: BUDGET - (time.time() - t0)
        prep = []
        for t in tasks:
            try:
                if isinstance(t.get("objective"), str) and _THOUSANDS_RE.search(t["objective"]):
                    t = dict(t, objective=_norm_thousands(t["objective"]))
                files = t.get("files") or []
                kind = route(files)
                if kind == "table" and TABLE_MODE in ("v1", "v1_then_v2", "v1_arbitrate", "v1_v2wins", "v1_then_v2_v2wins"):
                    V1 = _load_v1()
                    if V1 is not None:
                        prep.append(_prep_table_v1(V1, t, files, TABLE_MODE in ("v1_then_v2", "v1_then_v2_v2wins"),
                                                   {"v1_arbitrate": "llm", "v1_v2wins": "v2",
                                                    "v1_then_v2_v2wins": "v2"}.get(TABLE_MODE, False)))
                        continue
                dfs, dico = load_task(t["objective"], files)
                if kind == "nofile":
                    try:
                        rv = None if _rules_off("nofile") else calc_rules(t["objective"])
                    except Exception:
                        rv = None
                    if rv is not None and math.isfinite(rv):
                        if DEBUG:
                            print(f"[debug] {t['id']} calcul déterministe -> {rv}")
                        prep.append(dict(id=t["id"], obj=t["objective"], dfs=dfs, msgs=None, c=[(rv, 100.0)], ok=False))
                        continue
                if kind == "table" and not _rules_off("table"):
                    try:
                        rv = table_extra_rule(t["objective"], dfs, dico)
                    except Exception:
                        rv = None
                    if rv is not None and math.isfinite(rv):
                        if DEBUG:
                            print(f"[debug] {t['id']} gabarit supplémentaire -> {rv}")
                        prep.append(dict(id=t["id"], obj=t["objective"], dfs=dfs, msgs=None, c=[(rv, 100.0)], ok=False))
                        continue
                    try:
                        rv = table_stat_rule(t["objective"], dfs, dico)
                    except Exception:
                        rv = None
                    if rv is not None and math.isfinite(rv):
                        if DEBUG:
                            print(f"[debug] {t['id']} statistique déterministe -> {rv}")
                        prep.append(dict(id=t["id"], obj=t["objective"], dfs=dfs, msgs=None, c=[(rv, 100.0)], ok=False))
                        continue
                    try:
                        rv = table_group_rule(t["objective"], dfs, dico)
                    except Exception:
                        rv = None
                    if rv is not None and math.isfinite(rv):
                        if DEBUG:
                            print(f"[debug] {t['id']} agrégation déterministe -> {rv}")
                        prep.append(dict(id=t["id"], obj=t["objective"], dfs=dfs, msgs=None, c=[(rv, 100.0)], ok=False))
                        continue
                    try:
                        rv = table_join_rule(t["objective"], dfs, dico)
                    except Exception:
                        rv = None
                    if rv is not None and math.isfinite(rv):
                        if DEBUG:
                            print(f"[debug] {t['id']} jointure déterministe -> {rv}")
                        prep.append(dict(id=t["id"], obj=t["objective"], dfs=dfs, msgs=None, c=[(rv, 100.0)], ok=False))
                        continue
                    try:
                        rv = table_derive_rule(t["objective"], dfs, dico)
                    except Exception:
                        rv = None
                    if rv is not None and math.isfinite(rv):
                        if DEBUG:
                            print(f"[debug] {t['id']} ratio déterministe -> {rv}")
                        prep.append(dict(id=t["id"], obj=t["objective"], dfs=dfs, msgs=None, c=[(rv, 100.0)], ok=False))
                        continue
                    try:
                        rv = table_filter_rule(t["objective"], dfs, dico)
                    except Exception:
                        rv = None
                    if rv is not None and math.isfinite(rv):
                        if DEBUG:
                            print(f"[debug] {t['id']} filtre déterministe -> {rv}")
                        prep.append(dict(id=t["id"], obj=t["objective"], dfs=dfs, msgs=None, c=[(rv, 100.0)], ok=False))
                        continue
                if kind == "fit":
                    try:
                        rv = None if _rules_off("fit") else fit_rules(t["objective"], dfs, dico)
                    except Exception:
                        rv = None
                    if rv is not None and math.isfinite(rv):
                        if DEBUG:
                            print(f"[debug] {t['id']} règles -> {rv}")
                        prep.append(dict(id=t["id"], obj=t["objective"], dfs=dfs, msgs=None, c=[(rv, 100.0)], ok=False))
                        continue
                notes = ((align_keys(dfs, dico) if len(dfs) >= 2 else []) + entity_hints(t["objective"], dico)) if kind == "table" else (fit_hints(t["objective"], dico, dfs) if kind == "fit" else [])
                if kind == "nofile" and NOFILE_HINTS:
                    try:
                        notes = list(notes) + _nofile_hints(t["objective"])
                    except Exception:
                        pass
                # APPEL MIS À JOUR avec objective passé en paramètre
                msgs = build_messages(kind, build_user(kind, t["objective"], dfs, dico, notes), t["objective"], len(dfs))
                if DEBUG:
                    print(f"[debug] {t['id']} kind={kind} files={files} tables={ {k: v.shape for k, v in dfs.items()} } dico={len(dico)}c")
                if DEBUG and len(dfs) >= 2:
                    print("[debug] schéma vu par le modèle :\n" + msgs[-1]["content"].split("Objective:")[0][:1500])
                prep.append(dict(id=t["id"], obj=t["objective"], dfs=dfs, msgs=msgs, c=[], ok=True))
            except Exception:
                prep.append(dict(id=t.get("id"), obj="", dfs={}, msgs=None, c=[], ok=False))

        def absorb(p, o, weight):
            """Exécute une sortie du modèle ; garde la valeur ou mémorise l'échec. False si plus de temps."""
            if left() < 1.5:
                return False
            code = extract_code(o)
            mod = p.get("mod")
            if mod is not None:                 # tâche table v1 : son propre exécuteur
                v = mod.run_code(code, p["dfs"], min(EXEC_TIMEOUT, left()), p["obj"])
                err, state = mod.LAST_ERR, mod.LAST_STATE
            else:
                v = run_code(code, p["dfs"], min(EXEC_TIMEOUT, left()), p["obj"])
                err, state = LAST_ERR, LAST_STATE
            if v is None:
                p["fail"] = (code, err, state)
                if "```" not in o:
                    p["trunc"] = o              # sortie coupée avant la fin du bloc de code
            else:
                hc = p.get("hint_cols")
                if HINT_WEIGHT and hc:
                    frac = sum(_code_uses(code, c) for c in hc) / len(hc)
                    weight = weight * (0.5 + frac)  # 0,5 si aucune colonne repérée n'est utilisée, 1,5 si toutes
                p["c"].append((v, weight))
            if DEBUG:
                err = err if v is None else ""
                print(f"[debug] {p['id']} -> {v}  err={err}  (code: {len(code)} car.)\n{code[:350]}\n...\n{code[-150:]}\n" + "-" * 40)
            return True

        def run_round(n, temp, weight, max_new):
            live = [p for p in prep if p["ok"]]
            if not live or left() < 4:
                return
            try:
                gens = self.generate([p["msgs"] for p in live], n, temp, max_new=max_new, max_time=left() - 2)
            except Exception:
                if DEBUG:
                    import traceback
                    traceback.print_exc()
                return
            if DEBUG:
                print(f"[debug] génération {time.time() - t0:.1f}s écoulées, reste {left():.1f}s")
            for p, outs in zip(live, gens):
                for o in outs:
                    if not absorb(p, o, weight):
                        return

        def run_continue():
            """Les programmes coupés par max_new tokens reprennent là où ils se sont arrêtés."""
            todo = [p for p in prep if p["ok"] and not p["c"] and p.get("trunc")]
            if not todo or left() < 6:
                return
            tkw = {"enable_thinking": False} if "enable_thinking" in (getattr(self.tok, "chat_template", None) or "") else {}
            texts = [self.tok.apply_chat_template(p["msgs"], tokenize=False, add_generation_prompt=True, **tkw)
                     + "```python\n" + p["trunc"] for p in todo]
            try:
                gens = self.generate(texts, 1, 0.0, max_new=MAX_NEW, raw=True, max_time=min(8.0, left() - 2))
            except Exception:
                return
            for p, outs in zip(todo, gens):
                o = p.pop("trunc") + outs[0]
                if not absorb(p, o, 1.5):
                    return

        def run_repair():
            """Seconde chance : on renvoie au modèle son code, l'erreur et les vraies colonnes."""
            todo = [p for p in prep if p["ok"] and not p["c"] and p.get("fail")]
            if not todo or left() < 6:
                return
            msgs = []
            for p in todo:
                code, err, state = p["fail"]
                cols = "; ".join(f"{k}: {', '.join(map(str, d.columns))}" for k, d in p["dfs"].items())
                extra = ""
                # PATCH 5: Directives de self-repair enrichies
                if "Unalignable" in err or "IndexingError" in err:
                    extra = " Merge the tables first, then filter on the merged frame, never index one table with a mask from another."
                elif "KeyError" in err and p["dfs"]:
                    extra = " Use only the exact column names listed (after a merge, duplicated names get suffixes _x and _y)."
                elif "SyntaxError" in err:
                    extra = " Ensure all parentheses and brackets are properly closed, and strings are terminated."
                elif "AttributeError" in err and "round" in err:
                    extra = " Do not round the answer. Just return the raw float value."
                data_state = f"\nDataFrame state after the failure:\n{state}" if state else ""
                msgs.append(p["msgs"] + [
                    {"role": "assistant", "content": "```python\n" + code + "\n```"},
                    {"role": "user", "content": f"That code failed: {err}\nAvailable columns -> {cols}.{extra}"
                                                 f"{data_state}\nUse the observed dtypes, missing rates, and sample rows to fix the data logic (including joins/coercion), not only syntax.\n"
                                                 "Write the corrected program (code only, `answer` must be a finite number)."}])
            try:
                gens = self.generate(msgs, 1, 0.0, max_new=MAX_NEW, max_time=min(8.0, left() - 2))
            except Exception:
                return
            for p, outs in zip(todo, gens):
                for o in outs:
                    if not absorb(p, o, 1.5):
                        return

        t_prep = time.time() - t0
        t1 = time.time()
        run_round(1, 0.0, 1.5, MAX_NEW)            # passe gloutonne (prioritaire)
        d1 = time.time() - t1
        
        t2 = time.time()
        run_continue()                             # suite des programmes coupés
        if not os.environ.get("DSH_NO_REPAIR"):
            run_repair()                           # réparation des plantages
        d2 = time.time() - t2
        
        t3 = time.time()
        if left() > 6:                             # OPTIMISATION 2: Vote adaptatif
            for p in prep:
                if not p["ok"] or not p["c"]:
                    continue
                # Si les précédentes passes (gloutonne/réparée) sont déjà d'accord, on arrête d'échantillonner cette tâche
                vals = [round(v, 4) for v, w in p["c"] if v is not None]
                if len(vals) >= 2 and len(set(vals)) == 1:
                    p["ok"] = False
                    
            nsamp_adapt = int(os.environ.get("DSH_SAMPLES_ADAPTIVE", NSAMP))
            live = [p for p in prep if p["ok"]]
            if TABLE_SAMPLES > nsamp_adapt and live and len(live) <= LIVE_MAX and any(p.get("mod") is not None for p in live):
                nsamp_adapt = TABLE_SAMPLES        # peu de tâches au LLM : plus d'échantillons pour un vote stable
            run_round(nsamp_adapt, 0.5, 1.0, MAX_NEW)
            if EXTRA_ROUND and left() > 10:
                for p in prep:                     # second tour seulement pour les votes serrés
                    if not p["ok"]:
                        continue
                    nd_ = decimals(p["obj"])
                    tally = collections.Counter()
                    for v_, w_ in p["c"]:
                        tally[round(v_, nd_ if nd_ is not None else 4)] += w_
                    tot = sum(tally.values())
                    p["ok"] = (not tot) or max(tally.values()) / tot < 0.6
                if any(p["ok"] for p in prep):
                    run_round(nsamp_adapt, 0.6, 1.0, MAX_NEW) 
            
        print(f"[solve] prep {t_prep:.1f}s | glouton {d1:.1f}s | réparation {d2:.1f}s | échantillons "
              f"{time.time() - t3:.1f}s | total {time.time() - t0:.1f}s")

        res = []
        for p in prep:
            try:
                nd = decimals(p["obj"])
                cands = p["c"]
                if SANITY and p.get("msgs") is not None:
                    try:
                        cands = _sanitize(p["obj"], cands)
                    except Exception:
                        cands = p["c"]
                if p.get("arb"):
                    r1, r2 = p["arb"]
                    key = lambda x: round(x, nd if nd is not None else 6)
                    s1 = sum(w for x, w in cands if key(x) == key(r1))
                    s2 = sum(w for x, w in cands if key(x) == key(r2))
                    cands = [(r2 if s2 > s1 else r1, 1.0)]
                v = vote(cands, nd)
                if nd is not None:
                    v = round(v, nd)
                res.append({"id": p["id"], "answer": float(v)})
            except Exception:
                res.append({"id": p["id"], "answer": 0.0})
        return res
