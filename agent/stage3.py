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


_NUM_RE = re.compile(r"^[-+]?\s*\d[\d.,\s]*\s*[A-Za-z%$€£°/²³µ.]*$")
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


def run_code(code, dfs, timeout, obj=""):
    """Exécute le code généré, renvoie un float ou None."""
    from scipy import stats
    from sklearn.metrics import r2_score
    unreg = bool(re.search(r"without regularis|no regularis|unregularis|unpenali|without penal", obj.lower()))
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
                exec(code, ns)
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


def calc_rules(objective):
    """Exact solvers for recurring, unambiguous calculation templates; None delegates to the LLM."""
    o = " ".join(objective.lower().split())

    # Sum a comma-separated list of explicitly enumerated receipts / values.
    m = re.search(
        r"\brecorded\s+(?:these|the following)\s+[^:]{0,100}:\s*"
        r"(\d+(?:\.\d+)?(?:\s*,\s*\d+(?:\.\d+)?){1,})\s*[.;]?\s*"
        r"(?:what is the total|find the total|calculate the total)\b", o)
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
    m = re.search(r"probability\s+(\d+(?:\.\d+)?)\s*[.;,]?\s*among\s+(\d+)\s+(?:parts?|trials?|items?)[\s\S]*?at least\s+(\d+)\s+(?:defective|success)", o)
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
        return principal * (1 + rate) ** years

    # First full year in which a compound-interest balance reaches a target threshold.
    m = re.search(
        r"\bamount of\s+([\d.]+)\s+euros? earns\s+([\d.]+)%\s+compound interest per year"
        r"[\s\S]{0,120}?\bafter how many full years\b[\s\S]{0,100}?\bfirst reach(?:es)? at least\s+([\d.]+)", o)
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
    return None


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
        elif w == "carrier":
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
        r"\b(?:distinct|unique)\s+(.+?)(?=\s+(?:are|is|in|among|for|from|where|whose|with|of the|across|over)\b|[?.;]|$)",
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


def _resolve_dico_column(phrase, dfs, dico):
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
        rf"\bvalue\s+(-?\d+(?:[.,]\d+)?)\s+in the\s+(.+?)\s+column\s+means?\s+(?:it\s+)?(?:is\s+)?{missing}",
        rf"\b(?:a|an|the)\s+(.+?)\s+(?:of|equal(?:s)?|is)\s+(-?\d+(?:[.,]\d+)?)\s+means?\s+(?:it\s+)?(?:was\s+|is\s+)?{missing}",
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
    if len(dfs) != 1:
        return None
    o = " ".join(objective.lower().split())
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
        r"\bhow many\b.*?\b(?:have|has)\s+no\s+(?:recorded\s+)?(.+?)(?=[?.;]|$)", o)
    if not missing_count:
        missing_count = re.search(
            r"\bhow many\b.*?\bis\s+(?:the\s+)?(.+?)\s+(?:missing|not recorded|unrecorded)\b", o)
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

    if re.search(r"\bhow many\b", o) and re.search(r"\b(?:have|has)\s+(?:a |an )?(?:recorded )", o):
        if re.search(r"\b(?:among|whose|with|born|on or after|on or before)\b", o):
            return None
        m = re.search(r"\b(?:have|has)\s+(?:a |an )?(?:recorded )?(.+?)(?:\?|\.|$)", o)
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
            rf"\b(standard deviation|std\.?\s*dev(?:iation)?|variance|range|spread|maximum|minimum|median|mean|average|sum)\s+(?:of\s+)?(?:the\s+)?(.+?)(?={scope_end}|\s*\(|[?.]|$)", o)
        if not m:
            return None
        op, phrase = m.group(1), m.group(2)
        operation = {
            "maximum": "max", "minimum": "min", "average": "mean", "std dev": "std",
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
    """Keep the field phrase nearest to a filter operator/value."""
    low = prefix.lower()
    markers = (" and ", " with ", " whose ", " have ", " has ", " among ", " where ", " of ", " over ", " born ")
    cuts = [(low.rfind(mark), len(mark)) for mark in markers if low.rfind(mark) >= 0]
    for pos, size in sorted(cuts, reverse=True):
        value = low[pos + size:].strip()
        value = re.sub(r"^(?:the|a|an)\s+", "", value)
        value = re.sub(r"\s+(?:of|the|a|an)$", "", value).strip()
        if value:
            return value[-80:]
    return low[-80:].strip()


def _condition_column(context, dfs, dico, date_like=False):
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


def _parse_table_conditions(objective, dfs, dico):
    o = " ".join(objective.lower().split())
    out, date_spans = [], []
    date_re = r"(on or after|on or before|after|before)\s+((?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{4}|\d{4}-\d{2}-\d{2})"
    for m in re.finditer(date_re, o, re.I):
        ctx = _condition_context(o[:m.start()])
        if re.search(r"\bborn\b", o[max(0, m.start()-35):m.start()]):
            ctx = "date of birth"
        col = _condition_column(ctx, dfs, dico, date_like=True)
        if col:
            val = pd.to_datetime(m.group(2), dayfirst=bool(re.search(r"day/month|dd/mm|d/m/y|day-first|day first", o)), errors="coerce")
            if not pd.isna(val):
                op = {"on or after": "date_ge", "on or before": "date_le", "after": "date_gt", "before": "date_lt"}[m.group(1)]
                out.append((col[0], col[1], op, val, m.start()))
                date_spans.append(m.span())

    comparison_re = r"(strictly greater than|strictly less than|greater than|less than|at least|at most|no more than|no less than|not equal to|different from|equal to|equals?|above|below)\s+(-?\d+(?:\.\d+)?)"
    for m in re.finditer(comparison_re, o, re.I):
        if any(a <= m.start() < b for a, b in date_spans):
            continue
        ctx = _condition_context(o[:m.start()])
        col = _condition_column(ctx, dfs, dico)
        if not col:
            continue
        op = m.group(1)
        if op in ("not equal to", "different from"):
            op = "ne"
        elif op in ("equal", "equals", "equal to"):
            op = "eq"
        else:
            op = "gt" if ("greater than" in op or op == "above") else ("lt" if ("less than" in op or op == "below") else ("ge" if op in ("at least", "no less than") else "le"))
        out.append((col[0], col[1], op, float(m.group(2)), m.start()))

    # Explicit missingness tests on a named field.
    missing_re = r"\b(is not missing|is not blank|is not null|is missing|is blank|is null|is not recorded|is recorded)\b"
    for m in re.finditer(missing_re, o, re.I):
        ctx = _condition_context(o[:m.start()])
        col = _condition_column(ctx, dfs, dico)
        if col:
            op = "not_missing" if re.search(r"not|recorded", m.group(1)) and "not recorded" not in m.group(1) else "is_missing"
            if m.group(1) in ("is recorded",):
                op = "not_missing"
            out.append((col[0], col[1], op, None, m.start()))

    # Closed numeric ranges such as "with a sale amount between 10 and 20".
    for m in re.finditer(r"\bbetween\s+(-?\d+(?:\.\d+)?)\s+and\s+(-?\d+(?:\.\d+)?)", o):
        ctx = _condition_context(o[:m.start()])
        col = _condition_column(ctx, dfs, dico)
        if col and dfs[col[0]][col[1]].dtype.kind in "fiub":
            out.append((col[0], col[1], "between", (float(m.group(1)), float(m.group(2))), m.start()))

    # A date's calendar year, as in "pupils born in 2015".
    for m in re.finditer(r"\bborn in\s+(\d{4})\b", o):
        ctx = "date of birth"
        col = _condition_column(ctx, dfs, dico, date_like=True)
        if col:
            out.append((col[0], col[1], "year_eq", int(m.group(1)), m.start()))

    # Exact quoted categories, including a field named after the quote: 'N' in the scholarship field.
    for m in re.finditer(r"['\u2018\u201c]([^'\u2019\u201d]{1,80})['\u2019\u201d]", o):
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
    quoted_spans = [m.span() for m in re.finditer(r"['\u2018\u201c]([^'\u2019\u201d]{1,80})['\u2019\u201d]", o)]
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
        col = _condition_column(ctx, dfs, dico, date_like=True)
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
    if len(dfs) != 1:
        return None
    o = " ".join(objective.lower().split())
    if re.search(r"\brank\b|\bfor each\b", o):
        return None
    name, df = next(iter(dfs.items()))
    conditions = [c for c in _parse_table_conditions(o, dfs, dico) if c[0] == name]
    if not conditions:
        return None
    proportion = re.search(r"what (percentage|fraction)", o)
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
        r"^\s*(?:how many\b|what(?: is|'s) the (?:number|count) of\b|count the number of\b)|\bcount of\b", o)
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


def table_derive_rule(objective, dfs, dico):
    """Evaluate a named ratio (numerator divided by denominator) and aggregate the derived values."""
    if len(dfs) != 1:
        return None
    o = " ".join(objective.lower().split())
    name, df = next(iter(dfs.items()))
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
    conditions = [c for c in _parse_table_conditions(o, dfs, dico) if c[0] == name]
    rank_request = re.search(r"\bwhat is the rank of\b", o)
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
        descending = "highest" in o or "highest (rank 1)" in o
        return float(grouped_values.rank(method="min", ascending=not descending).loc[target_value])

    if summary_operation:
        vals = pd.to_numeric(grouped_values, errors="coerce").dropna()
        if vals.empty:
            return None
        return float({"mean": vals.mean, "median": vals.median, "sum": vals.sum,
                      "max": vals.max, "min": vals.min}[summary_operation]())

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
        if not id_col or not (entity_tokens & (_toks(fact) | _toks(id_desc))):
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
    kind = "logistic" if re.search(r"logistic regression", o) else ("linear" if re.search(r"least-squares", o) else None)
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
        m = re.search(r"intercept of the (.+?) on the (.+?), using", o)
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
    mc = re.search(r"coefficient of the (.+?) \(", o)
    if mc:
        item = mc.group(1).strip()
        sc = [(len(_toks(item) & _toks(desc.get(f, ""))), f) for f in feats]
        sc.sort(reverse=True)
        if not sc or sc[0][0] == 0 or (len(sc) > 1 and sc[0][0] == sc[1][0]):
            return None
        coef_idx = getattr(model, "_dsh_numeric_coef_idx", {}).get(sc[0][1])
        if coef_idx is None:
            return None  # a categorical field has several one-hot coefficients, not one scalar slope
        return float(np.ravel(model.coef_)[coef_idx])
    mn = re.search(r"How many [a-z ]+ of test\.csv have a predicted probability of [a-z ]+ of at least ([\d.]+)", o)
    if mn and kind == "logistic":
        t = te.dropna(subset=feats)
        return float((predict(t) >= float(mn.group(1))).sum())
    mm = re.search(r"mean predicted [a-z ]+ over the \w+ of test\.csv (?:with an? (.+?) of (\d{4}) or (earlier|later)|whose (.+?) is '([^']+)')", o)
    if mm and kind == "linear":
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
                files = t.get("files") or []
                kind = route(files)
                dfs, dico = load_task(t["objective"], files)
                if kind == "nofile":
                    try:
                        rv = calc_rules(t["objective"])
                    except Exception:
                        rv = None
                    if rv is not None and math.isfinite(rv):
                        if DEBUG:
                            print(f"[debug] {t['id']} calcul déterministe -> {rv}")
                        prep.append(dict(id=t["id"], obj=t["objective"], dfs=dfs, msgs=None, c=[(rv, 100.0)], ok=False))
                        continue
                if kind == "table":
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
                        rv = fit_rules(t["objective"], dfs, dico)
                    except Exception:
                        rv = None
                    if rv is not None and math.isfinite(rv):
                        if DEBUG:
                            print(f"[debug] {t['id']} règles -> {rv}")
                        prep.append(dict(id=t["id"], obj=t["objective"], dfs=dfs, msgs=None, c=[(rv, 100.0)], ok=False))
                        continue
                notes = ((align_keys(dfs, dico) if len(dfs) >= 2 else []) + entity_hints(t["objective"], dico)) if kind == "table" else (fit_hints(t["objective"], dico, dfs) if kind == "fit" else [])
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
            v = run_code(code, p["dfs"], min(EXEC_TIMEOUT, left()), p["obj"])
            if v is None:
                p["fail"] = (code, LAST_ERR, LAST_STATE)
                if "```" not in o:
                    p["trunc"] = o              # sortie coupée avant la fin du bloc de code
            else:
                p["c"].append((v, weight))
            if DEBUG:
                err = LAST_ERR if v is None else ""
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
            run_round(nsamp_adapt, 0.5, 1.0, MAX_NEW) 
            
        print(f"[solve] prep {t_prep:.1f}s | glouton {d1:.1f}s | réparation {d2:.1f}s | échantillons "
              f"{time.time() - t3:.1f}s | total {time.time() - t0:.1f}s")

        res = []
        for p in prep:
            try:
                nd = decimals(p["obj"])
                v = vote(p["c"], nd)
                if nd is not None:
                    v = round(v, nd)
                res.append({"id": p["id"], "answer": float(v)})
            except Exception:
                res.append({"id": p["id"], "answer": 0.0})
        return res
