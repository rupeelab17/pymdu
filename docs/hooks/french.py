""" French typography for MkDocs.

This plugin adds some French typography rules to your MkDocs project:

- Add a thin space before `!`, `?`, `:` and `;` punctuation marks.
- Use french quote (« and ») around quoted text.
- Use long dash (—) lists.
- Translate admonition titles.
"""

import re

RE_ADMONITION = re.compile(
    r'^(?P<pre>!!!\s*(?P<type>[\w\-]+)(?P<extra>(?: +[\w\-]+)*))(?: +"(?P<title>.*?)")? *$'
)
RE_PUNCT = re.compile(r"(<code>.*?</code>|<[^>]+>)", re.DOTALL)

translations = {}


def on_config(config):
    global translations
    translations = config["extra"]["admonition_translations"]


def on_page_markdown(markdown, page, config, files):
    out = []
    for line in markdown.splitlines():
        m = RE_ADMONITION.match(line)

        if m:
            type = m.group("type")
            if (
                m.group("title") is None or m.group("title").strip() == ""
            ) and type in translations:
                title = translations[type]
                line = m.group("pre") + f' "{title}"'
        out.append(line)
    markdown = "\n".join(out)
    return markdown


RE_PUNCT = re.compile(r"(?<=\w) ?([!?:;])")
RE_IGNORE = re.compile(r"<code[^>]*>.*?</code>|<[^>]+>|&\w+;|\w://|[!?:;]\w", re.DOTALL)


def process_html(html):
    parts = RE_IGNORE.split(html)
    entities = RE_IGNORE.findall(html)

    def process_part(part):
        part = RE_PUNCT.sub(r"&thinsp;\1", part)
        part = re.sub(r'"([^"]+)"', r"«&thinsp;\1&thinsp;»", part)
        return part

    processed_parts = [
        process_part(part) if not RE_IGNORE.fullmatch(part) else part for part in parts
    ]

    # Reconstruct the html with entities
    result = processed_parts[0]
    for entity, part in zip(entities, processed_parts[1:]):
        result += entity + part

    return result


def ligatures(html):
    map = {
        "coeur": "cœur",
        "soeur": "sœur",
        "boeuf": "bœuf",
        "coelacanthe": "cœlacanthe",
        "noeud": "nœud",
        "oeil": "œil",
        "oeuf": "œuf",
        "oeuvre": "œuvre",
        "oeuvrer": "œuvrer",
        "oedeme": "œdème",
        "oestrogène": "œstrogène",
        "oecuménique": "œcuménique",
        "oeillet": "œillet",
        "oe": "œ",
        "foetus": "fœtus",
        "oedipe": "œdipe",
        "caecum": "cæcum",
        "tænia": "tænia",
        "vitae": "vitæ",
        "ex aequo": "ex æquo",
        "cænotype": "cænotype",
        "voeu": "vœu",
    }

    abbreviations = {
        "cie": "C^{ie}",
    }


def on_page_content(html, page, config, files):
    return process_html(html)