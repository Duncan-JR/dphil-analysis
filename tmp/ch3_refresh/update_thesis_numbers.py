import pathlib

path = pathlib.Path('/Users/duncan/work/dphil-thesis/tex/ch3_anc.tex')
text = path.read_text()
replacements = {
    '$-6.58$~Mb in tsinfer~0.5': '$-6.59$~Mb in tsinfer~0.5',
    '19.64–49.41 kb for tsinfer 0.5': '10.55–26.13 kb for tsinfer 0.5',
    '0.37--0.75 percentage points': '0.006--0.025 percentage points',
    'it alone has [x] focal sites carried by [x] true and [x]\ninferred ancestors': 'it alone has 358,285 focal sites carried by 137,123 true and 228,445\ninferred ancestors',
    'less than [x] Mbp in tsinfer 0.5 versus\n[y] Mbp for tsinfer 0.4': 'less than $-12.54$ Mbp in tsinfer 0.5 versus\n$-12.54$ Mbp for tsinfer 0.4',
    'In total,[x]\\% of ancestors have at least one error and total pooled\nerror burden of [z]\\% for tsinfer 0.5, versus [y]\\% and [c] for tsinfer 0.4': 'In total,51.874\\% of ancestors have at least one error and total pooled\nerror burden of 0.00083007\\% for tsinfer 0.5, versus 51.863\\% and 0.00082992\\% for tsinfer 0.4',
    'matching is ~4X faster and sample matching is ~9X faster': 'matching is ~4.26X faster and sample matching is ~9.77X faster',
    'matching of ~4.5X in the 1000 Genomes Project exceeds the 3X speedup': 'matching of ~4.77X in the 1000 Genomes Project exceeds the 3.27X speedup',
}
for old, new in replacements.items():
    assert text.count(old) == 1, old
    text = text.replace(old, new)
path.write_text(text)
