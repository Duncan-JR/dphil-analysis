import pathlib
import time

import jupytext
import nbclient
import nbformat

root = pathlib.Path('/Users/duncan/work/dphil-analysis')
path = root / 'notebooks/ch3/figure_gen.py'
notebook = jupytext.read(path)
client = nbclient.NotebookClient(notebook, timeout=1800, resources={'metadata': {'path': str(root)}})
started = time.monotonic()
try:
    with client.setup_kernel():
        notebook.cells.append(nbformat.v4.new_code_cell("import sys; sys.path.insert(0, 'src')\n%matplotlib inline"))
        client.execute_cell(notebook.cells[-1], len(notebook.cells) - 1)
        notebook.cells.pop()
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == 'markdown' and '## Testing error buffering' in cell.source:
                print('Stopping before unrelated exploratory error-buffering figures', flush=True)
                break
            if cell.cell_type == 'code':
                print(f'{index}: {cell.source[:100]!r} ({time.monotonic() - started:.0f}s)', flush=True)
                client.execute_cell(cell, index)
finally:
    nbformat.write(notebook, path.with_suffix('.ipynb'))
print('Completed chapter figures and checks', flush=True)
