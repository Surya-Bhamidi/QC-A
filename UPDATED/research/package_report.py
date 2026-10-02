"""Synchronize active report entry points and create a portable source bundle."""
from pathlib import Path
import json
import zipfile


def main():
    for name in ['report','QViT_Overleaf_Report']:
        folder=Path(name)
        folder.mkdir(exist_ok=True)
        (folder/'main.tex').write_text('\\providecommand{\\projectroot}{..}\n\\input{../PROJECT_METHODOLOGY_REPORT.tex}\n')
        (folder/'references.bib').write_bytes(Path('references.bib').read_bytes())
        (folder/'README.md').write_text('This entry point includes the current root research paper. Compile from this directory or use the standalone source ZIP in artifacts/QCA_research_sources.zip. The earlier report is preserved in archive/.\n')
    Path('COMPARISON_REPORT.md').write_text('# Current comparison results\n\nThe earlier course comparison is preserved in archive/. Current evidence is in [central_summary.csv](artifacts/central_summary.csv), [measurement_summary.csv](artifacts/measurement_summary.csv), [paired primary tests](artifacts/primary_hypotheses.json), and [IMPLEMENTATION_REPORT.md](IMPLEMENTATION_REPORT.md).\n\nFive seeds and complete official tests are used. The primary adaptive allocator does not improve conditional attention-output error; quantum advantage and clinical utility are not established.\n')
    Path('dashboard/static/index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Research dashboard</title><p>The course dashboard has been replaced. <a href="/">Open the research dashboard</a>. Research only; not a clinical diagnostic tool.</p></html>\n')
    js=Path('dashboard/static/research.js')
    text=js.read_text(encoding='utf-8')
    text=text.replace("policy:$('policy').value});", "policy:$('policy').value,circuit_method:$('inferenceCircuit').value});")
    text=text.replace('Reference ancilla circuit:', 'Reference ${last.circuit_method} circuit:')
    js.write_text(text,encoding='utf-8')
    cells=[{'cell_type':'markdown','metadata':{},'source':['# Reconstructed fidelity-attention experiments\n','Research only. These cells read saved results; they do not claim quantum hardware execution. The complete shot sweep is classical probability emulation. The primary adaptive result is negative. See REPRODUCIBILITY.md and LIMITATIONS.md.']}]
    for code in ["from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nassert Path('artifacts/central_summary.csv').exists(), 'Run python -m research.analyze first'",
                 "display(pd.read_csv('artifacts/central_summary.csv'))",
                 "display(pd.read_csv('artifacts/measurement_summary.csv'))\ndisplay(Image(filename='artifacts/shot_tradeoff.png'))",
                 "display(pd.read_csv('artifacts/ablation_summary.csv'))",
                 "print(json.dumps(json.loads(Path('artifacts/primary_hypotheses.json').read_text()), indent=2))",
                 "display(Image(filename='artifacts/circuit_noise.png'))\nprint(Path('runs/circuits/summary.json').read_text())"]:
        cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':code.splitlines(keepends=True)})
    for i,cell in enumerate(cells):
        cell['id']=f'research-{i}'
    notebook={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.14.2'}},'nbformat':4,'nbformat_minor':5}
    Path('experiments_results.ipynb').write_text(json.dumps(notebook,indent=2),encoding='utf-8')
    paths=[Path(p) for p in ['main.tex','PROJECT_METHODOLOGY_REPORT.tex','RESEARCH_PRESENTATION.tex','references.bib','README.md','REPRODUCIBILITY.md','LIMITATIONS.md','METHOD.md','EXPERIMENT_PROTOCOL.md','NOVELTY_SEARCH.md','IMPLEMENTATION_REPORT.md']]
    paths+=[p for p in Path('artifacts').iterdir() if p.suffix in {'.tex','.pdf','.csv','.json'}]
    with zipfile.ZipFile('artifacts/QCA_research_sources.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in paths:
            if path.exists():
                z.write(path,path.as_posix())
    print('Updated active report copies, result notebook and portable paper source ZIP')


if __name__=='__main__':
    main()
