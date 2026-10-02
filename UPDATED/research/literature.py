"""Verified bibliographic metadata and conservative literature extraction, 2026-10-01.

NE = not established from the inspected primary text; it never means zero or absent.
"""
import csv
import json
from pathlib import Path

PAPERS=[
 ('henderson2019','Quanvolutional Neural Networks: Powering Image Recognition with Quantum Circuits','Maxwell Henderson and Samriddhi Shakya and Shashindra Pradhan and Tristan Cook',2019,'1904.04767',
  'image classification','MNIST','local image encoding','random local circuits','quanvolution, not transformer attention','simulation','NE','NE','NE','CNN and added nonlinearities','accuracy, training','local quantum features','not an attention allocator','historical context'),
 ('li2022','Quantum Self-Attention Neural Networks for Text Classification','Guangxi Li and Xuanqiang Zhao and Xin Wang',2022,'2205.05625',
  'text classification','public text benchmarks','quantum feature embedding','parameterized circuits','Gaussian projected attention','numerical simulation','NE','NE','low-level noise investigated','classical attention and QNLP','classification accuracy','QSANN','different quantum representation','prior quantum attention'),
 ('shi2022','QSAN: A Near-term Achievable Quantum Self-Attention Network','Jinjing Shi and Ren-Xin Zhao and Wenxuan Wang and Shichao Zhang and Xuelong Li',2022,'2207.07563',
  'binary image classification','MNIST subset','NE','quantum logic similarity circuit','QBSASM','PennyLane simulation','NE','NE','NE','hardware-efficient and QAOA ansatz','accuracy, convergence','quantum logic attention','restricted classification task','attention predates our study'),
 ('cherrat2022','Quantum Vision Transformers','El Amine Cherrat and Iordanis Kerenidis and Natansh Mathur and Jonas Landman and Martin Strahm and Yun Yvonna Li',2022,'2209.08167',
  'medical image classification','MedMNIST','vector/matrix loaders','orthogonal and compound layers','several quantum transformer designs','simulation and superconducting hardware','up to 6 hardware qubits','NE','hardware execution; details NE','classical ViT','classification performance, parameters','quantum transformer architectures','small-scale hardware tests','prior medical QViT; distinct attention design'),
 ('evans2024','Learning with SASQuaTCh: a Novel Variational Quantum Transformer Architecture with Kernel-Based Self-Attention','Ethan N. Evans and Matthew Cook and Zachary P. Bradshaw and Margarite L. LaBorde',2024,'2403.14753',
  'image classification','handwritten digits','amplitude image embedding','variational gates and multidimensional QFT','kernel-based attention channel','simulation and hardware','9 in abstract example','NE','NE','NE','accuracy and resource complexity','SASQuaTCh architecture','encoding/resource assumptions differ','prior kernel quantum attention'),
 ('unlu2024','Hybrid Quantum Vision Transformers for Event Classification in High Energy Physics','Eyup B. Unlu and Marcal Comajoan Cara and Gopal Ramesh Dahale and Zhongtian Dong and Roy T. Forestano and Sergei Gleyzer and Daniel Justice and Kyoungchul Kong and Tom Magorsch and Konstantin T. Matchev and Katia Matcheva',2024,'2402.00776',
  'photon/electron classification','electromagnetic calorimeter images','NE','hybrid variational modules','hybrid ViT','simulation','NE','NE','NE','similar-parameter classical ViT','classification performance','HEP hybrid variants','no shot allocator established','supports matched comparisons'),
 ('comajoan2024','Quantum Vision Transformers for Quark-Gluon Classification','Marcal Comajoan Cara and Gopal Ramesh Dahale and Zhongtian Dong and Roy T. Forestano and Sergei Gleyzer and Daniel Justice and Kyoungchul Kong and Tom Magorsch and Konstantin T. Matchev and Katia Matcheva and Eyup B. Unlu',2024,'2405.10284',
  'quark/gluon classification','CMS Open Data jet images','NE','VQCs in attention and MLP','hybrid ViT','numerical simulation','NE','NE','NE','similar-parameter classical architecture','classification performance','HEP QViT','comparable, not universal superiority','corrects conflated original citation'),
 ('chen2024','Quantum Mixed-State Self-Attention Network','Fu Chen and Qinglin Zhao and Li Feng and Chuangtao Chen and Yangbin Lin and Jianhong Lin',2024,'2403.02871',
  'text classification','Yelp, IMDb, Amazon','trainable quantum embedding and positional gates','mixed-state similarity','QMSAN','TensorCircuit simulation','NE','NE','depolarization, amplitude and phase damping in embedding','QSANN','accuracy, variation','mixed-state attention and positional encoding','specific embedding-noise study','noise robustness itself is not novel'),
 ('boucher2025','From $\\mathcal{O}(n^{2})$ to $\\mathcal{O}(n)$ Parameters: Quantum Self-Attention in Vision Transformers for Biomedical Image Classification','Thomas Boucher and John Whittle and Evangelos B. Mazomenos',2025,'2503.07294v2',
  'biomedical classification','eight MedMNIST datasets','angle RY encoding','RY/CNOT quantum QKV layers','QNN projections within attention','PennyLane/PyTorch GPU simulation','4 and 8','NE','NE','parameter-matched ViT and SOTA teachers','AUROC, accuracy, parameters, GFLOPs','efficient QSA and knowledge distillation','no clinical validation from benchmark','medical QViT and parameter matching already exist'),
 ('zhang2025hqvit','HQViT: Hybrid Quantum Vision Transformer for Image Classification','Hui Zhang and Qinglin Zhao and Mengchu Zhou and Li Feng',2025,'2504.02730',
  'image classification','MNIST and other vision datasets','whole-image amplitude encoding','parameterized query/key registers plus SWAP','SWAP-based pair coefficients','NE','logarithmic resource claims','NE','NE','ViT and quantum attention baselines','accuracy, resource complexity','hybrid SWAP attention','state preparation and readout cost matter','direct prior art against SWAP attention novelty'),
 ('patro2026','QiT: Quantum-Inspired Transformer for Visual Recognition Task','Badri N. Patro and Vijay Agneeswaran',2026,'2609.17789',
  'visual recognition','ImageNet-1K and other vision benchmarks','classical trigonometric features','none: tensor computation','periodic/cosine kernel','classical execution','not applicable','not applicable','not applicable','matched classical transformer','accuracy, parameters, GFLOPs','quantum-inspired inductive bias','no quantum computation or speedup claim','requires clear analytical baseline positioning'),
 ('shastry2022','Shot-frugal and Robust quantum kernel classifiers','Abhay Shastry and Abhijith Jayakumar and Apoorva Patel and Chiranjib Bhattacharyya',2022,'2210.06971',
  'kernel classification','NE','quantum kernels','kernel estimation','not transformer attention','NE','NE','variable measurement budget','unbiased kernel uncertainty','SVM formulations','classification reliability, shots','chance-constrained shot-frugal classification','different objective than attention output','shot efficiency is established prior work'),
 ('xu2026','AQKA: Active Quantum Kernel Acquisition Under a Shot Budget','Jian Xu and Chao Li and Delu Zeng and John Paisley and Qibin Zhao',2026,'2605.14672v2',
  'KRR/SVM kernel learning','synthetic sparse-sensitivity and real-data studies','ZZ quantum feature map','fidelity kernel acquisition','not softmax attention','simulation and IBM hardware reported','4 feature-map qubits in inspected setup','fixed-budget sweeps','noisy simulator and hardware','uniform, Nystrom-QKE, ShoFaR','accuracy, budget, kernel error','sensitivity times variance allocation','gains depend on sensitivity regime','major overlap; generic allocation novelty rejected'),
 ('zhang2025survey','A Survey of Quantum Transformers: Architectures, Challenges and Outlooks','Hui Zhang and Qinglin Zhao and Mengchu Zhou and Li Feng and Dusit Niyato and Shenggen Zheng and Lin Chen',2025,'2504.03192v5',
  'survey','multiple','multiple','PQC and QLA taxonomy','multiple mechanisms','survey of simulation/hardware','varies','varies','varies','literature comparison','resources and empirical evidence','organizes architectures and challenges','not an original matched benchmark','citation tracking and novelty context'),
 ('chen2025qasa','Quantum Adaptive Self-Attention for Quantum Transformer Models','Chi-Sheng Chen and En-Jui Kuo',2025,'2504.05336',
  'sequence learning','synthetic time series','PQC features','parameterized attention and residual projection','adaptive quantum representations','simulation','NE','NE','NE','standard and reduced transformers','generalization and convergence','QASA','adaptive representation does not establish shot allocation','terminological overlap needs distinction')
]


def main():
    columns=['key','title','authors','year','arxiv','task','dataset','encoding','circuit','attention','execution','qubits','shots','noise','baselines','metrics','contribution','limitations','relationship']
    rows=[dict(zip(columns,p)) for p in PAPERS]
    for r in rows:
        r['source']='https://arxiv.org/abs/'+r['arxiv']
        r['verified_on']='2026-10-01'
    out=Path('literature')
    out.mkdir(exist_ok=True)
    (out/'verified.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (out/'table.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    bib=[]
    for r in rows:
        bib.append('@misc{'+r['key']+',\n  title={'+r['title']+'},\n  author={'+r['authors']+'},\n  year={'+str(r['year'])+'},\n  eprint={'+r['arxiv']+'},\n  archivePrefix={arXiv},\n  url={'+r['source']+'}\n}')
    bib.append(r'''@article{yang2023medmnist,
  title={MedMNIST v2 - A large-scale lightweight benchmark for 2D and 3D biomedical image classification},
  author={Yang, Jiancheng and Shi, Rui and Wei, Donglai and Liu, Ziyun and Zhao, Lin and Ke, Bilian and Pfister, Hanspeter and Ni, Bingbing},
  journal={Scientific Data}, volume={10}, pages={41}, year={2023},
  doi={10.1038/s41597-022-01721-8}, url={https://www.nature.com/articles/s41597-022-01721-8.pdf}
}
@misc{bouakba2026,
  title={Quantum Destructive Self-Attention for NISQ-Era Transformers},
  author={Bouakba, Yousra and Belhadef, Hacene}, year={2026},
  howpublished={Research Square preprint}, doi={10.21203/rs.3.rs-8946631/v1},
  url={https://doi.org/10.21203/rs.3.rs-8946631/v1}
}
@article{kubler2020,
  title={An Adaptive Optimizer for Measurement-Frugal Variational Algorithms},
  author={K{\"u}bler, Jonas M. and Arrasmith, Andrew and Cincio, Lukasz and Coles, Patrick J.},
  journal={Quantum}, volume={4}, pages={263}, year={2020},
  doi={10.22331/q-2020-05-11-263}, url={https://quantum-journal.org/papers/q-2020-05-11-263/}
}''')
    Path('references.bib').write_text('\n\n'.join(bib)+'\n',encoding='utf-8')
    table=['# Verified literature extraction','', 'NE means not established from the inspected primary text, not absent. Full bibliographic metadata and all requested comparison fields are in `table.csv` and `verified.json`. Entries use verified preprint citations even where a later journal version exists.','', '| Citation | Task / data | Mechanism | Execution / resources | Relation and limitation |','|---|---|---|---|---|']
    for r in rows:
        table.append(f"| [{r['title']}]({r['source']}) ({r['year']}) | {r['task']}; {r['dataset']} | {r['encoding']}; {r['attention']} | {r['execution']}; qubits {r['qubits']}; shots {r['shots']} | {r['relationship']}; {r['limitations']} |")
    table.extend(['','Additional verified destructive-attention preprint: Yousra Bouakba and Hacene Belhadef (2026), *Quantum Destructive Self-Attention for NISQ-Era Transformers*, Research Square, DOI 10.21203/rs.3.rs-8946631/v1. Primary PDF inspected. NLP; ancilla-free destructive SWAP attention; dataset/shot/qubit/noise details remain unextracted. It excludes novelty claims for destructive attention itself.'])
    (out/'TABLE.md').write_text('\n'.join(table),encoding='utf-8')


if __name__=='__main__':
    main()
