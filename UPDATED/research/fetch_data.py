"""Download official archives through MedMNIST, after inspecting the dataset terms."""
import argparse
from pathlib import Path
import medmnist
from medmnist import INFO


def main():
    p=argparse.ArgumentParser()
    p.add_argument('datasets',nargs='+',choices=['pneumoniamnist','breastmnist','dermamnist','bloodmnist','retinamnist'])
    a=p.parse_args()
    Path('data').mkdir(exist_ok=True)
    for name in a.datasets:
        info=INFO[name]
        print(name,'license',info.get('license'),'source',info['url'],flush=True)
        cls=getattr(medmnist,info['python_class'])
        cls(split='train',root='data',download=True)


if __name__=='__main__':
    main()
