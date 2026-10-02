"""Frozen ImageNet image encoder feeding matched quantum/classical attention."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torchvision.models import resnet18,ResNet18_Weights
from .config import Config
from .data import load_splits
from .quantum_robust import QuantumRobustViT
from .quantum_robust_experiment import digest
from .runtime import json_write,torch_save_atomic,seed_all

MANIFEST=Path('configs/pretrained_quantum_accuracy.json')
CACHE=Path('artifacts/pretrained_quantum_features')


def head_config(spec,seed,kernel):
    return Config(**(spec['head']|{'dataset':spec['dataset'],'seed':seed,
                                 'model':'dot' if kernel=='dot' else 'fidelity'}))


class QuantumTransferViT(QuantumRobustViT):
    def __init__(self,c,kernel,feature_dim=512):
        super().__init__(c,kernel)
        del self.patch
        self.feature_projection=nn.Linear(feature_dim,c.embed_dim)

    def forward(self,x):
        x=self.feature_projection(x)
        x=self.drop(torch.cat([self.cls.expand(len(x),-1,-1),x],1)+self.position)
        for block in self.blocks:
            x=block(x)
        return self.head(self.norm(x[:,0]))


def make_encoder(spec):
    path=Path(spec['backbone']['weight_file'])
    assert digest(path).startswith('f37072fd'),'Unexpected pretrained weight hash'
    model=resnet18(weights=None)
    model.load_state_dict(torch.load(path,weights_only=True))
    encoder=nn.Sequential(*list(model.children())[:-2],nn.AdaptiveAvgPool2d(spec['backbone']['pool_size']))
    encoder.requires_grad_(False).eval()
    return encoder,ResNet18_Weights.IMAGENET1K_V1.transforms()


@torch.no_grad()
def image_features(encoder,transform,images,batch_size=32,flip=False):
    pieces=[]
    for start in range(0,len(images),batch_size):
        x=(images[start:start+batch_size]+1)/2
        if flip:
            x=x.flip(-1)
        x=transform(x.expand(-1,3,-1,-1))
        pieces.append(encoder(x).flatten(2).transpose(1,2).contiguous())
    return torch.cat(pieces)


def training_stats(train):
    # Only training features enter normalization; there is no target argument.
    return train.mean((0,1,2)),train.std((0,1,2)).clamp_min(1e-4)


def feature_hashes():
    return {p.name:digest(p) for p in sorted(CACHE.glob('*.pt'))}


def prepare():
    spec=json.loads(MANIFEST.read_text())
    seed_all(611,spec['backbone']['threads'])
    ds,_,_,data=load_splits(head_config(spec,0,'entangled'))
    encoder,transform=make_encoder(spec)
    CACHE.mkdir(parents=True,exist_ok=True)
    if (CACHE/'metadata.json').exists():
        meta=json.loads((CACHE/'metadata.json').read_text())
        assert meta['manifest_sha256']==digest(MANIFEST)
        assert meta['weights_sha256']==digest(spec['backbone']['weight_file'])
        assert meta['dataset_sha256']==data['sha256'] and meta['files_sha256']==feature_hashes()
        print('Reused complete feature cache',flush=True)
        return
    timings={};shapes={}
    for split in ['train','val','test']:
        start=time.perf_counter()
        images,labels=ds[split].tensors
        path=CACHE/(split+'.pt')
        if path.exists():
            bank=torch.load(path,weights_only=True)
            assert torch.equal(bank['labels'],labels)
        else:
            views=[]
            for flip in ([False,True] if split=='train' else [False]):
                before=time.perf_counter()
                views.append(image_features(encoder,transform,images,spec['backbone']['feature_batch_size'],flip))
                print('Extracted',split,'flip',flip,'images',len(images),'seconds',round(time.perf_counter()-before,1),flush=True)
            bank={'features':torch.stack(views),'labels':labels,'sample_ids':torch.arange(len(labels))}
            torch_save_atomic(bank,path)
        shapes[split]=list(bank['features'].shape)
        timings[split]=time.perf_counter()-start
    train=torch.load(CACHE/'train.pt',weights_only=True)['features']
    mean,std=training_stats(train)
    torch_save_atomic({'mean':mean,'std':std},CACHE/'normalization.pt')
    json_write(CACHE/'metadata.json',{'manifest_sha256':digest(MANIFEST),
        'weights_sha256':digest(spec['backbone']['weight_file']),'dataset_sha256':data['sha256'],
        'feature_source_sha256':digest(__file__),'files_sha256':feature_hashes(),
        'shapes':shapes,'seconds':timings,'preprocessing':str(transform),
        'frozen_encoder_parameters':sum(p.numel() for p in encoder.parameters()),
        'pretraining':'ImageNet-1K','normalization_fit':'training_views_only',
        'physical_hardware':False})
    print('Complete feature cache',shapes,flush=True)


def load_features():
    stats=torch.load(CACHE/'normalization.pt',weights_only=True)
    banks={}
    for split in ['train','val','test']:
        bank=torch.load(CACHE/(split+'.pt'),weights_only=True)
        bank['features']=(bank['features']-stats['mean'])/stats['std']
        banks[split]=bank
    return banks


if __name__=='__main__':
    prepare()
