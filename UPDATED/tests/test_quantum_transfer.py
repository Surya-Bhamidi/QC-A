import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
from research.config import Config
from research.quantum_transfer import QuantumTransferViT,training_stats
from research.quantum_robust import PennyLaneOverlap


class QuantumTransferTests(unittest.TestCase):
    def test_feature_heads_keep_quantum_path_and_matched_parameters(self):
        torch.set_num_threads(2)
        x=torch.randn(3,4,512)
        counts=[]
        for kernel in ['entangled','product','dot']:
            model=QuantumTransferViT(Config(patch_size=14,model='dot' if kernel=='dot' else 'fidelity'),kernel)
            counts.append(sum(p.numel() for p in model.parameters()))
            model.train();model(x).square().mean().backward()
            self.assertGreater(float(model.feature_projection.weight.grad.norm()),0)
            self.assertGreater(float(model.blocks[0].attn.q.weight.grad.norm()),0)
            model.eval();expected=model(x)
            if kernel!='dot':
                backend=PennyLaneOverlap(entangled=kernel=='entangled')
                model.set_circuit_backend(backend)
                torch.testing.assert_close(expected,model(x),atol=3e-6,rtol=3e-6)
                self.assertEqual(backend.circuit_settings,3*2*2*5*5)
        self.assertEqual(len(set(counts)),1)
        train=torch.randn(2,12,4,512)
        mean,std=training_stats(train)
        torch.testing.assert_close(((train-mean)/std).mean((0,1,2)),torch.zeros(512),atol=1e-6,rtol=0)
        self.assertTrue(torch.isfinite(std).all())

    def test_resume_preserves_random_feature_views_and_ema(self):
        from research import quantum_transfer_experiment as experiment
        spec=json.loads(Path('configs/pretrained_quantum_accuracy.json').read_text())
        spec['training'].update(epochs=3,batch_size=4,threads=2,warmup_epochs=1)
        generator=torch.Generator().manual_seed(72)
        banks={name:{'features':torch.randn(2 if name=='train' else 1,n,4,512,generator=generator),
                     'labels':torch.arange(n)%2}
               for name,n in [('train',12),('val',8),('test',8)]}
        with tempfile.TemporaryDirectory() as directory:
            base=Path(directory);(base/'metadata.json').write_text('{"frozen_encoder_parameters":0}')
            with patch.object(experiment,'CACHE',base),patch.object(experiment,'snapshot'):
                with patch.object(experiment,'ROOT',base/'full'):
                    full=experiment.run_one(spec,banks,'entangled',42)
                original=experiment.torch_save_atomic
                def interrupt(state,path):
                    original(state,path)
                    if Path(path).name=='last.pt':
                        raise RuntimeError('simulated interruption')
                with patch.object(experiment,'ROOT',base/'resumed'):
                    with patch.object(experiment,'torch_save_atomic',side_effect=interrupt):
                        with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                            experiment.run_one(spec,banks,'entangled',42)
                    resumed=experiment.run_one(spec,banks,'entangled',42)
            a=torch.load(base/'full/entangled-s42/last.pt',weights_only=False)
            b=torch.load(base/'resumed/entangled-s42/last.pt',weights_only=False)
            for key in ['model','ema']:
                for name in a[key]:
                    torch.testing.assert_close(a[key][name],b[key][name],atol=0,rtol=0)
            self.assertEqual(full['balanced_nll'],resumed['balanced_nll'])
            with np.load(base/'full/entangled-s42/test_predictions.npz') as x, \
                 np.load(base/'resumed/entangled-s42/test_predictions.npz') as y:
                np.testing.assert_array_equal(x['logits'],y['logits'])


if __name__=='__main__':
    unittest.main()
