import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import numpy as np
import torch
import pennylane as qml
from research.config import Config
from research.models import product_fidelity
from research.quantum_robust import (feature_states,feature_circuit,PennyLaneOverlap,
                                    QuantumRobustViT,SAMStep,augment)


class QuantumRobustTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_gate_simulation_and_compute_uncompute_agree(self):
        theta=torch.tensor([[.2,.5,1.2,2.],[1.1,.4,.7,.9]],dtype=torch.float64,requires_grad=True)
        dev=qml.device('default.qubit',wires=4)
        @qml.qnode(dev)
        def reference(angle):
            feature_circuit(angle)
            return qml.state()
        state=feature_states(theta)
        np.testing.assert_allclose(state.detach().numpy(),reference(theta.detach().numpy()),atol=1e-12)
        np.testing.assert_allclose(state.abs().square().sum(-1).detach().numpy(),1,atol=1e-12)
        actual=float((state[0].conj()@state[1]).abs().square().detach())
        backend=PennyLaneOverlap(chunk_size=1)
        measured=float(backend(theta.detach().numpy()[0:1],theta.detach().numpy()[1:2])[0])
        self.assertAlmostEqual(actual,measured,places=11)
        # Entanglement survives the final local rotations, and the second
        # encoding makes this overlap differ from the product-state ablation.
        schmidt=torch.linalg.svdvals(state[0].reshape(2,8))
        self.assertGreater(float(schmidt[1].detach()),1e-4)
        product=feature_states(theta,False)
        product_overlap=float((product[0].conj()@product[1]).abs().square().detach())
        self.assertGreater(abs(actual-product_overlap),1e-4)
        actual_kernel=lambda z:(feature_states(z)[0].conj()@feature_states(z)[1]).abs().square()
        self.assertTrue(torch.autograd.gradcheck(actual_kernel,theta))
        q,k=torch.randn(1,1,2,4,dtype=torch.float64),torch.randn(1,1,3,4,dtype=torch.float64)
        from research.models import angles
        qs,ks=feature_states(angles(q),False),feature_states(angles(k),False)
        torch.testing.assert_close((qs.conj()@ks.transpose(-1,-2)).abs().square(),product_fidelity(q,k),atol=1e-12,rtol=1e-12)

    def test_matching_parameters_and_classifier_circuit_backend(self):
        counts=[]
        x=torch.randn(2,1,28,28)
        for kernel in ['entangled','product','dot']:
            c=Config(model='dot' if kernel=='dot' else 'fidelity')
            model=QuantumRobustViT(c,kernel).eval()
            counts.append(sum(p.numel() for p in model.parameters()))
            expected=model(x)
            if kernel!='dot':
                backend=PennyLaneOverlap(kernel=='entangled')
                model.set_circuit_backend(backend)
                actual=model(x)
                torch.testing.assert_close(expected,actual,atol=3e-6,rtol=3e-6)
                self.assertEqual(backend.circuit_settings,2*2*2*17*17)
                self.assertGreater(backend.qnode_calls,0)
        self.assertEqual(len(set(counts)),1)

    def test_sam_restores_parameters_and_affine_padding(self):
        p=torch.nn.Parameter(torch.tensor([.1]))
        optimizer=torch.optim.SGD([p],lr=.1)
        sam=SAMStep(optimizer,rho=.05)
        calls=[]
        def loss():
            calls.append(float(p.detach()))
            return p.square().sum()
        sam.step(loss)
        self.assertEqual(len(calls),2)
        self.assertAlmostEqual(calls[1],.15,places=6)
        self.assertAlmostEqual(float(p.detach()),.07,places=6)
        x=torch.full((5,1,28,28),-.7)
        torch.manual_seed(4);a=augment(x)
        torch.manual_seed(4);b=augment(x)
        torch.testing.assert_close(a,b)
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.all((a>=-1)&(a<=1)))

    def test_resume_reproduces_stochastic_quantum_training(self):
        # Interrupt after an on-disk epoch checkpoint, then compare against an
        # uninterrupted run with augmentation, MixUp, dropout and EMA active.
        from research import quantum_robust_experiment as experiment
        from torch.utils.data import TensorDataset
        spec=json.loads(Path('configs/quantum_robust_accuracy.json').read_text())
        spec['training'].update(epochs=3,batch_size=4,threads=2,warmup_epochs=1)
        generator=torch.Generator().manual_seed(315)
        splits={name:TensorDataset(torch.rand(n,1,28,28,generator=generator)*2-1,
                                   torch.arange(n)%2)
                for name,n in [('train',12),('val',8),('test',8)]}
        datasets=(splits,1,2,{'sha256':'synthetic-resume-test'})
        with tempfile.TemporaryDirectory() as directory:
            for recipe in spec['recipes']:
                with self.subTest(recipe=recipe['name']):
                    full_root=Path(directory)/recipe['name']/'full'
                    resumed_root=Path(directory)/recipe['name']/'resumed'
                    with patch.object(experiment,'load_splits',return_value=datasets), \
                         patch.object(experiment,'snapshot'):
                        with patch.object(experiment,'ROOT',full_root):
                            full=experiment.run_one(spec,recipe,42,'entangled',test=True)
                            full_folder=experiment.run_folder(spec,recipe,42,'entangled')
                        original_save=experiment.torch_save_atomic
                        def interrupt_save(state,path):
                            original_save(state,path)
                            if Path(path).name=='last.pt':
                                raise RuntimeError('simulated interruption')
                        with patch.object(experiment,'ROOT',resumed_root):
                            with patch.object(experiment,'torch_save_atomic',side_effect=interrupt_save):
                                with self.assertRaisesRegex(RuntimeError,'simulated interruption'):
                                    experiment.run_one(spec,recipe,42,'entangled',test=True)
                            resumed=experiment.run_one(spec,recipe,42,'entangled',test=True)
                            resumed_folder=experiment.run_folder(spec,recipe,42,'entangled')
                    a=torch.load(full_folder/'last.pt',weights_only=False)
                    b=torch.load(resumed_folder/'last.pt',weights_only=False)
                    for key in ['model','ema']:
                        for name in a[key]:
                            torch.testing.assert_close(a[key][name],b[key][name],atol=0,rtol=0)
                    self.assertEqual(full['best_epoch'],resumed['best_epoch'])
                    self.assertEqual(full['balanced_nll'],resumed['balanced_nll'])
                    with np.load(full_folder/'test_predictions.npz') as x, \
                         np.load(resumed_folder/'test_predictions.npz') as y:
                        np.testing.assert_array_equal(x['logits'],y['logits'])


if __name__=='__main__':
    unittest.main()
