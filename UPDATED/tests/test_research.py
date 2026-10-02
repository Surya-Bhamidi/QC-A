import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from research.config import Config
from research.models import MatchedViT, product_fidelity, angles
from research.shots import BernoulliSampler, ShotConfig, estimate, ShotEstimator
from research.data import stratified_indices
from research.metrics import classification_metrics, delong_paired
from research.circuits import run_circuit, Noise


class ResearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_analytic_matches_nine_wire(self):
        rng=np.random.default_rng(9)
        for _ in range(3):
            q,k=rng.uniform(0,np.pi,(2,4))
            ideal=np.prod(np.cos((q-k)/2)**2)
            self.assertAlmostEqual(run_circuit(q,k)['fidelity'],ideal,places=10)
        self.assertAlmostEqual(run_circuit(np.zeros(4),np.zeros(4))['fidelity'],1.,places=10)
        self.assertAlmostEqual(run_circuit(np.zeros(4),np.array([np.pi,0,0,0]))['fidelity'],0.,places=10)

    def test_nonzero_gradient(self):
        q=torch.tensor([[[.1,.2,-.3,.4]]],dtype=torch.double,requires_grad=True)
        k=torch.zeros_like(q)
        self.assertTrue(torch.autograd.gradcheck(lambda z:product_fidelity(z,k),q))
        product_fidelity(q,k).sum().backward()
        self.assertGreater(q.grad.abs().sum().item(),0)

    def test_head_shapes_matching_and_row_sums(self):
        for heads in [1,2,4]:
            counts=[]
            for kind in ['dot','cosine','rbf','fidelity']:
                c=Config(heads=heads,model=kind)
                m=MatchedViT(c).eval()
                m.set_estimator(capture=True)
                self.assertEqual(m(torch.randn(2,1,28,28)).shape,(2,2))
                a=m.blocks[0].attn.last['attention']
                self.assertEqual(a.shape,(2,heads,17,17))
                torch.testing.assert_close(a.sum(-1),torch.ones(2,heads,17))
                counts.append(m.resources()['parameters'])
            self.assertEqual(len(set(counts)),1)

    def test_shot_convergence_in_distribution(self):
        rng=np.random.default_rng(2)
        p=.65
        # Distributional convergence, not a false requirement of monotonic error on one random trajectory.
        mse=[]
        for s in [16,1024]:
            samples=2*rng.binomial(s,p,size=20000)/s-1
            mse.append(np.mean((samples-(2*p-1))**2))
        self.assertLess(mse[1],mse[0]/30)

    def test_allocations_and_reproducibility(self):
        f=np.full((2,3,4,4),.4)
        v=np.arange(2*3*4*2).reshape(2,3,4,2)/10
        for policy in ['uniform','variance','sensitivity','combined']:
            cfg=ShotConfig(shots=32,policy=policy)
            first=None
            for _ in range(2):
                pred,audit=estimate(BernoulliSampler(f,np.random.default_rng(3)),v,2.,cfg)
                self.assertTrue(np.all(audit['shots'].sum(-1)==4*32))
                self.assertTrue(np.all(audit['shots']>=8))
                np.testing.assert_array_equal(sum(audit['decisions']),audit['shots'])
                if first is not None:
                    np.testing.assert_array_equal(pred,first)
                first=pred
            if policy=='uniform':
                self.assertTrue(np.all(audit['shots']==32))
        _,audit=estimate(BernoulliSampler(f,np.random.default_rng(3)),v,2.,ShotConfig(tolerance=1e3))
        self.assertEqual(audit['total_shots'],int(np.prod(f.shape))*8)

    def test_standard_destructive_uncompute_and_routing(self):
        q,k=np.array([.2,.7,1.1,2.]),np.array([1.,.2,.9,.1])
        ideal=np.prod(np.cos((q-k)/2)**2)
        for method in ['swap','destructive','uncompute']:
            r=run_circuit(q,k,method,shots=20000,seed=123)
            self.assertLess(abs(r['fidelity']-ideal),.025)
            self.assertAlmostEqual(run_circuit(q,k,method)['fidelity'],ideal,places=10)
        self.assertAlmostEqual(run_circuit(q[:2],k[:2],route_line=True)['fidelity'],
                               np.prod(np.cos((q[:2]-k[:2])/2)**2),places=10)

    def test_readout_channel(self):
        ideal=run_circuit([.3],[1.2])['fidelity']
        noisy=run_circuit([.3],[1.2],noise=Noise(readout=.1))['fidelity']
        self.assertAlmostEqual(noisy,.8*ideal,places=8)

    def test_nested_stratification_and_metrics(self):
        y=np.array([0]*31+[1]*69)
        a=stratified_indices(y,.1,9)
        b=stratified_indices(y,.5,9)
        self.assertTrue(set(a)<=set(b))
        self.assertEqual(set(y[a]),{0,1})
        m=classification_metrics(np.array([0,0,1,1]),np.array([[.9,.1],[.8,.2],[.1,.9],[.2,.8]]))
        self.assertEqual(m['auroc'],1.)
        self.assertEqual(m['specificity'],1.)
        self.assertEqual(delong_paired(np.array([0,0,1,1]),np.array([.1,.2,.9,.8]),np.array([.1,.2,.9,.8]))['p'],1.)

    def test_exact_resume_and_complete_test(self):
        from research.train import train
        from research.runtime import json_write
        import json
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            rng=np.random.default_rng(0)
            arrays={}
            for split,n in [('train',24),('val',8),('test',10)]:
                arrays[split+'_images']=rng.integers(0,256,(n,28,28),dtype=np.uint8)
                arrays[split+'_labels']=(np.arange(n)%2)[:,None]
            np.savez(root/'pneumoniamnist.npz',**arrays)
            c=Config(epochs=2,seed=44,batch_size=8,threads=2)
            uninterrupted=train(c,root/'a',root)
            train(c,root/'b',root,stop_after=1)
            resumed=train(c,root/'b',root)
            a=torch.load(uninterrupted/'last.pt',weights_only=False)['model']
            b=torch.load(resumed/'last.pt',weights_only=False)['model']
            for key in a:
                torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
            pred=np.load(resumed/'test_predictions.npz')
            self.assertEqual(len(pred['labels']),10)
            pred.close()
            self.assertEqual(json.loads((resumed/'test_metrics.json').read_text())['n'],10)

    def test_circuit_allocator_executes_and_signed_values(self):
        from research.circuits import CircuitSampler
        q=torch.tensor([[[[.3,.7],[.2,.5]]]])
        k=torch.tensor([[[[.1,.8],[.9,-.3]]]])
        v=np.array([[[[1.,0.],[0.,1.]]]])
        sampler=CircuitSampler(q,k,np.random.default_rng(91))
        f,audit=estimate(sampler,v,2.,ShotConfig(shots=16,rounds=1))
        self.assertEqual(audit['total_shots'],64)
        self.assertEqual(audit['circuit_calls'],8)
        self.assertEqual(audit['modeled_circuit_executions'],0)
        self.assertTrue(np.all(np.abs(f)<=1))


if __name__=='__main__':
    unittest.main()
