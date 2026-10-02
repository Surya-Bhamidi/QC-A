"""Verify that the hybrid's classical and fidelity controls can both learn."""
import unittest
import torch
from research.hybrid_accuracy import Architecture,ConvTokenClassifier


class HybridTests(unittest.TestCase):
    def test_parameter_matched_controls_and_full_gradient_path(self):
        torch.set_num_threads(2)
        x=torch.randn(3,1,28,28)
        y=torch.tensor([0,1,0])
        counts=[]
        for kernel in ['fidelity','dot']:
            model=ConvTokenClassifier(Architecture(model=kernel,seed=0,dropout=0))
            counts.append(sum(p.numel() for p in model.parameters()))
            logits=model(x)
            self.assertEqual(logits.shape,(3,2))
            torch.nn.functional.cross_entropy(logits,y).backward()
            for weight in [model.stem[0].weight,model.blocks[0].attn.q.weight,
                           model.blocks[0].attn.k.weight,model.head.weight]:
                self.assertTrue(torch.isfinite(weight.grad).all())
                self.assertGreater(float(weight.grad.abs().sum()),0)
        self.assertEqual(counts[0],counts[1])


if __name__=='__main__':
    unittest.main()
