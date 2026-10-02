import unittest
import numpy as np
from scipy.optimize import minimize
from research.circuits import Noise,run_circuit
from research.noisy_overlap import destructive_means,reference_angles,NoisyParitySampler
from research.bias_aware import (RiskConfig,POLICIES,attention_gram,continuous_counts,
                                 risk_components,solve_weights,surrogate,run_policy)


class BiasAwareTests(unittest.TestCase):
    def test_factorization_matches_full_noisy_circuit(self):
        rng=np.random.default_rng(311)
        noise=Noise(preparation=.004,one_qubit=.003,two_qubit=.02,amplitude=.002,
                    phase=.001,bitflip=.001,phaseflip=.002,readout=.03)
        for d in [2,4]:
            q,k=rng.uniform(0,np.pi,(2,d))
            actual=run_circuit(q,k,'destructive',noise=noise)['fidelity']
            self.assertAlmostEqual(float(destructive_means(q,k,noise)),actual,places=10)
        q,k,f=reference_angles(4)
        np.testing.assert_allclose(destructive_means(q,k),f,atol=1e-12)

    def test_softmax_bias_and_shared_covariance_geometry(self):
        f=np.array([[.1,.8,.4]])
        v=np.array([[[0.,1],[1.,-.4],[.2,.3]]])
        gram=attention_gram(f,v,2.)
        np.testing.assert_allclose(gram.sum(-1),0,atol=1e-13)
        _,bq,cq=risk_components(f,f,v,2.,np.array([1.]),np.array([.2]),np.array([[[0.,0.],[0.,.01]]]))
        self.assertAlmostEqual(float(bq.sum()),0.,places=12) # common bias cancels
        self.assertAlmostEqual(float(cq.sum()),0.,places=12) # common offset uncertainty cancels
        self.assertGreater(float(np.trace(bq[0])),0.) # a diagonal-only surrogate gets this wrong

    def test_continuous_allocation_and_box_qp(self):
        c=np.array([[1.,9.,.00001]])
        s=continuous_counts(c,np.array([11]))
        np.testing.assert_allclose(s.sum(1),11,atol=1e-8)
        self.assertAlmostEqual(s[0,1]/s[0,0],3.,places=6)
        self.assertAlmostEqual(s[0,2],1.,places=6)
        # Adding any count-independent bias constant cannot alter this optimizer.
        a=np.array([[[2.,-.4],[-.4,1.]]])
        q=np.array([[[.1,.03],[.03,.2]]])
        variance=np.array([[.2,.5]])
        correction=np.array([.7])
        got=solve_weights(a,q,variance,correction,np.zeros((1,2)),100)
        fun=lambda t:float(surrogate(t[None],np.ones((1,2)),a,q,variance,correction)[0])
        optimum=minimize(fun,np.zeros(2),bounds=[(0,1)]*2,tol=1e-12)
        self.assertLess(abs(fun(got[0])-optimum.fun),1e-8)

    def test_all_policies_budget_and_oracle_free_inputs(self):
        rng=np.random.default_rng(11)
        means=rng.uniform(0,.8,(6,5))
        values=rng.normal(size=(6,5,3))
        refs=np.broadcast_to(np.array([.03,.23,.52,.8]),(3,4))
        for policy in POLICIES:
            sampler=NoisyParitySampler(means,np.random.default_rng(90))
            reference=NoisyParitySampler(refs,np.random.default_rng(91))
            f,a=run_policy(sampler,reference,values,2.,policy,RiskConfig(shots=64),heads=2)
            self.assertTrue(np.isfinite(f).all())
            self.assertEqual(sampler.shots+reference.shots,3*2*5*64)
            self.assertTrue(np.all(a['shots_per_image']==2*5*64))
            if 'mitigation_weights' in a:
                self.assertTrue(np.all((a['mitigation_weights']>=0)&(a['mitigation_weights']<=1)))
                self.assertTrue(np.all(a['production_counts']>=1))
            sampler2=NoisyParitySampler(means,np.random.default_rng(90))
            reference2=NoisyParitySampler(refs,np.random.default_rng(91))
            f2,_=run_policy(sampler2,reference2,values,2.,policy,RiskConfig(shots=64),heads=2)
            np.testing.assert_array_equal(f,f2)


if __name__=='__main__':
    unittest.main()
