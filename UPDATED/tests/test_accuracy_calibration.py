import unittest
import numpy as np
from research.accuracy_calibration_audit import em_adjust,bbse_adjust,apply_methods


class CalibrationTests(unittest.TestCase):
    def test_shift_recovery_and_label_free_target_api(self):
        # Perfect source class-conditional predictions: target prevalence is identifiable.
        vp=np.eye(2)[[0]*40+[1]*60]
        vy=np.array([0]*40+[1]*60)
        tp=np.eye(2)[[0]*70+[1]*30]
        for method in [lambda:em_adjust(tp,np.array([.4,.6])),lambda:bbse_adjust(vp,vy,tp)]:
            p,meta=method()
            np.testing.assert_allclose(meta['target_prior_estimate'],[.7,.3],atol=1e-7)
            np.testing.assert_array_equal(p.argmax(1),tp.argmax(1))
        # Target labels cannot influence a method because no adaptation function accepts them.
        smooth=.98*vp+.01
        target=.98*tp+.01
        results=apply_methods(smooth,vy,target)
        for pred,p,metadata in results.values():
            self.assertEqual(pred.shape,(100,))
            self.assertTrue(np.isfinite(p).all())
            np.testing.assert_allclose(p.sum(1),1,atol=1e-12)
        np.testing.assert_array_equal(results['platt'][0],target.argmax(1))


if __name__=='__main__':
    unittest.main()
