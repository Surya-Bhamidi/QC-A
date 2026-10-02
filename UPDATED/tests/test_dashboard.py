import unittest
from dashboard.app import app


class DashboardTests(unittest.TestCase):
    def test_research_labels_and_missing_checkpoint(self):
        c=app.test_client()
        response=c.get('/')
        self.assertEqual(response.status_code,200)
        self.assertIn(b'not a clinical diagnostic tool',response.data)
        self.assertEqual(c.post('/api/predict',json={'run':'does-not-exist'}).status_code,400)

    def test_recorded_model_and_emulated_shot_accounting(self):
        c=app.test_client()
        models=c.get('/api/models').get_json()['models']
        fidelity=next((m for m in models if m['model']=='fidelity'),None)
        if fidelity is None:
            self.skipTest('No completed research checkpoint; train the central suite for dashboard integration test')
        response=c.post('/api/predict',json={'run':fidelity['id'],'sample':0,'mode':'emulator','shots':16,'policy':'combined'})
        self.assertEqual(response.status_code,200)
        r=response.get_json()
        self.assertEqual(r['actual_circuit_calls'],0)
        self.assertEqual(r['total_shots'],1156*16)
        self.assertAlmostEqual(sum(r['probabilities']),1.,places=6)
        self.assertEqual(c.post('/api/predict',json={'run':fidelity['id'],'sample':999999}).status_code,400)
        self.assertEqual(c.post('/api/predict',json={'run':fidelity['id'],'circuit_method':'invalid'}).status_code,400)
