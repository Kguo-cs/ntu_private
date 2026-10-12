"""Synthetic CPU protocol tests, excluded from actual search observations."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

here=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('independent_review',here/'overnight_independent_review_cpu.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def row(seed,gen,variant='control',ratio=1.):
    return {k:ratio for k in m.METRICS}|{'variant':variant,'seed':seed,'eval_seed':gen,
            'generated_onroad_fraction':.95,'num_generated_vehicles':1000,'num_gt_vehicles':1000}

class ReviewTests(unittest.TestCase):
    def test_actual_partial_epoch_exposure_bridges_batches(self):
        for s in (2048,4096,8192,10240,12288):
            self.assertEqual(m.exposure(s,512),m.exposure(s*2,256))
        self.assertEqual(m.exposure(4096,512),2097048)
        self.assertEqual(m.exposure(2048,512),1048524)
        self.assertNotEqual(m.exposure(4096,512),4096*512)
    def test_missing_gen_rejected(self):
        rows=[row(s,g) for s in (4817,5817) for g in (2817,3817)]
        with self.assertRaises(ValueError):m.seed_means(rows[:-1])
        rows[1]['eval_seed']=2817
        with self.assertRaises(ValueError):m.seed_means(rows)
    def test_gen_average_inside_training(self):
        rows=[row(4817,2817,ratio=.8),row(4817,3817,ratio=1.),row(5817,2817,ratio=.9),row(5817,3817,ratio=1.1)]
        means=m.seed_means(rows)
        self.assertAlmostEqual(means[4817]['speed_jsd'],.9)
        self.assertAlmostEqual(means[5817]['speed_jsd'],1.)
        self.assertAlmostEqual(m.mean_training(means)['speed_jsd'],.95)
    def test_pooled_gate_does_not_hide_single_training_failure(self):
        base=[row(s,g) for s in (4817,5817) for g in (2817,3817)]
        cand=copy.deepcopy(base)
        for r in cand:r['speed_jsd']=1.24 if r['seed']==4817 else .86
        result=m.paired_comparison(cand,base)
        self.assertTrue(result['pooled_gates_pass'])
        self.assertFalse(result['each_TRAIN_gates_pass'])
        self.assertFalse(result['eligible'])
        self.assertFalse(result['per_TRAIN']['4817']['eligible'])
    def test_wrong_medium_budget_rejected(self):
        exe={'label':'medium_candidate','completion_spec':{'name':'candidate','batch':256,'steps':4096,'seed':4817,'scenes':4096,'generations':[2817,3817],'budgets':[2048,4096]}}
        with self.assertRaises(ValueError):m.stage_plan(exe,{},None,{'promoted':['control','candidate']})
    def test_final_no_padding_and_materialization_evidence(self):
        old=m.ART
        with tempfile.TemporaryDirectory() as d:
            m.ART=Path(d);plan={'challenger':'candidate','actual_final_exposure_per_TRAIN':m.exposure(8192,512)}
            m.save(m.ART/'frozen_long_protocol.json',plan)
            records=[]
            for arm,seed,gen in __import__('itertools').product(('control','candidate'),(4817,5817),(2817,3817)):
                budget=8192 if arm=='control' else 16384
                r=row(seed,gen,arm,.95 if arm=='candidate' else 1.)
                r.update(budget=budget,seen_training_examples=plan['actual_final_exposure_per_TRAIN'])
                records.append({'phase':'long','row':r,'spec':{'steps':budget},'cohort':{'same':'actualFixture'}})
            result=m.final_review(records[:-1],{})
            self.assertFalse(result['matrix_complete']);self.assertEqual(len(result['missing_cells']),1)
            self.assertFalse(result['materialization_ready'])
            result=m.final_review(records,{})
            self.assertTrue(result['matrix_complete']);self.assertFalse(result['materialization_ready'])
            comp=result['comparison'];declaration={'complete_matrix':True,'frozen_challenger':'candidate','selected':'candidate','rows':[r['row'] for r in records],
              'ranking':[dict(name='candidate',R=comp['R'],eligible=comp['eligible'],per_training_seed=comp['per_TRAIN'])]}
            m.save(m.ART/'final_selection.json',declaration)
            result=m.final_review(records,{})
            self.assertTrue(result['materialization_ready']);self.assertTrue(result['selected_is_improvement'])
            declaration['selected']='control';m.save(m.ART/'final_selection.json',declaration)
            with self.assertRaises(ValueError):m.final_review(records,{})
        m.ART=old
    def test_full_requires_verified_final(self):
        records=[{'phase':'full50000','row':dict(row(4817,2817),budget=8192),'spec':{'steps':8192}}]
        with self.assertRaises(ValueError):m.full_review(records,{'matrix_complete':False})
    def test_frozen_protocol_mutation_is_flagged(self):
        old=m.ART
        with tempfile.TemporaryDirectory() as d:
            m.ART=Path(d);m.save(m.ART/'frozen_long_protocol.json',{'challenger':'first'})
            _,errors=m.pin_protocols();self.assertEqual(errors,[])
            m.save(m.ART/'frozen_long_protocol.json',{'challenger':'changed'})
            _,errors=m.pin_protocols();self.assertTrue(errors)
        m.ART=old

if __name__=='__main__':unittest.main()
