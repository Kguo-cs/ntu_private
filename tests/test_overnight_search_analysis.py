"""Statistical CPU checks for independent TRAIN counts and physical grouping."""
from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile

from tools import overnight_search_analysis as analysis


def row(variant='control', seed=4817, gen=2817, batch=512, budget=4096):
    return {'variant': variant, 'arm_id': variant + '@cfg', 'seed': seed, 'eval_seed': gen,
            'budget': budget, 'train_batch_size': batch, 'seen_training_examples': 2097048,
            'evaluation_weights': 'ema', 'planned_generation_seeds': [2817, 3817],
            'cohort_sha256': 'samephysical', 'cohort': {'matched': True, 'num_samples': 4096}, 'phase': 'long',
            'evaluation_only': False, 'initial_generator_sha256': f'seed{seed}',
            'stage': f'long_seed{seed}_{variant}', 'train_seconds': 1500., 'eval_seconds': 20.,
            **{key: 1. for key in analysis.METRICS}, 'generated_onroad_fraction': .8,
            'num_generated_vehicles': 100., 'num_gt_vehicles': 100.}


class AggregationTest(unittest.TestCase):
    def test_four_generation_observations_are_two_training_seeds_with_sample_sd(self):
        rows = [row(seed=seed, gen=gen) for seed in (4817, 5817) for gen in (2817, 3817)]
        for item, value in zip(rows, [1., 3., 5., 7.]):
            item['speed_jsd'] = value
        summary = analysis.statistics_by_arm(rows)
        self.assertEqual(summary['independent_training_N'], 2)
        self.assertEqual(summary['actual_GEN_observations'], 4)
        self.assertEqual(summary['metrics_native']['speed_jsd']['mean'], 4.)
        self.assertAlmostEqual(summary['metrics_native']['speed_jsd']['sd'], 2 ** 1.5)
        self.assertEqual(summary['metrics_raw_jsd_collision_percent']['speed_jsd']['mean'], .04)

    def test_single_training_seed_has_undefined_sd(self):
        summary = analysis.statistics_by_arm([row(gen=gen) for gen in (2817, 3817)])
        self.assertIsNone(summary['metrics_native']['speed_jsd']['sd'])
        self.assertFalse(summary['metrics_native']['speed_jsd']['sample_sd_defined'])

    def test_missing_or_duplicate_generation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing or duplicate GEN'):
            analysis.seed_means([row()])
        with self.assertRaisesRegex(ValueError, 'missing or duplicate GEN'):
            analysis.seed_means([row(), row(), row(gen=3817)])

    def test_batch_methods_bridge_only_actual_equal_exposure(self):
        rows = [row(gen=gen) for gen in (2817, 3817)] + [
            row('winner', gen=gen, batch=256, budget=8192) for gen in (2817, 3817)]
        groups = analysis.make_groups(rows)
        self.assertEqual(len(groups), 1)
        self.assertEqual(len(groups[0]['arms']), 2)
        self.assertTrue(groups[0]['arms'][1]['comparison']['ranking_eligible'])
        changed = [dict(item) for item in rows]
        changed[-1]['cohort_sha256'] = 'differentmodelsource'
        with self.assertRaisesRegex(ValueError, 'missing or duplicate GEN'):
            analysis.make_groups(changed)

    def test_unmatched_training_set_never_becomes_confirmation(self):
        candidate = [row('winner', gen=gen) for gen in (2817, 3817)]
        control = [row(seed=seed, gen=gen) for seed in (4817, 5817) for gen in (2817, 3817)]
        comparison = analysis.compare(candidate, control)
        self.assertFalse(comparison['ranking_eligible'])
        self.assertIn('No complete matched TRAIN', comparison['reason'])

    def test_failed_independent_seed_gate_cannot_hide_inside_pooled_mean(self):
        candidate = [row('winner', seed=seed, gen=gen) for seed in (4817, 5817) for gen in (2817, 3817)]
        control = [row(seed=seed, gen=gen) for seed in (4817, 5817) for gen in (2817, 3817)]
        for item in candidate:
            item['speed_jsd'] = .8 if item['seed'] == 4817 else 1.2
        comparison = analysis.compare(candidate, control)
        self.assertTrue(comparison['pooled_gates_pass'])
        self.assertFalse(comparison['each_training_seed_gates_pass'])
        self.assertFalse(comparison['ranking_eligible'])

    def test_fit_seconds_are_counted_once_not_per_gen(self):
        rows = [row(seed=seed, gen=gen) for seed in (4817, 5817) for gen in (2817, 3817)]
        timing = analysis.timing(rows, [])
        self.assertEqual(timing['total_fit_seconds'], 3000.)
        self.assertEqual(timing['total_actual_eval_seconds'], 80.)


class FigureStatisticsTest(unittest.TestCase):
    def test_publication_sizes_seed_counts_and_equal_gen_histogram_weighting(self):
        with tempfile.TemporaryDirectory(prefix='sim-overnight-chart-fixture-') as directory:
            root = Path(directory)
            (root / 'FIXTURE_SCOPE.txt').write_text('CPU synthetic statistics/rendering fixture only. Not training/search results.\n')
            observations = []
            # Partial N1 then complete N2 at a larger observed exposure.
            for exposure, budget, seeds in ((1048524, 2048, [4817]), (2097048, 4096, [4817, 5817])):
                for seed in seeds:
                    for variant, batch in (('control', 512), ('lane_center_normalized', 256)):
                        for gen in (2817, 3817):
                            value = row(variant, seed, gen, batch, budget * (512 // batch))
                            value['seen_training_examples'] = exposure
                            value['speed_jsd'] = 1. if seed == 4817 else 3.
                            observations.append(value)
            for index in range(18):
                value = row('control' if index == 0 else 'normalized_speed_p0_' + str(index))
                value.update(phase='screen', cohort_sha256='screenfixture',
                             planned_generation_seeds=[2817], seen_training_examples=393216)
                observations.append(value)
            # Different per-GEN vehicle counts exercise equal-normalized-GEN
            # averaging, which must differ from pooling raw vehicle counts.
            full = []
            for variant in ('control', 'lane_center_normalized'):
                for gen, counts in ((2817, [10, 10, 0]), (3817, [0, 10, 30])):
                    value = row(variant, gen=gen)
                    value.update(phase='full50000', num_samples=50000, cohort_sha256='fullfixture',
                                 num_gt_onroad_vehicles=80, gt_onroad_fraction=.8,
                                 num_generated_vehicles=sum(counts), report_source=str(root / variant / str(gen) / 'report.json'))
                    path = Path(value['report_source']).parent
                    path.mkdir(parents=True)
                    histogram = {key: {'minimum': 0., 'maximum': 3., 'bin_width': 1.,
                                      'matched_gt': [50, 25, 25], 'generated': counts} for key in analysis.ATTR}
                    (path / 'feature_histograms.json').write_text(json.dumps(histogram))
                    full.append(value)
            groups = analysis.make_groups(observations)
            metadata = analysis.figures(observations + full, groups, root / 'figures')
            self.assertEqual([len(page['arms']) for page in metadata['heatmaps']], [8, 8, 2])
            self.assertEqual(metadata['full50000']['status'], 'actual_complete')
            for item in metadata['layout_audit'].values():
                self.assertLessEqual(item['native_width_pt'], 511.000001)
                self.assertGreaterEqual(item['minimum_effective_font_at_report_box_pt'], 8.)
                self.assertEqual(item['nonempty_visible_text_outside_canvas'], [])
            curve = next(item for item in metadata['long_curves'] if item['metric'] == 'speed_jsd')
            for arm in curve['curves']:
                self.assertEqual(arm['independent_TRAIN_N'], [1, 2])
                self.assertEqual(arm['mean'], [.01, .02])
                self.assertIsNone(arm['sample_sd_across_TRAIN'][0])
                self.assertAlmostEqual(arm['sample_sd_across_TRAIN'][1], 2 ** .5 / 100)
            for item in metadata['full50000']['bin_statistics']:
                self.assertEqual(item['mean_normalized_bin_probability_across_GEN'], [.25, .375, .375])
            # A partially written fullscope stage cannot turn into a fake chart.
            skipped = analysis.full_histogram_figures(full[:3], root / 'figures',
                                                       __import__('tools.plot_frozen_map_search', fromlist=['save_figure']))
            self.assertEqual(skipped['status'], 'missing')


if __name__ == '__main__':
    unittest.main()
