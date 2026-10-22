"""Self-contained software checks using constructed values only."""
import copy
import json
import numpy as np
import pandas as pd
import sys
import tempfile
import unittest
import warnings
from nssi_model.data import adapt_inputs, align_modalities, generate_seeds, load_schema, read_feature_csv, read_inputs, read_seeds, remove_overlapping_blood, save_seeds
from nssi_model.evaluate import METRIC_NAMES, coefficient_shares, evaluate, main, metrics, paired_comparisons, selected_features, selection_table, stable_feature_tests, summarize
from nssi_model.model import ConvergedLogisticRegression, IntervalCodes, PANEL_MARKERS, RobustVotingSelector, UniversalStacking, clinical_model, experts, logistic, numeric_pipeline, panel_model, required_columns
from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from types import SimpleNamespace
from unittest.mock import patch


class MethodTests(unittest.TestCase):

    def test_seeds_generated_saved_read_without_ranking(self):
        a = generate_seeds(7)
        self.assertEqual(a, generate_seeds(7))
        self.assertNotEqual(a, generate_seeds(8))
        self.assertEqual(len(set(a)), 20)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'seeds.json'
            save_seeds(path, 7)
            self.assertEqual(read_seeds(path)[0], a)
            with self.assertRaises(FileExistsError):
                save_seeds(path, 7)

    def test_every_test_set_has_eleven_per_group(self):
        y = np.array([0] * 53 + [1] * 54)
        for seed in generate_seeds():
            tr, te = train_test_split(np.arange(107), test_size=0.2, stratify=y, random_state=seed)
            self.assertEqual(len(tr), 85)
            self.assertEqual(np.bincount(y[te]).tolist(), [11, 11])
            self.assertFalse(set(tr) & set(te))

    def test_interval_extrema_median_scaling_use_training_only(self):
        train = pd.DataFrame({'x': [1.0, 3.0, 'min', 'max', np.nan]})
        pipe = numeric_pipeline().fit(train)
        assert_allclose(pipe.named_steps['intervals'].low_, [0.9])
        assert_allclose(pipe.named_steps['intervals'].high_, [3.3])
        assert_allclose(pipe.named_steps['impute'].statistics_, [2.0])
        before = pipe.named_steps['scale'].mean_.copy()
        pipe.transform(pd.DataFrame({'x': [10000.0, 'max', 'min']}))
        assert_allclose(pipe.named_steps['scale'].mean_, before)
        with self.assertRaises(ValueError):
            IntervalCodes().fit(pd.DataFrame({'x': ['<5', 2]}))

    def test_oof_genetics_and_expert_training_membership(self):
        X = pd.DataFrame({'PRS_EUR': np.arange(40.0), 'Rare_Burden': np.arange(40.0) ** 2}, index=[f'id{i}' for i in range(40)])
        y = np.array([0, 1] * 20)
        model = UniversalStacking([('dummy', DummyClassifier(strategy='prior'))], logistic())
        self.assertIsInstance(clone(model), UniversalStacking)
        model.fit(X, y)
        seen = []
        for audit, transform in zip(model.fold_audit_, model.genetic_fold_transforms_):
            self.assertFalse(set(audit['train']) & set(audit['validation']))
            seen.extend(audit['validation'])
            expected = X.loc[audit['train']].mean().to_numpy()
            assert_allclose(transform.named_steps['scale'].mean_, expected)
            rows = X.index.get_indexer(audit['validation'])
            assert_allclose(model.oof_genetics_[rows], transform.transform(X.loc[audit['validation']]))
            train_y = y[X.index.get_indexer(audit['train'])]
            assert_allclose(model.oof_probabilities_[rows, 0], train_y.mean())
        self.assertCountEqual(seen, X.index)
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(model.oof_meta_features_.shape, (40, 5))
        assert_allclose(model.oof_meta_features_[:, 3:], model.oof_probabilities_ * model.oof_genetics_)
        before = model.genome_transform_.named_steps['scale'].mean_.copy()
        model.predict_proba(X * 100)
        assert_allclose(model.genome_transform_.named_steps['scale'].mean_, before)

    def test_five_experts_and_voting_membership(self):
        self.assertEqual([n for n, _ in experts()], ['Met_EN', 'Met_RF', 'Inf_EN', 'Inf_RF', 'Blood_RF'])
        rng = np.random.default_rng(42)
        y = np.array([0, 1] * 30)
        X = rng.normal(size=(60, 8)) + 3 * y[:, None]
        selector = RobustVotingSelector().fit(X, y)
        self.assertEqual(selector.rf_support_.sum(), 2)
        self.assertTrue(np.all(selector.get_support() == (selector.votes_ >= 2)))
        self.assertTrue(np.all(selector.p_values_ < 0.05))

    def test_fixed_unknown_columns(self):
        X = pd.DataFrame({'clinical__age': [10, 12, 14, 16], 'clinical__bdi_ii': [1, 2, 3, 4], 'clinical__scared': [2, 4, 1, 3], 'clinical__sex': ['Male'] * 4, 'clinical__depression': ['No', 'Yes', 'No', 'Yes'], 'clinical__anxiety': ['No'] * 4})
        model = clinical_model().fit(X, [0, 1, 0, 1])
        self.assertEqual(model.named_steps['preprocess'].transform(X).shape[1], 12)
        X.loc[0, 'clinical__sex'] = 'Unknown'
        self.assertTrue(np.isfinite(model.predict_proba(X)).all())

    def test_subgroup_metrics_use_frozen_probabilities(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.2, 0.6, 0.4, 0.8])
        result = metrics(y, p)
        self.assertEqual(result['Accuracy'], 0.5)
        subset = metrics(y[[0, 3]], p[[0, 3]])
        self.assertEqual(subset['Accuracy'], 1.0)
        self.assertAlmostEqual(subset['Brier'], 0.04)

    def test_alignment_does_not_impute_or_scale(self):
        a = pd.DataFrame({'met__x': [100.0, np.nan]}, index=['b', 'a'])
        b = pd.DataFrame({'inf__x': [3.0, 4.0]}, index=['a', 'b'])
        labels = pd.Series([0, 1], index=['a', 'b'])
        X, y, report = align_modalities({'met': a, 'inf': b}, labels)
        self.assertEqual(X.index.tolist(), ['a', 'b'])
        self.assertTrue(pd.isna(X.loc['a', 'met__x']))
        self.assertEqual(X.loc['b', 'met__x'], 100.0)

    def test_panel_uses_only_fixed_seven_columns(self):
        rng = np.random.default_rng(42)
        X = pd.DataFrame(rng.normal(size=(30, 8)), columns=[f'f{i}' for i in range(8)])
        model = panel_model(dict(zip(PANEL_MARKERS, X.columns[:7]))).fit(X, [0, 1] * 15)
        p = model.predict_proba(X)
        X['f7'] = 100000
        assert_allclose(model.predict_proba(X), p)
        with self.assertRaises(ValueError):
            panel_model(['a'] * 7)


class TrainingSpy(TransformerMixin, BaseEstimator):
    fits = []

    def fit(self, X, y=None):
        self.__class__.fits.append(set(X.index))
        return self

    def transform(self, X):
        return np.asarray(X)


class PaperContractTests(unittest.TestCase):

    def test_prediction_requires_consumed_features_only(self):
        model = SimpleNamespace(fitted_base_models_=[SimpleNamespace(named_steps={'columns': SimpleNamespace(columns_=['met__a', 'blood__b'])})], use_genomics=True, genomic_columns=['PRS_EUR', 'Rare_Burden'], feature_names_in_=['met__a', 'blood__b', 'clinical__age', 'panel__ratio', 'PRS_EUR', 'Rare_Burden'])
        self.assertEqual(required_columns(model), {'met__a', 'blood__b', 'PRS_EUR', 'Rare_Burden'})

    def test_prediction_csv_rejects_ambiguous_headers_and_missing_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'input.csv'
            for content in ['sample_id,PRS_EUR,PRS_EUR\na,1,2\n', 'sample_id,PRS_EUR\n,1\n', 'sample_id,PRS_EUR\n  ,1\n']:
                path.write_text(content)
                with self.assertRaises(ValueError):
                    read_feature_csv(path)
            path.write_text('sample_id,PRS_EUR\nb,2\na,1\n')
            self.assertEqual(read_feature_csv(path).index.tolist(), ['b', 'a'])

    def test_nonconverged_fit_cannot_be_scored(self):
        X = np.random.default_rng(3).normal(size=(40, 8))
        model = ConvergedLogisticRegression(solver='saga', max_iter=1, tol=1e-15, random_state=42)
        with self.assertRaisesRegex(RuntimeError, 'no candidate score or prediction'):
            model.fit(X, np.array([0, 1] * 20))
        self.assertEqual(clone(model).max_iter, 1)

    def test_later_failure_keeps_completed_seed_evidence(self):
        X = pd.DataFrame({'PRS_EUR': np.arange(40), 'Rare_Burden': np.arange(40) % 3}, index=[f'id{i}' for i in range(40)])
        y = pd.Series([0, 1] * 20, index=X.index)
        base = [('Dummy', Pipeline([('model', DummyClassifier())]))]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'results'
            with patch('nssi_model.evaluate.experts', return_value=base), patch('nssi_model.evaluate.genomic_model', side_effect=[DummyClassifier(), RuntimeError('second seed failed')]), patch('nssi_model.evaluate.selected_features', return_value={}):
                with self.assertRaisesRegex(RuntimeError, 'second seed failed'):
                    evaluate(X, y, [1, 2], output, clinical=False, ablations=False, software_test=True)
            self.assertEqual(json.loads((output / 'completed_seeds.json').read_text()), [1])
            for name in ['test_predictions', 'oof_meta_inputs', 'meta_coefficients']:
                self.assertEqual(set(pd.read_csv(output / f'{name}_partial.csv').seed), {1})
            self.assertTrue(json.loads((output / 'inner_folds_partial.json').read_text()))
            self.assertFalse((output / 'completed.json').exists())

    def test_paper_hyperparameters(self):
        base = dict(experts())
        self.assertEqual(len(base), 5)
        for name in ['Met_RF', 'Inf_RF', 'Blood_RF']:
            forest = base[name].named_steps['model']
            self.assertEqual(forest.n_estimators, 100)
            self.assertEqual(forest.class_weight, 'balanced')
        self.assertEqual(base['Blood_RF'].named_steps['model'].max_depth, 5)
        for name in ['Met_EN', 'Inf_EN']:
            search = base[name].named_steps['search']
            self.assertEqual(search.cv.n_splits, 5)
            self.assertEqual(search.param_grid['model__l1_ratio'], [0.5, 0.7])
            self.assertEqual(search.estimator.named_steps['model'].solver, 'saga')
        meta = logistic()
        self.assertEqual((meta.C, meta.max_iter, meta.tol, meta.solver), (0.1, 100, 0.0001, 'lbfgs'))

    def test_failed_run_retains_manifest_and_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data, mapping, output = (tmp / 'input.csv', tmp / 'map.json', tmp / 'result')
            data.write_text('placeholder')
            mapping.write_text(json.dumps({'version': 1}))
            X = pd.DataFrame({'met__a': [1, 2], 'inf__b': [2, 3], 'blood__c': [3, 4]})
            y = pd.Series([0, 1])

            def fail(*args, **kwargs):
                warnings.warn('deliberate convergence test', ConvergenceWarning)
                raise ValueError('deliberate failure')
            argv = ['evaluate', '--data', str(data), '--schema', str(mapping), '--seeds', 'unused.json', '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch('nssi_model.evaluate.adapt_inputs', return_value=(X, {'met__a': 'a', 'inf__b': 'b', 'blood__c': 'c'}, None)), patch('nssi_model.evaluate.read_inputs', return_value=(X, y)), patch('nssi_model.evaluate.read_seeds', return_value=(list(range(20)), {'seeds': list(range(20))})), patch('nssi_model.evaluate.evaluate', side_effect=fail):
                with self.assertRaisesRegex(ValueError, 'deliberate failure'):
                    main()
            status = json.loads((output / 'run_status.json').read_text())
            self.assertEqual(status['status'], 'failed')
            self.assertEqual(status['convergence_warnings'], 1)
            self.assertTrue((output / 'run_manifest.json').exists())
            manifest = json.loads((output / 'run_manifest.json').read_text())
            self.assertFalse(manifest['options']['omics_only'])
            self.assertIn('schema_sha256', manifest)
            self.assertFalse((output / 'completed.json').exists())

    def test_tuning_refits_preprocessing_on_each_tuning_train(self):
        X = pd.DataFrame(np.random.default_rng(2).normal(size=(30, 3)), columns=['met__a', 'met__b', 'met__c'])
        y = np.array([0, 1] * 15)
        expert = dict(experts())['Met_EN']
        search = expert.named_steps['search']
        search.set_params(param_grid={'model__C': [1.0], 'model__l1_ratio': [0.5]})
        search.estimator.set_params(preprocess=TrainingSpy())
        TrainingSpy.fits = []
        expert.fit(X, y)
        expected = [set(X.index[tr]) for tr, _ in StratifiedKFold(5).split(X, y)] + [set(X.index)]
        self.assertEqual(TrainingSpy.fits, expected)

    def test_selection_exact_nonzero_and_run_union(self):
        fake = SimpleNamespace(base_names_=['Met_EN'], fitted_base_models_=[SimpleNamespace(named_steps={'columns': SimpleNamespace(columns_=['met__tiny', 'met__zero']), 'search': SimpleNamespace(best_estimator_=SimpleNamespace(named_steps={'model': SimpleNamespace(coef_=np.array([[1e-09, 0.0]]))}))})])
        self.assertEqual(selected_features(fake)['Met_EN'], ['met__tiny'])
        runs = [{'met__a'} for _ in range(16)] + [set() for _ in range(4)]
        table = selection_table(['met__a', 'met__never'], runs).set_index('feature')
        self.assertTrue(table.loc['met__a', 'stable'])
        self.assertEqual(table.loc['met__never', 'count'], 0)
        runs[15] = set()
        self.assertFalse(selection_table(['met__a'], runs).iloc[0].stable)

    def test_fig2e_seven_groups_and_mean_absolute_formula(self):
        terms = ['Met_EN', 'Met_RF', 'Inf_EN', 'Inf_RF', 'Blood_RF', 'PRS_EUR', 'Rare_Burden', 'Met_EN_x_PRS_EUR', 'Met_EN_x_Rare_Burden']
        coefficients = pd.DataFrame([{'seed': seed, 'term': term, 'coefficient': sign} for seed, sign in [(1, 1), (2, -1)] for term in terms])
        shares = coefficient_shares(coefficients)
        self.assertEqual(len(shares), 7)
        self.assertAlmostEqual(shares.sum(), 100)
        self.assertAlmostEqual(shares['Metabolism'], 200 / 9)
        self.assertAlmostEqual(shares['PRS'], 100 / 9)

    def test_bh_applies_to_observed_stable_family(self):
        X = pd.DataFrame({'a': [1, 2, 3, 8, 9, 10], 'b': [1, 3, 5, 2, 4, 6]})
        y = pd.Series([0, 0, 0, 1, 1, 1])
        table = stable_feature_tests(X, y, ['a', 'b'])
        self.assertTrue((table.q_value >= table.p_value).all())
        self.assertEqual(len(table), 2)

    def test_overlap_is_removed_by_dictionary_not_similar_names(self):
        X = pd.DataFrame({'inf__IL6': [1], 'blood__cytokine': [2], 'blood__other': [3]})
        result, dropped = remove_overlapping_blood(X, {'inf__IL6': 'IL6', 'blood__cytokine': 'IL6', 'blood__other': 'OTHER'})
        self.assertEqual(dropped, ['blood__cytokine'])
        self.assertEqual(result.columns.tolist(), ['inf__IL6', 'blood__other'])
        with self.assertRaises(ValueError):
            remove_overlapping_blood(X, {})

    def test_labels_must_be_aligned_and_full_runs_need_20_seeds(self):
        X = pd.DataFrame({'PRS_EUR': range(20), 'Rare_Burden': range(20)})
        y = pd.Series([0, 1] * 10)
        model = UniversalStacking([('dummy', DummyClassifier())], logistic())
        with self.assertRaises(ValueError):
            model.fit(X, y.iloc[::-1])
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                evaluate(X, y, [42], Path(tmp) / 'output')

    def test_nonzero_paired_differences_not_silenced_by_allclose(self):
        scores = pd.DataFrame([{'seed': seed, 'model': model, 'subset': 'complete', 'AUC': 0.7 + (1e-09 if model == 'Omics' else 0), 'Brier': 0.2 - (1e-09 if model == 'Omics' else 0)} for seed in range(20) for model in ['Omics', 'Clinical']])
        self.assertTrue((paired_comparisons(scores).p_one_sided < 0.001).all())


class InterfaceTests(unittest.TestCase):
    def test_literal_categories_survive_csv_loading(self):
        cases = [(["NA", "NULL"], {"NA": "No", "NULL": "Yes"}, ["No", "Yes"]),
                 (["01", "02"], {"01": "No", "02": "Yes"}, ["No", "Yes"]),
                 (["1", ""], {"1": "Yes"}, ["Yes", None])]
        for raw, mapping, expected in cases:
            with self.subTest(raw=raw), tempfile.TemporaryDirectory() as tmp:
                schema = copy.deepcopy(self.schema)
                schema["clinical"] = {"depression_diagnosis": {"column": "diagnosis", "values": mapping}}
                frame = self.frame.assign(diagnosis=raw)
                frame["key"] = ["001", "NA"]
                result, _, _ = self.adapt(tmp, frame, schema)
                self.assertEqual(result.index.tolist(), ["001", "NA"])
                self.assertEqual(result["clinical__depression"].iloc[0], expected[0])
                if expected[1] is None:
                    self.assertTrue(pd.isna(result["clinical__depression"].iloc[1]))
                else:
                    self.assertEqual(result["clinical__depression"].iloc[1], expected[1])
                self.assertTrue(np.issubdtype(result["PRS_EUR"].dtype, np.number))

    def test_missing_tokens_are_explicit_and_cannot_conflict_with_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema = copy.deepcopy(self.schema)
            schema["clinical"] = {"depression_diagnosis": {"column": "diagnosis", "values": {"Yes": "Yes"}, "missing_values": ["", "NULL"]}}
            result, _, _ = self.adapt(tmp, self.frame.assign(diagnosis=["NULL", "Yes"]), schema)
            self.assertTrue(pd.isna(result["clinical__depression"].iloc[0]))
            schema["clinical"]["depression_diagnosis"]["values"]["NULL"] = "No"
            with self.assertRaisesRegex(ValueError, "overlap"):
                self.adapt(tmp, self.frame.assign(diagnosis=["NULL", "Yes"]), schema)


    def setUp(self):
        self.schema = {'version': 1, 'id_column': 'key', 'outcome': {'column': 'target', 'values': {'negative': 0, 'positive': 1}}, 'modalities': {block: {'alias': {'column': source, 'analyte_id': source}} for block, source in [('metabolomics', 'a'), ('inflammation', 'b'), ('blood', 'c')]}, 'genetics': {'polygenic_score': {'column': 'd'}, 'rare_variant_burden': {'column': 'e'}}}
        self.frame = pd.DataFrame({'key': ['test_b', 'test_a'], 'target': ['negative', 'positive'], 'a': [1.0, 2.0], 'b': [3.0, 4.0], 'c': [5.0, 6.0], 'd': [0.1, 0.2], 'e': [0, 1]})

    def adapt(self, directory, frame, schema, **kwargs):
        path = Path(directory) / 'input.csv'
        frame.to_csv(path, index=False)
        return adapt_inputs(path, schema, **kwargs)

    def test_arbitrary_source_names_have_identical_internal_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, _, _ = self.adapt(tmp, self.frame, self.schema)
            schema = copy.deepcopy(self.schema)
            schema['id_column'] = 'unrelated_identifier'
            schema['modalities']['metabolomics']['alias']['column'] = 'unrelated_measurement'
            second, _, _ = self.adapt(tmp, self.frame.rename(columns={'key': 'unrelated_identifier', 'a': 'unrelated_measurement'}), schema)
            assert_frame_equal(first, second)
            self.assertEqual(first.index.tolist(), ['test_b', 'test_a'])
            self.assertEqual(first['label'].tolist(), [0, 1])

    def test_inference_needs_no_outcome_or_unused_clinical_columns(self):
        schema = copy.deepcopy(self.schema)
        schema['clinical'] = {'age': {'column': 'not_available'}}
        required = {'met__alias', 'inf__alias', 'blood__alias', 'PRS_EUR', 'Rare_Burden'}
        with tempfile.TemporaryDirectory() as tmp:
            result, _, _ = self.adapt(tmp, self.frame.drop(columns='target'), schema, outcome=False, required=required)
            self.assertEqual(set(result), required)

    def test_reject_unmapped_outcome_and_duplicate_source_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            frame = self.frame.copy()
            frame.loc[0, 'target'] = 'undefined'
            with self.assertRaisesRegex(ValueError, 'Unmapped category'):
                self.adapt(tmp, frame, self.schema)
            schema = copy.deepcopy(self.schema)
            schema['genetics']['polygenic_score']['column'] = 'target'
            with self.assertRaisesRegex(ValueError, 'Outcome column'):
                self.adapt(tmp, self.frame, schema)

    def test_schema_rejects_duplicate_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'schema.json'
            path.write_text('{"version": 1, "version": 1}')
            with self.assertRaisesRegex(ValueError, 'Duplicate schema key'):
                load_schema(path)

    def test_panel_rejects_outcome_and_repeated_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            schema = copy.deepcopy(self.schema)
            schema['panel'] = {'CXCL5': schema['outcome']}
            with self.assertRaisesRegex(ValueError, 'Outcome column'):
                self.adapt(tmp, self.frame, schema)
            schema['panel'] = {'CXCL5': {'column': 'a'}, 'IL24': {'column': 'a'}}
            with self.assertRaisesRegex(ValueError, 'distinct source'):
                self.adapt(tmp, self.frame, schema)
            schema['panel'] = {'CXCL5': {'column': 'd'}}
            with self.assertRaisesRegex(ValueError, 'genetic or clinical'):
                self.adapt(tmp, self.frame, schema)
            schema['panel'] = {'CXCL5': {'column': 'a'}}
            result, _, panel = self.adapt(tmp, self.frame, schema)
            self.assertEqual(result[panel['CXCL5']].tolist(), self.frame.a.tolist())


class ExtendedMetricsTests(unittest.TestCase):
    def test_s12_metrics_against_hand_calculation(self):
        result = metrics([0, 0, 1, 1], [.1, .6, .4, .8])
        self.assertAlmostEqual(result["AUPRC"], 5/6)
        for name in ["Balanced_accuracy", "PPV", "NPV", "F1"]:
            self.assertAlmostEqual(result[name], .5)
        self.assertAlmostEqual(result["AUC"], .75)

    def test_undefined_metrics_and_summary_denominators(self):
        empty = metrics([], [])
        self.assertTrue(all(np.isnan(empty[name]) for name in METRIC_NAMES))
        negative = metrics([0, 0], [.1, .2])
        for name in ["AUC", "AUPRC", "Balanced_accuracy", "Sensitivity", "PPV", "F1"]:
            self.assertTrue(np.isnan(negative[name]))
        self.assertEqual(negative["NPV"], 1)
        positive = metrics([1, 1], [.8, .9])
        self.assertTrue(np.isnan(positive["NPV"]))
        self.assertTrue(np.isnan(positive["Specificity"]))
        self.assertEqual(positive["F1"], 1)
        rows = [{"model": "Omics", "subset": "depression", **r} for r in [negative, metrics([0, 1], [.1, .9])]]
        table = summarize(pd.DataFrame(rows)).set_index("metric")
        self.assertEqual(set(table.index), set(METRIC_NAMES))
        self.assertEqual(table.loc["AUPRC", "n_splits"], 1)
        self.assertEqual(table.loc["NPV", "n_splits"], 2)

    def test_evaluation_exports_all_s12_metrics_and_paired_changes(self):
        X = pd.DataFrame({"PRS_EUR": np.arange(40.), "Rare_Burden": np.arange(40.)%3,
                          "met__a": np.arange(40.), "inf__b": np.arange(40.), "blood__c": np.arange(40.),
                          "clinical__depression": ["Yes"]*40})
        y = pd.Series([0, 1]*20)
        base = [("Blood_RF", Pipeline([("model", DummyClassifier())]))]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"results"
            with patch("nssi_model.evaluate.experts", return_value=base), patch("nssi_model.evaluate.genomic_model", return_value=DummyClassifier()), patch("nssi_model.evaluate.selected_features", return_value={"Blood_RF": ["blood__c"]}):
                evaluate(X, y, [1, 2], output, clinical=False, ablations=False, software_test=True)
            self.assertTrue(set(METRIC_NAMES).issubset(pd.read_csv(output/"metrics.csv").columns))
            self.assertEqual(set(pd.read_csv(output/"summary.csv").metric), set(METRIC_NAMES))
            paired = pd.read_csv(output/"depression_paired_deltas.csv")
            self.assertTrue({"depression_minus_complete_"+m for m in METRIC_NAMES}.issubset(paired.columns))
            self.assertEqual(set(pd.read_csv(output/"depression_paired_delta_summary.csv").metric), set(METRIC_NAMES))


class SerializationTests(unittest.TestCase):
    def test_model_roundtrip_after_module_merge(self):
        import joblib
        X = pd.DataFrame({"PRS_EUR": np.arange(40.), "Rare_Burden": np.arange(40.) % 3})
        model = UniversalStacking([("dummy", DummyClassifier())], logistic()).fit(X, np.array([0, 1]*20))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"model.joblib"
            joblib.dump(model, path)
            restored = joblib.load(path)
            assert_allclose(restored.predict_proba(X), model.predict_proba(X))


if __name__ == "__main__":
    unittest.main()
