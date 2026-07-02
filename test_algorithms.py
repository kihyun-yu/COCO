import math
import unittest

import numpy as np

from algorithms import (
    COCOWithoutSlater2026,
    COCOWithoutSlaterConfig,
    YuNeelyWei2017,
    YuNeelyWei2017Doubling,
    YuNeelyWeiConfig,
)
from functions import AffineFunction
from functions import QuadraticFunction
from problems import (
    ConflictingStochasticConstraints,
    GradualSlaterStochasticConstraints,
    NoSlaterStochasticConstraints,
)
from sets import BoxSet


class AlgorithmTests(unittest.TestCase):
    def test_plain_python_lists_are_accepted_for_vectors(self):
        box = BoxSet(lower=[0.0, 0.0], upper=[1.0, 1.0])
        loss = AffineFunction(a=[1.0, 0.0])
        constraint = AffineFunction(a=[0.5, 0.5], b=-0.25)
        alg = YuNeelyWei2017(
            feasible_set=box,
            x=[0.5, 0.5],
            num_constraints=1,
            config=YuNeelyWeiConfig(V=1.0, alpha=2.0),
        )

        out = alg.step(loss, [constraint])

        self.assertEqual(out["t"], 1)
        self.assertTrue(np.all(alg.x >= box.lower))
        self.assertTrue(np.all(alg.x <= box.upper))

    def test_conflicting_problem_is_expected_feasible_but_not_round_wise_feasible(self):
        problem = ConflictingStochasticConstraints(dim=2)
        self.assertAlmostEqual(problem.expected_constraint(problem.comparator), 0.0)
        self.assertFalse(problem.round_wise_feasible_region_nonempty(np.array([1, 1, 1])))

    def test_conflicting_constraint_sampler_is_time_varying(self):
        problem = ConflictingStochasticConstraints(dim=2)
        early = problem.constraint_probabilities(round_index=1)
        later = problem.constraint_probabilities(round_index=80)

        self.assertAlmostEqual(float(np.sum(early)), 1.0)
        self.assertAlmostEqual(float(np.sum(later)), 1.0)
        self.assertTrue(np.all(early > 0.0))
        self.assertTrue(np.all(later > 0.0))
        self.assertFalse(np.allclose(early, later))

    def test_loss_schedule_changes_with_round_index(self):
        problem = ConflictingStochasticConstraints(dim=2)
        first_loss, _, _ = problem.sample_round(np.random.default_rng(0), round_index=1)
        later_loss, _, _ = problem.sample_round(np.random.default_rng(0), round_index=90)
        probe = np.array([0.4, 0.6])

        self.assertNotAlmostEqual(first_loss.value(probe), later_loss.value(probe))
        self.assertFalse(np.allclose(first_loss.gradient(probe), later_loss.gradient(probe)))

    def test_problem_complexity_switches_loss_and_constraints(self):
        simple = ConflictingStochasticConstraints(dim=2, complexity="simple")
        complicated = ConflictingStochasticConstraints(dim=2, complexity="complicated")
        simple_loss, simple_constraint, _ = simple.sample_round(np.random.default_rng(1), round_index=1)
        complicated_loss, complicated_constraint, _ = complicated.sample_round(
            np.random.default_rng(1), round_index=1
        )

        self.assertIsInstance(simple_loss, QuadraticFunction)
        self.assertIsInstance(simple_constraint, AffineFunction)
        self.assertNotIsInstance(complicated_loss, QuadraticFunction)
        self.assertNotIsInstance(complicated_constraint, AffineFunction)

    def test_no_slater_problem_has_only_boundary_feasible_comparator(self):
        problem = NoSlaterStochasticConstraints(dim=2)
        self.assertAlmostEqual(problem.expected_constraint(problem.comparator), 0.0)
        self.assertGreater(problem.expected_constraint(np.array([0.1, 0.0])), 0.0)
        self.assertFalse(problem.round_wise_feasible_region_nonempty(np.array([1, 1])))

    def test_gradual_slater_margin_controls_strict_feasibility(self):
        slater_problem = GradualSlaterStochasticConstraints(dim=2, margin=0.1)
        no_slater_limit = GradualSlaterStochasticConstraints(dim=2, margin=0.0)

        self.assertLess(slater_problem.expected_constraint(np.zeros(2)), 0.0)
        self.assertEqual(no_slater_limit.expected_constraint(np.zeros(2)), 0.0)

    def test_yu_neely_wei_2017_updates_inside_box_and_nonnegative_queue(self):
        box = BoxSet(lower=np.zeros(2), upper=np.ones(2))
        alg = YuNeelyWei2017(
            feasible_set=box,
            x=np.array([0.5, 0.5]),
            num_constraints=1,
            config=YuNeelyWeiConfig(V=1.0, alpha=2.0),
        )
        loss = AffineFunction(a=np.array([1.0, 0.0]))
        constraint = AffineFunction(a=np.array([0.5, 0.5]), b=-0.25)

        out = alg.step(loss, [constraint])

        self.assertEqual(out["t"], 1)
        self.assertTrue(np.all(alg.x >= box.lower))
        self.assertTrue(np.all(alg.x <= box.upper))
        self.assertGreaterEqual(alg.Q[0], 0.0)

    def test_yu_neely_wei_2017_doubling_advances_epochs(self):
        box = BoxSet(lower=np.zeros(2), upper=np.ones(2))
        alg = YuNeelyWei2017Doubling(
            feasible_set=box,
            x=np.array([0.5, 0.5]),
            num_constraints=1,
        )
        loss = AffineFunction(a=np.array([1.0, 0.0]))
        constraint = AffineFunction(a=np.array([0.5, 0.5]), b=-0.25)

        first = alg.step(loss, [constraint])
        second = alg.step(loss, [constraint])

        self.assertEqual(first["epoch_horizon"], 1)
        self.assertEqual(second["epoch_horizon"], 2)
        self.assertTrue(np.all(alg.x >= box.lower))
        self.assertTrue(np.all(alg.x <= box.upper))

    def test_coco_without_slater_2026_updates_inside_box(self):
        box = BoxSet(lower=np.zeros(2), upper=np.ones(2))
        alg = COCOWithoutSlater2026(
            feasible_set=box,
            x=np.array([0.5, 0.5]),
            config=COCOWithoutSlaterConfig(L=2.0, G=1.0, D=math.sqrt(2.0)),
        )
        loss = AffineFunction(a=np.array([1.0, -0.5]))
        constraint = AffineFunction(a=np.array([0.5, 0.5]), b=-0.4)

        out = alg.step(loss, constraint)

        self.assertEqual(out["t"], 1)
        self.assertGreater(out["eta"], 0.0)
        self.assertGreater(out["phi_prime"], 0.0)
        self.assertTrue(np.all(alg.x >= box.lower))
        self.assertTrue(np.all(alg.x <= box.upper))

    def test_coco_without_slater_2026_accepts_practical_gamma_scale(self):
        box = BoxSet(lower=np.zeros(2), upper=np.ones(2))
        alg = COCOWithoutSlater2026(
            feasible_set=box,
            x=np.array([0.5, 0.5]),
            config=COCOWithoutSlaterConfig(
                L=2.0,
                G=1.0,
                D=math.sqrt(2.0),
                gamma_scale=50.0,
                regularizer_scale=0.0,
            ),
        )
        loss = AffineFunction(a=np.array([1.0, -0.5]))
        constraint = AffineFunction(a=np.array([0.5, 0.5]), b=-0.4)

        out = alg.step(loss, constraint)

        self.assertGreater(out["gamma"], 1.0 / (24.0 * math.sqrt(2.0) * 2.0))


if __name__ == "__main__":
    unittest.main()
