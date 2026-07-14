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
from compare_algorithms import _exact_empirical_comparator
from problems import (
    ConflictingStochasticConstraints,
    GradualSlaterStochasticConstraints,
    NoSlaterStochasticConstraints,
)
from sets import BoxSet


class AlgorithmTests(unittest.TestCase):
    def test_empirical_comparator_uses_all_realized_objectives(self):
        losses = [
            QuadraticFunction(center=np.array([0.8, 0.2])),
            QuadraticFunction(center=np.array([0.8, 0.2])),
        ]

        comparator = _exact_empirical_comparator(
            losses, problem_name="gradual-slater", slater_margin=0.5
        )

        np.testing.assert_allclose(comparator, np.array([0.5, 0.0]))

    def test_problem_defaults_to_simple_five_dimensional_objective(self):
        problem = ConflictingStochasticConstraints()
        loss, constraint, _ = problem.sample_round(np.random.default_rng(0), round_index=1)

        self.assertEqual(problem.dim, 5)
        self.assertIsInstance(loss, QuadraticFunction)
        self.assertIsInstance(constraint, AffineFunction)
        self.assertEqual(loss.center.shape, (5,))
        self.assertAlmostEqual(loss.weight, 1.0)

    def test_solution_dimension_is_capped_at_ten(self):
        with self.assertRaisesRegex(ValueError, "between 2 and 10"):
            ConflictingStochasticConstraints(dim=11)

    def test_extra_comparator_coordinates_match_average_loss_center(self):
        problem = ConflictingStochasticConstraints(dim=5)
        np.testing.assert_allclose(problem.comparator[2:], 0.5)

    def test_constraint_stream_is_independent_of_solution_dimension(self):
        problem_2d = ConflictingStochasticConstraints(dim=2)
        problem_5d = ConflictingStochasticConstraints(dim=5)
        loss_rng_2d = np.random.default_rng(1)
        loss_rng_5d = np.random.default_rng(1)
        constraint_rng_2d = np.random.default_rng(2)
        constraint_rng_5d = np.random.default_rng(2)

        samples_2d = [
            problem_2d.sample_round(loss_rng_2d, t, constraint_rng_2d)[2]
            for t in range(1, 30)
        ]
        samples_5d = [
            problem_5d.sample_round(loss_rng_5d, t, constraint_rng_5d)[2]
            for t in range(1, 30)
        ]
        self.assertEqual(samples_2d, samples_5d)

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
        later_loss, _, _ = problem.sample_round(np.random.default_rng(0), round_index=45)
        probe = np.array([0.3, 0.4])

        self.assertNotAlmostEqual(first_loss.value(probe), later_loss.value(probe))
        self.assertFalse(np.allclose(first_loss.gradient(probe), later_loss.gradient(probe)))

    def test_simple_center_is_deterministic_and_cyclic(self):
        problem = GradualSlaterStochasticConstraints(dim=5, margin=0.1)
        centers = []
        for round_index in (1, 30, 31, 60, 61):
            loss, _, _ = problem.sample_round(
                np.random.default_rng(round_index),
                round_index=round_index,
                constraint_rng=np.random.default_rng(2),
            )
            centers.append(loss.center)

        np.testing.assert_allclose(centers[0], [0.8, 0.2, 0.2, 0.2, 0.2])
        np.testing.assert_allclose(centers[1], centers[0])
        np.testing.assert_allclose(centers[2], [0.2, 0.8, 0.8, 0.8, 0.8])
        np.testing.assert_allclose(centers[3], centers[2])
        np.testing.assert_allclose(centers[4], [0.2, 0.8, 0.2, 0.2, 0.2])

        first_a, _, _ = problem.sample_round(np.random.default_rng(10), round_index=1)
        first_b, _, _ = problem.sample_round(np.random.default_rng(20), round_index=1)
        np.testing.assert_allclose(first_a.center, first_b.center)

    def test_loss_switch_interval_is_configurable(self):
        problem = GradualSlaterStochasticConstraints(
            dim=2, margin=0.1, loss_switch_interval=2
        )
        centers = [
            problem.sample_round(np.random.default_rng(0), round_index=t)[0].center
            for t in range(1, 6)
        ]

        np.testing.assert_allclose(centers[0], [0.8, 0.2])
        np.testing.assert_allclose(centers[1], [0.8, 0.2])
        np.testing.assert_allclose(centers[2], [0.2, 0.8])
        np.testing.assert_allclose(centers[3], [0.2, 0.8])
        np.testing.assert_allclose(centers[4], [0.8, 0.2])

    def test_loss_switch_interval_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, "loss_switch_interval must be positive"):
            GradualSlaterStochasticConstraints(loss_switch_interval=0)

    def test_higher_dimensional_center_schedule_is_balanced(self):
        problem = GradualSlaterStochasticConstraints(
            dim=3, margin=0.1, loss_switch_interval=1
        )
        centers = np.stack([
            problem.sample_round(np.random.default_rng(0), round_index=t)[0].center
            for t in range(1, 7)
        ])

        np.testing.assert_allclose(np.mean(centers, axis=0), np.full(3, 0.5))
        self.assertEqual(len(np.unique(centers, axis=0)), 6)

    def test_sinusoidal_schedule_rotates_in_two_dimensions(self):
        problem = GradualSlaterStochasticConstraints(
            dim=2,
            margin=0.1,
            loss_schedule="sinusoidal",
            loss_rotation_period=200,
        )
        centers = [
            problem.sample_round(np.random.default_rng(0), round_index=t)[0].center
            for t in (1, 51, 101, 151)
        ]

        np.testing.assert_allclose(centers[0], [0.8, 0.5], atol=1e-12)
        np.testing.assert_allclose(centers[1], [0.5, 0.8], atol=1e-12)
        np.testing.assert_allclose(centers[2], [0.2, 0.5], atol=1e-12)
        np.testing.assert_allclose(centers[3], [0.5, 0.2], atol=1e-12)

    def test_sinusoidal_schedule_is_balanced_in_higher_dimensions(self):
        period = 120
        problem = GradualSlaterStochasticConstraints(
            dim=5,
            margin=0.1,
            loss_schedule="sinusoidal",
            loss_rotation_period=period,
        )
        centers = np.stack([
            problem.sample_round(np.random.default_rng(0), round_index=t)[0].center
            for t in range(1, period + 1)
        ])

        self.assertTrue(np.all(centers >= 0.2 - 1e-12))
        self.assertTrue(np.all(centers <= 0.8 + 1e-12))
        np.testing.assert_allclose(np.mean(centers, axis=0), np.full(5, 0.5))

    def test_loss_schedule_parameters_are_validated(self):
        with self.assertRaisesRegex(ValueError, "loss_schedule must be one of"):
            GradualSlaterStochasticConstraints(loss_schedule="unknown")
        with self.assertRaisesRegex(ValueError, "loss_rotation_period must be positive"):
            GradualSlaterStochasticConstraints(loss_rotation_period=0)

    def test_constraint_noise_is_uniformly_bounded(self):
        problem = GradualSlaterStochasticConstraints(dim=5, margin=0.1)
        _, constraint, _ = problem.sample_round(
            np.random.default_rng(1),
            round_index=1,
            constraint_rng=np.random.default_rng(2),
        )
        constraint_noise = constraint.b + problem.margin

        self.assertLessEqual(abs(constraint_noise), problem.noise_magnitude)

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

    def test_gradual_slater_constraint_uses_every_coordinate(self):
        problem = GradualSlaterStochasticConstraints(dim=5, margin=0.1)
        _, constraint, _ = problem.sample_round(
            np.random.default_rng(1),
            round_index=1,
            constraint_rng=np.random.default_rng(2),
        )

        np.testing.assert_allclose(constraint.gradient(np.zeros(5)), np.ones(5))
        np.testing.assert_allclose(problem.comparator, np.full(5, 0.02))
        self.assertAlmostEqual(problem.expected_constraint(problem.comparator), 0.0)

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
