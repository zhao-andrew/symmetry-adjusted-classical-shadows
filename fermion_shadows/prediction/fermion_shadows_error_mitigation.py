import numpy as np
from itertools import product, combinations
from typing import Sequence, Callable, Union

from matplotlib import use
import matplotlib.pyplot as plt

from fermion_shadows_prediction import diagonal_majoranas, compute_expectation_values_from_majorana_observables


def mitigate_majorana_expectations_via_particle_number_full_system(
        majorana_expectations: dict,
        n_qubits: int,
        n_particles: int) -> dict:
    full_system_diagonal_majoranas = diagonal_majoranas(n_qubits, 2)
    one_body_diagonal_majoranas = [term for term in full_system_diagonal_majoranas if len(term) == 2]
    two_body_diagonal_majoranas = [term for term in full_system_diagonal_majoranas if len(term) == 4]
    one_body_conserved_quantity = particle_number_conserved_quantity_for_majoranas(n_qubits, n_particles, 2)
    two_body_conserved_quantity = particle_number_conserved_quantity_for_majoranas(n_qubits, n_particles, 4)

    if one_body_conserved_quantity == 0.0:
        one_body_rescaling_factor = 1.0
        print('One-body conserved quantity is zero, leaving one-body expectation values untouched.')

    else:
        noisy_one_body_quantity = 0.0
        for term in one_body_diagonal_majoranas:
            noisy_one_body_quantity += majorana_expectations[term]
        one_body_rescaling_factor = one_body_conserved_quantity / noisy_one_body_quantity

    if two_body_conserved_quantity == 0.0:
        two_body_rescaling_factor = 1.0
        print('Two-body conserved quantity is zero, leaving two-body expectation values untouched.')

    else:
        noisy_two_body_quantity = 0.0
        for term in two_body_diagonal_majoranas:
            noisy_two_body_quantity += majorana_expectations[term]
        two_body_rescaling_factor = two_body_conserved_quantity / noisy_two_body_quantity

    mitigated_expectations = {}
    for term, val in majorana_expectations.items():
        if len(term) == 2:
            mitigated_expectations[term] = val * one_body_rescaling_factor
        elif len(term) == 4:
            mitigated_expectations[term] = val * two_body_rescaling_factor

    return mitigated_expectations


def mitigate_majorana_expectations_via_particle_number_spin_adapted(
        majorana_expectations: dict,
        up_qubits: Union[int, Sequence[int]],
        down_qubits: Union[int, Sequence[int]],
        n_up_particles: int,
        n_down_particles: int) -> dict:
    if isinstance(up_qubits, int):
        n_up_qubits = up_qubits
        # up_qubits = list(range(up_qubits))
    else:
        n_up_qubits = len(up_qubits)

    if isinstance(down_qubits, int):
        n_down_qubits = down_qubits
        # down_qubits = list(range(n_up_qubits, n_up_qubits + n_down_qubits))
    else:
        n_down_qubits = len(down_qubits)

    n_qubits = n_up_qubits + n_down_qubits

    up_spin_diagonal_majoranas = diagonal_majoranas(n_up_qubits, 2)
    down_spin_diagonal_majoranas = diagonal_majoranas(n_down_qubits, 2)
    
    diagonal_majoranas_up = [[()],
                             [term for term in up_spin_diagonal_majoranas if len(term) == 2],
                             [term for term in up_spin_diagonal_majoranas if len(term) == 4]]
    diagonal_majoranas_down = [[()],
                               [term for term in down_spin_diagonal_majoranas if len(term) == 2],
                               [term for term in down_spin_diagonal_majoranas if len(term) == 4]]
    conserved_quantity_up = [1.0,
                             particle_number_conserved_quantity_for_majoranas(n_up_qubits, n_up_particles, 2),
                             particle_number_conserved_quantity_for_majoranas(n_up_qubits, n_up_particles, 4)]
    conserved_quantity_down = [1.0,
                               particle_number_conserved_quantity_for_majoranas(n_down_qubits, n_down_particles, 2),
                               particle_number_conserved_quantity_for_majoranas(n_down_qubits, n_down_particles, 4)]

    mitigated_expectations = {}

    for k, l in [(1, 0), (0, 1), (1, 1), (2, 0), (0, 2)]:
        diag_ops_up = diagonal_majoranas_up[k]
        T_up = conserved_quantity_up[k]
        diag_ops_down = diagonal_majoranas_down[l]
        T_down = conserved_quantity_down[l]

        T = T_up * T_down

        if T == 0.0:
            rescaling_factor = 1.0
#             print(
#                 'The ({}, {}) conserved quantity is zero, leaving those expectation values untouched.'.format(k, l)
#             )

        else:
            noisy_T = 0.0
            for term_up, term_down_shifted in product(diag_ops_up, diag_ops_down):
                term_down = tuple([p + 2 * n_up_qubits for p in term_down_shifted])
                term = term_up + term_down
                noisy_T += majorana_expectations[term]
            if np.isclose(noisy_T, 0.0):
                print('Noisy conserved quantity for ({}, {}) is {}'.format(k, l, noisy_T))
                rescaling_factor = 1.0
            else:
                rescaling_factor = T / noisy_T

        for term_up, term_down in product(combinations(range(2 * n_up_qubits), 2 * k),
                                          combinations(range(2 * n_up_qubits, 2 * n_qubits), 2 * l)):
            term = term_up + term_down

            try:
                val = majorana_expectations[term]
                mitigated_expectations[term] = val * rescaling_factor
            except KeyError:
                pass

    return mitigated_expectations


def particle_number_conserved_quantity_for_majoranas(n_modes: int,
                                                     n_particles: int,
                                                     majorana_degree: int
                                                     ) -> float:
    assert n_modes > 0
    assert n_particles <= n_modes
    assert majorana_degree <= 2 * n_modes

    if majorana_degree == 0:
        conserved_quantity = 1.0

    elif majorana_degree == 2:
        conserved_quantity = n_modes - 2 * n_particles

    elif majorana_degree == 4:
        conserved_quantity = 0.5 * n_modes * (n_modes - 1) - 2 * n_particles * (n_modes - n_particles)

    else:
        if majorana_degree % 2:
            raise ValueError('Odd-degree Majorana operators are unphysical fermionic operators.')
        else:
            raise ValueError('Error mitigation for 2k-degree Majorana operators, k > 2, is currently not implemented.')

    return conserved_quantity


def bootstrap_error_bars(data: Sequence[dict], observables: Sequence[dict],
                         mitigation_function: Callable, args: Sequence = (),
                         n_resamples: int = 200,
                         compute_sample_mean: bool = True,
                         plot_histogram: bool = False,
                         observable_labels: Sequence[str] = None) -> Union[np.ndarray, Sequence[np.ndarray]]:
    """Requires the use of a mitigation strategy. Don't use if not performing error mitigation,
    just use np.std instead."""

    if plot_histogram and observable_labels is not None:
        assert len(observables) == len(observable_labels)

    n_samples = len(data)

    resampled_observables_means = np.zeros((n_resamples, len(observables)))
    for j in range(n_resamples):
        resampled_data_index = np.random.randint(n_samples, size=n_samples)

        resampled_data_means = {term : 0.0 for term in data[0].keys()}
        for i in resampled_data_index:
            resampled_data = data[i]
            for term, val in resampled_data.items():
                resampled_data_means[term] += val / n_samples

        resampled_data_means_mitigated = mitigation_function(resampled_data_means, *args)

        resampled_observables_means[j] = compute_expectation_values_from_majorana_observables(
            resampled_data_means_mitigated, observables)

    observable_stdev_from_resamples = np.std(resampled_observables_means, axis=0, ddof=1)
    
    if compute_sample_mean:
        sample_mean = {term : 0 for term in data[0].keys()}
        for data_point in data:
            for term, val in data_point.items():
                sample_mean[term] += val / n_samples
        
        sample_mean_mitigated = mitigation_function(sample_mean, *args)
        
        sample_mean_obs = compute_expectation_values_from_majorana_observables(
            sample_mean_mitigated, observables)
    else:
        sample_mean_obs = None

    if plot_histogram:
        try:
            use('Qt5Agg')
        except ImportError:
            pass
        
        for j in range(len(observables)):
            plt.figure()
            plt.hist(resampled_observables_means[:, j])
            plt.xlabel('Expectation value')
            if observable_labels is None:
                title = '{}'.format(tuple(observables[j].keys()))
            else:
                title = observable_labels[j]
            plt.title(title)

            plt.savefig(title + ' bootstrap-hist_B={}.png'.format(n_resamples),
                        dpi=200, bbox_inches='tight')
    
    if compute_sample_mean:
        return np.array(sample_mean_obs), observable_stdev_from_resamples
    else:
        return observable_stdev_from_resamples
