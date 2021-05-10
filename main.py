import copy
import platform

import numpy as np
from scipy.optimize import linear_sum_assignment

from generate_tests import create_problem

np.random.seed(987654321)


def calc_load(hosts, vms, mapping):
    load = np.zeros([len(hosts), 2])
    for vm, loc in zip(vms, mapping):
        load[int(loc)] += vm
    return load


def active_score(load):
    return np.sum(np.all(load > 0, axis=1)) / len(load)


def migration_score(mapping1, mapping2, ram):
    return np.inner(mapping1 != mapping2, ram) / np.sum(ram)


def dummy_reorder(hosts, vms, mapping=None):
    return mapping


def solver_reorder(hosts, vms, mapping=None):
    from pyvpsolver.solvers import vbpsolver
    W = hosts[0]
    w, b = np.unique(vms, return_counts=True, axis=0)
    obj, lst_sol = vbpsolver.solve(W, w, b, script="vpsolver_coinor.sh", verbose=False)
    flat_sol = []
    host_indices = [[] for _ in range(len(w))]
    for host_count, host in lst_sol[0]:
        for _ in range(host_count):
            flat_sol.append(host)
    for host_ind, host in enumerate(flat_sol):
        for vm_ind, _ in host:
            host_indices[vm_ind].append(host_ind)
    new_mapping = np.zeros(len(vms), dtype=np.uint)
    for vm_ind, vm in enumerate(w):
        new_mapping[np.all(vms == vm, axis=1)] = np.array(host_indices[vm_ind], dtype=np.uint)
    return new_mapping


def sercon_reorder(hosts, vms, mapping=None):
    new_mapping = copy.copy(mapping)
    host_loads = calc_load(hosts, vms, mapping)
    avg_host_loads = np.average(host_loads, axis=1)
    permutation = np.argsort(avg_host_loads)

    freed_hosts = 1
    while freed_hosts > 0:
        freed_hosts = 0
        for current_host_ind in permutation:
            if not np.all(host_loads[current_host_ind] == 0):
                # host is not already free
                vm_inds_of_this_host = []
                for vm_ind in range(len(new_mapping)):
                    if new_mapping[vm_ind] == current_host_ind:
                        vm_inds_of_this_host.append(vm_ind)

                # trying to unload them
                backup_mapping = copy.copy(new_mapping)
                backup_host_loads = copy.copy(host_loads)
                good = np.any(host_loads > 0, axis=1)
                good[current_host_ind] = False
                for vm_ind in vm_inds_of_this_host:
                    possible_locations = good & np.all(host_loads + vms[vm_ind] <= hosts, axis=1)
                    if np.all(possible_locations == 0):
                        # failed to reorder, return existing mapping
                        new_mapping = backup_mapping
                        host_loads = backup_host_loads
                        break

                    loc = np.arange(0, len(hosts))[possible_locations][0]
                    new_mapping[vm_ind] = loc
                    host_loads[loc] += vms[vm_ind]
                    host_loads[current_host_ind] -= vms[vm_ind]
                else:
                    freed_hosts += 1

    return new_mapping


def ffd_reorder(hosts, vms, mapping=None):
    # returns mapping, hosts and vms are the same

    new_resources = np.zeros([len(hosts), 2])
    new_mapping = np.zeros(len(vms))

    permutation = np.argsort(vms, axis=0)[:, 1][::-1]
    for i in range(len(vms)):
        vm = vms[permutation][i]
        possible_locations = np.all(new_resources + vm <= hosts, axis=1)
        if np.all(possible_locations == 0):
            # failed to reorder, return existing mapping
            return mapping
        loc = (np.arange(0, len(hosts))[possible_locations])[0]
        new_mapping[permutation[i]] = loc
        new_resources[loc] += vm

    return new_mapping


def migopt_reorder(hosts, vms, mapping, old_mapping):
    resources = calc_load(hosts, vms, mapping)
    holes = []
    for i in range(len(vms)):
        holes.append((mapping[i], vms[i]))
    max_vm = np.max(vms, axis=0)
    for j in range(len(hosts)):
        remaining = hosts[j] - resources[j]
        if np.any(remaining != 0) and np.any(resources[j] != 0):
            times = np.min(remaining // max_vm)
            for _ in range(int(times)):
                holes.append((j, max_vm))
            holes.append((j, remaining - max_vm * times))

    matrix = np.zeros([len(vms), len(holes)])
    for i in range(len(vms)):
        vm = vms[i]
        old_location = old_mapping[i]
        for k in range(len(holes)):
            hole_location, hole_size = holes[k]
            if np.all(vm <= hole_size):
                if old_location == hole_location:
                    matrix[i][k] = 0
                else:
                    matrix[i][k] = vm[1]
            else:
                matrix[i][k] = 10 ** 9

    # hole == [where, size]
    row_ind, col_ind = linear_sum_assignment(matrix)
    updated_mapping = np.zeros(len(vms))
    for i in range(len(vms)):
        hole_location, hole_size = holes[col_ind[i]]
        updated_mapping[i] = hole_location

    return updated_mapping


def do_shrink(vms):
    unique_vms, indices = np.unique(vms, axis=0, return_inverse=True)
    multiplier = np.clip(np.random.random([len(unique_vms), 2]) + 0.25, 0.5, 1)
    new_unique_vms = unique_vms * multiplier
    return np.ceil(new_unique_vms[indices]).astype(np.uint)


def with_migopt(algo):
    def inner(hosts, vms, init_mapping):
        new_mapping = algo(hosts, vms, init_mapping)
        return migopt_reorder(hosts, vms, new_mapping, init_mapping)

    return inner


def report_algorithm(algo_name, algo_fn, hosts, vms, init_mapping):
    new_mapping = algo_fn(hosts, vms, init_mapping)
    assert new_mapping is not None
    new_resources = calc_load(hosts, vms, new_mapping)
    assert np.all(new_resources <= hosts)
    print(f'{algo_name} results:')
    print('active score:', active_score(new_resources))
    print('migration score:', migration_score(init_mapping, new_mapping, vms[:, 1]))
    print()
    return new_mapping


def main():
    setups = [
        #   Type, Hosts, Shrink
        [3, 100, True],
        [-3, 100, False]
    ]

    for task_type, host_count, shrink in setups:
        print('=' * 80)
        print('TEST SETTINGS:')
        print(f'Task type: {task_type}')
        print(f'Host count: {host_count}')
        print(f'Shrink: {shrink}')

        hosts, vms, init_mapping = create_problem(task_type, host_count)
        if shrink:
            new_vms = do_shrink(vms)
            print('Shrinking VMs by random factor in [0.5; 1], in real life this corresponds to real usage statistics:')
            print('Most customers do not fully consume their quotas - this makes resources overbooking possible.')
            print('Now, VM requirements are smaller than before, so that we can rearrange them in order to free hosts.')
            print()
        else:
            new_vms = vms
        algorithms = [
            ['Initial', dummy_reorder],
            ['FirstFitDecreasing', ffd_reorder],
            ['Sercon', sercon_reorder]
        ]

        if platform.system() == 'Linux':
            algorithms.append(['PyVPSolver', solver_reorder])

        for algo_name, algo_fn in algorithms:
            report_algorithm(algo_name, algo_fn, hosts, new_vms, init_mapping)
            report_algorithm(f'{algo_name} + migopt', with_migopt(algo_fn), hosts, new_vms, init_mapping)


if __name__ == '__main__':
    main()
