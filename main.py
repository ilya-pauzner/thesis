import random

import numpy as np
from scipy.optimize import linear_sum_assignment

random.seed(123456789)


def calc_load(hosts, vms, mapping):
    load = np.zeros([len(hosts), 2])
    for vm, loc in zip(vms, mapping):
        load[loc] += vm
    return load


def active_score(load):
    return np.sum(np.all(load > 0, axis=1)) / len(load)


def migration_score(mapping1, mapping2, ram):
    return np.inner(mapping1 != mapping2, ram) / np.sum(ram)


def random_reorder(hosts, vms, mapping=None):
    # returns mapping, hosts and vms are the same

    init_mapping = np.zeros(len(vms), dtype=np.int)
    init_resources = np.zeros([len(hosts), 2], dtype=np.int)

    for i, vm in enumerate(vms):
        possible_locations = np.all(init_resources + vm <= hosts, axis=1)
        if np.all(possible_locations == 0):
            return mapping
        loc = random.choice(np.arange(0, len(hosts))[possible_locations])
        init_mapping[i] = loc
        init_resources[loc] += vm

    return init_mapping


def ffd_reorder(hosts, vms, mapping=None):
    # returns mapping, hosts and vms are the same

    new_resources = np.zeros([len(hosts), 2], dtype=np.int)
    new_mapping = np.zeros(len(vms), dtype=np.int)

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
        if np.any(remaining != 0):
            times = np.min(remaining // max_vm)
            for _ in range(int(times)):
                holes.append((j, max_vm))
            holes.append((j, remaining - max_vm * times))

    matrix = np.zeros([len(vms), len(holes)], dtype=np.int)
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
    updated_mapping = np.zeros(len(vms), dtype=np.int)
    for i in range(len(vms)):
        hole_location, hole_size = holes[col_ind[i]]
        updated_mapping[i] = hole_location

    return updated_mapping


def main():
    hosts = np.array([[100, 200] for _ in range(5)], dtype=np.int)
    vms = np.array([[random.randint(10, 30), random.randint(20, 60)] for _ in range(15)], dtype=np.int)
    init_mapping = random_reorder(hosts, vms, None)
    assert init_mapping is not None

    # hosts, vms, init_mapping = create_problem(3, 50)
    init_resources = calc_load(hosts, vms, init_mapping)
    assert np.all(init_resources <= hosts)
    print('Initial active score:', active_score(init_resources))

    new_mapping = ffd_reorder(hosts, vms, init_mapping)
    assert new_mapping is not None
    new_resources = calc_load(hosts, vms, new_mapping)
    assert np.all(new_resources <= hosts)
    print('FirstFitDecreasing active score:', active_score(new_resources))
    print('Migration score incurred by transition:', migration_score(init_mapping, new_mapping, vms[:, 1]))

    updated_mapping = migopt_reorder(hosts, vms, new_mapping, init_mapping)
    assert updated_mapping is not None
    updated_resources = calc_load(hosts, vms, updated_mapping)
    assert np.all(updated_resources <= hosts)
    print('Migration score after optimisation:', migration_score(init_mapping, updated_mapping, vms[:, 1]))


if __name__ == '__main__':
    main()
