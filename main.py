import random

import numpy as np
from scipy.optimize import linear_sum_assignment

random.seed(123456789)


def active_score(load):
    return np.sum(np.all(load > 0, axis=1)) / len(load)


def migration_score(mapping1, mapping2, ram):
    return np.inner(mapping1 != mapping2, ram) / np.sum(ram)


vm_count = 15
host_count = 5
hosts = np.array([[100, 200] for _ in range(host_count)], dtype=np.int)
vms = np.array([[random.randint(10, 30), random.randint(20, 60)] for _ in range(vm_count)], dtype=np.int)

init_mapping = np.zeros(vm_count, dtype=np.int)
init_resources = np.zeros([host_count, 2], dtype=np.int)

for i, vm in enumerate(vms):
    possible_locations = np.all(init_resources + vm <= hosts, axis=1)
    loc = random.choice(np.arange(0, host_count)[possible_locations])
    init_mapping[i] = loc
    init_resources[loc] += vm

assert np.all(init_resources <= hosts)

print('RandomFit active score:', active_score(init_resources))

new_mapping = np.zeros(vm_count, dtype=np.int)
new_resources = np.zeros([host_count, 2], dtype=np.int)

permutation = np.argsort(vms, axis=0)[:, 1][::-1]
for i in range(len(vms)):
    vm = vms[permutation][i]
    possible_locations = np.all(new_resources + vm <= hosts, axis=1)
    loc = (np.arange(0, host_count)[possible_locations])[0]
    new_mapping[permutation[i]] = loc
    new_resources[loc] += vm

assert np.all(new_resources <= hosts)

print('FirstFitDecreasing active score:', active_score(new_resources))
print('Migration score incurred by transition:', migration_score(init_mapping, new_mapping, vms[:, 1]))

holes = []
for i in range(len(vms)):
    holes.append((new_mapping[i], vms[i]))
max_vm = np.max(vms, axis=0)
for j in range(len(hosts)):
    remaining = hosts[j] - new_resources[j]
    if np.any(remaining != 0):
        times = np.min(remaining // max_vm)
        for _ in range(int(times)):
            holes.append((j, max_vm))
        holes.append((j, remaining - max_vm * times))

matrix = np.zeros([len(vms), len(holes)], dtype=np.int)
for i in range(len(vms)):
    vm = vms[i]
    init_location = init_mapping[i]
    for k in range(len(holes)):
        hole_location, hole_size = holes[k]
        if np.all(vm <= hole_size):
            if init_location == hole_location:
                matrix[i][k] = 0
            else:
                matrix[i][k] = vm[1]
        else:
            matrix[i][k] = 10 ** 9

# hole == [where, size]
row_ind, col_ind = linear_sum_assignment(matrix)
updated_mapping = np.zeros(vm_count, dtype=np.int)
updated_resources = np.zeros([host_count, 2], dtype=np.int)
for i in range(len(vms)):
    hole_location, hole_size = holes[col_ind[i]]
    updated_mapping[i] = hole_location
    updated_resources[hole_location] += vms[i]

assert np.all(updated_resources <= hosts)

print('Migration score after optimisation:', migration_score(init_mapping, updated_mapping, vms[:, 1]))
