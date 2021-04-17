import random

import numpy as np

random.seed(123456789)


def active_score(load):
    return np.sum(np.all(load > 0, axis=1)) / len(load)


def migration_score(mapping1, mapping2, ram):
    return np.inner(mapping1 != mapping2, ram) / np.sum(ram)


vm_count = 15
host_count = 5
hosts = np.array([[100, 200] for _ in range(host_count)])
vms = np.array([[random.randint(10, 30), random.randint(20, 60)] for _ in range(vm_count)])

init_mapping = np.zeros(vm_count)
init_resources = np.zeros([host_count, 2])

for i, vm in enumerate(vms):
    possible_locations = np.all(init_resources + vm <= hosts, axis=1)
    loc = random.choice(np.arange(0, host_count)[possible_locations])
    init_mapping[i] = loc
    init_resources[loc] += vm

print('RandomFit active score:', active_score(init_resources))

new_mapping = np.zeros(vm_count)
new_resources = np.zeros([host_count, 2])

permutation = np.argsort(vms, axis=0)[:, 1][::-1]
for i in range(len(vms)):
    vm = vms[permutation][i]
    possible_locations = np.all(new_resources + vm <= hosts, axis=1)
    loc = (np.arange(0, host_count)[possible_locations])[0]
    new_mapping[i] = loc
    new_resources[loc] += vm

print('FirstFitDecreasing active score:', active_score(new_resources))
print('Migration score incurred by transition:', migration_score(init_mapping, new_mapping, vms[:, 1]))
