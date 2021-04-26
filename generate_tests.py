import numpy as np

np.random.seed(987654321)

# generally useless, easily packs to (250, xxx) - best possible
host1 = [250, 700]
vm1 = [[1, 1], [2, 4], [2, 8], [8, 16], [4, 8], [4, 16], [8, 32]]
freq1 = [24, 8, 2, 1, 4, 1, 1]

host2 = [275, 325]
vm2 = [[1, 1], [1, 2], [2, 4], [2, 8], [8, 32], [4, 8], [1, 4], [4, 16], [48, 32], [8, 32], [6, 4], [12, 8], [96, 64]]
freq2 = [24, 24, 8, 2, 2, 4, 2, 2, 2, 1, 4, 2, 4]

# generally useless, easily packs to (74, 296) - best possible
host3 = [75, 325]
vm3 = [[4, 16], [2, 8], [16, 64], [32, 128], [64, 256], [8, 32]]
freq3 = [4, 6, 1, 2, 1, 21]

host4 = [75, 150]
vm4 = [[64, 128], [16, 32], [2, 4], [32, 64], [4, 8]]
freq4 = [1, 5, 2, 5, 1]

hosts = [host1, host2, host3, host4]
vms = [vm1, vm2, vm3, vm4]
freqs = [freq1, freq2, freq3, freq4]


def create_problem(problem_type, host_count):
    host = np.array(hosts[problem_type])
    vm = np.array(vms[problem_type])
    freq = np.array(freqs[problem_type])

    result_resources = np.zeros([host_count, 2], dtype=np.int)
    result_hosts = np.array([host for i in range(host_count)])
    result_vms = []
    result_mapping = []
    while True:
        selected_vm = vm[np.random.choice(np.arange(0, len(vm)), p=(freq / freq.sum()))]
        possible_locations = np.all(result_resources + selected_vm <= result_hosts, axis=1)
        if np.all(possible_locations == 0):
            return result_hosts, np.array(result_vms), np.array(result_mapping)
        loc = (np.arange(0, host_count)[possible_locations])[0]
        result_resources[loc] += selected_vm
        result_mapping.append(loc)
        result_vms.append(selected_vm)


create_problem(0, 100)
