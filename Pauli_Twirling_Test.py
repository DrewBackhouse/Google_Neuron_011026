import cirq
import numpy as np
from Functions import PauliTwirling

Twirl = True
num_instances = 3

circuit = cirq.Circuit(cirq.cphase(0.1)(cirq.q(0), cirq.q(1)), cirq.M(cirq.q(0), cirq.q(1), key='m'))
print(circuit)

if Twirl == True:
    circuits = PauliTwirling(circuit, num_instances)
    for i in range(num_instances):
        print(circuits[i])


sampler = cirq.Simulator() 

if Twirl == True:
    results = sampler.run_batch(circuits, repetitions = 5)
    for i in range(num_instances):
        print(results[i][0])
    result = np.concatenate([res[0].measurements['m'] for res in results])
    print(result)
else: 
    result = sampler.run(circuit, repetitions = 5)
    print(result)