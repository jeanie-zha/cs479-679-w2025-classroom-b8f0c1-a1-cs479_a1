import numpy as np
import matplotlib.pyplot as plt
import pytest

import neuron_models as nm
import ans as ans

def test_slope():
    A = ans.LIFNeuron(Tau_m=0.03, label='A')
    A.set_input_buffer(2.1)
    A.set_v(-0.6)
    A.set_s(1.2)
    A.slope()

    assert A._LIFNeuron__dvdt == pytest.approx(60., abs=1e-4)
    assert A._LIFNeuron__dsdt == pytest.approx(-24., abs=1e-4)

def test_step():
    A = ans.LIFNeuron(Tau_m=0.03, label='A')
    A.set_input_buffer(2.1)
    A.set_v(-0.6)
    A.set_s(1.8)
    A.slope()
    A.step(0.02)

    assert A._LIFNeuron__v == pytest.approx(1, abs=1e-4)
    assert A._LIFNeuron__s == pytest.approx(43.08, abs=1e-4)

def test_input_current():
    # Test current variable, s
    net = nm.SpikingNetwork()
    # One spike at t=0, which will affect A's input current in the next time step.
    inA = nm.InputNeuron([0.], label='inA')
    A = ans.LIFNeuron(label='A')
    net.add_neuron(inA)     # neuron index 0
    net.add_neuron(A)       # neuron index 1
    inA.connect_to(A, 0.5)  # connect inA -> A
    net.simulate(0.003, 0.001)

    assert net.neur[1].get_s_history() == pytest.approx(np.array([0., 0., 10.]), abs=1e-4)

def test_spike_interpolation():
    # Test interpolated spike time
    net = nm.SpikingNetwork()
    A = ans.LIFNeuron(label='A')
    net.add_neuron(A)

    # Set voltage and current so it's just about to spike.
    A.set_v(0.95)
    A.set_s(10.)

    net.simulate(0.002, 0.001)

    assert A.get_spikes()[0] == pytest.approx(0.00011, abs=1e-5)
    assert A.get_v() == pytest.approx(0., abs=1e-5)
