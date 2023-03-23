import pytest
import numpy as np
import pydantic

import tidy3d as td

from numpy.random import default_rng

from tidy3d.plugins.resonance import ResonanceFinder
from tidy3d import ScalarFieldTimeDataArray, FieldTimeData, FieldTimeMonitor

RTOL = 1e-2
NTIME = 10000


def generate_signal(freqs, decays, amplitudes, phases, time_step):
    t = np.arange(NTIME)

    complex_amplitudes = amplitudes * np.exp(1j * phases)
    complex_freqs = 2 * np.pi * freqs - 1j * decays

    signal = np.zeros(len(t), dtype=complex)

    for i in range(len(freqs)):
        signal += complex_amplitudes[i] * np.exp(-1j * complex_freqs[i] * t * time_step)
    return signal


def check_resonances(freqs, decays, amplitudes, phases, resonances):
    inds = np.argsort(freqs)
    freqs = freqs[inds]
    decays = decays[inds]
    amplitudes = amplitudes[inds]
    phases = phases[inds]
    assert len(freqs) == resonances.dims["freq"]

    complex_amplitudes = amplitudes * np.exp(1j * phases)

    for i in range(len(freqs)):
        resonance = resonances.isel(freq=i)
        assert np.isclose(np.abs(resonance.freq), freqs[i], rtol=RTOL, atol=0)
        assert np.isclose(resonance.decay, decays[i], rtol=RTOL, atol=0)
        assert np.isclose(resonance.amplitude, amplitudes[i], rtol=RTOL, atol=0)
        assert np.isclose(
            resonance.amplitude * np.exp(1j * resonance.phase),
            complex_amplitudes[i],
            rtol=RTOL,
            atol=0,
        )


def test_simple():
    freqs = np.array([0.1, 0.2])
    decays = np.array([0.002, 0.0005])
    amplitudes = np.array([2, 3])
    phases = np.array([0, np.pi / 2])

    f_min = 0.05
    f_max = 0.25
    time_step = 1

    signal = generate_signal(freqs, decays, amplitudes, phases, time_step)
    resonance_finder = ResonanceFinder(freq_window=(f_min, f_max))
    resonances = resonance_finder.run_raw_signal(signal, time_step)
    check_resonances(freqs, decays, amplitudes, phases, resonances)


@pytest.mark.parametrize("rng_seed", np.arange(0, 10, 3))
def test_random_sinusoids(rng_seed):
    """tests resonance solver on a random sum of sinusoids"""
    rng = default_rng(rng_seed)
    time_step = 1

    num_sines = 20
    f_min = 0.1
    f_max = 0.2
    decay_min = 1e-5
    decay_max = 1e-3
    amp_min = 1e-2
    amp_max = 1e1

    amplitudes = amp_min + (amp_max - amp_min) * rng.random(num_sines)
    phases = -np.pi + 2 * np.pi * rng.random(num_sines)
    freqs = f_min + (f_max - f_min) * rng.random(num_sines)
    decays = decay_min + (decay_max - decay_min) * rng.random(num_sines)

    signal = generate_signal(freqs, decays, amplitudes, phases, time_step)
    resonance_finder = ResonanceFinder(freq_window=(f_min, f_max), init_num_freqs=200)
    resonances = resonance_finder.run_raw_signal(signal, time_step)
    check_resonances(freqs, decays, amplitudes, phases, resonances)


def test_scalar_field_time():
    time_step = 1

    freqs = np.array([0.4, 0.3])
    decays = np.array([0.0001, 0.005])
    amplitudes = np.array([1, 1])
    phases = np.array([0, 0])

    t = np.arange(NTIME) / time_step
    signal = generate_signal(freqs, decays, amplitudes, phases, time_step)
    coords = dict(x=[0], y=[0], z=[0], t=t)
    fd = ScalarFieldTimeDataArray(np.reshape(signal, (1, 1, 1, len(signal))), coords=coords)
    resonance_finder = ResonanceFinder(freq_window=(0.2, 0.5), init_num_freqs=100)
    resonances = resonance_finder.run_scalar_field_time(fd)
    check_resonances(freqs, decays, amplitudes, phases, resonances)


def test_field_time_single():
    time_step = 1

    freqs = np.array([0.4, 0.3])
    decays = np.array([0.0001, 0.005])
    amplitudes = np.array([1, 1])
    phases = np.array([0, 0])

    t = np.arange(NTIME) / time_step
    signal = generate_signal(freqs, decays, amplitudes, phases, time_step)
    coords = dict(x=[0], y=[0], z=[0], t=t)
    fd = ScalarFieldTimeDataArray(np.reshape(signal, (1, 1, 1, len(signal))), coords=coords)
    fd2 = ScalarFieldTimeDataArray(np.reshape(signal * 2, (1, 1, 1, len(signal))), coords=coords)
    monitor = FieldTimeMonitor(size=(0, 0, 0), interval=1, name="field", fields=["Hx", "Hy"])
    field = FieldTimeData(monitor=monitor, Hx=fd, Hy=fd2)
    resonance_finder = ResonanceFinder(freq_window=(0.2, 0.5), init_num_freqs=100)
    resonances = resonance_finder.run(field)
    amplitudes = 3 * amplitudes
    check_resonances(freqs, decays, amplitudes, phases, resonances)


def test_field_time_mult():
    time_step = 1

    freqs = np.array([0.4, 0.3])
    decays = np.array([0.0001, 0.005])
    amplitudes = np.array([1, 1])
    phases = np.array([0, 0])

    t = np.arange(NTIME) / time_step
    signal = generate_signal(freqs, decays, amplitudes, phases, time_step)
    coords = dict(x=[0], y=[0], z=[0], t=t)
    fd = ScalarFieldTimeDataArray(np.reshape(signal, (1, 1, 1, len(signal))), coords=coords)
    fd2 = ScalarFieldTimeDataArray(np.reshape(signal * 2, (1, 1, 1, len(signal))), coords=coords)
    monitor = FieldTimeMonitor(size=(0, 0, 0), interval=1, name="field", fields=["Hx", "Hy"])
    field = FieldTimeData(monitor=monitor, Hx=fd, Hy=fd2)
    field2 = FieldTimeData(monitor=monitor, Hx=fd2, Hy=fd)
    resonance_finder = ResonanceFinder(freq_window=(0.2, 0.5), init_num_freqs=100)
    resonances = resonance_finder.run((field, field2))
    amplitudes = 6 * amplitudes
    check_resonances(freqs, decays, amplitudes, phases, resonances)


def test_field_time_e_and_m():
    time_step = 1

    freqs = np.array([0.4, 0.3])
    decays = np.array([0.0001, 0.005])
    amplitudes = np.array([1, 1])
    phases = np.array([0, 0])

    t = np.arange(NTIME) / time_step
    signal = generate_signal(freqs, decays, amplitudes, phases, time_step)
    coords = dict(x=[0], y=[0], z=[0], t=t)
    fd = ScalarFieldTimeDataArray(np.reshape(signal, (1, 1, 1, len(signal))), coords=coords)
    fd2 = ScalarFieldTimeDataArray(np.reshape(signal * 2, (1, 1, 1, len(signal))), coords=coords)
    monitor = FieldTimeMonitor(size=(0, 0, 0), interval=1, name="field", fields=["Ex", "Hy"])
    field = FieldTimeData(monitor=monitor, Ex=fd, Hy=fd2)
    field2 = FieldTimeData(monitor=monitor, Ex=fd, Hy=fd2)
    resonance_finder = ResonanceFinder(freq_window=(0.2, 0.5), init_num_freqs=100)
    resonances = resonance_finder.run((field, field2))
    amplitudes = 2 * amplitudes
    check_resonances(freqs, decays, amplitudes, phases, resonances)


def test_field_time_use_e_only():
    time_step = 1

    freqs = np.array([0.4, 0.3])
    decays = np.array([0.0001, 0.005])
    amplitudes = np.array([1, 1])
    phases = np.array([0, 0])

    t = np.arange(NTIME) / time_step
    signal = generate_signal(freqs, decays, amplitudes, phases, time_step)
    coords = dict(x=[0], y=[0], z=[0], t=t)
    fd = ScalarFieldTimeDataArray(np.reshape(signal, (1, 1, 1, len(signal))), coords=coords)
    fd2 = ScalarFieldTimeDataArray(np.reshape(signal * 2, (1, 1, 1, len(signal))), coords=coords)
    monitor = FieldTimeMonitor(size=(0, 0, 0), interval=1, name="field", fields=["Hy"])
    monitor2 = FieldTimeMonitor(size=(0, 0, 0), interval=1, name="field", fields=["Ex"])
    field = FieldTimeData(monitor=monitor, Hy=fd2)
    field2 = FieldTimeData(monitor=monitor2, Ex=fd)
    resonance_finder = ResonanceFinder(freq_window=(0.2, 0.5), init_num_freqs=100)
    resonances = resonance_finder.run((field, field2))
    amplitudes = amplitudes
    check_resonances(freqs, decays, amplitudes, phases, resonances)
