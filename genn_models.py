import numpy as np
from pygenn import create_neuron_model, create_postsynaptic_model

glif3 = create_neuron_model(
    "glif3",
    params=[
        "C",
        "G",
        "El",
        ("spike_cut_length", "int"),
        "th_inf",
        "V_reset",
        "asc_amp_array_1",
        "asc_amp_array_2",
        "asc_stable_coeff_1",
        "asc_stable_coeff_2",
        "asc_decay_rates_1",
        "asc_decay_rates_2",
        "asc_refractory_decay_rates_1",
        "asc_refractory_decay_rates_2"],
    vars=[
        ("V", "scalar"),
        ("refractory_countdown", "int"),
        ("ASC_1", "scalar"),
        ("ASC_2", "scalar")],
    sim_code="""
    // Sum after spike currents
    const scalar sum_of_ASC = ASC_1*asc_stable_coeff_1 + ASC_2*asc_stable_coeff_2;

    // Voltage
    if (refractory_countdown <= 0) {
        V+=1/C*(Isyn+sum_of_ASC-G*(V-El))*dt;
    }

    // ASCurrents
    if (refractory_countdown <= 0) {
        ASC_1 *= asc_decay_rates_1;
        ASC_2 *= asc_decay_rates_2;
        }


    // Decrement refractory_countdown by 1; Do not decrement past -1
    if (refractory_countdown > -1) {
        refractory_countdown -= 1;
    }
    """,
    threshold_condition_code="V > th_inf",
    reset_code="""
    V=V_reset;
    ASC_1 = asc_amp_array_1 + ASC_1 * asc_refractory_decay_rates_1;
    ASC_2 = asc_amp_array_2 + ASC_2 * asc_refractory_decay_rates_2;
    refractory_countdown = spike_cut_length;
    """)

psc_alpha = create_postsynaptic_model(
    class_name="Alpha",
    sim_code="""
    injectCurrent(x);
    x = decay * ((dt * inSyn * exp(1.0f) / tau) + x);
    inSyn*=decay;
    """,
    vars=[("x", "scalar")],
    params=[("tau")],
    derived_params=[("decay", lambda pars, dt: np.exp(-dt / pars["tau"]))])
