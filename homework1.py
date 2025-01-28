import numpy as np
from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt

from python_propagate.Scenario import Scenario
from python_propagate.Environment.Planets import Earth
from python_propagate.Agents.spacecraft import Spacecraft
from python_propagate.Agents import State
from datetime import datetime, timedelta
from python_propagate.Dynamics.keplerian import Keplerian
from python_propagate.Dynamics.J2 import J2
from python_propagate.Dynamics.J3 import J3
from python_propagate.Dynamics.drag import Drag
from python_propagate.Utilities.units import DEG2RAD, RAD2DEG


def initialize_scenario():
    earth = Earth()

    start_time = datetime.strptime("2025-01-15T12:30:00","%Y-%m-%dT%H:%M:%S")
    duration = timedelta(seconds=86400)
    dt = timedelta(seconds=30)

    scenario = Scenario(central_body = earth, start_time = start_time, duration = duration, dt = dt)

    position = [1340.745, -6663.403, -132.528]
    velocity = [5.457807, 1.368701, -5.614317]
    initial_state = State(position=position,velocity=velocity)


    return scenario, initial_state


def problem1():

    scenario, initial_state = initialize_scenario()

    initial_state.time = scenario.start_time
    sma,ecc,inc,raan,arg,nu = initial_state.to_keplerian(scenario.central_body.mu) 
    period = 2*np.pi*np.sqrt(sma**3/scenario.central_body.mu)

    r = np.linalg.norm(initial_state.position)
    v = np.linalg.norm(initial_state.velocity)
    Utb = scenario.central_body.mu / r
    UJ2 = -3/2 * scenario.central_body.J2 * Utb * (scenario.central_body.radius / r)**2 * ((initial_state.z_ECI/r)**2 - 1/3) 
    UJ3 = -1/2 * scenario.central_body.J3 * Utb * (scenario.central_body.radius / r)**3 * (5 * (initial_state.z_ECI/r)**3 - 3 * (initial_state.z_ECI/r))
    energy0 = (v**2 / 2 - (Utb + UJ2 + UJ3))

    sat = Spacecraft(initial_state,
                     start_time=scenario.start_time,
                     duration=scenario.duration,
                     dt = scenario.dt)
    
    sat.set_scenario(scenario=scenario)
    dynamics = ('kepler','J2','J3',)
    sat.add_dynamics(dynamics=dynamics)

    sat.propagate(tolerance=1e-14)

    final_state = sat.state.compile()[np.newaxis,:]
    data = loadmat("C:/Users/ajber/Desktop/College Classes/Spring_2025/Space_Debris/Homework/homewrok1/HW01_ComparisonResults.mat")
    expected_end = data['endState_TwoBody_J2_J3']

    diff = (final_state - expected_end)

    # final_state_table, difference_table = generate_latex_table(final_state, expected_end,'Test')

    final_state_dict = {'X [km]': final_state[0,0],'Y [km]': final_state[0,1],'Z [km]': final_state[0,2],'Vx [km/s]': final_state[0,3],'Vy [km/s]': final_state[0,4],'Vz [km/s]': final_state[0,5]}
    diff_state_dict = {'X [km]': diff[0,0],'Y [km]': diff[0,1],'Z [km]': diff[0,2],'Vx [km/s]': diff[0,3],'Vy [km/s]': diff[0,4],'Vz [km/s]': diff[0,5]}

    final_state_df = pd.DataFrame(final_state_dict, index = ['Final State']).transpose()
    diff_state_df = pd.DataFrame(diff_state_dict, index = ['Difference']).transpose()

    final_state_df.to_latex('final_state_p1.tex',position='H',label='tab:final_state_p1',caption='Final State for Problem 1')
    diff_state_df.to_latex('difference_p1.tex',position='H',label='tab:difference_p1',caption='Difference between final state and expected final state for Problem 1',
                           float_format="%.12e")
    
    sma_data = []
    ecc_data = []
    inc_data = []
    raan_data = []
    arg_data = []
    time_data = []
    energy_data = []

    for i,state in enumerate(sat.state_data):
        sma,ecc,inc,raan,arg,nu = state.to_keplerian(scenario.central_body.mu)
        r = np.linalg.norm(state.position)
        v = np.linalg.norm(state.velocity)
        Utb = scenario.central_body.mu / r
        UJ2 = - 3/2 * scenario.central_body.J2 * Utb * (scenario.central_body.radius / r)**2 * ((state.z_ECI/r)**2 - 1/3) 
        UJ3 = - 1/2 * scenario.central_body.J3 * Utb * (scenario.central_body.radius / r)**3 * (5 * (state.z_ECI/r)**3 - 3 * (state.z_ECI/r))
        energy_data.append((v**2 / 2 - (Utb + UJ2 + UJ3)) - energy0)

        sma_data.append(sma)
        ecc_data.append(ecc)
        inc_data.append(inc*RAD2DEG)
        raan_data.append(raan*RAD2DEG)
        arg_data.append(arg*RAD2DEG)
        time_data.append(i*scenario.dt.total_seconds())



    fig = plt.figure(figsize=(12,8))
    ax_sma = fig.add_subplot(511)
    ax_ecc = fig.add_subplot(512)
    ax_inc = fig.add_subplot(513)
    ax_raan = fig.add_subplot(514)
    ax_arg = fig.add_subplot(515)

    ax_sma.plot(time_data,sma_data,'g',label='Semi-Major Axis')
    ax_ecc.plot(time_data,ecc_data,'b',label='Eccentricity')
    ax_inc.plot(time_data,inc_data,'r',label='Inclination')
    ax_raan.plot(time_data,raan_data,'y',label='RAAN')
    ax_arg.plot(time_data,arg_data,'m',label='Argument of Perigee')

    for ax in [ax_sma, ax_ecc, ax_inc, ax_raan, ax_arg]:
        ax.axvline(x=period, color='k', linestyle='--', label='Period')
        # ax.text(period, ax.get_ylim()[1] * 0.9, f'{period} s', rotation=90, verticalalignment='center')

    ax_sma.set_ylabel('a [km]')
    ax_ecc.set_ylabel('ecc')
    ax_inc.set_ylabel('inc [deg]')
    ax_raan.set_ylabel(r'$\Omega$ [deg]')
    ax_arg.set_ylabel(r'$\omega$ [deg]')

    ax_sma.grid()
    ax_ecc.grid()
    ax_inc.grid()
    ax_raan.grid()
    ax_arg.grid()

    ax_sma.legend()
    ax_ecc.legend()
    ax_inc.legend()
    ax_raan.legend()
    ax_arg.legend()

    ax_sma.set_title('Period: {:.4f} hrs'.format(period/3600))
    
    plt.savefig('figures/keplerian_elements_vs_time_p1.png')
    plt.show()

    fig = plt.figure(figsize=(10,6))
    ax_energy = fig.add_subplot()
    ax_energy.plot(time_data,energy_data,'r',label='Energy') 
    ax_energy.set_ylabel('Energy [kJ/kg]')
    ax_energy.set_xlabel('Time [s]')    
    ax_energy.grid()

    plt.savefig('figures/energy_vs_time_p1.png')
    plt.show()

    return sma_data, ecc_data, inc_data, raan_data, arg_data



def problem2(sma_data, ecc_data, inc_data, raan_data, arg_data):

    scenario, initial_state = initialize_scenario()

    initial_state.time = scenario.start_time
    sma,ecc,inc,raan,arg,nu = initial_state.to_keplerian(scenario.central_body.mu) 
    period = 2*np.pi*np.sqrt(sma**3/scenario.central_body.mu)

    r = np.linalg.norm(initial_state.position)
    v = np.linalg.norm(initial_state.velocity)
    Utb = scenario.central_body.mu / r
    UJ2 = -3/2 * scenario.central_body.J2 * Utb * (scenario.central_body.radius / r)**2 * ((initial_state.z_ECI/r)**2 - 1/3) 
    UJ3 = -1/2 * scenario.central_body.J3 * Utb * (scenario.central_body.radius / r)**3 * (5 * (initial_state.z_ECI/r)**3 - 3 * (initial_state.z_ECI/r))
    energy0 = (v**2 / 2 - (Utb + UJ2 + UJ3))

    coefficent_of_drag = 2.0
    mass = 1350
    area = 3.6 / (1000 ** 2)
    

    sat = Spacecraft(initial_state,
                     start_time=scenario.start_time,
                     duration=scenario.duration,
                     dt = scenario.dt, 
                     coefficent_of_drag=coefficent_of_drag,
                     mass = mass,
                     area = area)
    
    sat.set_scenario(scenario=scenario)
    dynamics = ('kepler','J2','J3','drag')
    sat.add_dynamics(dynamics=dynamics)

    sat.propagate(tolerance=1e-14)

    sma_data_diff = []
    ecc_data_diff = []
    inc_data_diff = []
    raan_data_diff = []
    arg_data_diff = []
    time_data = []
    energy_data = []

    for i,state in enumerate(sat.state_data):
        sma,ecc,inc,raan,arg,nu = state.to_keplerian(scenario.central_body.mu)
        r = np.linalg.norm(state.position)
        v = np.linalg.norm(state.velocity)
        Utb = scenario.central_body.mu / r
        UJ2 = - 3/2 * scenario.central_body.J2 * Utb * (scenario.central_body.radius / r)**2 * ((state.z_ECI/r)**2 - 1/3) 
        UJ3 = - 1/2 * scenario.central_body.J3 * Utb * (scenario.central_body.radius / r)**3 * (5 * (state.z_ECI/r)**3 - 3 * (state.z_ECI/r))
        energy_data.append((v**2 / 2 - (Utb + UJ2 + UJ3)) - energy0)

        sma_data_diff.append(sma - sma_data[i])
        ecc_data_diff.append(ecc - ecc_data[i])
        inc_data_diff.append(inc*RAD2DEG - inc_data[i])
        raan_data_diff.append(raan*RAD2DEG - raan_data[i])
        arg_data_diff.append(arg*RAD2DEG - arg_data[i])
        time_data.append(i*scenario.dt.total_seconds())

    fig = plt.figure(figsize=(12,8))
    ax_sma = fig.add_subplot(511)
    ax_ecc = fig.add_subplot(512)
    ax_inc = fig.add_subplot(513)
    ax_raan = fig.add_subplot(514)
    ax_arg = fig.add_subplot(515)

    ax_sma.plot(time_data,sma_data_diff,'g',label='Semi-Major Axis')
    ax_ecc.plot(time_data,ecc_data_diff,'b',label='Eccentricity')
    ax_inc.plot(time_data,inc_data_diff,'r',label='Inclination')
    ax_raan.plot(time_data,raan_data_diff,'y',label='RAAN')
    ax_arg.plot(time_data,arg_data_diff,'m',label='Argument of Perigee')

    ax_sma.set_ylabel('Difference in a [km]')
    ax_ecc.set_ylabel('Differenc in ecc')
    ax_inc.set_ylabel('Difference inc [deg]')
    ax_raan.set_ylabel(r'Difference $\Omega$ [deg]')
    ax_arg.set_ylabel(r'Difference $\omega$ [deg]')

    ax_sma.grid()
    ax_ecc.grid()
    ax_inc.grid()
    ax_raan.grid()
    ax_arg.grid()

    ax_sma.legend()
    ax_ecc.legend()
    ax_inc.legend()
    ax_raan.legend()
    ax_arg.legend()

    ax_sma.set_title('Period: {:.4f} hrs'.format(period/3600))
    
    plt.savefig('figures/keplerian_elements_vs_time_p2.png')
    plt.show()

    fig = plt.figure(figsize=(10,6))
    ax_energy = fig.add_subplot()
    ax_energy.plot(time_data,energy_data,'r',label='Energy') 
    ax_energy.set_ylabel('Energy [kJ/kg]')
    ax_energy.set_xlabel('Time [s]')    
    ax_energy.grid()

    plt.savefig('figures/energy_vs_time_p2.png')
    plt.show()

def problem3():

    earth = Earth()

    start_time = datetime.strptime("2025-01-15T12:30:00","%Y-%m-%dT%H:%M:%S")
    duration = timedelta(seconds=86400)
    dt = timedelta(seconds=30)

    scenario = Scenario(central_body = earth, start_time = start_time, duration = duration, dt = dt)
    position = np.array([1340.745, -6663.403, -132.528])
    velocity = np.array([5.457807, 1.368701, -5.614317])

    state1 = State(position=np.array(position),velocity=np.array(velocity),stm = np.eye(6))
    state2 = State(position=np.array(position)* (1+1e-7),velocity=np.array(velocity)* (1+1e-7),stm = np.eye(6))

    coefficent_of_drag = 2.0
    mass = 1350
    area = 3.6 * 1e-6

    jah_sat1 = Spacecraft(state1,
                     start_time=start_time,
                     duration=duration,
                     dt = scenario.dt, 
                     coefficent_of_drag=coefficent_of_drag,
                     mass = mass,
                     area = area)
    
    jah_sat2 = Spacecraft(state2,
                     start_time=start_time,
                     duration=duration,
                     dt = scenario.dt, 
                     coefficent_of_drag=coefficent_of_drag,
                     mass = mass,
                     area = area)
    
    jah_sat1.set_scenario(scenario=scenario)
    jah_sat2.set_scenario(scenario=scenario)

    dynamics = ('kepler','J2','J3','drag','stm')
    jah_sat1.add_dynamics(dynamics=dynamics)
    jah_sat2.add_dynamics(dynamics=dynamics)

    
    delta_x0 = np.hstack((state1.position * 1e-7, state1.velocity * 1e-7))
    
    

    #First propagate
    jah_sat1.propagate(tolerance=1e-14)
    xf = jah_sat1.state
    stmf = xf.stm
    delta_xf = stmf @ delta_x0



    #second propagate
    jah_sat2.propagate(tolerance=1e-14)

    xf_test = jah_sat2.state

    #test
    delta_xf2_pos  = -(xf.position - xf_test.position)
    delta_xf2_velo = -(xf.velocity - xf_test.velocity)

    delta_xf2 = np.hstack((delta_xf2_pos,delta_xf2_velo))

    diff = delta_xf2 - delta_xf

    diff_state_dict = {'X [km]': diff[0],'Y [km]': diff[1],'Z [km]': diff[2],'Vx [km/s]': diff[3],'Vy [km/s]': diff[4],'Vz [km/s]': diff[5]}

    diff_state_df = pd.DataFrame(diff_state_dict, index = ['Difference']).transpose()

    diff_state_df.to_latex('difference_p3.tex',position='H',label='tab:difference_p1',caption='Difference between STM and Non-linear solution for Problem 1',
                           float_format="%.12e")




if __name__ == '__main__':

    # sma_data, ecc_data, inc_data, raan_data, arg_data = problem1()
    # problem2(sma_data=sma_data, ecc_data=ecc_data, inc_data=inc_data, raan_data=raan_data, arg_data=arg_data)
    problem3()


