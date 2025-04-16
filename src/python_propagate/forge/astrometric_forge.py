import pandas as pd
import numpy as np
import h5py
from collections import namedtuple
import sqlite3
from datetime import datetime, timedelta
from copy import deepcopy

from sgp4.api import Satrec

from python_propagate.utilities.units import RAD2DEG, ARC2DEG, DEG2RAD
from python_propagate.forge import Forge
from python_propagate.utilities.load_spice import load_spice
from python_propagate.states.proper_orbital_elements import ProperElements

from python_propagate.agents import Agent
from python_propagate.utilities.string_format import DATESTR
from python_propagate.states import OrbitalElements, State

from python_propagate.utilities.transforms import mean2true


class AstroForge(Forge):

    def __init__(
        self,
        scenario,
        genes,
        output_directory,
        data_types,
        output_types,
        add_noise=False,
        plots=None,
        name="Forge",
    ):
        super().__init__(
            scenario,
            genes,
            output_directory,
            data_types,
            output_types,
            add_noise,
            plots,
            name,
        )

    def process_agent(
        self, agent, scenario, datatypes, add_noise=False, propagate=True
    ):
        load_spice()
        if propagate:
            agent.propagate()  # Update agent state

        elements = [
            state.to_keplerian(agent.scenario.central_body.mu)
            for state in agent.state_data
        ]

        proper_elements = ProperElements(
            orbital_element_data=elements, dt=agent.dt, duration=agent.duration
        )

        mean_oe = np.array(proper_elements.mean_orbital_elements_data).T
        proper_oe = np.array(proper_elements.proper_orbital_elements_data).T

        data_agent = []

        for i, (state, oe, moe, poe) in enumerate(
            zip(agent.state_data, elements, mean_oe, proper_oe)
        ):
            for station in scenario.stations:
                az, el = station.calculate_azimuth_and_elevation(state=state)
                az *= RAD2DEG
                el *= RAD2DEG

                az += np.random.normal(0, 5 * ARC2DEG)
                el += np.random.normal(0, 5 * ARC2DEG)

                ra, dec = station.calculate_ra_and_dec(state=state)
                ra *= RAD2DEG
                dec *= RAD2DEG

                ra += np.random.normal(0, 5 * ARC2DEG)
                dec += np.random.normal(0, 5 * ARC2DEG)

                rho, rhodot = station.calculate_range_and_range_rate_from_target(
                    state=state
                )
                rho += np.random.normal(0, 1e-3)
                rhodot += np.random.normal(0, 1e-6)
                cnt = 0

                if add_noise:

                    az_noise = np.random.normal(0, 5 * ARC2DEG)
                    el_noise = np.random.normal(0, 5 * ARC2DEG)

                    ra_noise = np.random.normal(0, 5 * ARC2DEG)
                    dec_noise = np.random.normal(0, 5 * ARC2DEG)

                    rho_noise = np.random.normal(0, 1e-3)
                    rhodot_noise = np.random.normal(0, 1e-6)

                    sma_noise = np.random.normal(0, 10)
                    ecc_noise = np.random.normal(0, 0.001)
                    inc_noise = np.random.normal(0, 2)
                    arg_noise = np.random.normal(0, 2)
                    raan_noise = np.random.normal(0, 2)
                    nu_noise = np.random.normal(0, 2)

                else:

                    az_noise = 0
                    el_noise = 0

                    ra_noise = 0
                    dec_noise = 0

                    rho_noise = 0
                    rhodot_noise = 0

                    sma_noise = 0
                    ecc_noise = 0
                    inc_noise = 0
                    arg_noise = 0
                    raan_noise = 0
                    nu_noise = 0

                if el > station.minimum_elevation_angle:
                    data_entry = {
                        "agent": agent.name,
                        "index": i,
                        "epoch_time": state.time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "time_sec": i * agent.dt.total_seconds(),
                        "station": station.name,
                        "station_id": station.identity,
                        "RA_DEG": (ra + ra_noise) % 360,
                        "DEC_DEG": max(-90, min(90, dec + dec_noise)),
                        "AZ_DEG": az + az_noise,
                        "EL_DEG": el + el_noise,
                        "RANGE_KM": rho + rho_noise,
                        "RANGE_RATE_KMS": rhodot + rhodot_noise,
                        "X_INERTIAL_KM": state.position[0],
                        "Y_INERTIAL_KM": state.position[1],
                        "Z_INERTIAL_KM": state.position[2],
                        "VX_INERTIAL_KMS": state.velocity[0],
                        "VY_INERTIAL_KMS": state.velocity[1],
                        "VZ_INERTIAL_KMS": state.velocity[2],
                        "LAT_DEG": state.latlong[0] * RAD2DEG,
                        "LON_DEG": state.latlong[1] * RAD2DEG,
                        "ALT_KM": np.linalg.norm(state.position)
                        - agent.scenario.central_body.radius,
                        "SMA_KM": oe.sma + sma_noise,
                        "ECC_KM": oe.ecc + ecc_noise,
                        "INC_DEG": (oe.inc * RAD2DEG + inc_noise) % 180,
                        "ARG_DEG": (oe.arg * RAD2DEG + arg_noise) % 360,
                        "RAAN_DEG": (oe.raan * RAD2DEG + raan_noise) % 360,
                        "NU_DEG": (oe.nu * RAD2DEG + nu_noise) % 360,
                        "MEAN_SMA_KM": moe[0] + sma_noise,
                        "MEAN_ECC_KM": moe[1] + ecc_noise,
                        "MEAN_INC_DEG": (moe[2] * RAD2DEG + inc_noise) % 180,
                        "MEAN_ARG_DEG": (moe[3] * RAD2DEG + arg_noise) % 360,
                        "MEAN_RAAN_DEG": (moe[4] * RAD2DEG + raan_noise) % 360,
                        "MEAN_NU_DEG": (moe[5] * RAD2DEG + nu_noise) % 360,
                        "PROP_SMA_KM": poe[0] + sma_noise,
                        "PROP_ECC_KM": poe[1] + ecc_noise,
                        "PROP_INC_DEG": (poe[2] * RAD2DEG + inc_noise) % 180,
                        "PROP_ARG_DEG": (poe[3] * RAD2DEG + arg_noise) % 360,
                        "PROP_RAAN_DEG": (poe[4] * RAD2DEG + raan_noise) % 360,
                        "PROP_NU_DEG": (poe[5] * RAD2DEG + nu_noise) % 360,
                    }

                    filtered_data_entry = {
                        key: value
                        for key, value in data_entry.items()
                        if key in datatypes
                        or key
                        in {
                            "agent",
                            "index",
                            "time_sec",
                            "epoch_time",
                            "station",
                            "station_id",
                        }
                    }
                    # for data_type in datatypes:
                    #     if data_type in data_entry:
                    #         pass

                    data_agent.append(filtered_data_entry)
                    cnt += 1

        return data_agent, agent
