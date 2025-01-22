import numpy as np
from python_propagate.Scenario import Scenario
from python_propagate.Agents import Agent

def stm(state: np.array, time: float, scenario:Scenario, agent:Agent):

    rx = state[0]
    ry = state[1]
    rz = state[2]
    
    vx = state[3]
    vy = state[4]
    vz = state[5]

    r = np.sqrt(rx**2 + ry**2 + rz**2)

    R = scenario.central_body.radius
    mu = scenario.central_body.mu
    J2 = scenario.central_body.J2
    J3 = scenario.central_body.J3

    density = scenario.central_body.atmosphere_model(r) 
    Cd = agent.coefficet_of_drag
    A = agent.area
    M = agent.mass

    t2 = rx * scenario.central_body.angular_velocity
    t3 = ry * scenario.central_body.angular_velocity
    t4 = R**2
    t5 = R**3
    t6 = rz * 3.0
    t7 = rx**2
    t8 = ry**2
    t9 = rz**2
    t10 = rz**3
    t12 = vx * 2.0
    t13 = vy * 2.0
    t14 = vz**2
    t17 = 1.0 / M
    t11 = t9**2
    t15 = t2 * 2.0
    t16 = t3 * 2.0
    t18 = t3 + vx
    t19 = -t2
    t22 = t7 * (3.0 / 5.0)
    t23 = t8 * (3.0 / 5.0)
    t25 = t9 * (27.0 / 5.0)
    t29 = (t2 - vy)**2
    t30 = t7 + t8 + t9
    t20 = -t15
    t21 = t19 + vy
    t24 = t18**2
    t26 = t12 + t16
    t27 = -t25
    t31 = 1.0 / t30
    t33 = 1.0 / t30**(3.0 / 2.0)
    t34 = 1.0 / t30**(5.0 / 2.0)
    t35 = 1.0 / t30**(7.0 / 2.0)
    t37 = 1.0 / t30**(11.0 / 2.0)
    t28 = t13 + t20
    t32 = t31**2
    t36 = t33**3
    t38 = mu * t33
    t39 = rz * t31 * 10.0
    t40 = t9 * t31 * 5.0
    t41 = t10 * t31 * 7.0
    t42 = t11 * t31 * 7.0
    t47 = t9 * t31 * 21.0
    t48 = t14 + t24 + t29
    t51 = mu * rx * ry * t34 * 3.0
    t52 = mu * rx * t6 * t34
    t53 = mu * ry * t6 * t34
    t60 = J3 * mu * rx * ry * t5 * t10 * t37 * 35.0
    t43 = -t38
    t44 = -t41
    t45 = t10 * t32 * 10.0
    t46 = t11 * t32 * 14.0
    t50 = -t47
    t54 = t40 - 1.0
    t57 = np.sqrt(t48)
    t59 = J2 * mu * rx * ry * t4 * t9 * t36 * 15.0
    t62 = -t60
    t68 = t22 + t23 + t27 + t42
    t49 = -t45
    t55 = t54 - 2.0
    t56 = t6 + t44
    t58 = 1.0 / t57
    t61 = -t59
    t63 = (A * Cd * density * t17 * t57) / 2.0
    t67 = t46 + t50 + 3.0
    t69 = J2 * mu * t4 * t34 * t54 * (3.0 / 2.0)
    t72 = J2 * mu * rx * ry * t4 * t35 * t54 * (15.0 / 2.0)
    t64 = -t63
    t65 = t63 * scenario.central_body.angular_velocity
    t66 = t39 + t49
    t70 = J3 * mu * t5 * t35 * t56 * (5.0 / 2.0)
    t73 = A * Cd * density * t17 * t18 * t58 * scenario.central_body.angular_velocity * (t2 - vy) * (-1.0 / 2.0)
    t74 = -t72
    t75 = J3 * mu * rx * ry * t5 * t36 * t56 * (35.0 / 2.0)
    t71 = -t70

    mt1 = [
        0.0,
        0.0,
        0.0,
        t43 + t69 + t71 + t73 + mu * t7 * t34 * 3.0 - J2 * mu * t4 * t7 * t9 * t36 * 15.0 - J3 * mu * t5 * t7 * t10 * t37 * 35.0 - J2 * mu * t4 * t7 * t35 * t54 * (15.0 / 2.0) + J3 * mu * t5 * t7 * t36 * t56 * (35.0 / 2.0),
        t51 + t61 + t62 + t65 + t74 + t75 + (A * Cd * density * t17 * t29 * t58 * scenario.central_body.angular_velocity) / 2.0,
        t52 + J3 * mu * t5 * t35 * (rx * (6.0 / 5.0) - rx * t11 * t32 * 14.0) * (5.0 / 2.0) - J2 * mu * rx * t4 * t10 * t36 * 15.0 - J3 * mu * rx * t5 * t36 * t68 * (35.0 / 2.0) - J2 * mu * rx * rz * t4 * t35 * t55 * (15.0 / 2.0) - (A * Cd * density * t17 * t58 * scenario.central_body.angular_velocity * vz * (t2 - vy)) / 2.0,
        0.0,
        0.0,
        0.0,
        t51 + t61 + t62 + t74 + t75 - (A * Cd * density * t17 * t57 * scenario.central_body.angular_velocity) / 2.0 - (A * Cd * density * t17 * t24 * t58 * scenario.central_body.angular_velocity) / 2.0
    ]

    mt2 = [
        t43 + t69 + t71 + mu * t8 * t34 * 3.0 - J2 * mu * t4 * t8 * t9 * t36 * 15.0 - J3 * mu * t5 * t8 * t10 * t37 * 35.0 - J2 * mu * t4 * t8 * t35 * t54 * (15.0 / 2.0) + J3 * mu * t5 * t8 * t36 * t56 * (35.0 / 2.0) + (A * Cd * density * t17 * t18 * t58 * scenario.central_body.angular_velocity * (t2 - vy)) / 2.0,
        t53 + J3 * mu * t5 * t35 * (ry * (6.0 / 5.0) - ry * t11 * t32 * 14.0) * (5.0 / 2.0) - J2 * mu * ry * t4 * t10 * t36 * 15.0 - J3 * mu * ry * t5 * t36 * t68 * (35.0 / 2.0) - J2 * mu * ry * rz * t4 * t35 * t55 * (15.0 / 2.0) - (A * Cd * density * t17 * t18 * t58 * scenario.central_body.angular_velocity * vz) / 2.0,
        0.0,
        0.0,
        0.0,
        t52 + J2 * mu * rx * t4 * t34 * t66 * (3.0 / 2.0) - J3 * mu * rx * t5 * t35 * t67 * (5.0 / 2.0) - J2 * mu * rx * rz * t4 * t35 * t54 * (15.0 / 2.0) + J3 * mu * rx * rz * t5 * t36 * t56 * (35.0 / 2.0)
    ]

    mt3 = [
        t53 + J2 * mu * ry * t4 * t34 * t66 * (3.0 / 2.0) - J3 * mu * ry * t5 * t35 * t67 * (5.0 / 2.0) - J2 * mu * ry * rz * t4 * t35 * t54 * (15.0 / 2.0) + J3 * mu * ry * rz * t5 * t36 * t56 * (35.0 / 2.0),
        t43 + mu * rz * t6 * t34 - J3 * mu * t5 * t35 * (rz * (54.0 / 5.0) - t10 * t31 * 28.0 + rz**5 * t32 * 14.0) * (5.0 / 2.0) + J2 * mu * t4 * t34 * t55 * (3.0 / 2.0) + J2 * mu * rz * t4 * t34 * t66 * (3.0 / 2.0) - J3 * mu * rz * t5 * t36 * t68 * (35.0 / 2.0) - J2 * mu * t4 * t9 * t35 * t55 * (15.0 / 2.0),
        1.0,
        0.0,
        0.0,
        t64 - (A * Cd * density * t17 * t18 * t26 * t58) / 4.0,
        (A * Cd * density * t17 * t26 * t58 * (t2 - vy)) / 4.0,
        A * Cd * density * t17 * t26 * t58 * vz * (-1.0 / 4.0),
        0.0,
        1.0,
        0.0,
        A * Cd * density * t17 * t18 * t28 * t58 * (-1.0 / 4.0)
    ]

    mt4 = [
        t64 + (A * Cd * density * t17 * t28 * t58 * (t2 - vy)) / 4.0,
        A * Cd * density * t17 * t28 * t58 * vz * (-1.0 / 4.0),
        0.0,
        0.0,
        1.0,
        A * Cd * density * t17 * t18 * t58 * vz * (-1.0 / 2.0),
        (A * Cd * density * t17 * t58 * vz * (t2 - vy)) / 2.0,
        t64 - (A * Cd * density * t14 * t17 * t58) / 2.0
    ]

    A_matrix_total = np.reshape([mt1, mt2, mt3, mt4], (6, 6))
    return A_matrix_total
