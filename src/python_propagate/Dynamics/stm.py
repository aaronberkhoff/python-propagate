import numpy as np
from python_propagate.Scenario import Scenario
from python_propagate.Dynamics import Dynamic
from python_propagate.Agents.state import State
from numpy import sqrt

#TODO: explore specifying the difference between the classes for normal dynamics and STMs

class STM(Dynamic):

    def __init__(self, scenario:Scenario, agent=None, stm=True):
        super().__init__(scenario, agent, stm)

    def function(self,state: State, time: float):

        stm = state.stm
        A_matrix = self.A_matrix(state, time)

        stm_dot = A_matrix @ stm


        return State(stm_dot=stm_dot)

    def A_matrix(self,state: State, time: float):

        rx, ry, rz = state.extract_position()
        vx, vy, vz = state.extract_velocity()

        r = np.sqrt(rx**2 + ry**2 + rz**2)

        R =  self.scenario.central_body.radius
        mu = self.scenario.central_body.mu
        J2 = self.scenario.central_body.J2
        J3 = self.scenario.central_body.J3

        densityA = self.scenario.central_body.atmosphere_model(r) 
        Cd = self.agent.coefficet_of_drag
        A =  self.agent.area
        M =  self.agent.mass
        w = self.scenario.central_body.angular_velocity

        # A_matrix_total = np.array([[0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1], [0.5*A*Cd*densityA*w*(-rx*w + vy)*(ry*w + vx)/(M*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)) - 21/2*J2*R**2*mu*rx**2*(-rx**2 - ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(9/2) - 3*J2*R**2*mu*rx**2/(rx**2 + ry**2 + rz**2)**(7/2) + (3/2)*J2*R**2*mu*(-rx**2 - ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(7/2) - 45/2*J3*R**3*mu*rx**2*rz*(-3*rx**2 - 3*ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(11/2) - 15*J3*R**3*mu*rx**2*rz/(rx**2 + ry**2 + rz**2)**(9/2) + (5/2)*J3*R**3*mu*rz*(-3*rx**2 - 3*ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(9/2) + 3*mu*rx**2/(rx**2 + ry**2 + rz**2)**(5/2) - mu/(rx**2 + ry**2 + rz**2)**(3/2), -0.5*A*Cd*densityA*w*(ry*w + vx)**2/(M*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)) - 0.5*A*Cd*densityA*w*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)/M - 21/2*J2*R**2*mu*rx*ry*(-rx**2 - ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(9/2) - 3*J2*R**2*mu*rx*ry/(rx**2 + ry**2 + rz**2)**(7/2) - 45/2*J3*R**3*mu*rx*ry*rz*(-3*rx**2 - 3*ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(11/2) - 15*J3*R**3*mu*rx*ry*rz/(rx**2 + ry**2 + rz**2)**(9/2) + 3*mu*rx*ry/(rx**2 + ry**2 + rz**2)**(5/2), -21/2*J2*R**2*mu*rx*rz*(-rx**2 - ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(9/2) + 12*J2*R**2*mu*rx*rz/(rx**2 + ry**2 + rz**2)**(7/2) - 45/2*J3*R**3*mu*rx*rz**2*(-3*rx**2 - 3*ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(11/2) + 20*J3*R**3*mu*rx*rz**2/(rx**2 + ry**2 + rz**2)**(9/2) + (5/2)*J3*R**3*mu*rx*(-3*rx**2 - 3*ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(9/2) + 3*mu*rx*rz/(rx**2 + ry**2 + rz**2)**(5/2), -0.5*A*Cd*densityA*(ry*w + vx)**2/(M*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)) - 0.5*A*Cd*densityA*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)/M, -0.5*A*Cd*densityA*(-rx*w + vy)*(ry*w + vx)/(M*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)), -0.5*A*Cd*densityA*vz*(ry*w + vx)/(M*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2))], [0.5*A*Cd*densityA*w*(-rx*w + vy)**2/(M*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)) + 0.5*A*Cd*densityA*w*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)/M - 21/2*J2*R**2*mu*rx*ry*(-rx**2 - ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(9/2) - 3*J2*R**2*mu*rx*ry/(rx**2 + ry**2 + rz**2)**(7/2) - 45/2*J3*R**3*mu*rx*ry*rz*(-3*rx**2 - 3*ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(11/2) - 15*J3*R**3*mu*rx*ry*rz/(rx**2 + ry**2 + rz**2)**(9/2) + 3*mu*rx*ry/(rx**2 + ry**2 + rz**2)**(5/2), -0.5*A*Cd*densityA*w*(-rx*w + vy)*(ry*w + vx)/(M*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)) - 21/2*J2*R**2*mu*ry**2*(-rx**2 - ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(9/2) - 3*J2*R**2*mu*ry**2/(rx**2 + ry**2 + rz**2)**(7/2) + (3/2)*J2*R**2*mu*(-rx**2 - ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(7/2) - 45/2*J3*R**3*mu*ry**2*rz*(-3*rx**2 - 3*ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(11/2) - 15*J3*R**3*mu*ry**2*rz/(rx**2 + ry**2 + rz**2)**(9/2) + (5/2)*J3*R**3*mu*rz*(-3*rx**2 - 3*ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(9/2) + 3*mu*ry**2/(rx**2 + ry**2 + rz**2)**(5/2) - mu/(rx**2 + ry**2 + rz**2)**(3/2), -21/2*J2*R**2*mu*ry*rz*(-rx**2 - ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(9/2) + 12*J2*R**2*mu*ry*rz/(rx**2 + ry**2 + rz**2)**(7/2) - 45/2*J3*R**3*mu*ry*rz**2*(-3*rx**2 - 3*ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(11/2) + 20*J3*R**3*mu*ry*rz**2/(rx**2 + ry**2 + rz**2)**(9/2) + (5/2)*J3*R**3*mu*ry*(-3*rx**2 - 3*ry**2 + 4*rz**2)/(rx**2 + ry**2 + rz**2)**(9/2) + 3*mu*ry*rz/(rx**2 + ry**2 + rz**2)**(5/2), -0.5*A*Cd*densityA*(-rx*w + vy)*(ry*w + vx)/(M*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)), -0.5*A*Cd*densityA*(-rx*w + vy)**2/(M*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)) - 0.5*A*Cd*densityA*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)/M, -0.5*A*Cd*densityA*vz*(-rx*w + vy)/(M*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2))], [0.5*A*Cd*densityA*vz*w*(-rx*w + vy)/(M*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)) - 21/2*J2*R**2*mu*rx*rz*(-3*rx**2 - 3*ry**2 + 2*rz**2)/(rx**2 + ry**2 + rz**2)**(9/2) - 9*J2*R**2*mu*rx*rz/(rx**2 + ry**2 + rz**2)**(7/2) - 45/2*J3*R**3*mu*rx*(7*rz**4 + (0.6*rx**2 + 0.6*ry**2 - 5.4*rz**2)*(rx**2 + ry**2 + rz**2))/(rx**2 + ry**2 + rz**2)**(11/2) + (5/2)*J3*R**3*mu*(2*rx*(0.6*rx**2 + 0.6*ry**2 - 5.4*rz**2) + 1.2*rx*(rx**2 + ry**2 + rz**2))/(rx**2 + ry**2 + rz**2)**(9/2) + 3*mu*rx*rz/(rx**2 + ry**2 + rz**2)**(5/2), -0.5*A*Cd*densityA*vz*w*(ry*w + vx)/(M*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)) - 21/2*J2*R**2*mu*ry*rz*(-3*rx**2 - 3*ry**2 + 2*rz**2)/(rx**2 + ry**2 + rz**2)**(9/2) - 9*J2*R**2*mu*ry*rz/(rx**2 + ry**2 + rz**2)**(7/2) - 45/2*J3*R**3*mu*ry*(7*rz**4 + (0.6*rx**2 + 0.6*ry**2 - 5.4*rz**2)*(rx**2 + ry**2 + rz**2))/(rx**2 + ry**2 + rz**2)**(11/2) + (5/2)*J3*R**3*mu*(2*ry*(0.6*rx**2 + 0.6*ry**2 - 5.4*rz**2) + 1.2*ry*(rx**2 + ry**2 + rz**2))/(rx**2 + ry**2 + rz**2)**(9/2) + 3*mu*ry*rz/(rx**2 + ry**2 + rz**2)**(5/2), -21/2*J2*R**2*mu*rz**2*(-3*rx**2 - 3*ry**2 + 2*rz**2)/(rx**2 + ry**2 + rz**2)**(9/2) + 6*J2*R**2*mu*rz**2/(rx**2 + ry**2 + rz**2)**(7/2) + (3/2)*J2*R**2*mu*(-3*rx**2 - 3*ry**2 + 2*rz**2)/(rx**2 + ry**2 + rz**2)**(7/2) - 45/2*J3*R**3*mu*rz*(7*rz**4 + (0.6*rx**2 + 0.6*ry**2 - 5.4*rz**2)*(rx**2 + ry**2 + rz**2))/(rx**2 + ry**2 + rz**2)**(11/2) + (5/2)*J3*R**3*mu*(28*rz**3 + 2*rz*(0.6*rx**2 + 0.6*ry**2 - 5.4*rz**2) - 10.8*rz*(rx**2 + ry**2 + rz**2))/(rx**2 + ry**2 + rz**2)**(9/2) + 3*mu*rz**2/(rx**2 + ry**2 + rz**2)**(5/2) - mu/(rx**2 + ry**2 + rz**2)**(3/2), -0.5*A*Cd*densityA*vz*(ry*w + vx)/(M*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)), -0.5*A*Cd*densityA*vz*(-rx*w + vy)/(M*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)), -0.5*A*Cd*densityA*vz**2/(M*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)) - 0.5*A*Cd*densityA*sqrt(vz**2 + (-rx*w + vy)**2 + (ry*w + vx)**2)/M]])

        t2 = rx*w
        t3 = ry*w
        t4 = R**2
        t5 = R**3
        t6 = rz*3.0
        t7 = rx**2
        t8 = ry**2
        t9 = rz**2
        t10 = rz**3
        t12 = vx*2.0
        t13 = vy*2.0
        t14 = vz**2
        t17 = 1.0/M
        t11 = t9**2
        t15 = t2*2.0
        t16 = t3*2.0
        t18 = t3+vx
        t19 = -t2
        t22 = t7*(3.0/5.0)
        t23 = t8*(3.0/5.0)
        t25 = t9*(2.7e+1/5.0)
        t29 = (t2-vy)**2
        t30 = t7+t8+t9
        t20 = -t15
        t21 = t19+vy
        t24 = t18**2
        t26 = t12+t16
        t27 = -t25
        t31 = 1.0/t30
        t33 = 1.0/t30**(3.0/2.0)
        t34 = 1.0/t30**(5.0/2.0)
        t35 = 1.0/t30**(7.0/2.0)
        t37 = 1.0/t30**(1.1e+1/2.0)
        t28 = t13+t20
        t32 = t31**2
        t36 = t33**3
        t38 = mu*t33
        t39 = rz*t31*1.0e+1
        t40 = t9*t31*5.0
        t41 = t10*t31*7.0
        t42 = t11*t31*7.0
        t47 = t9*t31*2.1e+1
        t48 = t14+t24+t29
        t51 = mu*rx*ry*t34*3.0
        t52 = mu*rx*t6*t34
        t53 = mu*ry*t6*t34
        t60 = J3*mu*rx*ry*t5*t10*t37*3.5e+1
        t43 = -t38
        t44 = -t41
        t45 = t10*t32*1.0e+1
        t46 = t11*t32*1.4e+1
        t50 = -t47
        t54 = t40-1.0
        t57 = sqrt(t48)
        t59 = J2*mu*rx*ry*t4*t9*t36*1.5e+1
        t62 = -t60
        t68 = t22+t23+t27+t42
        t49 = -t45
        t55 = t54-2.0
        t56 = t6+t44
        t58 = 1.0/t57
        t61 = -t59
        t63 = (A*Cd*densityA*t17*t57)/2.0
        t67 = t46+t50+3.0
        t69 = J2*mu*t4*t34*t54*(3.0/2.0)
        t72 = J2*mu*rx*ry*t4*t35*t54*(1.5e+1/2.0)
        t64 = -t63
        t65 = t63*w
        t66 = t39+t49
        t70 = J3*mu*t5*t35*t56*(5.0/2.0)
        t73 = A*Cd*densityA*t17*t18*t58*w*(t2-vy)*(-1.0/2.0)
        t74 = -t72
        t75 = J3*mu*rx*ry*t5*t36*t56*(3.5e+1/2.0)
        t71 = -t70
        mt1 = [0.0,0.0,0.0,t43+t69+t71+t73+mu*t7*t34*3.0-J2*mu*t4*t7*t9*t36*1.5e+1-J3*mu*t5*t7*t10*t37*3.5e+1-J2*mu*t4*t7*t35*t54*(1.5e+1/2.0)+J3*mu*t5*t7*t36*t56*(3.5e+1/2.0),t51+t61+t62+t65+t74+t75+(A*Cd*densityA*t17*t29*t58*w)/2.0,t52+J3*mu*t5*t35*(rx*(6.0/5.0)-rx*t11*t32*1.4e+1)*(5.0/2.0)-J2*mu*rx*t4*t10*t36*1.5e+1-J3*mu*rx*t5*t36*t68*(3.5e+1/2.0)-J2*mu*rx*rz*t4*t35*t55*(1.5e+1/2.0)-(A*Cd*densityA*t17*t58*vz*w*(t2-vy))/2.0,0.0,0.0,0.0,t51+t61+t62+t74+t75-(A*Cd*densityA*t17*t57*w)/2.0-(A*Cd*densityA*t17*t24*t58*w)/2.0]
        mt2 = [t43+t69+t71+mu*t8*t34*3.0-J2*mu*t4*t8*t9*t36*1.5e+1-J3*mu*t5*t8*t10*t37*3.5e+1-J2*mu*t4*t8*t35*t54*(1.5e+1/2.0)+J3*mu*t5*t8*t36*t56*(3.5e+1/2.0)+(A*Cd*densityA*t17*t18*t58*w*(t2-vy))/2.0,t53+J3*mu*t5*t35*(ry*(6.0/5.0)-ry*t11*t32*1.4e+1)*(5.0/2.0)-J2*mu*ry*t4*t10*t36*1.5e+1-J3*mu*ry*t5*t36*t68*(3.5e+1/2.0)-J2*mu*ry*rz*t4*t35*t55*(1.5e+1/2.0)-(A*Cd*densityA*t17*t18*t58*vz*w)/2.0,0.0,0.0,0.0,t52+J2*mu*rx*t4*t34*t66*(3.0/2.0)-J3*mu*rx*t5*t35*t67*(5.0/2.0)-J2*mu*rx*rz*t4*t35*t54*(1.5e+1/2.0)+J3*mu*rx*rz*t5*t36*t56*(3.5e+1/2.0)]
        mt3 = [t53+J2*mu*ry*t4*t34*t66*(3.0/2.0)-J3*mu*ry*t5*t35*t67*(5.0/2.0)-J2*mu*ry*rz*t4*t35*t54*(1.5e+1/2.0)+J3*mu*ry*rz*t5*t36*t56*(3.5e+1/2.0),t43+mu*rz*t6*t34-J3*mu*t5*t35*(rz*(5.4e+1/5.0)-t10*t31*2.8e+1+rz**5*t32*1.4e+1)*(5.0/2.0)+J2*mu*t4*t34*t55*(3.0/2.0)+J2*mu*rz*t4*t34*t66*(3.0/2.0)-J3*mu*rz*t5*t36*t68*(3.5e+1/2.0)-J2*mu*t4*t9*t35*t55*(1.5e+1/2.0),1.0,0.0,0.0,t64-(A*Cd*densityA*t17*t18*t26*t58)/4.0,(A*Cd*densityA*t17*t26*t58*(t2-vy))/4.0,A*Cd*densityA*t17*t26*t58*vz*(-1.0/4.0),0.0,1.0,0.0,A*Cd*densityA*t17*t18*t28*t58*(-1.0/4.0)]
        mt4 = [t64+(A*Cd*densityA*t17*t28*t58*(t2-vy))/4.0,A*Cd*densityA*t17*t28*t58*vz*(-1.0/4.0),0.0,0.0,1.0,A*Cd*densityA*t17*t18*t58*vz*(-1.0/2.0),(A*Cd*densityA*t17*t58*vz*(t2-vy))/2.0,t64-(A*Cd*densityA*t14*t17*t58)/2.0]
        A_matrix_total = np.reshape((mt1+mt2+mt3+mt4),(6,6))

    


        return A_matrix_total
