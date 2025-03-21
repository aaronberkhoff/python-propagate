from python_propagate.states import State

class OrbitalElements(State):

    def __init__(self, frame="inertial", **kwargs):
        super().__init__(frame, **kwargs)

    