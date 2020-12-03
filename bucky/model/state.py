"""Provide a class to hold the internal state vector to the compartment model (and track compartment indices)."""

import contextlib

from ..numerical_libs import reimport_numerical_libs, xp


class buckyState:  # pylint: disable=too-many-instance-attributes
    """Class to manage the state of the bucky compartments (and their indices)."""

    def __init__(self, consts, Nij, state=None):
        """Initialize the compartment indices and the state vector using the calling modules numerical libs"""

        reimport_numerical_libs("model.state.buckyState.__init__")

        self.En = consts["En"]  # TODO rename these to like gamma shape or something
        self.Im = consts["Im"]
        self.Rhn = consts["Rhn"]
        self.consts = consts

        indices = {"S": 0}
        indices["E"] = slice(1, self.En + 1)
        indices["I"] = slice(indices["E"].stop, indices["E"].stop + self.Im)
        indices["Ic"] = slice(indices["I"].stop, indices["I"].stop + self.Im)
        indices["Ia"] = slice(indices["Ic"].stop, indices["Ic"].stop + self.Im)
        indices["R"] = slice(indices["Ia"].stop, indices["Ia"].stop + 1)
        indices["Rh"] = slice(indices["R"].stop, indices["R"].stop + self.Rhn)
        indices["D"] = slice(indices["Rh"].stop, indices["Rh"].stop + 1)

        indices["Itot"] = xp.concatenate([xp.r_[v] for k, v in indices.items() if k in ("I", "Ia", "Ic")])
        indices["H"] = xp.concatenate([xp.r_[v] for k, v in indices.items() if k in ("Ic", "Rh")])

        indices["incH"] = slice(indices["D"].stop, indices["D"].stop + 1)
        indices["incC"] = slice(indices["incH"].stop, indices["incH"].stop + 1)

        indices["N"] = slice(0, indices["D"].stop)

        self.indices = indices

        self.n_compartments = xp.to_cpu(indices["incC"].stop)

        self.n_age_grps, self.n_nodes = Nij.shape

        if state is None:
            self.state = xp.zeros(self.state_shape)
        else:
            self.state = state

    def __getattribute__(self, attr):
        """Allow for . access to the compartment indices, otherwise return the 'normal' attribute."""
        with contextlib.suppress(AttributeError):
            if attr in super().__getattribute__("indices"):
                out = self.state[self.indices[attr]]
                if out.shape[0] == 1:
                    out = xp.squeeze(out, axis=0)
                return out

        return super().__getattribute__(attr)

    def __setattr__(self, attr, x):
        """Allow setting of compartments using . notation, otherwise default to normal attribute behavior."""
        try:
            if attr in super().__getattribute__("indices"):
                # TODO check that its a slice otherwise this wont work so we should warn
                self.state[self.indices[attr]] = x
            else:
                super().__setattr__(attr, x)
        except AttributeError:
            super().__setattr__(attr, x)

    @property
    def state_shape(self):
        """Return the shape of the internal state ndarray."""
        return (self.n_compartments, self.n_age_grps, self.n_nodes)

    def init_S(self):
        """Init the S compartment such that N=1."""
        self.S = 1.0 - xp.sum(self.state, axis=0)
