"""Provide a class to hold the internal state vector to the compartment model (and track compartment indices)."""

import contextlib
import copy

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

        # Build a dict of bin counts per evolved compartment
        bin_counts = {}
        for name in ("S", "R", "D", "incH", "incC"):
            bin_counts[name] = 1
        for name in ("I", "Ic", "Ia"):
            bin_counts[name] = self.Im
        bin_counts["E"] = self.En
        bin_counts["Rh"] = self.Rhn

        # calculate slices for each compartment
        indices = {}
        current_index = 0
        for name, nbins in bin_counts.items():
            indices[name] = slice(current_index, current_index + nbins)
            current_index += nbins

        # define some combined compartment indices
        indices["N"] = xp.concatenate([xp.r_[v] for k, v in indices.items() if "inc" not in k])
        indices["Itot"] = xp.concatenate([xp.r_[v] for k, v in indices.items() if k in ("I", "Ia", "Ic")])
        indices["H"] = xp.concatenate([xp.r_[v] for k, v in indices.items() if k in ("Ic", "Rh")])

        self.indices = indices

        self.n_compartments = sum([n for n in bin_counts.values()])

        self.n_age_grps, self.n_nodes = Nij.shape

        if state is None:
            self.state = xp.zeros(self.state_shape)
        else:
            self.state = state

    def zeros_like(self):
        ret = copy.copy(self)
        ret.state = xp.zeros_like(self.state)
        return ret

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
