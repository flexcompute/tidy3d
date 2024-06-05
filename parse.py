#!/usr/bin/env -S poetry run python
# ruff: noqa: F401

import re
import numpy as np
import matplotlib.pyplot as plt


def main():
    with open("opt_ag.log") as f_ag, open("opt_aj.log") as f_aj:
        log_ag = f_ag.read()
        log_aj = f_aj.read()

    J_ag = list(map(float, re.findall(r"J = ([-]?\d+\.\d+e-\d\d)", log_ag)))
    J_aj = list(map(float, re.findall(r"J = ([-]?\d+\.\d+e-\d\d)", log_aj)))

    p_ag = list(map(float, re.findall(r"with value (\d+.\d+)", log_ag)))
    p_aj = list(map(float, re.findall(r"penalty: Traced<ConcreteArray\((\d+.\d+),", log_aj)))

    w_ag = list(map(float, re.findall(r"penalty_weight: (\d+.\d+)", log_ag)))
    w_aj = list(map(float, re.findall(r"penalty_weight: (\d+.\d+)", log_aj)))

    x = range(20)
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, J_ag, ls="-", c="tab:blue", label="J")
    ax.plot(x, J_aj, ls="-", c="tab:red", label="J")
    ax.plot(x, p_ag, ls="--", c="tab:blue", label="p")
    ax.plot(x, p_aj, ls="--", c="tab:red", label="p")
    ax.plot(x, np.array(p_ag) * w_ag, ls=":", c="tab:blue", label="p * w")
    ax.plot(x, np.array(p_aj) * w_aj, ls=":", c="tab:red", label="p * w")
    ax.set_xticks(range(0, 20, 2))
    ax.set_xlabel("iteration")
    h, l = ax.get_legend_handles_labels()
    ph = [plt.plot([], marker="", ls="")[0]] * 2
    handles = ph + h
    labels = ["autograd", "adjoint"] + l
    ax.legend(handles, labels, ncol=4, markerfirst=False)
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
