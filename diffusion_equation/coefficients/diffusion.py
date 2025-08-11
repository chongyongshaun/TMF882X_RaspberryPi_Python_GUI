def D(mua, musp, mua_independent):
    """
    Calculate the diffusion coefficient.

    :param mua: `\mu_a`
    :param musp: `\mu_s'`
    :param mua_independent: Whether the `\mu_a` independent form of the diffusion coefficient equation is used
        rather than the `\mu_a` dependent form.
    :returns: D, the diffusion coefficient
    """
    return 1/(3*float(musp)) if bool(mua_independent) else 1/(3*(float(musp)+float(mua)))
