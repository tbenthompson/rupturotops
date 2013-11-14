def mult_with_smoothness(cells, half_width, padded, smoothness):
    beta = [[0.0 for col in range(half_width)] for row in range(cells)]
    for l in range(cells):
        for r in range(half_width):
            for i in range(0, half_width):
                for j in range(0, half_width):
                    beta[l][half_width - 1 - r] += smoothness[l][r][i][j] * \
                            padded[half_width - 1 + l - r + i] * \
                            padded[half_width - 1 + l - r + j]
    #note that this result is in the backwards pyWENO form
    #retval = [beta3, beta2, beta1] in the Shu 2009 notation
    return beta

