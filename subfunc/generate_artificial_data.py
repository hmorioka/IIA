"""Data generation"""

import numpy as np
from subfunc.showdata import *


# =============================================================
# =============================================================
def generate_artificial_data(num_comp,
                             num_data,
                             num_layer,
                             num_basis,
                             modulate_range1,
                             modulate_range2=None,
                             mix_mode='dyn',
                             num_data_test=None,
                             negative_slope=0.2,
                             x_limit=1e2,
                             eig_zerotolerance=1e-7,
                             random_seed=0):
    """Generate artificial data.
    Args:
        num_comp: number of components
        num_data: number of data points
        num_layer: number of layers of mixing-MLP
        num_basis: number of basis of modulation
        modulate_range1: range of the normalization for the modulations (scale)
        modulate_range2: range of the normalization for the modulations (mean)
        mix_mode: 'dyn' for dynamical model, 'inst' for instantaneous mixture model
        num_data_test: (option) number of data points (testing data, if necessary)
        negative_slope: negative slope of leaky ReLU
        x_limit: if x exceed this range, re-generate it
        eig_zerotolerance: acceptable minimum eigenvalue ratio of data
        random_seed: (option) random seed
    Returns:
        x: observed signals [num_comp, num_data]
        s: source signals [num_comp, num_data]
        label: labels [num_data]
        x_te: observed signals (test data) [num_comp, num_data]
        s_te: source signals (test data) [num_comp, num_data]
        label_te: labels (test data) [num_data]
        mlplayer: parameters of MLP
        modulate1: modulation parameter1 [num_comp, num_data]
        modulate2: modulation parameter2 [num_comp, num_data]
    """

    if mix_mode == 'dyn':
        cat_input = True
    else:
        cat_input = False

    random_seed_orig = random_seed
    stable_flag = False
    cnt = 0
    while not stable_flag:
        # Change random seed
        random_seed = random_seed_orig + num_data + num_layer * 100 + cnt * 10000

        # Generate MLP parameters
        mlplayer = gen_mlp_parms(num_comp,
                                 num_layer,
                                 cat_input=cat_input,
                                 negative_slope=negative_slope,
                                 random_seed=random_seed)

        # Generate source signal (k=2)
        x, s, label, modulate1, modulate2 = gen_x_gauss_scale_mean(num_comp,
                                                                   num_data,
                                                                   modulate_range1=modulate_range1,
                                                                   modulate_range2=modulate_range2,
                                                                   mlplayer=mlplayer,
                                                                   num_basis=num_basis,
                                                                   cat_input=cat_input,
                                                                   negative_slope=negative_slope,
                                                                   x_limit=x_limit,
                                                                   random_seed=random_seed)

        # Check stability
        x_max = np.max(np.max(x, axis=1), axis=0)
        if x_max < x_limit:
            stable_flag = True
        # Check eigenvalues (do not allow zero eigenvalue)
        d, V = np.linalg.eigh(np.cov(x))
        if np.sum((d / d[-1]) < eig_zerotolerance) > 0:
            stable_flag = False

        cnt = cnt + 1

    # Add test data (option)
    if num_data_test is not None:
        stable_flag = False
        cnt = 0
        while not stable_flag:
            # Change random seed
            random_seed_test = random_seed + (cnt+1)*10000

            x_te, s_te, label_te, _, _ = gen_x_gauss_scale_mean(num_comp,
                                                                num_data_test,
                                                                modulate1=modulate1,
                                                                modulate2=modulate2,
                                                                mlplayer=mlplayer,
                                                                num_basis=num_basis,
                                                                cat_input=cat_input,
                                                                negative_slope=negative_slope,
                                                                x_limit=x_limit,
                                                                random_seed=random_seed_test)

            # Check stability
            x_max = np.max(np.max(x_te, axis=1), axis=0)
            if x_max < x_limit:
                stable_flag = True
            # Check eigenvalues (do not allow zero eigenvalue)
            d, V = np.linalg.eigh(np.cov(x_te))
            if np.sum((d / d[-1]) < eig_zerotolerance) > 0:
                stable_flag = False

            cnt = cnt + 1

    else:
        x_te = None
        s_te = None
        label_te = None

    return x, s, label, x_te, s_te, label_te, mlplayer, modulate1, modulate2


# =============================================================
# =============================================================
def gen_x_gauss_scale_mean(num_comp,
                           num_data,
                           mlplayer,
                           num_basis,
                           modulate_range1=None,
                           modulate_range2=None,
                           modulate1=None,
                           modulate2=None,
                           cat_input=False,
                           negative_slope=None,
                           x_limit=None,
                           random_seed=0):
    """Generate source signal for PCL.
    Args:
        num_comp: number of components
        num_data: number of data points
        mlplayer: MLP parameters by gen_mlp_parms()
        num_basis: number of frequencies of fourier bases
        modulate_range1: range of modulation1
        modulate_range2: range of modulation2
        modulate1: (option) pre-generated modulation (scale)
        modulate2: (option) pre-generated modulation (bias)
        cat_input: concatenate x and s for input (i.e. AR model) or not
        negative_slope: negative slope of leaky ReLU
        x_limit: if x exceed this range, re-generate it
        random_seed: (option) random seed
    Returns:
        x: observed signals [num_comp, num_data]
        s: source signals [num_comp, num_data]
        label: labels [num_data]
        modulate1: modulation parameter1 [num_comp, num_data]
        modulate2: modulation parameter2 [num_comp, num_data]
    """

    print('Generating source...')

    if modulate_range1 is None:
        modulate_range1 = [-1, 1]
    if modulate_range2 is None:
        modulate_range2 = [-1, 1]

    # Initialize random generator
    np.random.seed(random_seed)

    # Generate innovations (not modulated)
    innov = np.random.normal(0, 1, [num_comp, num_data])
    t_basis = 2 * np.pi * np.arange(1, num_basis + 1).reshape([-1, 1]) * np.arange(0, num_data).reshape([1, -1]) / num_data
    fourier_basis = np.concatenate([np.sin(t_basis), np.cos(t_basis)], axis=0)

    if modulate1 is None:
        # generate modulation based on fourier bases
        modulate1 = np.random.uniform(-1, 1, [num_comp, num_basis * 2])
        modulate1 = np.dot(modulate1, fourier_basis)
        # normalize
        modulate1 = modulate1 - np.min(modulate1, axis=1, keepdims=True)
        modulate1 = modulate1 / np.max(modulate1, axis=1, keepdims=True) * (modulate_range1[1] - modulate_range1[0]) + modulate_range1[0]
        # apply exp (necessary for calculating log(lambda))
        modulate1 = np.exp(modulate1)

    if modulate2 is None:
        # generate modulation based on fourier bases
        modulate2 = np.random.uniform(-1, 1, [num_comp, num_basis * 2])
        modulate2 = np.dot(modulate2, fourier_basis)
        # normalize
        modulate2 = modulate2 - np.min(modulate2, axis=1, keepdims=True)
        modulate2 = modulate2 / np.max(modulate2, axis=1, keepdims=True) * (modulate_range2[1] - modulate_range2[0]) + modulate_range2[0]

    # make label
    label = np.arange(0, num_data)

    # Modulate innovations
    scale = 1 / np.sqrt(2 * modulate1)
    mean = - modulate2 / 2
    innov = innov * scale
    innov = innov + mean
    s = innov

    # Generate source signal
    x = np.zeros([num_comp, num_data])
    if cat_input:
        for i in range(1, num_data):
            xim1 = np.copy(x[:, i-1])
            sim1 = np.copy(s[:, i-1])
            x[:, i] = apply_mlp(np.concatenate([xim1, sim1], axis=0), mlplayer, negative_slope=negative_slope).reshape(-1)
            if x_limit is not None and np.max(x[:, i]) > x_limit:
                print('bad parameter')
                break
    else:
        for i in range(num_data):
            sim1 = np.copy(s[:, i])
            x[:, i] = apply_mlp(sim1, mlplayer, negative_slope=negative_slope).reshape(-1)
            if x_limit is not None and np.max(x[:, i]) > x_limit:
                print('bad parameter')
                break

    return x, s, label, modulate1, modulate2


# =============================================================
# =============================================================
def gen_mlp_parms(num_comp,
                  num_layer,
                  num_input=None,
                  cat_input=False,
                  iter4condthresh=10000,
                  cond_thresh_ratio=0.25,
                  layer_name_base='ip',
                  negative_slope=None,
                  last_scale=1,
                  random_seed=0):
    """Generate MLP and Apply it to source signal.
    Args:
        num_comp: number of components (number of nodes of hidden layers)
        num_layer: number of layers
        num_input: dimension of input (if not the same with the number of components)
        cat_input: concatenate x and s for input (i.e. AR model) or not
        iter4condthresh: (option) number of random iteration to decide the threshold of condition number of mixing matrices
        cond_thresh_ratio: (option) percentile of condition number to decide its threshold
        layer_name_base: (option) layer name
        negative_slope: (option) parameter of leaky-ReLU
        last_scale: scaling factor for the last output
        random_seed: (option) random seed
    Returns:
        mixlayer: parameters of mixing layers
    """

    print('Generating mlp parameters...')

    # Initialize random generator
    np.random.seed(random_seed)

    if num_input is None:
        num_input = num_comp

    def genA(num_in, num_out, in_double=False, nonlin=True):
        a = np.random.uniform(-1, 1, [num_out, num_in])
        if in_double:
            if nonlin:
                a = a * np.sqrt(3/((1 + negative_slope**2)*num_in))
            else:
                a = a * np.sqrt(3/(num_in*2))
        else:
            if nonlin:
                a = a * np.sqrt(6/((1 + negative_slope**2)*num_in))
            else:
                a = a * np.sqrt(6/(num_in*2))
        return a

    # Determine condThresh ------------------------------------
    condlist = np.zeros([iter4condthresh])
    for i in range(iter4condthresh):
        A = genA(num_comp, num_comp)
        condlist[i] = np.linalg.cond(A)

    condlist.sort()  # Ascending order
    condthresh = condlist[int(iter4condthresh * cond_thresh_ratio)]
    print('    cond thresh: %f' % condthresh)

    mixlayer = []
    for ln in range(num_layer):
        # A
        condA = condthresh + 1
        while condA > condthresh:
            if num_layer == 1:
                A = genA(num_input, num_comp, nonlin=False, in_double=cat_input)
            else:
                if ln == 0:  # 1st layer
                    A = genA(num_input, num_comp, nonlin=True, in_double=cat_input)
                elif ln == num_layer-1:  # last layer
                    A = genA(num_comp, num_comp, nonlin=False)
                else:
                    A = genA(num_comp, num_comp, nonlin=True)
            condA = np.linalg.cond(A)
            print('    L%d: cond=%f' % (ln, condA))
        # b
        if ln == 0:
            b = np.zeros([num_input]).reshape([1, -1]).T
        else:
            b = np.zeros([num_comp]).reshape([1, -1]).T

        if ln == 0 and cat_input:  # concatenate As for dynamical model
            condAs = condthresh + 1
            while condAs > condthresh:
                if num_layer == 1:
                    As = genA(num_input, num_comp, nonlin=False, in_double=cat_input)
                else:
                    As = genA(num_input, num_comp, nonlin=True, in_double=cat_input)
                condAs = np.linalg.cond(As)
                print('    L%ds: cond=%f' % (ln, condAs))
            #
            A = np.concatenate([A, As], axis=1)
            b = np.concatenate([b, b], axis=0)

        # totoal scaling
        if ln == num_layer - 1:
            A = A * last_scale

        # Storege ---------------------------------------------
        layername = layer_name_base + str(ln)
        mixlayer.append({"name": layername, "A": A.copy(), "b": b.copy()})

    return mixlayer


# =============================================================
# =============================================================
def apply_mlp(x,
              mlplayer,
              nonlinear_type='ReLU',
              negative_slope=None):
    """Generate MLP and Apply it to source signal.
    Args:
        x: input signals. 2D ndarray [num_comp, num_data]
        mlplayer: parameters of MLP generated by gen_mlp_parms
        nonlinear_type: (option) type of nonlinearity
        negative_slope: (option) parameter of leaky-ReLU
    Returns:
        y: mixed signals. 2D ndarray [num_comp, num_data]
    """

    num_layer = len(mlplayer)

    # Generate mixed signal -----------------------------------
    y = x.copy()

    if y.ndim == 1:
        y = np.reshape(y, [-1, 1])

    for ln in range(num_layer):

        # Apply bias and mixing matrix ------------------------
        y = y + mlplayer[ln]['b']
        y = np.dot(mlplayer[ln]['A'], y)

        # Apply nonlinearity ----------------------------------
        if ln != num_layer-1:  # No nolinearity for the last layer
            if nonlinear_type == "ReLU":  # Leaky-ReLU
                y[y < 0] = negative_slope * y[y < 0]
            else:
                raise ValueError

    return y

