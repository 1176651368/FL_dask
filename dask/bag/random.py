import heapq
import math
import random as rnd
from functools import partial
from itertools import islice

from dask.bag.core import Bag


def sample(population, k, split_every=None):
    """Chooses k unique random elements from a bag.

    Returns a new bag containing elements from the population while
    leaving the original population unchanged.

    Parameters
    ----------
    population: Bag
        Elements to sample.
    k: integer, optional
        Number of elements to sample.
    split_every: int (optional)
        Group partitions into groups of this size while performing reduction.
        Defaults to 8.

    Examples
    --------
    >>> import dask.bag as db
    >>> from dask.bag import random
    >>> b = db.from_sequence(range(5), npartitions=2)
    >>> list(random.sample(b, 3).compute())  # doctest: +SKIP
    [1, 3, 5]
    """
    res = _sample(population=population, k=k, split_every=split_every)
    return res.map_partitions(_finalize_sample, k)


def choices(population, k=1, split_every=None):
    """
    Return a k sized list of elements chosen with replacement.

    Parameters
    ----------
    population: Bag
        Elements to sample.
    k: integer, optional
        Number of elements to sample.
    split_every: int (optional)
        Group partitions into groups of this size while performing reduction.
        Defaults to 8.

    Examples
    --------
    >>> import dask.bag as db
    >>> from dask.bag import random
    >>> b = db.from_sequence(range(5), npartitions=2)
    >>> list(random.choices(b, 3).compute())  # doctest: +SKIP
    [1, 1, 5]
    """
    res = _sample_with_replacement(population=population, k=k, split_every=split_every)
    return res.map_partitions(_finalize_sample, k)


def _sample_reduce(reduce_iter, k, replace):
    """
    Reduce function used on the sample and choice functions.

    Parameters
    ----------
    reduce_iter : iterable
        Each element is a tuple coming generated by the _sample_map_partitions function.
    replace: bool
        If True, sample with replacement. If False, sample without replacement.

    Returns a sequence of uniformly distributed samples;
    """
    ns_ks = []
    s = []
    n = 0
    # unfolding reduce outputs
    for i in reduce_iter:
        (s_i, n_i) = i
        s.extend(s_i)
        n += n_i
        k_i = len(s_i)
        ns_ks.append((n_i, k_i))

    if k > n and not replace:
        return s, n

    # creating the probability array
    p = []
    for n_i, k_i in ns_ks:
        if k_i > 0:
            p_i = n_i / (k_i * n)
            p += [p_i] * k_i

    sample_func = rnd.choices if replace else _weighted_sampling_without_replacement
    return sample_func(population=s, weights=p, k=k), n


def _weighted_sampling_without_replacement(population, weights, k):
    """
    Source:
        Weighted random sampling with a reservoir, Pavlos S. Efraimidis, Paul G. Spirakis
    """
    elt = [(math.log(rnd.random()) / weights[i], i) for i in range(len(weights))]
    return [population[x[1]] for x in heapq.nlargest(k, elt)]


def _sample(population, k, split_every):
    if k < 0:
        raise ValueError("Cannot take a negative number of samples")
    return population.reduction(
        partial(_sample_map_partitions, k=k),
        partial(_sample_reduce, k=k, replace=False),
        out_type=Bag,
        split_every=split_every,
    )


def _finalize_sample(reduce_iter, k):
    sample = reduce_iter[0]
    if len(sample) < k:
        raise ValueError("Sample larger than population")
    return sample


def _sample_map_partitions(population, k):
    """
    Reservoir sampling strategy based on the L algorithm
    See https://en.wikipedia.org/wiki/Reservoir_sampling#An_optimal_algorithm
    """

    reservoir, stream_length = [], 0
    stream = iter(population)
    for e in islice(stream, k):
        reservoir.append(e)
        stream_length += 1

    w = math.exp(math.log(rnd.random()) / k)
    nxt = (k - 1) + _geometric(w)

    for i, e in enumerate(stream, k):
        if i == nxt:
            reservoir[rnd.randrange(k)] = e
            w *= math.exp(math.log(rnd.random()) / k)
            nxt += _geometric(w)
        stream_length += 1

    return reservoir, stream_length


def _sample_with_replacement(population, k, split_every):
    return population.reduction(
        partial(_sample_with_replacement_map_partitions, k=k),
        partial(_sample_reduce, k=k, replace=True),
        out_type=Bag,
        split_every=split_every,
    )


def _sample_with_replacement_map_partitions(population, k):
    """
    Reservoir sampling with replacement, the main idea is to use k reservoirs of size 1
    See Section Applications in http://utopia.duth.gr/~pefraimi/research/data/2007EncOfAlg.pdf
    """

    stream = iter(population)
    e = next(stream)
    reservoir, stream_length = [e for _ in range(k)], 1

    w = [rnd.random() for _ in range(k)]
    nxt = [_geometric(wi) for wi in w]
    min_nxt = min(nxt)

    for i, e in enumerate(stream, 1):
        if i == min_nxt:
            for j, n in enumerate(nxt):
                if n == min_nxt:
                    reservoir[j] = e
                    w[j] *= rnd.random()
                    nxt[j] += _geometric(w[j])
            min_nxt = min(nxt)

        stream_length += 1

    return reservoir, stream_length


def _geometric(p):
    return int(math.log(rnd.uniform(0, 1)) / math.log(1 - p)) + 1
