import fractions
import math
import sys
from dask.array.core import Array
import dask.array as da
import numpy as np


class EncodedNumber(object):
    BASE = 16
    """Base to use when exponentiating. Larger `BASE` means
    that :attr:`exponent` leaks less information. If you vary this,
    you'll have to manually inform anyone decoding your numbers.
    """
    LOG2_BASE = math.log(BASE, 2)
    FLOAT_MANTISSA_BITS = sys.float_info.mant_dig

    def __init__(self, public_key, encoding, exponent):
        self.public_key = public_key
        self.encoding = encoding
        self.exponent = exponent

    @classmethod
    def encode(cls, public_key, scalar, precision=None, max_exponent=None):
        """Return an encoding of an int or float.

        This encoding is carefully chosen so that it supports the same
        operations as the Paillier cryptosystem.

        If *scalar* is a float, first approximate it as an int, `int_rep`:

            scalar = int_rep * (:attr:`BASE` ** :attr:`exponent`),

        for some (typically negative) integer exponent, which can be
        tuned using *precision* and *max_exponent*. Specifically,
        :attr:`exponent` is chosen to be equal to or less than
        *max_exponent*, and such that the number *precision* is not
        rounded to zero.

        Having found an integer representation for the float (or having
        been given an int `scalar`), we then represent this integer as
        a non-negative integer < :attr:`~PaillierPublicKey.n`.

        Paillier homomorphic arithemetic works modulo
        :attr:`~PaillierPublicKey.n`. We take the convention that a
        number x < n/3 is positive, and that a number x > 2n/3 is
        negative. The range n/3 < x < 2n/3 allows for overflow
        detection.

        Args:
          public_key (PaillierPublicKey): public key for which to encode
            (this is necessary because :attr:`~PaillierPublicKey.n`
            varies).
          scalar: an int or float to be encrypted.
            If int, it must satisfy abs(*value*) <
            :attr:`~PaillierPublicKey.n`/3.
            If float, it must satisfy abs(*value* / *precision*) <<
            :attr:`~PaillierPublicKey.n`/3
            (i.e. if a float is near the limit then detectable
            overflow may still occur)
          precision (float): Choose exponent (i.e. fix the precision) so
            that this number is distinguishable from zero. If `scalar`
            is a float, then this is set so that minimal precision is
            lost. Lower precision leads to smaller encodings, which
            might yield faster computation.
          max_exponent (int): Ensure that the exponent of the returned
            `EncryptedNumber` is at most this.

        Returns:
          EncodedNumber: Encoded form of *scalar*, ready for encryption
          against *public_key*.
        """
        # Calculate the maximum exponent for desired precision
        if precision is not None:
            if isinstance(scalar,(int,float)):
                prec_exponent = math.floor(math.log(precision, cls.BASE))
            elif isinstance(scalar,Array):
                prec_exponent = da.floor(da.log(precision, cls.BASE))
            elif isinstance(scalar,np.ndarray):
                prec_exponent = np.floor(np.log(precision, cls.BASE))
            else:
                raise ValueError()

        else:
            if isinstance(scalar,int):
                prec_exponent = 0

            if isinstance(scalar,float):
                bin_flt_exponent = math.frexp(scalar)[1]
                bin_lsb_exponent = bin_flt_exponent - cls.FLOAT_MANTISSA_BITS
                prec_exponent = math.floor(bin_lsb_exponent / cls.LOG2_BASE)

            if isinstance(scalar,Array):
                if np.issubdtype(scalar.dtype, np.int16) or np.issubdtype(scalar.dtype, np.int32) \
                        or np.issubdtype(scalar.dtype, np.int64):
                    prec_exponent = da.zeros(shape=scalar.shape)
                elif np.issubdtype(scalar.dtype, np.float16) or np.issubdtype(scalar.dtype, np.float32) \
                        or np.issubdtype(scalar.dtype, np.float64):
                    bin_flt_exponent = da.frexp(scalar)[1]
                    bin_lsb_exponent = bin_flt_exponent - cls.FLOAT_MANTISSA_BITS
                    prec_exponent = da.floor(bin_lsb_exponent / cls.LOG2_BASE)
                else:
                    scalar = scalar.astype(float)
                    bin_flt_exponent = da.frexp(scalar)[1]
                    bin_lsb_exponent = bin_flt_exponent - cls.FLOAT_MANTISSA_BITS
                    prec_exponent = da.floor(bin_lsb_exponent / cls.LOG2_BASE)

            if isinstance(scalar,np.ndarray):
                if np.issubdtype(scalar.dtype, np.int16) or np.issubdtype(scalar.dtype, np.int32) \
                        or np.issubdtype(scalar.dtype, np.int64):
                    prec_exponent = np.zeros(shape=scalar.shape)
                elif np.issubdtype(scalar.dtype, np.float16) or np.issubdtype(scalar.dtype, np.float32) \
                        or np.issubdtype(scalar.dtype, np.float64):
                    bin_flt_exponent = np.frexp(scalar)[1]
                    bin_lsb_exponent = bin_flt_exponent - cls.FLOAT_MANTISSA_BITS
                    prec_exponent = np.floor(bin_lsb_exponent / cls.LOG2_BASE)
                else:
                    raise ValueError()

        if max_exponent is None:
            exponent = prec_exponent
        else:
            if isinstance(scalar,(float,int)):
                exponent = min(max_exponent, prec_exponent)
            elif isinstance(scalar,np.ndarray):
                exponent = np.minimum(max_exponent, prec_exponent)
            elif isinstance(scalar,Array):
                exponent = da.minimum(max_exponent, prec_exponent)
            else:
                raise TypeError()

        if isinstance(scalar, (float, int)):
            int_rep = round(fractions.Fraction(scalar)
                            * fractions.Fraction(cls.BASE) ** -exponent)
            return cls(public_key, int_rep % public_key.n, exponent)

        elif isinstance(scalar, np.ndarray):
            # todo not this type
            int_rep = np.frompyfunc(lambda x, y, z: int(x * pow(y, -z)),
                                    3, 1)(scalar, cls.BASE, exponent)
            exponent = exponent.astype(np.int)
            return cls(public_key, int_rep % public_key.n, exponent)

        elif isinstance(scalar, Array):
            # exponent = da.frompyfunc(int,1,1)(exponent)
            exponent = exponent.astype(dtype=int)
            int_rep = (scalar * pow(cls.BASE, -exponent)).astype(dtype=int)
            new_int_rep = (int_rep % public_key.n)
            new_int_rep = da.frompyfunc(int,1,1)(new_int_rep)
            return cls(public_key, new_int_rep, exponent)

    def decode(self):
        return da.frompyfunc(self.decode_one,3,1)(self.encoding,self.exponent,self.public_key)

    def decode_one(self,encoding,exponent,public_key):
        """Decode plaintext and return the result.

        Returns:
          an int or float: the decoded number. N.B. if the number
            returned is an integer, it will not be of type float.

        Raises:
          OverflowError: if overflow is detected in the decrypted number.
        """
        if encoding >= public_key.n:
            # Should be mod n
            raise ValueError('Attempted to decode corrupted number')
        elif encoding <= public_key.max_int:
            # Positive
            mantissa = encoding
        elif encoding >= public_key.n - public_key.max_int:
            # Negative
            mantissa = encoding - public_key.n
        else:
            raise OverflowError('Overflow detected in decrypted number')

        if exponent >= 0:
            # Integer multiplication. This is exact.
            return mantissa * self.BASE ** exponent
        else:
            # BASE ** -e is an integer, so below is a division of ints.
            # Not coercing mantissa to float prevents some overflows.
            try:
                return mantissa / self.BASE ** -exponent
            except OverflowError as e:
                raise OverflowError(
                    'decoded result too large for a float') from e

    def decrease_exponent_to(self, new_exp):
        """Return an `EncodedNumber` with same value but lower exponent.

        If we multiply the encoded value by :attr:`BASE` and decrement
        :attr:`exponent`, then the decoded value does not change. Thus
        we can almost arbitrarily ratchet down the exponent of an
        :class:`EncodedNumber` - we only run into trouble when the encoded
        integer overflows. There may not be a warning if this happens.

        This is necessary when adding :class:`EncodedNumber` instances,
        and can also be useful to hide information about the precision
        of numbers - e.g. a protocol can fix the exponent of all
        transmitted :class:`EncodedNumber` to some lower bound(s).

        Args:
          new_exp (int): the desired exponent.

        Returns:
          EncodedNumber: Instance with the same value and desired
            exponent.

        Raises:
          ValueError: You tried to increase the exponent, which can't be
            done without decryption.
        """
        if new_exp > self.exponent:
            raise ValueError('New exponent %i should be more negative than'
                             'old exponent %i' % (new_exp, self.exponent))
        factor = pow(self.BASE, self.exponent - new_exp)
        new_enc = self.encoding * factor % self.public_key.n
        return self.__class__(self.public_key, new_enc, new_exp)
