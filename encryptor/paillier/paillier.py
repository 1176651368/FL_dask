import datetime

import dask.array as da
from dask.array.core import Array
import numpy as np
from numpy import ndarray
from .util import getprimeover, mulmod, powmod, invert, isqrt
from .encoding import EncodedNumber

DEFAULT_KEYSIZE = 1024
SUPPORT_TYPE = (float, int, ndarray, Array)


def generate_paillier_keypair(n_length=DEFAULT_KEYSIZE):
    """Return a new :class:`PaillierPublicKey` and :class:`PaillierPrivateKey`.

    Add the private key to *private_keyring* if given.

    Args:
      private_keyring (PaillierPrivateKeyring): a
        :class:`PaillierPrivateKeyring` on which to store the private
        key.
      n_length: key size in bits.

    Returns:
      tuple: The generated :class:`PaillierPublicKey` and
      :class:`PaillierPrivateKey`
    """
    p = q = n = None
    n_len = 0
    while n_len != n_length:
        p = getprimeover(n_length // 2)
        q = p
        while q == p:
            q = getprimeover(n_length // 2)
        n = p * q
        n_len = n.bit_length()

    public_key = PaillierPublicKey(n)
    private_key = PaillierPrivateKey(public_key, p, q)

    return public_key, private_key


class PaillierPublicKey:

    def __init__(self, n):
        self.g = n + 1
        self.n = n
        self.nsquare = n * n
        self.max_int = n // 3 - 1

    def __repr__(self):
        publicKeyHash = hex(hash(self))[2:]
        return "<PaillierPublicKey {}>".format(publicKeyHash[:10])

    def __eq__(self, other):
        return self.n == other.n

    def __hash__(self):
        return hash(self.n)

    def get_nude(self, plaintext):
        if not isinstance(plaintext, int):
            raise TypeError('Expected int type plaintext but got: %s' %
                            type(plaintext))

        if self.n - self.max_int <= plaintext < self.n:
            # Very large plaintext, take a sneaky shortcut using inverses
            neg_plaintext = self.n - plaintext  # = abs(plaintext - nsquare)
            # avoid using gmpy2's mulmod when a * b < c
            neg_ciphertext = (self.n * neg_plaintext + 1) % self.nsquare
            # todo :may some problem
            nude_ciphertext = invert(neg_ciphertext, self.nsquare)
        else:
            # we chose g = n + 1, so that we can exploit the fact that
            # (n+1)^plaintext = n*plaintext + 1 mod n^2
            nude_ciphertext = (self.n * plaintext + 1) % self.nsquare
        return nude_ciphertext

    def raw_encrypt(self, plaintext, r_value=None):
        """Paillier encryption of a positive integer plaintext < :attr:`n`.

        You probably should be using :meth:`encrypt` instead, because it
        handles positive and negative ints and floats.

        Args:
          plaintext (int): a positive integer < :attr:`n` to be Paillier
            encrypted. Typically this is an encoding of the actual
            number you want to encrypt.
          r_value (int): obfuscator for the ciphertext; by default (i.e.
            r_value is None), a random value is used.

        Returns:
          int: Paillier encryption of plaintext.

        Raises:
          TypeError: if plaintext is not an int.
        """

        if isinstance(plaintext, Array):
            nude_ciphertext = da.frompyfunc(self.get_nude, 1, 1)(plaintext)
            r = r_value or self.get_random_lt_n()
            obfuscator = powmod(r, self.n, self.nsquare)
            # a * b % c
            return da.frompyfunc(mulmod, 3, 1)(nude_ciphertext, obfuscator, self.nsquare)
        elif isinstance(plaintext, (int, float)):
            nude_ciphertext = self.get_nude(plaintext)
            r = r_value or self.get_random_lt_n()
            obfuscator = powmod(r, self.n, self.nsquare)
            return mulmod(nude_ciphertext, obfuscator, self.nsquare)
        else:
            raise TypeError(f'todo : {type(plaintext)}')

    def get_random_lt_n(self):
        import random
        """Return a cryptographically random number less than :attr:`n`"""
        return random.SystemRandom().randrange(1, self.n)

    def encrypt(self, value, precision=None, r_value=None):
        """Paillier encrypt an encoded value.

        Args:
          encoding: The EncodedNumber instance.
          r_value (int): obfuscator for the ciphertext; by default (i.e.
            if *r_value* is None), a random value is used.

        Returns:
          EncryptedNumber: An encryption of *value*.
        """
        # If r_value is None, obfuscate in a call to .obfuscate() (below)
        st = datetime.datetime.now()

        if not isinstance(value, SUPPORT_TYPE):
            raise TypeError(f'not support type {type(value)}, we just support {SUPPORT_TYPE}')
        encoding = EncodedNumber.encode(self, value, precision)
        print(datetime.datetime.now() - st)
        ciphertext = self.raw_encrypt(encoding.encoding, r_value=1)
        print(datetime.datetime.now() - st)

        encrypted_number = EncryptedNumber(self, ciphertext, encoding.exponent)
        print(datetime.datetime.now() - st)

        if r_value is None:
            encrypted_number.obfuscate()
        print(datetime.datetime.now() - st)

        return encrypted_number


class EncryptedNumber():
    """Represents the Paillier encryption of a float or int.

    Typically, an `EncryptedNumber` is created by
    :meth:`PaillierPublicKey.encrypt`. You would only instantiate an
    `EncryptedNumber` manually if you are de-serializing a number
    someone else encrypted.


    Paillier encryption is only defined for non-negative integers less
    than :attr:`PaillierPublicKey.n`. :class:`EncodedNumber` provides
    an encoding scheme for floating point and signed integers that is
    compatible with the partially homomorphic properties of the Paillier
    cryptosystem:

    1. D(E(a) * E(b)) = a + b
    2. D(E(a)**b)     = a * b

    where `a` and `b` are ints or floats, `E` represents encoding then
    encryption, and `D` represents decryption then decoding.

    Args:
      public_key (PaillierPublicKey): the :class:`PaillierPublicKey`
        against which the number was encrypted.
      ciphertext (int): encrypted representation of the encoded number.
      exponent (int): used by :class:`EncodedNumber` to keep track of
        fixed precision. Usually negative.

    Attributes:
      public_key (PaillierPublicKey): the :class:`PaillierPublicKey`
        against which the number was encrypted.
      exponent (int): used by :class:`EncodedNumber` to keep track of
        fixed precision. Usually negative.

    Raises:
      TypeError: if *ciphertext* is not an int, or if *public_key* is
        not a :class:`PaillierPublicKey`.
    """

    def __init__(self, public_key, ciphertext, exponent=0):
        # super(EncryptedNumber, self).__init__()
        self.public_key = public_key
        self.ciphertext = ciphertext
        self.exponent = exponent
        self.__is_obfuscated = False

    def compute(self):
        self.exponent = self.exponent.compute()
        self.ciphertext = self.ciphertext.compute()
        return self

    def __add__(self, other):
        """Add an int, float, `EncryptedNumber` or `EncodedNumber`."""
        if isinstance(other, EncryptedNumber):
            return self._add_encrypted(other)
        elif isinstance(other, EncodedNumber):
            return self._add_encoded(other)
        else:
            return self._add_scalar(other)

    def __radd__(self, other):
        """Called when Python evaluates `34 + <EncryptedNumber>`
        Required for builtin `sum` to work.
        """
        return self.__add__(other)

    def __mul__(self, other):
        """Multiply by an int, float, or EncodedNumber."""
        if isinstance(other, EncryptedNumber):
            raise NotImplementedError('Good luck with that...')

        if isinstance(other, EncodedNumber):
            encoding = other
        else:
            encoding = EncodedNumber.encode(self.public_key, other)

        product = self._raw_mul(encoding.encoding)
        exponent = self.exponent + encoding.exponent

        return EncryptedNumber(self.public_key, product, exponent)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return other + (self * -1)

    def __truediv__(self, scalar):
        return self.__mul__(1 / scalar)

    def decrease_exponent_to(self, new_exp):
        """Return an EncryptedNumber with same value but lower exponent.

        If we multiply the encoded value by :attr:`EncodedNumber.BASE` and
        decrement :attr:`exponent`, then the decoded value does not change.
        Thus we can almost arbitrarily ratchet down the exponent of an
        `EncryptedNumber` - we only run into trouble when the encoded
        integer overflows. There may not be a warning if this happens.

        When adding `EncryptedNumber` instances, their exponents must
        match.

        This method is also useful for hiding information about the
        precision of numbers - e.g. a protocol can fix the exponent of
        all transmitted `EncryptedNumber` instances to some lower bound(s).

        Args:
          new_exp (int): the desired exponent.

        Returns:
          EncryptedNumber: Instance with the same plaintext and desired
            exponent.

        Raises:
          ValueError: You tried to increase the exponent.
        """
        if new_exp > self.exponent:
            raise ValueError('New exponent %i should be more negative than '
                             'old exponent %i' % (new_exp, self.exponent))
        multiplied = self * pow(EncodedNumber.BASE, self.exponent - new_exp)
        multiplied.exponent = new_exp

        return multiplied

    def obfuscate(self):
        """Disguise ciphertext by multiplying by r ** n with random r.

        This operation must be performed for every `EncryptedNumber`
        that is sent to an untrusted party, otherwise eavesdroppers
        might deduce relationships between this and an antecedent
        `EncryptedNumber`.

        For example::

            enc = public_key.encrypt(1337)
            send_to_nsa(enc)       # NSA can't decrypt (we hope!)
            product = enc * 3.14
            send_to_nsa(product)   # NSA can deduce 3.14 by bruteforce attack
            product2 = enc * 2.718
            product2.obfuscate()
            send_to_nsa(product)   # NSA can't deduce 2.718 by bruteforce attack
        """
        if isinstance(self.ciphertext, Array):
            r = self.public_key.get_random_lt_n()
            r_pow_n = powmod(r, self.public_key.n, self.public_key.nsquare)
            # self.ciphertext = da.mod(self.ciphertext * r_pow_n, self.public_key.nsquare)
            self.ciphertext = da.frompyfunc(mulmod,3,1)(self.ciphertext, r_pow_n, self.public_key.nsquare)

    def _add_scalar(self, scalar):
        """Returns E(a + b), given self=E(a) and b.

        Args:
          scalar: an int or float b, to be added to `self`.

        Returns:
          EncryptedNumber: E(a + b), calculated by encrypting b and
            taking the product of E(a) and E(b) modulo
            :attr:`~PaillierPublicKey.n` ** 2.

        Raises:
          ValueError: if scalar is out of range or precision.
        """
        encoded = EncodedNumber.encode(self.public_key, scalar,
                                       max_exponent=self.exponent)
        return self._add_encoded(encoded)

    def _add_encoded(self, encoded):
        """Returns E(a + b), given self=E(a) and b.

        Args:
          encoded (EncodedNumber): an :class:`EncodedNumber` to be added
            to `self`.

        Returns:
          EncryptedNumber: E(a + b), calculated by encrypting b and
            taking the product of E(a) and E(b) modulo
            :attr:`~PaillierPublicKey.n` ** 2.

        Raises:
          ValueError: if scalar is out of range or precision.
        """
        if self.public_key != encoded.public_key:
            raise ValueError("Attempted to add numbers encoded against "
                             "different public keys!")

        # In order to add two numbers, their exponents must match.
        a, b = self, encoded
        if a.exponent > b.exponent:
            a = self.decrease_exponent_to(b.exponent)
        elif a.exponent < b.exponent:
            b = b.decrease_exponent_to(a.exponent)

        # Don't bother to salt/obfuscate in a basic operation, do it
        # just before leaving the computer.
        encrypted_scalar = a.public_key.raw_encrypt(b.encoding, 1)
        sum_ciphertext = a._raw_add(a.ciphertext, encrypted_scalar)

        return EncryptedNumber(a.public_key, sum_ciphertext, a.exponent)

    def _add_encrypted(self, other):
        """Returns E(a + b) given E(a) and E(b).

        Args:
          other (EncryptedNumber): an `EncryptedNumber` to add to self.

        Returns:
          EncryptedNumber: E(a + b), calculated by taking the product
            of E(a) and E(b) modulo :attr:`~PaillierPublicKey.n` ** 2.

        Raises:
          ValueError: if numbers were encrypted against different keys.
        """
        if self.public_key != other.public_key:
            raise ValueError("Attempted to add numbers encrypted against "
                             "different public keys!")

        # In order to add two numbers, their exponents must match.
        a, b = self, other
        if a.exponent > b.exponent:
            a = self.decrease_exponent_to(b.exponent)

        elif a.exponent < b.exponent:
            b = b.decrease_exponent_to(a.exponent)
        sum_ciphertext = a._raw_add(a.ciphertext, b.ciphertext)
        return EncryptedNumber(a.public_key, sum_ciphertext, a.exponent)

    def _raw_add(self, e_a, e_b):
        """Returns the integer E(a + b) given ints E(a) and E(b).

        N.B. this returns an int, not an `EncryptedNumber`, and ignores
        :attr:`ciphertext`

        Args:
          e_a (int): E(a), first term
          e_b (int): E(b), second term

        Returns:
          int: E(a + b), calculated by taking the product of E(a) and
            E(b) modulo :attr:`~PaillierPublicKey.n` ** 2.
        """
        return mulmod(e_a, e_b, self.public_key.nsquare)

    def _raw_mul(self, plaintext):
        """Returns the integer E(a * plaintext), where E(a) = ciphertext

        Args:
          plaintext (int): number by which to multiply the
            `EncryptedNumber`. *plaintext* is typically an encoding.
            0 <= *plaintext* < :attr:`~PaillierPublicKey.n`

        Returns:
          int: Encryption of the product of `self` and the scalar
            encoded in *plaintext*.

        Raises:
          TypeError: if *plaintext* is not an int.
          ValueError: if *plaintext* is not between 0 and
            :attr:`PaillierPublicKey.n`.
        """
        if isinstance(plaintext, int):
            if plaintext < 0 or plaintext >= self.public_key.n:
                raise ValueError('Scalar out of bounds: %i' % plaintext)

            if self.public_key.n - self.public_key.max_int <= plaintext:
                # Very large plaintext, play a sneaky trick using inverses
                neg_c = invert(self.ciphertext, self.public_key.nsquare)
                neg_scalar = self.public_key.n - plaintext
                return powmod(neg_c, neg_scalar, self.public_key.nsquare)
            else:
                return powmod(self.ciphertext, plaintext, self.public_key.nsquare)
        else:
            if da.any(plaintext >= self.public_key.n - self.public_key.max_int):
                # Very large plaintext, play a sneaky trick using inverses
                neg_c = invert(self.ciphertext, self.public_key.nsquare)
                neg_scalar = self.public_key.n - plaintext
                return powmod(neg_c, neg_scalar, self.public_key.nsquare)
            else:
                return da.frompyfunc(powmod, 3, 1)(self.ciphertext, plaintext, self.public_key.nsquare)


class PaillierPrivateKey(object):
    def __init__(self, public_key, p, q):
        if not p * q == public_key.n:
            raise ValueError('given public key does not match the given p and q.')
        if p == q:
            # check that p and q are different, otherwise we can't compute p^-1 mod q
            raise ValueError('p and q have to be different')
        self.public_key = public_key
        if q < p:  # ensure that p < q.
            self.p = q
            self.q = p
        else:
            self.p = p
            self.q = q
        self.psquare = self.p * self.p

        self.qsquare = self.q * self.q
        self.p_inverse = invert(self.p, self.q)
        self.hp = self.h_function(self.p, self.psquare)
        self.hq = self.h_function(self.q, self.qsquare)

    @staticmethod
    def from_totient(public_key, totient):
        """given the totient, one can factorize the modulus

        The totient is defined as totient = (p - 1) * (q - 1),
        and the modulus is defined as modulus = p * q

        Args:
          public_key (PaillierPublicKey): The corresponding public
            key
          totient (int): the totient of the modulus

        Returns:
          the :class:`PaillierPrivateKey` that corresponds to the inputs

        Raises:
          ValueError: if the given totient is not the totient of the modulus
            of the given public key
        """
        p_plus_q = public_key.n - totient + 1
        p_minus_q = isqrt(p_plus_q * p_plus_q - public_key.n * 4)
        q = (p_plus_q - p_minus_q) // 2
        p = p_plus_q - q
        if not p * q == public_key.n:
            raise ValueError('given public key and totient do not match.')
        return PaillierPrivateKey(public_key, p, q)

    def __repr__(self):
        pub_repr = repr(self.public_key)
        return "<PaillierPrivateKey for {}>".format(pub_repr)

    def decrypt(self, encrypted_number, Encoding=None):
        """Return the :class:`EncodedNumber` decrypted from *encrypted_number*.

        Args:
          encrypted_number (EncryptedNumber): an
            :class:`EncryptedNumber` with a public key that matches this
            private key.
          Encoding (class): A class to use instead of :class:`EncodedNumber`, the
            encoding used for the *encrypted_number* - used to support alternative
            encodings.

        Returns:
          :class:`EncodedNumber`: The decrypted plaintext.

        Raises:
          TypeError: If *encrypted_number* is not an
            :class:`EncryptedNumber`.
          ValueError: If *encrypted_number* was encrypted against a
            different key.
        """
        if isinstance(encrypted_number, EncryptedNumber) or \
                isinstance(encrypted_number, Array) or isinstance(encrypted_number, ndarray):
            pass
        else:
            raise TypeError('Expected encrypted_number to be an EncryptedNumber'
                            ' not: %s' % type(encrypted_number))

        # if self.public_key != encrypted_number.public_key:
        #     raise ValueError('encrypted_number was encrypted against a '
        #                      'different key!')
        def t(x):
             try:
                 return x.exponent
             except:
                 return None

        if Encoding is None:
            Encoding = EncodedNumber
        encoded = self.raw_decrypt(encrypted_number)
        new_exponent = da.frompyfunc(lambda x:t(x),1,1)(encrypted_number)
        result = Encoding(self.public_key, encoded,new_exponent).decode()
        return result

    def raw_decrypt(self, ciphertext):
        """Decrypt raw ciphertext and return raw plaintext.

        Args:
          ciphertext (int): (usually from :meth:`EncryptedNumber.ciphertext()`)
            that is to be Paillier decrypted.

        Returns:
          int: Paillier decryption of ciphertext. This is a positive
          integer < :attr:`public_key.n`.

        Raises:
          TypeError: if ciphertext is not an int.
        """

        def t(x,y,z):
             try:
                 return powmod(x.ciphertext, y, z)
             except:
                 return None

        decrypt_to_p = mulmod(
            self.l_function(
                da.frompyfunc(lambda x,y,z:t(x,y,z),3,1)(ciphertext,self.p - 1,self.psquare),
                self.p),
            self.hp,
            self.p)

        decrypt_to_q = mulmod(
            self.l_function(
                da.frompyfunc(lambda x,y,z:t(x,y,z),3,1)(ciphertext, self.q - 1, self.qsquare),
                self.q),
            self.hq,
            self.q)
        return self.crt(decrypt_to_p, decrypt_to_q)

    def h_function(self, x, xsquare):
        """Computes the h-function as defined in Paillier's paper page 12,
        'Decryption using Chinese-remaindering'.
        """
        return invert(self.l_function(powmod(self.public_key.g, x - 1, xsquare), x), x)

    def l_function(self, x, p):
        """Computes the L function as defined in Paillier's paper. That is: L(x,p) = (x-1)/p"""
        return (x - 1) // p

    def crt(self, mp, mq):
        """The Chinese Remainder Theorem as needed for decryption. Returns the solution modulo n=pq.

        Args:
           mp(int): the solution modulo p.
           mq(int): the solution modulo q.
       """
        u = mulmod(mq - mp, self.p_inverse, self.q)
        return mp + (u * self.p)

    def __eq__(self, other):
        return self.p == other.p and self.q == other.q

    def __hash__(self):
        return hash((self.p, self.q))


def toArray(data):
    public_key, ciphertext, exponent = data.public_key, data.ciphertext, data.exponent
    r = da.frompyfunc(EncryptedNumber, 3, 1)(public_key, ciphertext, exponent)
    r.public_key, r.ciphertext, r.exponent = public_key, ciphertext, exponent
    return r


