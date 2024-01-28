// Author: dWallet Labs, Ltd.
// SPDX-License-Identifier: BSD-3-Clause-Clear

use crypto_bigint::subtle::Choice;
use crypto_bigint::subtle::ConstantTimeLess;
use crypto_bigint::CheckedAdd;
use crypto_bigint::{rand_core::CryptoRngCore, CheckedMul, Uint};
use crypto_bigint::{NonZero, RandomMod};
use group::{
    GroupElement, KnownOrderGroupElement, KnownOrderScalar, Samplable,
    StatisticalSecuritySizedNumber,
};
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::ops::BitAnd;

/// An error in encryption related operations.
#[derive(thiserror::Error, Clone, Debug, PartialEq)]
pub enum Error {
    #[error("group error")]
    GroupInstantiation(#[from] group::Error),
    #[error("zero dimension: cannot evalute a zero-dimension linear combination")]
    ZeroDimension,
    #[error("an internal error that should never have happened and signifies a bug")]
    InternalError,
    #[error("the requested function cannot be securely evaluated")]
    SecureFunctionEvaluation,
}

/// The Result of the `new()` operation of types implementing the
/// [`AdditivelyHomomorphicEncryptionKey`] trait.
pub type Result<T> = std::result::Result<T, Error>;

/// An Encryption Key of an Additively Homomorphic Encryption scheme.
pub trait AdditivelyHomomorphicEncryptionKey<const PLAINTEXT_SPACE_SCALAR_LIMBS: usize>:
    PartialEq + Clone + Debug + Eq
{
    type PlaintextSpaceGroupElement: KnownOrderScalar<PLAINTEXT_SPACE_SCALAR_LIMBS> + Samplable;
    type RandomnessSpaceGroupElement: GroupElement + Samplable;
    type CiphertextSpaceGroupElement: GroupElement;

    /// The public parameters of the encryption scheme.
    ///
    /// Includes the public parameters of the plaintext, randomness and ciphertext groups.
    ///
    /// Used in [`Self::encrypt()`] to define the encryption algorithm.
    /// As such, it uniquely identifies the encryption-scheme (alongside the type `Self`) and will
    /// be used for Fiat-Shamir Transcripts).
    type PublicParameters: AsRef<
            GroupsPublicParameters<
                PlaintextSpacePublicParameters<PLAINTEXT_SPACE_SCALAR_LIMBS, Self>,
                RandomnessSpacePublicParameters<PLAINTEXT_SPACE_SCALAR_LIMBS, Self>,
                CiphertextSpacePublicParameters<PLAINTEXT_SPACE_SCALAR_LIMBS, Self>,
            >,
        > + Serialize
        + for<'r> Deserialize<'r>
        + PartialEq
        + Clone
        + Debug
        + Eq;

    /// Instantiate the encryption key from the public parameters of the encryption scheme.
    fn new(public_parameters: &Self::PublicParameters) -> Result<Self>;

    /// $\Enc(pk, \pt; \eta_{\sf enc}) \to \ct$: Encrypt `plaintext` to `self` using
    /// `randomness`.
    ///
    /// A deterministic algorithm that on input a public key $pk$, a plaintext $\pt \in \calP_{pk}$
    /// and randomness $\eta_{\sf enc} \in \calR_{pk}$, outputs a ciphertext $\ct \in \calC_{pk}$.
    fn encrypt_with_randomness(
        &self,
        plaintext: &Self::PlaintextSpaceGroupElement,
        randomness: &Self::RandomnessSpaceGroupElement,
        public_parameters: &Self::PublicParameters,
    ) -> Self::CiphertextSpaceGroupElement;

    /// $\Enc(pk, \pt)$: a probabilistic algorithm that first uniformly samples `randomness`
    /// $\eta_{\sf enc} \in \calR_{pk}$ from `rng` and then calls [`Self::
    /// encrypt_with_randomness()`] to encrypt `plaintext` to `self` using the sampled randomness.
    fn encrypt(
        &self,
        plaintext: &Self::PlaintextSpaceGroupElement,
        public_parameters: &Self::PublicParameters,
        rng: &mut impl CryptoRngCore,
    ) -> Result<(
        Self::RandomnessSpaceGroupElement,
        Self::CiphertextSpaceGroupElement,
    )> {
        let randomness = Self::RandomnessSpaceGroupElement::sample(
            &public_parameters.randomness_space_public_parameters(),
            rng,
        )?;

        let ciphertext = self.encrypt_with_randomness(plaintext, &randomness, public_parameters);

        Ok((randomness, ciphertext))
    }

    /// Efficient homomorphic evaluation of the linear
    /// combination defined by `coefficients` and `ciphertexts`.
    /// Returns $a_1 \odot \ct_1 \oplus \ldots \oplus a_\ell \odot \ct_\ell$.
    /// For an affine transformation, prepend ciphertexts with $\ct_0 = \Enc(1)$.
    ///
    /// SECURITY NOTE: This method *doesn't* assure circuit privacy.
    /// For circuit private implementation, use [`Self::securely_evaluate_linear_combination`].
    fn evaluate_linear_combination<const DIMENSION: usize>(
        coefficients: &[Self::PlaintextSpaceGroupElement; DIMENSION],
        ciphertexts: &[Self::CiphertextSpaceGroupElement; DIMENSION],
    ) -> Result<Self::CiphertextSpaceGroupElement> {
        if DIMENSION == 0 {
            return Err(Error::ZeroDimension);
        }

        let neutral = ciphertexts[0].neutral();

        Ok(coefficients.iter().zip(ciphertexts.iter()).fold(
            neutral,
            |curr, (coefficient, ciphertext)| {
                curr + ciphertext.scalar_mul(&coefficient.value().into())
            },
        ))
    }

    /// $\Eval(pk,f, \ct_1,\ldots,\ct_\ell; \omega, \eta)$: Secure function evaluation.
    ///
    /// This function securely computes an efficient homomorphic evaluation of the
    /// linear combination defined by `coefficients` and `ciphertexts`:
    /// $f(x_1,\ldots,x_\ell)=\sum_{i=1}^{\ell}{a_i x_i}$ where
    /// $a_i\in [0,q)$, and $\ell$ ciphertexts $\ct_1,\ldots,\ct_\ell$.
    ///
    /// For an affine transformation, prepend ciphertexts with $\ct_0 = \Enc(1)$.
    ///
    /// _Secure function evaluation_ states that giving the resulting ciphertext of the above evaluation
    /// to the decryption key owner for decryption does not reveal anything about $f$
    /// except what can be learned from the (decrypted) result alone.
    ///
    /// This is ensured by masking the linear combination with a random (`mask`)
    /// multiplication $\omega$ of the `modulus` $q$, and adding a fresh `randomness` $\eta$, which can be thought of as decrypting and re-encrypting with a fresh randomness:
    /// \ct = \Enc(pk, \omega q; \eta) \bigoplus_{i=1}^\ell \left(  a_i \odot \ct_i  \right)
    ///
    /// Let $\PT_i$ be the upper bound associated with $\ct_i$ (that is, this is the maximal value
    /// one obtains from decrypting $\ct_i$, but without reducing modulo $q$),
    /// where $\omega$ is uniformly chosen from $[0,2^s\PTsum)$ and $\eta$ is uniformly chosen from $\ZZ_N^*$.
    /// Then, the upper bound associated with the resulting $\ct$ is
    /// $$ \PT_{\sf eval} = (2^s+1)\cdot q\cdot \PTsum $$ and
    /// Correctness is assured as long as $\PT_{\sf eval}<N$.
    ///
    /// In more detail, these steps are taken to generically assure circuit privacy:
    /// 1. Re-randomization. This should be done by adding an encryption of zero with fresh
    ///    (uniformly sampled) randomness to the outputted ciphertext.
    ///
    /// 2. Masking. Our evaluation should be masked by a random multiplication of the homomorphic
    ///    evaluation group order $q$.
    ///
    ///    While the decryption modulo $q$ will remain correct,
    ///    assuming that the mask was "big enough", i.e. $\omega$ is uniformly chosen from $[0,2^s\PTsum)$,
    ///    the decryption will also be statistically indistinguishable from random.
    ///
    ///    *NOTE*: this function does not (and in fact, cannot) guarantee that
    ///    each of the given ciphertexts $\ct_i$ are in fact bounded by its corresponding upper bound $\PT_i$.
    ///    Instead, this responsibility is on the caller, which needs to assure that
    ///    by verifying appropriate zero-knowledge (and range) proofs.
    ///    An exception to the above is when the ciphertext was encrypted by the caller,
    ///    in which case the caller knows the corresponding plaintext.
    ///
    /// 3. No modulations. The size of our evaluation $\PT_{\sf eval}$ should be smaller than the order of
    ///    the encryption plaintext group $N$ in order to assure it does not go through modulation
    ///    in the plaintext space.
    ///
    /// In the case that the plaintext order is the same as the evaluation `modulus`, steps 2, 3 are
    /// skipped.
    ///
    /// See: Definition $2.1, B.2, B.3, D.1$ in "2PC-MPC: Threshold ECDSA in $\calO(1)$".
    fn securely_evaluate_linear_combination_with_randomness<const DIMENSION: usize>(
        &self,
        coefficients: &[Self::PlaintextSpaceGroupElement; DIMENSION],
        ciphertexts_and_upper_bounds: [(
            Self::CiphertextSpaceGroupElement,
            Uint<PLAINTEXT_SPACE_SCALAR_LIMBS>,
        ); DIMENSION],
        modulus: &Uint<PLAINTEXT_SPACE_SCALAR_LIMBS>,
        mask: &Self::PlaintextSpaceGroupElement,
        randomness: &Self::RandomnessSpaceGroupElement,
        public_parameters: &Self::PublicParameters,
    ) -> Result<Self::CiphertextSpaceGroupElement> {
        if DIMENSION == 0 {
            return Err(Error::ZeroDimension);
        }

        let plaintext_order: Uint<PLAINTEXT_SPACE_SCALAR_LIMBS> =
            Self::PlaintextSpaceGroupElement::order_from_public_parameters(
                public_parameters.plaintext_space_public_parameters(),
            );

        let ciphertexts = ciphertexts_and_upper_bounds
            .clone()
            .map(|(ciphertext, _)| ciphertext);

        let linear_combination = Self::evaluate_linear_combination(coefficients, &ciphertexts)?;

        // Re-randomization is performed in any case, and a masked multiplication of the modulus is
        // added only if the order of the plaintext space differs from `modulus`.
        let plaintext = if &plaintext_order == modulus {
            coefficients[0].neutral()
        } else {
            // First, verify that each coefficient $a_i$ is smaller then the modulus $q$.
            if !bool::from(
                coefficients
                    .iter()
                    .fold(Choice::from(1u8), |choice, coefficient| {
                        choice.bitand(coefficient.value().into().ct_lt(modulus))
                    }),
            ) {
                return Err(Error::SecureFunctionEvaluation);
            }

            // Finally, verify that the evaluation upper bound $\PT_{\sf eval}$ is smaller than the plaintext modulus $N$.
            let evaluation_upper_bound =
                Self::evaluation_upper_bound(&ciphertexts_and_upper_bounds, modulus)?;
            let secure_evaluation_upper_bound = Option::<Uint<PLAINTEXT_SPACE_SCALAR_LIMBS>>::from(
                evaluation_upper_bound.checked_add(&mask.value().into()),
            )
            .ok_or(Error::SecureFunctionEvaluation)?;

            if secure_evaluation_upper_bound >= plaintext_order {
                return Err(Error::SecureFunctionEvaluation);
            }
            let modulus = Self::PlaintextSpaceGroupElement::new(
                Uint::<PLAINTEXT_SPACE_SCALAR_LIMBS>::from(modulus).into(),
                &coefficients[0].public_parameters(),
            )?;

            modulus * mask
        };

        let encryption_with_fresh_randomness =
            self.encrypt_with_randomness(&plaintext, randomness, public_parameters);

        Ok(linear_combination + encryption_with_fresh_randomness)
    }

    /// $$ q\cdot \PTsum $$: Computes the evaluation upper bound of the linear combination of `ciphertexts`,
    /// each bounded by the corresponding `upper_bounds`, and coefficients, each bounded by `modulus`.
    ///
    /// Not to be confused with the secure evaluation upper bound $\PT_{\sf eval}$ to which a mask is added.
    fn evaluation_upper_bound<const DIMENSION: usize>(
        ciphertexts_and_upper_bounds: &[(
            Self::CiphertextSpaceGroupElement,
            Uint<PLAINTEXT_SPACE_SCALAR_LIMBS>,
        ); DIMENSION],
        modulus: &Uint<PLAINTEXT_SPACE_SCALAR_LIMBS>,
    ) -> Result<Uint<PLAINTEXT_SPACE_SCALAR_LIMBS>> {
        let upper_bounds_sum = ciphertexts_and_upper_bounds
            .iter()
            .map(|(_, upper_bound)| upper_bound)
            .fold(Some(Uint::ZERO), |sum, upper_bound| {
                sum.and_then(|sum| sum.checked_add(upper_bound).into())
            })
            .ok_or(Error::SecureFunctionEvaluation)?;

        Option::<Uint<PLAINTEXT_SPACE_SCALAR_LIMBS>>::from(upper_bounds_sum.checked_mul(&modulus))
            .ok_or(Error::SecureFunctionEvaluation)
    }

    /// Samples the mask $\omega$ is uniformly from $[0,2^s\PTsum)$, as required for secure function evaluation.
    fn sample_mask_for_secure_function_evaluation<const DIMENSION: usize>(
        ciphertexts_and_upper_bounds: &[(
            Self::CiphertextSpaceGroupElement,
            Uint<PLAINTEXT_SPACE_SCALAR_LIMBS>,
        ); DIMENSION],
        modulus: &Uint<PLAINTEXT_SPACE_SCALAR_LIMBS>,
        public_parameters: &Self::PublicParameters,
        rng: &mut impl CryptoRngCore,
    ) -> Result<Self::PlaintextSpaceGroupElement> {
        let evaluation_upper_bound =
            Self::evaluation_upper_bound(&ciphertexts_and_upper_bounds, modulus)?;

        let mask_upper_bound = evaluation_upper_bound.checked_mul(
            &(Uint::<PLAINTEXT_SPACE_SCALAR_LIMBS>::ONE << StatisticalSecuritySizedNumber::BITS),
        );

        let mask_upper_bound = Option::<NonZero<_>>::from(mask_upper_bound.and_then(NonZero::new))
            .ok_or(Error::SecureFunctionEvaluation)?;

        let mask = Uint::<PLAINTEXT_SPACE_SCALAR_LIMBS>::random_mod(rng, &mask_upper_bound);

        Ok(Self::PlaintextSpaceGroupElement::new(
            mask.into(),
            public_parameters.plaintext_space_public_parameters(),
        )?)
    }

    /// $\Eval(pk,f, \ct_1,\ldots,\ct_t; \eta_{\sf eval})$: Secure function evaluation.
    ///
    /// This is the probabilistic linear combination algorithm which samples `mask` and `randomness`
    /// from `rng` and calls [`Self::securely_evaluate_linear_combination_with_randomness()`].
    fn securely_evaluate_linear_combination<const DIMENSION: usize>(
        &self,
        coefficients: &[Self::PlaintextSpaceGroupElement; DIMENSION],
        ciphertexts_and_upper_bounds: [(
            Self::CiphertextSpaceGroupElement,
            Uint<PLAINTEXT_SPACE_SCALAR_LIMBS>,
        ); DIMENSION],
        modulus: &Uint<PLAINTEXT_SPACE_SCALAR_LIMBS>,
        public_parameters: &Self::PublicParameters,
        rng: &mut impl CryptoRngCore,
    ) -> Result<(
        Self::PlaintextSpaceGroupElement,
        Self::RandomnessSpaceGroupElement,
        Self::CiphertextSpaceGroupElement,
    )> {
        let randomness = Self::RandomnessSpaceGroupElement::sample(
            &public_parameters.randomness_space_public_parameters(),
            rng,
        )?;

        let mask = Self::sample_mask_for_secure_function_evaluation(
            &ciphertexts_and_upper_bounds,
            modulus,
            public_parameters,
            rng,
        )?;

        let evaluated_ciphertext = self.securely_evaluate_linear_combination_with_randomness(
            coefficients,
            ciphertexts_and_upper_bounds,
            modulus,
            &mask,
            &randomness,
            public_parameters,
        )?;

        Ok((mask, randomness, evaluated_ciphertext))
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Serialize, Deserialize)]
pub struct GroupsPublicParameters<
    PlaintextSpacePublicParameters,
    RandomnessSpacePublicParameters,
    CiphertextSpacePublicParameters,
> {
    pub plaintext_space_public_parameters: PlaintextSpacePublicParameters,
    pub randomness_space_public_parameters: RandomnessSpacePublicParameters,
    pub ciphertext_space_public_parameters: CiphertextSpacePublicParameters,
}

pub trait GroupsPublicParametersAccessors<
    'a,
    PlaintextSpacePublicParameters: 'a,
    RandomnessSpacePublicParameters: 'a,
    CiphertextSpacePublicParameters: 'a,
>:
    AsRef<
    GroupsPublicParameters<
        PlaintextSpacePublicParameters,
        RandomnessSpacePublicParameters,
        CiphertextSpacePublicParameters,
    >,
>
{
    fn plaintext_space_public_parameters(&'a self) -> &'a PlaintextSpacePublicParameters {
        &self.as_ref().plaintext_space_public_parameters
    }

    fn randomness_space_public_parameters(&'a self) -> &'a RandomnessSpacePublicParameters {
        &self.as_ref().randomness_space_public_parameters
    }

    fn ciphertext_space_public_parameters(&'a self) -> &'a CiphertextSpacePublicParameters {
        &self.as_ref().ciphertext_space_public_parameters
    }
}

impl<
        'a,
        PlaintextSpacePublicParameters: 'a,
        RandomnessSpacePublicParameters: 'a,
        CiphertextSpacePublicParameters: 'a,
        T: AsRef<
            GroupsPublicParameters<
                PlaintextSpacePublicParameters,
                RandomnessSpacePublicParameters,
                CiphertextSpacePublicParameters,
            >,
        >,
    >
    GroupsPublicParametersAccessors<
        'a,
        PlaintextSpacePublicParameters,
        RandomnessSpacePublicParameters,
        CiphertextSpacePublicParameters,
    > for T
{
}

impl<
        PlaintextSpacePublicParameters,
        RandomnessSpacePublicParameters,
        CiphertextSpacePublicParameters,
    > AsRef<Self>
    for GroupsPublicParameters<
        PlaintextSpacePublicParameters,
        RandomnessSpacePublicParameters,
        CiphertextSpacePublicParameters,
    >
{
    fn as_ref(&self) -> &Self {
        self
    }
}

pub type PlaintextSpaceGroupElement<const PLAINTEXT_SPACE_SCALAR_LIMBS: usize, E> =
<E as AdditivelyHomomorphicEncryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS>>::PlaintextSpaceGroupElement;
pub type PlaintextSpacePublicParameters<const PLAINTEXT_SPACE_SCALAR_LIMBS: usize, E> =
group::PublicParameters<<E as AdditivelyHomomorphicEncryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS>>::PlaintextSpaceGroupElement>;
pub type PlaintextSpaceValue<const PLAINTEXT_SPACE_SCALAR_LIMBS: usize, E> =
group::Value<<E as AdditivelyHomomorphicEncryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS>>::PlaintextSpaceGroupElement>;

pub type RandomnessSpaceGroupElement<const PLAINTEXT_SPACE_SCALAR_LIMBS: usize, E> =
<E as AdditivelyHomomorphicEncryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS>>::RandomnessSpaceGroupElement;
pub type RandomnessSpacePublicParameters<const PLAINTEXT_SPACE_SCALAR_LIMBS: usize, E> =
group::PublicParameters<<E as AdditivelyHomomorphicEncryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS>>::RandomnessSpaceGroupElement>;
pub type RandomnessSpaceValue<const PLAINTEXT_SPACE_SCALAR_LIMBS: usize, E> =
group::Value<<E as AdditivelyHomomorphicEncryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS>>::RandomnessSpaceGroupElement>;
pub type CiphertextSpaceGroupElement<const PLAINTEXT_SPACE_SCALAR_LIMBS: usize, E> =
<E as AdditivelyHomomorphicEncryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS>>::CiphertextSpaceGroupElement;
pub type CiphertextSpacePublicParameters<const PLAINTEXT_SPACE_SCALAR_LIMBS: usize, E> =
group::PublicParameters<<E as AdditivelyHomomorphicEncryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS>>::CiphertextSpaceGroupElement>;
pub type CiphertextSpaceValue<const PLAINTEXT_SPACE_SCALAR_LIMBS: usize, E> =
group::Value<<E as AdditivelyHomomorphicEncryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS>>::CiphertextSpaceGroupElement>;
pub type PublicParameters<const PLAINTEXT_SPACE_SCALAR_LIMBS: usize, E> =
    <E as AdditivelyHomomorphicEncryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS>>::PublicParameters;
