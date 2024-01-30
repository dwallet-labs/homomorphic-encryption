use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::BitAnd;

use crypto_bigint::subtle::{Choice, CtOption};
use crypto_bigint::CheckedAdd;
use crypto_bigint::{rand_core::CryptoRngCore, CheckedMul, Uint};
use crypto_bigint::{NonZero, RandomMod};
// Author: dWallet Labs, Ltd.
// SPDX-License-Identifier: BSD-3-Clause-Clear
use crypto_bigint::subtle::ConstantTimeLess;
use group::{
    GroupElement, KnownOrderGroupElement, KnownOrderScalar, PartyID, Samplable,
    StatisticalSecuritySizedNumber,
};
use serde::{Deserialize, Serialize};

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
            // Verify that the secure evaluation upper bound $\PT_{\sf eval}$ is smaller than the plaintext modulus $N$.
            // This is done first by multiplying each of the coefficients by the corresponding upper bound:
            let evaluation_upper_bound: Uint<PLAINTEXT_SPACE_SCALAR_LIMBS> =
                ciphertexts_and_upper_bounds
                    .iter()
                    .map(|(_, upper_bound)| upper_bound)
                    .zip(coefficients.iter())
                    .map(|(upper_bound, coefficient)| {
                        coefficient.value().into().checked_mul(upper_bound)
                    })
                    .reduce(|a, b| a.and_then(|a| b.and_then(|b| a.checked_add(&b))))
                    .and_then(|evaluation_upper_bound| evaluation_upper_bound.into())
                    .ok_or(Error::SecureFunctionEvaluation)?;

            // And then adding the mask by modulus $ \omega q $, to result with the secure evaluation upper bound $\PT_{\sf eval}$:
            let secure_evaluation_upper_bound = Option::<Uint<PLAINTEXT_SPACE_SCALAR_LIMBS>>::from(
                mask.value()
                    .into()
                    .checked_mul(modulus)
                    .and_then(|mask_by_modulus| {
                        evaluation_upper_bound.checked_add(&mask_by_modulus)
                    }),
            )
            .ok_or(Error::SecureFunctionEvaluation)?;

            // And finally checking that it is smaller than the plaintext order $ $\PT_{\sf eval}$ < N $:
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

    /// Samples the mask $\omega$ is uniformly from $[0,2^s\PTsum)$, as required for secure function evaluation.
    fn sample_mask_for_secure_function_evaluation<const DIMENSION: usize>(
        ciphertexts_and_upper_bounds: &[(
            Self::CiphertextSpaceGroupElement,
            Uint<PLAINTEXT_SPACE_SCALAR_LIMBS>,
        ); DIMENSION],
        public_parameters: &Self::PublicParameters,
        rng: &mut impl CryptoRngCore,
    ) -> Result<Self::PlaintextSpaceGroupElement> {
        let upper_bounds_sum = ciphertexts_and_upper_bounds
            .iter()
            .map(|(_, upper_bound)| upper_bound)
            .fold(Some(Uint::ZERO), |sum, upper_bound| {
                sum.and_then(|sum| sum.checked_add(upper_bound).into())
            })
            .ok_or(Error::SecureFunctionEvaluation)?;

        let mask_upper_bound = upper_bounds_sum.checked_mul(
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

        // Then sample the mask uniformly from $[0,2^s\PTsum)$.
        let mask = Self::sample_mask_for_secure_function_evaluation(
            &ciphertexts_and_upper_bounds,
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

/// A Decryption Key of an Additively Homomorphic Encryption scheme.
pub trait AdditivelyHomomorphicDecryptionKey<
    const PLAINTEXT_SPACE_SCALAR_LIMBS: usize,
    EncryptionKey: AdditivelyHomomorphicEncryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS>,
>: AsRef<EncryptionKey> + Clone + PartialEq
{
    /// The decryption key used for decryption.
    type SecretKey;

    /// Instantiate the decryption key from the public parameters of the encryption scheme,
    /// and the secret key.
    fn new(
        secret_key: Self::SecretKey,
        public_parameters: &EncryptionKey::PublicParameters,
    ) -> Result<Self>;

    /// $\Dec(sk, \ct) \to \pt$: Decrypt `ciphertext` using `decryption_key`.
    /// A deterministic algorithm that on input a secret key $sk$ and a ciphertext $\ct \in
    /// \calC_{pk}$ outputs a plaintext $\pt \in \calP_{pk}$.
    ///
    /// SECURITY NOTE: in some decryption schemes, like RLWE-based schemes, decryption can fail, and this could in turn leak secret data if not handled carefully.
    /// In this case, this function must execute in constant time. However, that isn't sufficient; the caller must also handle the results in constant time.
    /// One way is by verifying zero-knowledge proofs before decrypting, so you only decrypt when you know you've succeeded.
    /// Another is the classic way of handling `CtOption`, which is to perform some computation over garbage (e.g. `Default`) values if `.is_none()`.
    /// An example for this is RLWE-based key-exchange protocols, where you decrypt and if you fail you perform the computation over a garbage value and send it anyway.
    fn decrypt(
        &self,
        ciphertext: &EncryptionKey::CiphertextSpaceGroupElement,
        public_parameters: &EncryptionKey::PublicParameters,
    ) -> CtOption<EncryptionKey::PlaintextSpaceGroupElement>;
}

/// A Decryption Key Share of a Threshold Additively Homomorphic Encryption scheme
pub trait AdditivelyHomomorphicDecryptionKeyShare<
    const PLAINTEXT_SPACE_SCALAR_LIMBS: usize,
    EncryptionKey: AdditivelyHomomorphicEncryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS>,
>: AsRef<EncryptionKey> + Clone + PartialEq
{
    /// The decryption key share used for decryption.
    type SecretKeyShare;
    /// A decryption share of a ciphertext in the process of Threshold Decryption.
    type DecryptionShare: Clone + Debug + PartialEq + Eq;
    /// A proof that a decryption share was correctly computed on a ciphertext using the decryption
    /// key share `Self`.
    type PartialDecryptionProof: Clone + Debug + PartialEq + Eq;
    /// A lagrange coefficient used for Threshold Decryption.
    /// These values are passed to the `Self::combine_decryption_shares` methods
    /// separately from `Self::PublicParameters` as they depend on the decrypter set.
    type LagrangeCoefficient: Clone + Debug + PartialEq + Eq;
    /// The public parameters of the threshold decryption scheme.
    type PublicParameters: AsRef<EncryptionKey::PublicParameters>
        + Serialize
        + for<'r> Deserialize<'r>
        + PartialEq
        + Clone
        + Debug
        + Eq;

    /// An error in threshold decryption.
    type Error: Debug;

    /// Instantiate the decryption key share from the public parameters of the threshold decryption scheme,
    /// and the secret key share.
    fn new(
        party_id: PartyID,
        secret_key_share: Self::SecretKeyShare,
        public_parameters: &Self::PublicParameters,
    ) -> std::result::Result<Self, Self::Error>;

    /// The Semi-honest variant of Partial Decryption, returns the decryption share without proving
    /// correctness.
    ///
    /// SECURITY NOTE: see corresponding note in [`AdditivelyHomomorphicDecryptionKey::decrypt`]; the same applies here.
    fn generate_decryption_share_semi_honest(
        &self,
        ciphertext: &EncryptionKey::CiphertextSpaceGroupElement,
        public_parameters: &Self::PublicParameters,
    ) -> CtOption<Self::DecryptionShare>;

    /// Performs the Maliciously-secure Partial Decryption in which decryption shares are computed
    /// and proven correct.
    ///
    /// SECURITY NOTE: see corresponding note in [`AdditivelyHomomorphicDecryptionKey::decrypt`]; the same applies here.
    fn generate_decryption_shares(
        &self,
        ciphertexts: Vec<EncryptionKey::CiphertextSpaceGroupElement>,
        public_parameters: &Self::PublicParameters,
        rng: &mut impl CryptoRngCore,
    ) -> CtOption<(Vec<Self::DecryptionShare>, Self::PartialDecryptionProof)>;

    /// Compute the lagrange coefficient of party `party_id`.
    /// Used for threshold decryption, where the lagrange coefficients of the current decrypters set is required.
    ///
    /// Since the number of subsets of size `threshold` of the parties set whose size is `number_of_parties` grow super-exponentially,
    /// these values cannot be computed ahead of time and included in `Self::PublicParameters`.
    ///
    /// Instead, they should be lazily computed according to the participating parties in a given threshold decryption session.
    /// That being said, these can be cached so if there is a default set of decrypters, or one that decrypts more than once,
    /// these values can be computed only once for that set.
    fn compute_lagrange_coefficient(
        party_id: PartyID,
        number_of_parties: PartyID,
        decrypters: Vec<PartyID>,
        public_parameters: &Self::PublicParameters,
    ) -> Self::LagrangeCoefficient;

    /// Finalizes the Threshold Decryption protocol by combining decryption shares. This is the
    /// Semi-Honest variant in which no proofs are verified.
    ///
    /// Correct decryption isn't assured upon success,
    /// and one should be able to verify the output independently or trust the process was done correctly.
    fn combine_decryption_shares_semi_honest(
        decryption_shares: HashMap<PartyID, Self::DecryptionShare>,
        lagrange_coefficients: HashMap<PartyID, Self::LagrangeCoefficient>,
        public_parameters: &Self::PublicParameters,
    ) -> std::result::Result<EncryptionKey::PlaintextSpaceGroupElement, Self::Error>;

    /// Finalizes the Threshold Decryption protocol by combining decryption shares. This is the
    /// Maliciously-secure variant in which the corresponding zero-knowledge proofs are verified,
    /// and correct decryption is assured upon success.
    fn combine_decryption_shares(
        ciphertexts: Vec<EncryptionKey::CiphertextSpaceGroupElement>,
        decryption_shares_and_proofs: HashMap<
            PartyID,
            (Vec<Self::DecryptionShare>, Self::PartialDecryptionProof),
        >,
        lagrange_coefficients: HashMap<PartyID, Self::LagrangeCoefficient>,
        public_parameters: &Self::PublicParameters,
        rng: &mut impl CryptoRngCore,
    ) -> std::result::Result<Vec<EncryptionKey::PlaintextSpaceGroupElement>, Self::Error>;
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

#[allow(clippy::erasing_op)]
#[allow(clippy::identity_op)]
pub mod tests {
    use crypto_bigint::{Uint, U64};
    use group::{GroupElement, Value};

    use super::*;

    pub fn encrypt_decrypts<
        const PLAINTEXT_SPACE_SCALAR_LIMBS: usize,
        EncryptionKey: AdditivelyHomomorphicEncryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS>,
        DecryptionKey: AdditivelyHomomorphicDecryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS, EncryptionKey>,
    >(
        decryption_key: DecryptionKey,
        public_parameters: &EncryptionKey::PublicParameters,
        rng: &mut impl CryptoRngCore,
    ) {
        let encryption_key = decryption_key.as_ref();

        let plaintext: Uint<PLAINTEXT_SPACE_SCALAR_LIMBS> = (&U64::from(42u64)).into();
        let plaintext: EncryptionKey::PlaintextSpaceGroupElement =
            EncryptionKey::PlaintextSpaceGroupElement::new(
                plaintext.into(),
                public_parameters.plaintext_space_public_parameters(),
            )
            .unwrap();

        let (_, ciphertext) = encryption_key
            .encrypt(&plaintext, &public_parameters, rng)
            .unwrap();

        assert_eq!(
            plaintext,
            decryption_key
                .decrypt(&ciphertext, &public_parameters)
                .unwrap(),
            "decrypted ciphertext should match the plaintext"
        );
    }

    pub fn evaluates<
        const EVALUATION_GROUP_SCALAR_LIMBS: usize,
        const PLAINTEXT_SPACE_SCALAR_LIMBS: usize,
        EvaluationGroupElement: KnownOrderScalar<EVALUATION_GROUP_SCALAR_LIMBS>
            + From<Value<EncryptionKey::PlaintextSpaceGroupElement>>,
        EncryptionKey: AdditivelyHomomorphicEncryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS>,
        DecryptionKey,
    >(
        decryption_key: DecryptionKey,
        evaluation_group_public_parameters: &EvaluationGroupElement::PublicParameters,
        public_parameters: &PublicParameters<PLAINTEXT_SPACE_SCALAR_LIMBS, EncryptionKey>,
        rng: &mut impl CryptoRngCore,
    ) where
        DecryptionKey:
            AdditivelyHomomorphicDecryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS, EncryptionKey>,
    {
        let encryption_key = decryption_key.as_ref();

        let zero: Uint<PLAINTEXT_SPACE_SCALAR_LIMBS> = (&U64::from(0u64)).into();
        let zero = EncryptionKey::PlaintextSpaceGroupElement::new(
            zero.into(),
            public_parameters.plaintext_space_public_parameters(),
        )
        .unwrap();

        let one: Uint<PLAINTEXT_SPACE_SCALAR_LIMBS> = (&U64::from(1u64)).into();
        let one = EncryptionKey::PlaintextSpaceGroupElement::new(
            one.into(),
            public_parameters.plaintext_space_public_parameters(),
        )
        .unwrap();
        let two: Uint<PLAINTEXT_SPACE_SCALAR_LIMBS> = (&U64::from(2u64)).into();
        let two = EncryptionKey::PlaintextSpaceGroupElement::new(
            two.into(),
            public_parameters.plaintext_space_public_parameters(),
        )
        .unwrap();
        let five: Uint<PLAINTEXT_SPACE_SCALAR_LIMBS> = (&U64::from(5u64)).into();
        let five = EncryptionKey::PlaintextSpaceGroupElement::new(
            five.into(),
            public_parameters.plaintext_space_public_parameters(),
        )
        .unwrap();
        let seven: Uint<PLAINTEXT_SPACE_SCALAR_LIMBS> = (&U64::from(7u64)).into();
        let seven = EncryptionKey::PlaintextSpaceGroupElement::new(
            seven.into(),
            public_parameters.plaintext_space_public_parameters(),
        )
        .unwrap();
        let seventy_three: Uint<PLAINTEXT_SPACE_SCALAR_LIMBS> = (&U64::from(73u64)).into();
        let seventy_three = EncryptionKey::PlaintextSpaceGroupElement::new(
            seventy_three.into(),
            public_parameters.plaintext_space_public_parameters(),
        )
        .unwrap();

        let (_, encrypted_two) = encryption_key
            .encrypt(&two, &public_parameters, rng)
            .unwrap();

        let (_, encrypted_five) = encryption_key
            .encrypt(&five, &public_parameters, rng)
            .unwrap();

        let (_, encrypted_seven) = encryption_key
            .encrypt(&seven, &public_parameters, rng)
            .unwrap();

        let evaluted_ciphertext = encrypted_five.scalar_mul(&U64::from(1u64))
            + encrypted_seven.scalar_mul(&U64::from(0u64))
            + encrypted_two.scalar_mul(&U64::from(73u64));

        let expected_evaluation_result: Uint<PLAINTEXT_SPACE_SCALAR_LIMBS> =
            (&U64::from(1u64 * 5 + 0 * 7 + 73 * 2)).into();
        let expected_evaluation_result = EncryptionKey::PlaintextSpaceGroupElement::new(
            expected_evaluation_result.into(),
            public_parameters.plaintext_space_public_parameters(),
        )
        .unwrap();

        assert_eq!(
            expected_evaluation_result,
            decryption_key
                .decrypt(&evaluted_ciphertext, &public_parameters)
                .unwrap()
        );

        let randomness = EncryptionKey::RandomnessSpaceGroupElement::sample(
            public_parameters.randomness_space_public_parameters(),
            rng,
        )
        .unwrap();

        let evaluation_order = (&EvaluationGroupElement::order_from_public_parameters(
            &evaluation_group_public_parameters,
        ))
            .into();

        let ciphertexts_and_upper_bounds = [
            (encrypted_five, evaluation_order),
            (encrypted_seven, evaluation_order),
            (encrypted_two, evaluation_order),
        ];

        let mask = EncryptionKey::sample_mask_for_secure_function_evaluation(
            &ciphertexts_and_upper_bounds,
            public_parameters,
            rng,
        )
        .unwrap();

        let privately_evaluted_ciphertext = encryption_key
            .securely_evaluate_linear_combination_with_randomness(
                &[one, zero, seventy_three],
                ciphertexts_and_upper_bounds,
                &evaluation_order,
                &mask,
                &randomness,
                public_parameters,
            )
            .unwrap();

        assert_ne!(
            evaluted_ciphertext, privately_evaluted_ciphertext,
            "privately evaluating the linear combination should result in a different ciphertext due to added randomness"
        );

        assert_ne!(
            decryption_key.decrypt(&evaluted_ciphertext, &public_parameters).unwrap(),
            decryption_key.decrypt(&privately_evaluted_ciphertext, &public_parameters).unwrap(),
            "decryptions of privately evaluated linear combinations should be statistically indistinguishable from straightforward ones"
        );

        assert_eq!(
            EvaluationGroupElement::from(decryption_key.decrypt(&evaluted_ciphertext, &public_parameters).unwrap().value()),
            EvaluationGroupElement::from(decryption_key.decrypt(&privately_evaluted_ciphertext, &public_parameters).unwrap().value()),
            "decryptions of privately evaluated linear combinations should match straightforward ones modulu the evaluation group order"
        );
    }
}
