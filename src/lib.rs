// Author: dWallet Labs, LTD.
// SPDX-License-Identifier: BSD-3-Clause-Clear
use std::collections::HashMap;
use std::fmt::Debug;

use crypto_bigint::{rand_core::CryptoRngCore, Uint};
use group::{GroupElement, KnownOrderGroupElement, KnownOrderScalar, PartyID, Samplable};
use serde::{Deserialize, Serialize};

/// An error in encryption related operations
#[derive(thiserror::Error, Clone, Debug, PartialEq)]
pub enum Error {
    #[error("group error")]
    GroupInstantiation(#[from] group::Error),
    #[error("zero dimension: cannot evalute a zero-dimension linear combination")]
    ZeroDimension,
    #[error("an internal error that should never have happened and signifies a bug")]
    InternalError,
}

/// The Result of the `new()` operation of types implementing the
/// `AdditivelyHomomorphicEncryptionKey` trait
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
        + Clone
        + PartialEq;

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

    /// $\Eval(pk,f, \ct_1,\ldots,\ct_t; \eta_{\sf eval})$: Efficient homomorphic evaluation of the
    /// linear combination defined by `coefficients` and `ciphertexts`.
    ///
    /// This method *does not assure circuit privacy*.
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

    /// $\Eval(pk,f, \ct_1,\ldots,\ct_t; \eta_{\sf eval})$: Efficient homomorphic evaluation of the
    /// linear combination defined by `coefficients` and `ciphertexts`.
    ///
    /// In order to perform an affine evaluation, the free variable should be paired with an
    /// encryption of one.
    ///
    /// This method ensures circuit privacy by masking the linear combination with a random (`mask`)
    /// multiplication of the `modulus` $q$ using fresh `randomness`:
    ///
    /// $\ct = \Enc(pk, \omega q; \eta) \bigoplus_{i=1}^\ell \left(  a_i \odot \ct_i \right)$
    ///
    /// In more detail, these steps are taken to genrically assure circuit privacy:
    /// 1. Rerandomization. This should be done by adding an encryption of zero with fresh
    ///    randomness to the outputted ciphertext.
    ///
    /// 2. Masking. Our evaluation should be masked by a random multiplication of the homomorphic
    ///    evaluation group order $q$.
    ///
    ///    While the decryption modulo $q$ will remain correct,
    ///    assuming that the mask was "big enough", it will be statistically indistinguishable from
    ///    random.
    ///
    ///   We do not perform any security checks to gaurantee soundness or even correctness, and
    ///  "big enough" should be checked outside this function in the context of the broader protocol
    ///   for which security was proven.
    ///
    /// 3. No modulations. The size of our evaluation $l*B^2$ should be smaller than the order of
    ///    the encryption plaintext group $N$ in order to assure it does not go through modulation
    ///    in the plaintext space.
    ///
    /// In the case that the plaintext order is the same as the evaluation `modulus`, steps 2, 3 are
    /// skipped.
    fn evaluate_circuit_private_linear_combination_with_randomness<
        const DIMENSION: usize,
        const MODULUS_LIMBS: usize,
    >(
        &self,
        coefficients: &[Self::PlaintextSpaceGroupElement; DIMENSION],
        ciphertexts: &[Self::CiphertextSpaceGroupElement; DIMENSION],
        modulus: &Uint<MODULUS_LIMBS>,
        mask: &Self::PlaintextSpaceGroupElement,
        randomness: &Self::RandomnessSpaceGroupElement,
        public_parameters: &Self::PublicParameters,
    ) -> Result<Self::CiphertextSpaceGroupElement> {
        if DIMENSION == 0 {
            return Err(Error::ZeroDimension);
        }

        let plaintext_order: Uint<PLAINTEXT_SPACE_SCALAR_LIMBS> = coefficients[0].order();

        let linear_combination = Self::evaluate_linear_combination(coefficients, ciphertexts)?;

        // Rerandomization is performed in any case, and a masked multiplication of the modulus is
        // added only if the order of the plaintext space differs from `modulus`.
        let plaintext =
            if PLAINTEXT_SPACE_SCALAR_LIMBS == MODULUS_LIMBS && plaintext_order == modulus.into() {
                coefficients[0].neutral()
            } else {
                Self::PlaintextSpaceGroupElement::new(
                    Uint::<PLAINTEXT_SPACE_SCALAR_LIMBS>::from(modulus).into(),
                    &coefficients[0].public_parameters(),
                )? * mask
            };

        let encryption_with_fresh_randomness =
            self.encrypt_with_randomness(&plaintext, randomness, public_parameters);

        Ok(linear_combination + encryption_with_fresh_randomness)
    }
}

/// A Decryption Key of an Additively Homomorphic Encryption scheme
pub trait AdditivelyHomomorphicDecryptionKey<
    const PLAINTEXT_SPACE_SCALAR_LIMBS: usize,
    EncryptionKey: AdditivelyHomomorphicEncryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS>,
>: Into<EncryptionKey> + Clone + PartialEq
{
    /// The decryption key used for decryption.
    type SecretKey;

    /// Instantiate the decryption key from the public parameters of the encryption scheme,
    /// and the secret key.
    fn new(
        encryption_scheme_public_parameters: &EncryptionKey::PublicParameters,
        secret_key: Self::SecretKey,
    ) -> Result<Self>;

    /// $\Dec(sk, \ct) \to \pt$: Decrypt `ciphertext` using `decryption_key`.
    /// A deterministic algorithm that on input a secret key $sk$ and a ciphertext $\ct \in
    /// \calC_{pk}$ outputs a plaintext $\pt \in \calP_{pk}$.
    fn decrypt(
        &self,
        ciphertext: &EncryptionKey::CiphertextSpaceGroupElement,
    ) -> EncryptionKey::PlaintextSpaceGroupElement;
}

/// A Decryption Key Share of a Threshold Additively Homomorphic Encryption scheme
pub trait AdditivelyHomomorphicDecryptionKeyShare<
    const PLAINTEXT_SPACE_SCALAR_LIMBS: usize,
    EncryptionKey: AdditivelyHomomorphicEncryptionKey<PLAINTEXT_SPACE_SCALAR_LIMBS>,
>: Into<EncryptionKey> + Clone + PartialEq
{
    /// A decryption share of a ciphertext in the process of Threshold Decryption.
    type DecryptionShare: Clone + Debug + PartialEq + Eq;
    /// A proof that a decryption share was correctly computed on a ciphertext using the decryption
    /// key share `Self`.
    type PartialDecryptionProof: Clone + Debug + PartialEq + Eq;
    /// Precomputed values used for Threshold Decryption.
    type PrecomputedValues: Clone + Debug + PartialEq + Eq;

    /// The Semi-honest variant of Partial Decryption, returns the decryption share without proving
    /// correctness.
    fn generate_decryption_share_semi_honest(
        &self,
        ciphertext: &EncryptionKey::CiphertextSpaceGroupElement,
    ) -> Result<Self::DecryptionShare>;

    /// Performs the Maliciously-secure Partial Decryption in which decryption shares are computed
    /// and proven correct.
    fn generate_decryption_shares(
        &self,
        ciphertexts: Vec<EncryptionKey::CiphertextSpaceGroupElement>,
        rng: &mut impl CryptoRngCore,
    ) -> Result<(Vec<Self::DecryptionShare>, Self::PartialDecryptionProof)>;

    /// Finalizes the Threshold Decryption protocol by combining decryption shares. This is the
    /// Semi-Honest variant in which no proofs are verified.
    fn combine_decryption_shares_semi_honest(
        decryption_shares: HashMap<PartyID, Self::DecryptionShare>,
        encryption_key: &EncryptionKey,
        precomputed_values: Self::PrecomputedValues,
    ) -> Result<EncryptionKey::PlaintextSpaceGroupElement>;
}

#[derive(Debug, PartialEq, Clone, Serialize, Deserialize)]
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
