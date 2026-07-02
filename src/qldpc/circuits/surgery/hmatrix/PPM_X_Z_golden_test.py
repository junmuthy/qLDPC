"""Golden byte-identity regression for the single-gadget builders.

Freezes build_gadget (edge-expanded, Ide, Gowda, Nadkarni, Dauphinais
arXiv:2410.02753 Algorithm 3, deterministic at seed=0) and
build_gadget_augmented (closed form, Cain et al. arXiv:2603.28627 §B.1)
output across a fixed basket as SHA-256 hashes — a determinism guard against
silent construction drift. Regenerate the _GOLDEN dict by pasting the output
of _regenerate_golden() against a known-good tree.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

import numpy as np

from qldpc import codes
from qldpc.circuits.surgery.conftest import (
    build_generalised_bicycle_code,
    load_webster_seed_set,
)
from qldpc.circuits.surgery.hmatrix.PPM_X_Z import (
    GadgetLayout,
    build_gadget,
    build_gadget_augmented,
)
from qldpc.codes.common import CSSCode
from qldpc.objects import Pauli, PauliXZ

_GOLDEN: dict[str, dict[str, str]] = {'Steane|X|aug': {'HX_merged': 'cfebe9aa02016e62fbf893a0434a78f2a1e3b2b9460875a0e8a3cffe0c9c8067',
                  'HZ_merged': '2c00472ac0d2ece84c56dfefc717410d144cfa853345cbc8b64f5bf4d6690007',
                  'Q_prime': '5f4bd761808119ae5104bffc6820c6d1ebb47119c623f85c9165414aa1d57e87',
                  'data_checks': '356b13aa56a822752d87979f23f0a27467b51eef8fb0395c7c52de173ef2571e',
                  'incidence': 'bb47d09764e3687a5d089b70ec0f84f91e960f39e7eb1e2593af9b5d33262ebf',
                  'partial_0': '800ddf39dfe0598d22b404e1ce711ae7b9c25128afeb55944abd416f991f2315',
                  'support': '9798549418cb6f2fc21948af401a9e76b0d76be4decd1832af85ea57ce6c2082'},
 'Steane|X|plain': {'HX_merged': '4bdb995400f54d1a3b31b2c2ce00161724ec255a93f2e865ad50f0add49a9468',
                    'HZ_merged': '59dbd7c3d60edd7f180a1ed3c2c885b0fab671bed39fbd2b191efde95d84d6d9',
                    'Q_prime': 'c3cfab2cb19ff893e979d8c2e443bdc496416fc8e9ad578c62f5c5db9b77a89a',
                    'data_checks': '6b1e0170df3836491ffcfa025ec4437fa3ea883392cf7fc5fd8797a4da01b1f2',
                    'incidence': 'a1a1cc2299c58b0cd0bfc937862f9c0d8299ac6f8e5c0527bada7239c4b7caad',
                    'partial_0': '2f1535a4022a4d7483b3b6b1953de8cf6ebb2c1e93a5198a7c145ea687902537',
                    'support': '9798549418cb6f2fc21948af401a9e76b0d76be4decd1832af85ea57ce6c2082'},
 'Steane|Z|aug': {'HX_merged': '11ad6da5bb5622d9727e85df02106c33c7bbdfe63619c9767867d5e209330a5e',
                  'HZ_merged': 'b5091113da5e53c9262992c77ee7659dfa313afaaea1d6a404b2f7820ae4d487',
                  'Q_prime': '10a1bfffc0b96f3df91df3f47323cfc5a80abce5e3051fa5f172b27d2872c975',
                  'data_checks': '92846f53a80d0e61eab48a38d32bea54266d0e589aa15c68ba8ee4f8e2b77751',
                  'incidence': 'c2e0d93c673739a3a7fd6410233fabd3b820535e163c3a757b8fbcf76f50f316',
                  'partial_0': 'ffedf96c2e238fb52f00eb794078aaab51feb5d1ae000d504e002a57f8160ee4',
                  'support': '21f98a1faba3ccf33fb30d737299bc45f52cd099d43bfcbf5e4c2da7eeb76eac'},
 'Steane|Z|plain': {'HX_merged': 'aa0aea4b5ce27ceff33fb92a0db056bdc9fd099b9cf14b5ddd5aef213a0d9f13',
                    'HZ_merged': 'a1a3657619e8e17d4702f35d881c1cbd3a72de936f2f54e3cf232814bc2caf08',
                    'Q_prime': '0beef7e903fce1e25b443e987bf86c1466cd3a48b991fa656afa62cf7931e4c7',
                    'data_checks': '3c534f19458efdbe3400ee91caacf825ddaa6398808cfdff85cf01ba78ce1504',
                    'incidence': '614d552da5a5c27c0d355bf05ee824dd82c4bcf2418231cd36a51747d475fdcd',
                    'partial_0': '1f64673779413bef9a029dc55b8d5d14cb3039370b90148a649a5baa4584a2b2',
                    'support': '21f98a1faba3ccf33fb30d737299bc45f52cd099d43bfcbf5e4c2da7eeb76eac'},
 'Webster0|X|aug': {'HX_merged': 'f2d3d95acc243d6f6971ffcde093e0eba387b2d8b35d7bf74ff8121c3a8efec8',
                    'HZ_merged': '57362c7710666966411068aa0aad47f636b3966ccf9bec3cae4bbc67267ecb47',
                    'Q_prime': '2ebb3180fbadced19db474c80b30a0dcaf4ff709af7eddc812957e04229595f1',
                    'data_checks': '3e1b15f64607afdfa28116b8566fb8d2ec95df0c9fb4c47dcef0f687884eafdb',
                    'incidence': '3c5795223c32c900cab8a1413f635792c1297e1ecb76d041c1d541d553a31ffb',
                    'partial_0': '2b1530fc07968cc90c4a407c0aa1130d72fcd0b76ed7c3edf19045e928f8c72d',
                    'support': '8f0cb21653e4611b4ec38d5cc558eb8f3a1b8c0658ba3622db6327ff76cc8c90'},
 'Webster0|X|plain': {'HX_merged': 'a382a22c7fc9c4113b5b2a9394fff7f36bc4c4e5ad26c5bd5be1e239c6409b66',
                      'HZ_merged': '87be205636ada1778f64174c7b2e2b1f5790baab63612dc5cfb24df2c3759df8',
                      'Q_prime': '2ebb3180fbadced19db474c80b30a0dcaf4ff709af7eddc812957e04229595f1',
                      'data_checks': '3e1b15f64607afdfa28116b8566fb8d2ec95df0c9fb4c47dcef0f687884eafdb',
                      'incidence': 'f0ef586b0165812ff64cca8d5963aadd04030bff24b13f982922a0fdf6446de7',
                      'partial_0': '79e44673eded42d19b46320f4c16f44424a017c5fe0e6b1f4b5998b12342b892',
                      'support': '8f0cb21653e4611b4ec38d5cc558eb8f3a1b8c0658ba3622db6327ff76cc8c90'},
 'Webster0|Z|aug': {'HX_merged': 'b36584f21f6f1758d63615c0822152a09fe1868e54a959504ab0375c3fdefa4f',
                    'HZ_merged': '65e0da623c44442de5034bb083eb62a1a706df3142de6345758e14bb71d22bd6',
                    'Q_prime': '425bfe59bea765af84ffd81f9062ea6bcd8ec49d325363b6537cb99935518f6d',
                    'data_checks': '11d0a4f18efea2becd060c5ea60ed5004adb9309a55babfb7cefc64c9dff5726',
                    'incidence': '6e0cac30e0dfc50146781b18e8d0d45cf16d96df080b2f31c63128e78820fdf0',
                    'partial_0': 'c8230f5e47fc9f860f3923be99ba68e688f45bc451751ef7f8c9f1438125998b',
                    'support': '2275ddb8918d641f134d11a8d06284ce4605bb9348b86f256459fc87aa3f42c9'},
 'Webster0|Z|plain': {'HX_merged': 'e7948cbc1a932878419434a6d25f64d81f4218f8d88410bf9f0d309f4129aaa5',
                      'HZ_merged': 'a5195e6b612f69a61531c53da4738ad9a8d5eb5bcb6a65d9df34f133ee2317e0',
                      'Q_prime': '9ac0bd126bb5917200101d3c1dad92b3e06addbbb758f6d7fb870da8c58e2b80',
                      'data_checks': '2dc3b71f5d1f0ae085791df46085e52e1f2ed40c3f792513d17c71d5084e9e85',
                      'incidence': 'ccc8d4721e9d23b8b44e0e5075fc943ff757ac3bee819d98189b3aa290276358',
                      'partial_0': 'a41b9e0978690839fdcac9e43d37fc852b7c68048190ffb2651ba83595b8d310',
                      'support': '2275ddb8918d641f134d11a8d06284ce4605bb9348b86f256459fc87aa3f42c9'},
 'Webster1|X|aug': {'HX_merged': '270d89755078c09dc7368c613bf3eb6a1a87cbafca8f6b48157270d7b3f1338b',
                    'HZ_merged': '372155597cefb79718caee8e9b9bc9497c9a1f502c7aa4496cde44df6cd44e76',
                    'Q_prime': '495fc767f1844a8adb79ebdc2b09f155e2d6f6061b451c5f2be0ae88f5e20e51',
                    'data_checks': 'a712b71b032876a546f99680c795eae8534bd4c003eb64b95cbccb1b0e44c85c',
                    'incidence': '67673f9155036043590f155856cc802ef6c66af4eed1baeeb3e97f2bfdad2a0d',
                    'partial_0': 'ed52c2bbf002c53a6527285a85bdfc55c71ffc27ee2083ad00b0d7f454321fe7',
                    'support': 'ae2ce0629ed389c4fbce926530f4c1af06e26e4a73c79c279366bac4f82a5385'},
 'Webster1|Z|aug': {'HX_merged': '30c97efc9c2fb4c611c21ac3262fea630728ab1f4901f768d7b5e0569f08b46f',
                    'HZ_merged': '40b4a46d3f019375944bb6c5419c8b8b8f919411a3bc21171a8a433088c77567',
                    'Q_prime': '0fcc415aa4739e76edfecfbb7a3c4fa726f4c274b985464f71a0579aaa4938f9',
                    'data_checks': '1c86e5071b561acaf7590ff922b770f1ead5cb39d6e66397700e2cb3fa79434b',
                    'incidence': '1ead04b798cbd6a3ff9c8390c9a637e8b86e1a5cddaf1d58c40ffbc2d6943b6f',
                    'partial_0': 'c7404570fbdfb011a3239ee726d93ebe793d7618fec58803e72a493c408e1f96',
                    'support': 'defe473f91d5e8df4cdf7134e6d7ffc5ed816ff43df2fa588998f69c454dd0b8'},
 'Webster2|X|aug': {'HX_merged': 'b0c99137f9d50596af695b1b0895ccb15fb9bde3da04c29673489f794baeecc7',
                    'HZ_merged': '7012cccdabbe618905359d735a417bfe66052e3179a1b4a3bd77322b52b4a86a',
                    'Q_prime': 'a8d3044051f0b284e99507d548a32334d95cf822024735e67e00dde71daaf608',
                    'data_checks': 'a57fa7be78cff407206a5890abe03a32e6a72a564f9b8d4b7bbb114245787a84',
                    'incidence': '2121dc153cbf3ae73f7fef07a2fa0b7eab9df0387ea5572c1f3e8a5e69191d0d',
                    'partial_0': '5a1b07ecc60b56fa608cf1a34089322cc186fd4379540619f70c454318540641',
                    'support': '58bdc4805b4d3df04a1af3613db87a0c205c18fe3609169ac2e6297aeab7e2d4'},
 'Webster2|Z|aug': {'HX_merged': '027a427ef8ee161deeb3dbab9353dfbab076b8788c9c9fb47eb57c6d1a4cf5fb',
                    'HZ_merged': '5e7aeba7829b48048e6835f98439af9510e032c1d9bbb8db449d2e91f08ed1fb',
                    'Q_prime': '33e17a4506e89df7e29048f6e9f1bb43cc21b29db77bf4f98bf4d85d9226167d',
                    'data_checks': '73a60268e02170e2061fa320b72c7769ac321ea2573d97ad2bda221f9da2d157',
                    'incidence': 'a3b9bf1efa67dcadd3144354c5959558862ffc8aacfe02ff35a5a99d71c32f36',
                    'partial_0': '449c0d3b7f29be7d609a83324d36b2c0b1c102d909db17a6c12f2c03fa5887d0',
                    'support': 'a75d21840a027839a4f3478615b9d63ef97daba61df4e1d17c55235c31edf7a8'},
 'Webster3|X|aug': {'HX_merged': 'd1a35fa03474038da7e179204bd27bb0060291ec267733a27ca0b037727040bb',
                    'HZ_merged': '69fe63d4e6fb8334bda3d0e300ea2e5676c3ad4bb3a7d1b5d2cdb6e71c4976da',
                    'Q_prime': '570ec0d16c55de3b4b820d3fe32ea68a6470acab92eb618dc215b12ce50f0834',
                    'data_checks': '2c98f407458151c64be5bcdd01def638558dddadc2d3095454c4f954e673d8d6',
                    'incidence': '46f16cd975a29da37dfd9bb27a0d70b1030d051204420ba40f4c02d63d117607',
                    'partial_0': 'd94c25475a549d2b4540debb5024b2c824fcd7ef96a63b9e696dcbcb90167336',
                    'support': '68652cc71ca70a734c334e5170f97030d737c6e60d84b6165c1c0decd2523ff5'},
 'Webster3|Z|aug': {'HX_merged': '70eb90877df6a5d2d5f09a6fccab14185900af98307f090a3fba78ac199cc4c3',
                    'HZ_merged': '1fb100decb4a1b7c9c5e7a397c958cb4e69f213a0e5dd8b07ce765b6c5b438f7',
                    'Q_prime': '73260891f33a9fc13f72d066c37712d49e08d942d78c185c6ce6002d3efd77be',
                    'data_checks': 'ff48b9314762864509dc2bebc61c4a639d351a283a4f1e9dbaf62264c17365d4',
                    'incidence': 'b2bdb540ae296825d2313312559fd6639b69593716006b7b0e23f5af1f99db41',
                    'partial_0': '7dc938a5e0a5db75848a7b8f61df8e0fae31d9ca3ca59bcd89b2b87d7fa185dd',
                    'support': 'c80a3ee693771d56b88dc96dc96d5e622b33b2b6395644881744165574c02fc0'}}
_FIELDS = (
    "support", "data_checks", "incidence", "partial_0",
    "HX_merged", "HZ_merged", "Q_prime",
)


def _canon(value: Any) -> str:
    """Shape-aware canonical SHA-256 of an array/tuple field (int64 bytes)."""
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.int64))
    return hashlib.sha256(arr.tobytes() + repr(arr.shape).encode()).hexdigest()


def _golden_extra(support_len: int) -> np.ndarray:
    """Deterministic weight-2 incidence_extra: rows (0,1) and (1,2) mod L."""
    rows = []
    for j in (0, 1):
        r = np.zeros(support_len, dtype=np.uint8)
        r[j % support_len] = 1
        r[(j + 1) % support_len] = 1
        rows.append(r)
    return np.array(rows, dtype=np.uint8)


# build_gadget (edge-expanded) runs the exact exhaustive Cheeger sweep, which is
# O(2^|V0|): fine for |V0| ≤ 16, minutes for Webster3's |V0| = 26. The 'plain'
# (new-construction) cases are therefore pinned only where they stay fast; the
# closed-form 'aug' cases (no Cheeger sweep) cover every code.
_PLAIN_MAX_SUPPORT = 16


def _golden_cases() -> Iterator[tuple[str, CSSCode, np.ndarray, PauliXZ, np.ndarray | None]]:
    """Yield (tag, code, x, basis, incidence_extra | None)."""
    entries: list[tuple[str, CSSCode]] = [("Steane", codes.SteaneCode())]
    for ci in range(4):
        d = load_webster_seed_set(ci)
        entries.append((f"Webster{ci}", build_generalised_bicycle_code(d["l"], d["A"], d["B"])))
    for name, code in entries:
        for basis in (Pauli.X, Pauli.Z):
            x = np.asarray(code.get_logical_ops(basis)[0]).astype(np.uint8)
            if int(np.count_nonzero(x)) <= _PLAIN_MAX_SUPPORT:
                yield (f"{name}|{basis.name}|plain", code, x, basis, None)
            extra = _golden_extra(int(np.count_nonzero(x)))
            yield (f"{name}|{basis.name}|aug", code, x, basis, extra)


def _layout_for(
    code: CSSCode, x: np.ndarray, basis: PauliXZ, extra: np.ndarray | None
) -> GadgetLayout:
    if extra is None:
        # Golden freezes the edge-expanded construction (deterministic at seed=0).
        return build_gadget(code, x, basis=basis, seed=0)
    return build_gadget_augmented(code, x, extra, basis=basis)


def _hashes() -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for tag, code, x, basis, extra in _golden_cases():
        g = _layout_for(code, x, basis, extra)
        out[tag] = {f: _canon(getattr(g, f)) for f in _FIELDS}
    return out


def test_gadget_builders_byte_identical_to_golden() -> None:
    expected = _GOLDEN
    actual = _hashes()
    assert actual.keys() == expected.keys(), (
        f"case set drift: extra={set(actual) - set(expected)}, "
        f"missing={set(expected) - set(actual)}"
    )
    for tag in expected:
        for field in _FIELDS:
            assert actual[tag][field] == expected[tag][field], (
                f"byte mismatch at {tag}.{field}"
            )


def _regenerate_golden() -> None:  # pragma: no cover - manual, run on good tree
    import pprint

    print("_GOLDEN: dict[str, dict[str, str]] = " + pprint.pformat(_hashes(), sort_dicts=True, width=100))
