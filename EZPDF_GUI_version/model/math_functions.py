# model/math_functions.py

import numpy as np
import warnings
from scipy.spatial.distance import cdist
from scipy import interpolate
from model.elem_data import ElementData
from scipy.linalg import cho_factor, cho_solve, LinAlgError
import scipy.sparse as sp
from math import factorial
import io

data = ElementData()


# Messages already reported by _warn_once(), so that a warning about the same
# situation is not repeated on every recalculation (parameter sliders trigger
# many recalculations in a row).
_WARNED_MESSAGES = set()


def _warn_once(message):
    """Emit a warning the first time this exact message occurs."""
    if message in _WARNED_MESSAGES:
        return
    _WARNED_MESSAGES.add(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)
    print("[Warning] " + message)


def reset_warning_history():
    """Allow previously reported warnings to be shown again.

    Call this when the user changes files, so a genuinely new situation is
    reported even if the wording matches an earlier warning.
    """
    _WARNED_MESSAGES.clear()


def create_atom_distance_matrix(atom_positions):
    distance_matrix = cdist(atom_positions, atom_positions)
    return distance_matrix


def __cal_fi(scat_values, q):
    """
    Calculate scattering form factor (Vectorized)
    q can be a scalar or a numpy array.
    """
    # [Vectorization] q가 배열일 경우 자동으로 배열 연산 수행
    k_sq = (0.25 * q / np.pi) ** 2
    fi1 = scat_values[0] * np.exp(-scat_values[1] * k_sq)
    fi2 = scat_values[2] * np.exp(-scat_values[3] * k_sq)
    fi3 = scat_values[4] * np.exp(-scat_values[5] * k_sq)
    fi4 = scat_values[6] * np.exp(-scat_values[7] * k_sq)
    fi5 = scat_values[8] * np.exp(-scat_values[9] * k_sq)
    fic = scat_values[10]
    fi = fi1 + fi2 + fi3 + fi4 + fi5 + fic
    return fi


def __cal_compton_fi(compton_scattering_factors, q):
    """
    Calculate Compton scattering form factor (Vectorized support)
    """
    k_sq = (0.25 * q / np.pi) ** 2
    fi1 = compton_scattering_factors[0] * np.exp(-compton_scattering_factors[6] * k_sq)
    fi2 = compton_scattering_factors[1] * np.exp(-compton_scattering_factors[7] * k_sq)
    fi3 = compton_scattering_factors[2] * np.exp(-compton_scattering_factors[8] * k_sq)
    fi4 = compton_scattering_factors[3] * np.exp(-compton_scattering_factors[9] * k_sq)
    fi5 = compton_scattering_factors[4] * np.exp(-compton_scattering_factors[10] * k_sq)
    fic = compton_scattering_factors[5]
    fi = fi1 + fi2 + fi3 + fi4 + fi5 + fic
    return fi


def compton_cal_exp(atom_indices, compton_scat_parms, compton_scattering_factors,
                    atomic_number, qmin=0.2, qmax=20, qstep=0.1, wavelength=0.1665, alpha=3,
                    composition_weights=None):

    #============================================================================
    #  Compton (incoherent) X-ray scattering calculation
    #  컴프턴(비탄성) X선 산란 강도 계산
    # ----------------------------------------------------------------------------
    #  REFERENCE / 참고문헌
    #  H. H. M. Balyuzi, "Analytic Approximations to Incoherently Scattered
    #  X-Ray Intensities", Acta Cryst. (1975). A31, 600-602.
    #
    #  Source data of the tabulated coefficients (compton_parameter_only.txt):
    #  계수 테이블의 원 출처:
    #    DABAX file "CrossSec_Compton_Balyuzi.dat", extracted from sf_inc.f of
    #    the library by S. Brennan and P. L. Cowan, Rev. Sci. Instrum. 63, 850
    #    (1992). Coefficients originally from Cromer & Mann, J. Chem. Phys. 47,
    #    1892 (1967) and Cromer, J. Chem. Phys. 50, 4857 (1969), fitted by
    #    Balyuzi (1975).
    #
    #  KEY IDEA / 핵심 개념
    #  --------------------------------------------------------------------------
    #  EN: Balyuzi did NOT fit the incoherent intensity I_inc(s) directly.
    #      He fitted the function  F(s) = Z - I_inc(s)  to a sum of five
    #      Gaussians (no constant term, c = 0):
    #
    #          F_fit(s) = c + sum_{i=1..5} a_i * exp(-b_i * s^2),   c = 0
    #
    #      where  s = sin(theta)/lambda = Q / (4*pi),  and  Z = atomic number.
    #      Therefore the incoherent (Compton) intensity per atom is recovered by
    #      a simple subtraction:
    #
    #          I_inc(s) = Z - F_fit(s)
    #
    #      F_fit is ALREADY (Z - I_inc); it is NOT an atomic form factor, so it
    #      must not be squared or divided by Z. (This was the previous bug.)
    #
    #  KR: Balyuzi는 비탄성 강도 I_inc(s)를 직접 피팅한 것이 아니라,
    #      함수  F(s) = Z - I_inc(s)  를 5개의 가우시안 합(상수항 c = 0)으로
    #      피팅했다:
    #
    #          F_fit(s) = c + Σ_{i=1..5} a_i · exp(-b_i · s^2),   c = 0
    #
    #      여기서  s = sin(theta)/lambda = Q / (4π),  Z = 원자번호 이다.
    #      따라서 원자당 비탄성(컴프턴) 강도는 단순한 뺄셈으로 얻는다:
    #
    #          I_inc(s) = Z - F_fit(s)
    #
    #      F_fit 은 그 자체가 이미 (Z - I_inc) 값이며, 원자 form factor가
    #      아니다. 그러므로 제곱하거나 Z로 나누면 안 된다. (이전 버그의 원인)
    #
    #  VALIDITY RANGE / 유효 범위
    #  --------------------------------------------------------------------------
    #  EN: The fit is accurate for s = sin(theta)/lambda <= ~1.4-1.5 A^-1,
    #      i.e. Q <= ~18-19 A^-1. Beyond this, values are extrapolated.
    #  KR: 이 피팅은 s = sin(theta)/lambda <= 약 1.4~1.5 A^-1 (즉 Q <= 약
    #      18~19 A^-1) 범위에서 정확하다. 그 이상은 외삽 영역이다.
    # ============================================================================

    # parameters for Breit-Dirac recoil factor
    me = 9.109534e-31
    C = 2.99792458e18  # 8+10 m to A
    h = 6.62607015e-14  # -34+20 , m^2 to A^2
    part_A = 2.0 * h * wavelength / me / C

    num_fact = len(compton_scattering_factors)
    q_range = np.arange(qmin, qmax, qstep)

    # 원소별 조성량(가중치) 결정.
    # composition_weights가 주어지면 원소당 한 개씩의 조성량으로 사용하고,
    # 그렇지 않으면 원자를 나열한 atom_indices를 세어 가중치를 만든다.
    # 두 방식은 같은 결과를 주며, 전자는 소수 조성도 처리할 수 있다.
    if composition_weights is not None:
        weights = np.asarray(composition_weights, dtype=float)
        if weights.size != num_fact:
            raise ValueError(
                f"composition_weights has {weights.size} entries but there are "
                f"{num_fact} compton scattering factors"
            )
    else:
        weights = np.bincount(np.asarray(atom_indices),
                              minlength=num_fact).astype(float)

    num_atom = weights.sum()

    # [Optimization Note] This loop is kept as is, but can be vectorized if needed.
    # Typically Compton calculation is done once and is fast enough.
    list_compton_scat = []
    compton_scat = 0.0

    # 원소별 원자번호를 가중치에 맞춰 미리 합산 (q에 무관하므로 루프 밖에서 계산)
    atomic_number_arr = np.asarray(
        [atomic_number[k] for k in range(num_fact)], dtype=float)
    atomic_number_sum = float(np.dot(weights, atomic_number_arr))

    for q in q_range:
        part_B = (q / (4.0 * np.pi)) ** 2.0
        BD_recoil_fact = (part_A * part_B + 1) ** (-alpha)  # Breit-Dirac recoil factor
        list_fi = []
        for k in range(num_fact):
            list_fi.append(__cal_compton_fi(compton_scat_parms[atomic_number[k] - 1], q))
        list_fi = np.asarray(list_fi)
        # I_inc = Z - F_fit (Balyuzi 1975); F_fit is already (Z - I_inc)
        ffit_sum = float(np.dot(weights, list_fi))
        compton_scat = BD_recoil_fact * (1 / num_atom * atomic_number_sum
                                         - 1 / num_atom * ffit_sum)
        list_compton_scat.append(compton_scat)
    return q_range, list_compton_scat


def cal_Iq(atom_indices, scattering_factors, atom_distance_matrix,
           qmin=0.5, qmax=20, qstep=0.05):
    # Theoretical calculation loop (kept as loop to prevent memory overflow for large N)
    num_atom = len(atom_indices)
    num_fact = len(scattering_factors)
    diag_idx = np.diag_indices(num_atom)
    # sin(q*r)/(q*r) -> 1 as r -> 0. Zero distances occur on the diagonal
    # (self-pairs) and for any coincident atoms sharing identical coordinates;
    # both are set to 1 below so the 0/0 division never produces NaN/warnings.
    zero_mask = (atom_distance_matrix == 0.0)
    list_Iq = []
    q_range = np.arange(qmin, qmax, qstep)
    for q in q_range:
        fi_mat = np.zeros((num_atom, num_atom))
        list_fi = []
        for k in range(num_fact):
            list_fi.append(__cal_fi(scattering_factors[k], q))
        list_fi = np.asarray(list_fi)
        for i, idx in enumerate(atom_indices):
            fi = list_fi[idx]
            fi_mat[i, :] = fi
        if q == 0:
            sin_mat = np.ones(np.shape(atom_distance_matrix))
        else:
            qr = q * atom_distance_matrix
            sin_mat = np.divide(np.sin(qr), qr,
                                out=np.ones_like(atom_distance_matrix, dtype=float),
                                where=~zero_mask)
        sin_mat[zero_mask] = 1.0
        Iq = fi_mat * np.transpose(fi_mat) * sin_mat
        list_Iq.append(np.sum(Iq))
    return q_range, list_Iq


def cal_Sq(atom_indices, scattering_factors, atom_distance_matrix,
           qmin=0.5, qmax=20, qstep=0.05, return_Iq=False):
    # Theoretical calculation loop
    num_atom = len(atom_indices)
    num_fact = len(scattering_factors)
    diag_idx = np.diag_indices(num_atom)
    # sin(q*r)/(q*r) -> 1 as r -> 0. Zero distances occur on the diagonal
    # (self-pairs) AND for any coincident atoms sharing identical coordinates.
    # Both must be set to this limit; otherwise the 0/0 division yields NaN
    # (with an "invalid value encountered in divide" warning) that propagates
    # into I(q)/S(q)/F(q)/G(r).
    zero_mask = (atom_distance_matrix == 0.0)

    list_Iq = []
    sq_mean_fi = []
    mean_sq_fi = []
    q_range = np.arange(qmin, qmax, qstep)

    for q in q_range:
        fi_mat = np.zeros((num_atom, num_atom))
        list_fi = []
        for k in range(num_fact):
            list_fi.append(__cal_fi(scattering_factors[k], q))
        list_fi = np.asarray(list_fi)
        fi_sum, fi2_sum = 0.0, 0.0
        for i, idx in enumerate(atom_indices):
            fi = list_fi[idx]
            fi_mat[i, :] = fi
            fi_sum = fi_sum + fi
            fi2_sum = fi2_sum + fi ** 2
        if q == 0:
            sin_mat = np.ones(np.shape(atom_distance_matrix))
        else:
            qr = q * atom_distance_matrix
            sin_mat = np.divide(np.sin(qr), qr,
                                out=np.ones_like(atom_distance_matrix, dtype=float),
                                where=~zero_mask)
        sin_mat[zero_mask] = 1.0
        Iq = fi_mat * np.transpose(fi_mat) * sin_mat
        list_Iq.append(np.sum(Iq))
        sq_mean_fi.append((fi_sum / num_atom) ** 2)
        mean_sq_fi.append(fi2_sum / num_atom)

    list_Iq = np.asarray(list_Iq)
    mean_sq_fi = np.asarray(mean_sq_fi)  # <f^2>
    sq_mean_fi = np.asarray(sq_mean_fi)  # <f>^2
    list_Sq = (list_Iq - num_atom * mean_sq_fi) / (num_atom * sq_mean_fi) + 1
    list_Fq = q_range * (list_Sq - 1)

    if return_Iq:
        return q_range, list_Iq, list_Sq, list_Fq, mean_sq_fi, sq_mean_fi
    else:
        return q_range, list_Sq


def cal_Gr_integral(q, Sq, rmin=0, rmax=100, rstep=0.02, qdamp=0.0):
    """
    [KR] 코드에서 직접 호출하는 직접 사인 적분 함수입니다.
         XYZ 모델에서 cal_Sq로 계산한 q, Sq를 그대로 전달할 수 있습니다.
         GUI 연결, FFT, 제로 패딩 및 저Q 선형 외삽을 사용하지 않습니다.
         q, Sq는 같은 길이의 1차원 NumPy 배열이어야 합니다.
         q는 2점 이상이며, 일정한 양의 간격으로 증가해야 합니다.

    [EN] Code API: direct sine integration of S(Q), including XYZ-derived S(Q).
         Requires equal-length 1D NumPy arrays on a uniform increasing Q grid
         with at least two samples. No FFT, zero padding or low-Q extrapolation.

    초보자를 위한 코드 사용 예 / Beginner example:
        # 1. 프로젝트 루트(model 폴더가 있는 위치)에서 실행합니다.
        #    model/math_functions.py에 있는 함수를 불러옵니다.
        from model.math_functions import cal_Gr_integral

        # 2. 먼저 q와 Sq를 준비해야 합니다. 변수 이름은 대소문자를 구분합니다.
        #    q: Q 좌표 배열(단위: 1/Angstrom)
        #    Sq: 각 q 위치에 대응하는 S(Q) 배열
        #    XYZ 모델의 경우 cal_Sq(...)가 반환한 q와 Sq를 사용합니다.
        #    두 배열은 같은 길이의 1차원 NumPy 배열이어야 합니다.
        #    q는 2점 이상이며 일정한 양의 간격으로 증가해야 합니다.

        # 3. 아래 코드를 호출하면 직접 적분으로 G(r)을 계산합니다.
        #    q, Sq를 준비하지 않고 실행하면 NameError가 발생합니다.
        r, gr = cal_Gr_integral(q, Sq, rmin=0, rmax=30, rstep=0.01)

        # r: 원자 사이 거리 좌표(Angstrom), gr: 각 r에 대응하는 G(r) 값
        # rmin=0, rmax=30: 거리 범위, rstep=0.01: 거리 간격
        # qdamp의 기본값은 0.0이므로 감쇠는 적용되지 않습니다.

    [EN] Calculate the Pair Distribution Function G(r) from Structure Factor S(q) using direct sine integral transform.

         This method performs a discrete integration (Riemann sum) for each r value.
         It avoids FFT padding; numerical accuracy depends on Q spacing and the integration range.
         Formula: G(r) = (2/pi) * integral [ Q * (S(Q)-1) * sin(Q*r) ] dQ.

    [KR] 구조 인자 S(q)로부터 직접 사인 적분 변환을 사용하여 쌍 분포 함수 G(r)을 계산합니다.
         각 r 값에 대해 이산 적분(리만 합)을 수행합니다.
         FFT 패딩을 사용하지 않으며, 수치 정확도는 Q 간격과 적분 범위에 따라 달라집니다.
         공식: G(r) = (2/pi) * integral [ Q * (S(Q)-1) * sin(Q*r) ] dQ

    Args:
        q (numpy.ndarray):
            [EN] The momentum transfer axis Q (inverse Angstrom). 1D array of floats.
            [KR] 운동량 전달 축 Q (단위: 1/A). 실수형 1차원 배열.

        Sq (numpy.ndarray):
            [EN] The Structure Factor S(q) corresponding to the q axis. 1D array of floats.
            [KR] Q 축에 해당하는 구조 인자 S(q). 실수형 1차원 배열.

        rmin (float, optional):
            [EN] Minimum value for the real-space distance r (Angstrom). Default is 0.
            [KR] 실공간 거리 r의 최소값 (단위: A). 기본값은 0.

        rmax (float, optional):
            [EN] Maximum value for the real-space distance r (Angstrom). Default is 100.
            [KR] 실공간 거리 r의 최대값 (단위: A). 기본값은 100.

        rstep (float, optional):
            [EN] Step size for the r axis. Default is 0.02.
            [KR] r 축의 간격. 기본값은 0.02.

        qdamp (float, optional):
            [EN] Gaussian damping factor to simulate limited instrumental resolution.
                 Applied as envelope: exp(-0.5 * (r * qdamp)^2). Default is 0.0 (no damping).
            [KR] 제한된 기기 해상도를 시뮬레이션하기 위한 가우시안 감쇠 인자.
                 적용 식: exp(-0.5 * (r * qdamp)^2). 기본값은 0.0 (감쇠 없음).

    Returns
    -------
        tuple (numpy.ndarray, numpy.ndarray):
            [EN] Returns a tuple containing:
                 - list_r: The generated r-axis (Angstrom).
                 - list_Gr: The calculated G(r).
            [KR] 다음을 포함하는 튜플을 반환합니다:
                 - list_r: 생성된 r 축 (단위: A).
                 - list_Gr: 계산된 G(r) 값.
    """
    list_r = np.arange(rmin, rmax + rstep, rstep)
    Fq = (Sq - 1) * q
    qstep = q[1] - q[0]
    list_Gr = np.zeros_like(list_r)
    list_qFq = Fq * qstep
    for i, r in enumerate(list_r):
        list_Gr[i] = np.sum(list_qFq * np.sin(q * r)) * 2.0 / np.pi
    if qdamp != 0.0:
        list_Gr = np.exp(-0.5 * (list_r * qdamp) ** 2) * list_Gr
    return list_r, list_Gr


def cal_expSq(atom_indices, scattering_factors, expqiq, bkgqiq, qmin=0, qmax=25, qstep=0.01,
              background_scale=1.1, poly_order=11.0, return_Iq=False,
              composition_weights=None):
    """
    [VECTORIZED] Optimized version for Experimental S(q) calculation.
    Removed 'for q in q_range' loop for significant speedup.
    """
    # -------------------------------------------------------------------
    # 1. Experimental Data 처리
    # -------------------------------------------------------------------
    if isinstance(expqiq, str):
        data_exp = load_qiq_file(expqiq, min_cols=2, usecols=(0, 1))
        exp_q = data_exp[:, 0]
        exp_Iq = data_exp[:, 1]
    else:
        exp_q = expqiq[0]
        exp_Iq = expqiq[1]

    # -------------------------------------------------------------------
    # 2. Background Data 처리
    # -------------------------------------------------------------------
    # Physically, a background can only be subtracted when it was measured and
    # integrated with exactly the same settings as the sample, so the sample
    # and background must share the same q-axis. If the q-axes differ, the
    # subtraction is meaningless, so we refuse to guess and raise a clear error
    # instead of silently producing a wrong result.
    bkg_q = None
    if isinstance(bkgqiq, str):
        data_bkg = load_qiq_file(bkgqiq, min_cols=2, usecols=(0, 1))
        bkg_q = data_bkg[:, 0]
        bkg_Iq = data_bkg[:, 1]
    elif bkgqiq is not None:
        if hasattr(bkgqiq, 'shape') and len(bkgqiq.shape) > 1:
            bkg_q = bkgqiq[0]
            bkg_Iq = bkgqiq[1]
        elif isinstance(bkgqiq, (list, tuple)) and len(bkgqiq) == 2 and hasattr(bkgqiq[0], '__len__'):
            bkg_q = bkgqiq[0]
            bkg_Iq = bkgqiq[1]
        else:
            bkg_Iq = bkgqiq
    else:
        bkg_Iq = np.zeros_like(exp_Iq)

    exp_q = np.asarray(exp_q, dtype=float)
    exp_Iq = np.asarray(exp_Iq, dtype=float)
    bkg_Iq = np.asarray(bkg_Iq, dtype=float)

    # Verify that the background shares the sample's q-axis (the normal case
    # when both come from the same integration setup). If the q-axes differ we
    # do NOT stop the calculation — we warn the user and proceed, leaving the
    # decision of whether the result is trustworthy to them.
    have_real_bkg = np.any(bkg_Iq != 0.0)
    bkg_use_own_q = False   # interpolate the background from its own q-axis?

    if have_real_bkg:
        if bkg_q is not None:
            bkg_q = np.asarray(bkg_q, dtype=float)
            same_length = (len(bkg_q) == len(exp_q))

            if same_length:
                max_abs_diff = float(np.max(np.abs(bkg_q - exp_q)))
                exact = np.allclose(bkg_q, exp_q, rtol=0.0, atol=0.0)
                tiny = np.allclose(bkg_q, exp_q, rtol=1e-5, atol=1e-6)

                if not tiny:
                    # Genuinely different q grid. Warn, but continue by
                    # interpolating the background onto the common grid using
                    # its own q-axis.
                    bkg_use_own_q = True
                    _msg = (
                        "The sample and background q-axes are different "
                        f"(largest q difference = {max_abs_diff:.6g}). "
                        "Background subtraction normally requires the sample "
                        "and background to share the same q values. The "
                        "calculation will still proceed by interpolating the "
                        "background onto the sample q-grid, but please check "
                        "whether the result is meaningful."
                    )
                    _warn_once(_msg)
                elif not exact:
                    # Values match only to within rounding: proceed, but note
                    # the small difference.
                    _msg = (
                        "The sample and background q-axes are not exactly "
                        f"equal; they differ by up to {max_abs_diff:.3g} in q. "
                        "Proceeding with subtraction on the shared q-grid."
                    )
                    _warn_once(_msg)
            else:
                # Different number of points. Warn and continue using the
                # background's own q-axis for interpolation.
                bkg_use_own_q = True
                _msg = (
                    "The sample and background have different numbers of "
                    f"points (sample: {len(exp_q)}, background: "
                    f"{len(bkg_Iq)}). Background subtraction normally requires "
                    "the same q values. The calculation will still proceed by "
                    "interpolating the background onto the sample q-grid, but "
                    "please check whether the result is meaningful."
                )
                _warn_once(_msg)
        else:
            # No background q-axis available.
            if len(bkg_Iq) != len(exp_q):
                # Cannot align without a q-axis; warn and skip the background
                # rather than crashing.
                _msg = (
                    "The background has a different number of points from the "
                    f"sample (sample: {len(exp_q)}, background: {len(bkg_Iq)}) "
                    "and no background q-axis is available, so it cannot be "
                    "aligned. The background will be ignored for this "
                    "calculation."
                )
                _warn_once(_msg)
                have_real_bkg = False
                bkg_Iq = np.zeros_like(exp_Iq)

    # q-grid: use the experimental data's own q grid, so the
    # q resolution comes from the measurement and a user qstep can't oversample
    # it. Resample to a uniform grid of the SAME number of points spanning the
    # data's q range (removes minor non-uniformity, keeps the point count), then
    # restrict to [qmin, qmax]. cal_expGr_fft derives its qstep from this grid.
    # [KR] q 격자: 실험 데이터 자체의 q 격자를 그대로 사용한다. 따라서 q 분해능은
    #      측정에서 정해지며, 사용자가 준 qstep으로 데이터를 과도하게 촘촘히
    #      보간하지 않는다. 데이터의 q 범위에 걸쳐 점 개수는 그대로 유지하면서
    #      균일한 간격으로 다시 배치하고(약간의 불균일만 없앰), [qmin, qmax]
    #      구간으로 자른다. cal_expGr_fft는 이 격자에서 qstep을 가져간다.
    xreg = np.linspace(exp_q[0], exp_q[-1], len(exp_q))
    q_range = xreg[(xreg >= qmin) & (xreg <= qmax)]
    if q_range.size < 2:
        q_range = xreg

    scaled_expIq = np.interp(q_range, exp_q, exp_Iq)
    if have_real_bkg:
        if bkg_use_own_q and bkg_q is not None:
            # q-axes differ: interpolate the background from its own q-axis.
            scaled_bkgIq = np.interp(q_range, bkg_q, bkg_Iq)
        else:
            # Shared q-axis: interpolate from the sample q-axis.
            scaled_bkgIq = np.interp(q_range, exp_q, bkg_Iq)
    else:
        scaled_bkgIq = np.zeros_like(q_range)

    list_scaled_bkgIq = background_scale * scaled_bkgIq
    list_Iq = scaled_expIq - list_scaled_bkgIq

    # -------------------------------------------------------------------
    # 3. Form Factor Calculation (Vectorized, weighted by composition)
    # -------------------------------------------------------------------
    # Only the unique elements and how much of each is present matter here:
    #
    #     <f>   = sum_k(c_k * f_k)   / sum_k(c_k)
    #     <f^2> = sum_k(c_k * f_k^2) / sum_k(c_k)
    #
    # `atom_indices` gives one entry per atom (the original form), e.g.
    # Co38O119P1 -> 158 entries; the amount of each element is recovered by
    # counting. Alternatively `composition_weights` gives one amount per unique
    # element directly, which also allows fractional compositions such as
    # Li0.2Co0.36Mn0.37Ni0.07.
    #
    # Both give the same averages, since scaling every element by the same
    # factor cancels in the normalisation:
    #   Li0.2Co0.36Mn0.37Ni0.07  ==  Li20Co36Mn37Ni7
    atom_indices = np.asarray(atom_indices)
    num_fact = len(scattering_factors)

    # (1) 모든 Unique Element(num_fact)에 대해 전체 q_range의 Form Factor 미리 계산
    # 결과 shape: (num_fact, len(q_range))
    # __cal_fi 함수가 numpy array 연산을 지원하므로 for loop 없이 한 번에 계산
    list_fi_all_q = np.array([__cal_fi(sf, q_range) for sf in scattering_factors])

    # (2) 각 Unique Element의 조성량(가중치) 결정
    # 결과 shape: (num_fact,)
    if composition_weights is not None:
        # Explicit amount per unique element (allows fractional compositions).
        weights = np.asarray(composition_weights, dtype=float)
        if weights.size != num_fact:
            raise ValueError(
                f"composition_weights has {weights.size} entries but there are "
                f"{num_fact} scattering factors"
            )
    else:
        # One entry per atom: count how many of each unique element.
        weights = np.bincount(atom_indices, minlength=num_fact).astype(float)

    total_weight = weights.sum()
    num_atom = total_weight

    # (3) <f> 및 <f^2> 계산 (조성으로 가중 평균)
    # weights[:, None] 로 broadcast 하여 (len(q_range), ) 크기의 1차원 배열 생성
    sum_fi = np.sum(weights[:, None] * list_fi_all_q, axis=0)  # Sum of c*f
    sum_fi_sq = np.sum(weights[:, None] * list_fi_all_q ** 2, axis=0)  # Sum of c*f^2

    sq_mean_fi = (sum_fi / total_weight) ** 2  # <f>^2
    mean_sq_fi = sum_fi_sq / total_weight  # <f^2>

    # S(q) 계산 (배열 연산)
    # [EN] X-ray scattering-factor normalization following the ad hoc
    #      data-reduction approach of PDFgetX3. The experimental I(q) is in
    #      arbitrary units (not normalized per incident flux nor per number of
    #      scatterers), so a least-squares scale factor is required to bring the
    #      normalized intensity onto the normal scattering-factor curve before
    #      computing S(q). Without this scale the S(q) amplitude is wrong
    #      (the deviation from 1 is off by a large factor).
    #        normalized_intensity     = I / <f>^2
    #        normal_scattering_factor = <f^2> / <f>^2
    #        normalization_scale = dot(normalized_intensity, normal_scattering_factor)
    #                              / dot(normalized_intensity, normalized_intensity)
    #        S(q) = normalization_scale * normalized_intensity
    #               - normal_scattering_factor + 1
    # [KR] PDFgetX3의 ad hoc 데이터 처리 방식을 따르는 X-ray 산란인자 정규화.
    #      실험 I(q)는 임의 단위(입사 강도/산란체 수로 정규화되지 않음)이므로,
    #      정규화 강도를 normal scattering-factor 곡선에 맞추는 최소자승 스케일
    #      상수가 필요하다. 이 스케일이 없으면 S(q) 진폭이(1로부터의 편차가)
    #      큰 배수로 어긋난다.
    #
    # Reference:
    #   Juhas, P., Davis, T., Farrow, C. L. & Billinge, S. J. L. (2013).
    #   PDFgetX3: a rapid and highly automatable program for processing powder
    #   diffraction data into total scattering pair distribution functions.
    #   J. Appl. Cryst. 46, 560-566.

    normalized_intensity = list_Iq / sq_mean_fi          # I / <f>^2   (sq_mean_fi = <f>^2)
    normal_scattering_factor = mean_sq_fi / sq_mean_fi   # <f^2> / <f>^2 (mean_sq_fi = <f^2>)
    normalization_scale = (np.dot(normalized_intensity, normal_scattering_factor)
                           / np.dot(normalized_intensity, normalized_intensity))

    # print(f"[DEBUG2] sum_fi[0]={sum_fi[0]:.4f}, sum_fi_sq[0]={sum_fi_sq[0]:.4f}")
    # print(f"[DEBUG2] sq_mean_fi[0]={sq_mean_fi[0]:.4f}, mean_sq_fi[0]={mean_sq_fi[0]:.4f}")
    # print(f"[DEBUG2] scattering_factors.shape={np.array(scattering_factors).shape}")
    # print(f"[DEBUG2] atom_fi_all_q.shape={atom_fi_all_q.shape}")
    # print(f"[DEBUG2] list_fi_all_q.shape={list_fi_all_q.shape}")
    # print(f"[DEBUG2] normalized_intensity[0]={normalized_intensity[0]:.6f}, normal_scattering_factor[0]={normal_scattering_factor[0]:.6f}")

    list_Sq = normalization_scale * normalized_intensity - normal_scattering_factor + 1.0
    # print(f"[DEBUG] normalization_scale={normalization_scale}, list_Sq range={list_Sq.min():.3f} to {list_Sq.max():.3f}, num_atom={num_atom}")

    # F(q) 공간 다항 적합(Vandermonde, 상수항 제거)
    qmax_for_sq = float(np.max(q_range))

    def _poly_for_sq_at_order(order_int: int):
        order_int = max(int(order_int), 1)
        qscaled = q_range / qmax_for_sq
        V = np.vander(qscaled, order_int + 1)
        A = V[:, :-1]
        b = q_range * (list_Sq - 1.0)
        p, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        vand2 = np.vander([qmax_for_sq], order_int + 1)[0, :-1]
        new_p = p / vand2
        final_p = np.poly1d(new_p)
        return final_p(q_range)

    # Poly order 처리
    po = float(poly_order)
    if po.is_integer():
        polynomial_for_sq = _poly_for_sq_at_order(int(round(po)))
    else:
        lo = int(np.floor(po))
        hi = int(np.ceil(po))
        w_hi = po - lo
        w_lo = 1.0 - w_hi
        poly_lo = _poly_for_sq_at_order(lo)
        poly_hi = _poly_for_sq_at_order(hi)
        polynomial_for_sq = w_lo * poly_lo + w_hi * poly_hi

    # 정규화 & F(q)
    norm_list_Sq = list_Sq - polynomial_for_sq
    list_Fq = q_range * (norm_list_Sq - 1.0)

    if return_Iq:
        return (q_range, list_Iq, scaled_expIq, list_scaled_bkgIq,
                list_Sq, norm_list_Sq, list_Fq, mean_sq_fi, sq_mean_fi)
    else:
        return (q_range, list_Iq, scaled_expIq, list_scaled_bkgIq,
                list_Sq, norm_list_Sq, list_Fq, mean_sq_fi, sq_mean_fi)

    normalized_intensity = list_Iq / sq_mean_fi          # I / <f>^2   (sq_mean_fi = <f>^2)
    normal_scattering_factor = mean_sq_fi / sq_mean_fi   # <f^2> / <f>^2 (mean_sq_fi = <f^2>)
    normalization_scale = (np.dot(normalized_intensity, normal_scattering_factor)
                           / np.dot(normalized_intensity, normalized_intensity))

    # np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/sq_mean_fi.txt', sq_mean_fi)
    # np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/mean_sq_fi.txt', mean_sq_fi)
    # np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/normalized_intensity.txt', normalized_intensity)
    # np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/normal_scattering_factor.txt', normal_scattering_factor)
    # np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/normalization_scale.txt', normalization_scale)
    # np.savetxt('D:/1-Manuscript_2014/EZPDF_EZPIT/CoPi_test/scaled_experimental_I(q).txt', normalization_scale * list_Iq )


def lorch_window(q, qmax):
    """Lorch modification function M(q) = sin(pi*q/qmax) / (pi*q/qmax).

    M(0) = 1 and M(qmax) = 0. Used to damp the finite-q_max truncation before
    the F(q) -> G(r) sine transform, reducing termination ripple.
    """
    q = np.asarray(q, dtype=float)
    if qmax <= 0:
        return np.ones_like(q)
    x = np.pi * q / qmax
    m = np.ones_like(x)
    nz = x != 0.0
    m[nz] = np.sin(x[nz]) / x[nz]
    return m


# ============================================================================
# BEGINNER'S MATHEMATICAL GUIDE / 초보자를 위한 수학 설명
# Applies to the active cal_expGr_fft below, not the commented legacy code.
# 아래 실행 함수의 설명입니다. 주석으로 보존된 과거 코드와 구분하십시오.
# Equations use plain ASCII so they remain readable in a Python editor.
# ============================================================================
# 1. What is transformed? / 무엇을 변환하는가?
# EN: S(Q) is dimensionless. F(Q) = Q * [S(Q) - 1]. If is_Fq=True,
#     the supplied values already ARE F(Q); do not multiply by Q again.
# KO: S(Q)는 무차원이며 F(Q)=Q*[S(Q)-1]입니다. is_Fq=True이면 입력이
#     이미 F(Q)이므로 Q를 다시 곱하지 않습니다.
#     Q: 1/Angstrom; r: Angstrom; F(Q): 1/Angstrom; G(r): 1/Angstrom^2.
#     Q*r is dimensionless / Q*r은 무차원입니다.
#
# 2. Target integral / 계산하려는 적분
#     G(r) = (2/pi) * integral[F(Q)*sin(Q*r) dQ, Qlo .. Qhi]
# EN: Here Qlo=q[0], Qhi=q[-1], the first and last supplied measurements.
#     The function receives an array, not a separate nominal Qmin setting.
# KO: 이 함수의 경계는 전달된 배열의 첫 Q와 마지막 Q입니다. GUI 설정의
#     Qmin과 첫 측정 Q는 격자 때문에 조금 다를 수 있습니다.
# EN: This is a finite-Q transform, not the unavailable integral to infinity.
#     Truncation can produce termination ripples; zero filling does not remove them.
# KO: 유한한 Q 범위의 변환이므로 끝단 진동이 생길 수 있습니다. 제로 채움이
#     무한한 Q 범위의 측정을 복구하거나 끝단 진동을 없애지는 않습니다.
#
# 3. A grid anchored at Q=0 / Q=0을 기준으로 한 계산 격자
#     dQ = median(diff(q))/4; Q[k] = k*dQ.
# EN: The code can reduce dQ further for a large requested rmax. It linearly
#     interpolates the measured F onto these EXACT Q positions. Merely placing
#     the first measured value in bin ceil(q[0]/dQ) would shift its coordinate.
# KO: 기본 계산 간격은 원본 Q 간격 중앙값의 1/4입니다. 큰 rmax에서는 더
#     줄어들 수 있습니다. 정확한 Q[k] 위치에서 보간하여 좌표 이동을 방지합니다.
# EN: positive starts as zeros. Only Qlo <= Q[k] <= Qhi gets interpolated data.
#     Thus 0 <= Q[k] < Qlo stays zero: no inferred low-Q signal is added.
# KO: positive 배열을 먼저 0으로 만들고 측정 범위 안만 보간값으로 채웁니다.
#     저Q의 0은 계산에서 그 구간을 제외한다는 뜻이지, 실제 시료의 F가 0이라는
#     주장이 아닙니다. Qlo가 격자점과 일치하면 그 원본값을 사용합니다.
# EN: The Q=0 array entry is set to zero for odd symmetry; its sine contribution
#     is zero regardless. Other measured endpoints are not forced to zero.
# KO: Q=0은 홀대칭을 위해 0으로 지정합니다. sin(0*r)=0이므로 사인 변환에
#     기여하지 않습니다. 다른 측정 끝점의 값을 강제로 0으로 만들지는 않습니다.
# EN: Interpolation adds sampling points, not new physical information.
# KO: 촘촘한 보간은 계산 표본을 늘릴 뿐 실험 분해능을 높이지 않습니다.
#
# 4. Odd extension / 홀함수 확장
#     F(-Q) = -F(Q)
#     positive example: [0, 1, 3, 2]
#     full odd array:   [0, 1, 3, 2, 0, -2, -3, -1]
# EN: Negative-Q values sit at the END of the periodic FFT array. The center
#     contains high-|Q| zeros. M is the FULL array length including both signs.
# KO: 음의 Q 데이터는 FFT 배열 끝에 역순으로 넣고 부호를 바꿉니다.
#     가운데 0은 고Q 패딩입니다. 저Q에 넣는 0과 목적 및 위치가 다릅니다.
# EN: M is rounded up to a power of two, large enough for the odd data and
#     the desired internal r sampling. This is one implementation of Pavol's
#     construction, not a claim of identical code or preprocessing to PDFgetX3.
# KO: 홀함수 데이터를 담고 필요한 r 간격을 얻도록 길이 M을 정한 뒤 2의
#     거듭제곱으로 올립니다. PDFgetX3의 전체 처리와 동일하다는 의미는 아닙니다.
#
# 5. Why the imaginary FFT part? / 왜 FFT의 허수부를 사용하는가?
# EN: NumPy's forward DFT has the following sign convention:
#     Z[j] = sum(odd[k] * exp(-2*pi*i*j*k/M), k=0..M-1)
#     theta = 2*pi*j*k/M
#     F[k]*exp(-i*theta) - F[k]*exp(+i*theta) = -2*i*F[k]*sin(theta)
#     Therefore Im(Z[j]) = -2 * sum(F[k]*sin(theta), positive k).
# KO: 양/음의 Q 항을 더하면 cos 항은 상쇄되고 sin 항만 남습니다.
#     허수부의 음수 부호는 NumPy FFT 지수의 음수 부호에서 나옵니다.
#     실수부는 이상적으로 0이며 부동소수점 계산에서는 작은 잔차가 가능합니다.
#
# 6. Scale factor / 정규화 계수 유도
#     G(r[j]) ~= (2*dQ/pi) * sum(F[k]*sin(Q[k]*r[j]))
#              = -(dQ/pi) * Im(Z[j])
#     Implemented: gr_fine = -(dq/pi) * fft(odd).imag[:M//2]
# EN: Odd extension supplies the factor of 2. Do NOT multiply this by 2 again.
# KO: 홀함수 확장에서 이미 두 배의 기여가 생겼으므로 2를 다시 곱하지 않습니다.
# EN: NumPy IFFT uses the opposite sign and divides by M, so equivalently:
#     gr_fine = (M*dQ/pi) * ifft(odd).imag[:M//2]
# KO: IFFT로 바꿀 때는 부호와 M 계수도 함께 바꿔야 합니다. 함수 이름만
#     fft에서 ifft로 바꾸면 결과가 틀립니다. 두 방식의 정확도는 본질적으로 같습니다.
# EN: The OLD positive-Q-only IFFT needs 2*M*dQ/pi instead. That coefficient
#     must not be copied onto the new odd-extended array.
# KO: 과거 양의 Q만 사용한 IFFT의 계수 2*M*dQ/pi와 혼동하지 마십시오.
#
# 7. Correct radial coordinates / 올바른 r 좌표
#     Q[k]*r[j] = 2*pi*k*j/M
#     r[j] = j*2*pi/(M*dQ); dr_fft = 2*pi/(M*dQ)
# EN: Use the ACTUAL M after padding. If N=M/2 is the retained half length,
#     the same formula is dr_fft=pi/(N*dQ). These are not conflicting formulas.
# KO: 패딩 후 실제 전체 길이 M을 사용해야 합니다. 반환하는 절반의 길이 N을
#     사용하면 pi/(N*dQ)로 쓰며, 두 식은 동일합니다.
# EN: The first half covers 0 <= r < pi/dQ. CubicSpline maps these results
#     onto the user's r grid. rstep is an output sampling interval, not
#     experimental spatial resolution. More padding samples r more finely.
# KO: 앞쪽 절반을 사용하고 사용자 r 격자로 3차 스플라인 보간합니다.
#     제로 패딩과 작은 rstep은 곡선을 촘촘히 표시할 뿐 실험 분해능을 높이지 않습니다.
#
# 8. Optional Lorch window / 선택적인 Lorch 창
# EN: Only when use_lorch=True, multiply the interpolated signal by
#     sin(pi*Q/Qhi)/(pi*Q/Qhi), with limit 1 at Q=0, BEFORE transforming.
# KO: use_lorch=True일 때만 변환 전에 Lorch 창을 곱합니다. 끝단 진동을
#     완화할 수 있지만 피크 폭/높이도 달라지므로 비교 계산의 조건을 맞춰야 합니다.
#
# 9. Independent verification / 독립적인 검증과 한계
# EN: On the SAME Q grid, compare FFT against the direct sine SUM in section 6.
#     Separately, on ORIGINAL Q coordinates, compare with trapezoidal integration:
#     G(r) ~= (2/pi)*sum( (q[k+1]-q[k])/2 *
#         (F[k]*sin(q[k]*r) + F[k+1]*sin(q[k+1]*r)) ).
# KO: 같은 격자의 직접 사인 합산은 FFT 부호/정규화를 검증합니다. 원본 Q의
#     사다리꼴 적분은 격자 보간 및 경계 처리까지 포함하여 비교합니다.
# EN: These discretizations need not match exactly. Qlo and Qhi may fall
#     between FFT bins; interpolation, boundary sampling and quadrature all
#     contribute errors. Finer dQ does not guarantee monotonically smaller
#     error. Check convergence for each dataset. One successful sample is
#     not a universal accuracy guarantee or validation of experimental corrections.
# KO: 보간, 경계점 위치, 적분 방식 때문에 작은 차이가 남을 수 있습니다.
#     dQ를 줄일 때 오차가 항상 단조 감소하지는 않습니다. 시료별로 수렴성을
#     확인하십시오. 변환 검증이 배경 제거/조성 등 실험 처리까지 보증하지는 않습니다.
# ============================================================================

# EN: extrapolate_type="linear" is a legacy, UNUSED compatibility argument.
#     It does NOT enable extrapolation. F(Q)=0 below the first input Q.
#     np.interp performs linear interpolation ONLY within the measured Q range.
# KO: extrapolate_type="linear"는 기존 호출과의 호환성을 위해 남긴 미사용 인자입니다.
#     "linear"가 표시되어 있어도 저Q 외삽은 하지 않습니다.
#     첫 입력 Q(q[0])보다 낮은 구간은 F(Q)=0으로 채웁니다.
#     측정 구간 내부의 선형 보간(np.interp)은 저Q 외삽과 다릅니다.
# EN: Despite "exp" in its name, this function transforms BOTH experimental
#     S(Q)/F(Q) and S(Q)/F(Q) calculated from XYZ atomic coordinates.
#     It does NOT read XYZ files or compute atomic scattering itself.
#     GUI XYZ route: get_xyz_graphs() -> cal_Sq(); then get_gr() -> this function.
#     is_Fq=False: supply S(Q). is_Fq=True: supply F(Q).
#     The GUI currently uses this Pavol FFT, not a direct-integral routine.
# KO: 함수 이름에 exp가 있지만 실험 S(Q)/F(Q)뿐 아니라 XYZ 원자 좌표에서
#     계산한 S(Q)/F(Q)도 이 함수로 G(r)로 변환합니다.
#     XYZ 파일 읽기나 원자 산란 계산 자체를 이 함수에서 수행하는 것은 아닙니다.
#     GUI의 XYZ 경로: get_xyz_graphs() -> cal_Sq(), 이후 get_gr() -> 이 함수.
#     S(Q)는 is_Fq=False, F(Q)는 is_Fq=True로 전달합니다.
#     현재 GUI는 직접 적분 함수가 아니라 이 Pavol FFT를 사용합니다.
# EN: Direct quadrature generally costs O(N_Q*N_r), versus O(M*log(M)) for
#     the FFT stage (M includes padding). Actual speed depends on array sizes,
#     implementation and hardware; small jobs need not benefit from FFT.
# KO: 직접 적분은 보통 O(N_Q*N_r), FFT 단계는 O(M*log(M))의 계산이 필요합니다.
#     M은 패딩을 포함한 길이입니다. 큰 격자에서는 보통 직접 적분이 느리지만,
#     실제 속도는 점 개수, 구현, 하드웨어에 따라 달라집니다.
def cal_expGr_fft(q, Sq_or_Fq, rmin=0, rmax=100, rstep=0.01,
                  extrapolate_type="linear", is_Fq=False, pad_mode="zero",
                  use_lorch=False):
    """Pavol-style odd-extension FFT with zero signal outside measured Q.

    Resample onto Q=k*dQ at one quarter of the median input spacing.
    This interpolation does not add experimental resolution. The measured
    first/last Q define the integration support; no low-Q extrapolation.
    extrapolate_type is retained solely for compatibility with existing calls.
    Output r coordinates follow the requested rmin, rmax and rstep convention.
    """
    try:
        q = np.asarray(q, dtype=float)
        values = np.asarray(Sq_or_Fq, dtype=float)
        if q.ndim != 1 or values.shape != q.shape or q.size < 2:
            raise ValueError("Q and signal must be equal-length 1D arrays (at least 2 points)")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(values)):
            raise ValueError("Q and signal must be finite")
        if q[0] < 0 or np.any(np.diff(q) <= 0):
            raise ValueError("Q must be nonnegative and strictly increasing")
        if not np.all(np.isfinite([rmin, rmax, rstep])) or rmin < 0 or rmax < rmin or rstep <= 0:
            raise ValueError("Require 0 <= rmin <= rmax and rstep > 0")
        if pad_mode != "zero":
            raise ValueError("Pavol transform requires pad_mode='zero'")
        Fq = values.copy() if is_Fq else q * (values - 1.0)
        r_list = np.arange(rmin, rmax + rstep, rstep)
        # EN: Respect original Q positions; do not shift them by prepending bins.
        # KO: 원본 Q 좌표에서 보간하며, 정수 개수의 0을 붙여 좌표를 이동하지 않습니다.
        dq = float(np.median(np.diff(q))) / 4.0
        dq = min(dq, np.pi / (r_list[-1] + 2.0 * rstep))
        Q = np.arange(int(np.ceil(q[-1] / dq)) + 1, dtype=float) * dq
        positive = np.zeros(Q.size)
        measured = (Q >= q[0]) & (Q <= q[-1])
        positive[measured] = np.interp(Q[measured], q, Fq)
        # F(0)=0 for the odd extension; sin(0*r) contributes zero in any case.
        positive[0] = 0.0
        if use_lorch:
            positive *= lorch_window(Q, q[-1])
        # [0, F1, F2, ..., zeros in the middle, ..., -F2, -F1]
        dr_fft_target = min(rstep, np.pi / (q[-1] * 80.0))
        required = max(2 * Q.size, int(np.ceil(2 * np.pi / (dq * dr_fft_target))))
        M = 1 << (required - 1).bit_length()
        odd = np.zeros(M)
        odd[:Q.size] = positive
        odd[-(Q.size - 1):] = -positive[1:][::-1]
        # EN: The previous implementation used IFFT on positive-Q data.
        #     This version uses FFT on an odd-extended array (Pavol's method).
        #     Both compute the same sine transform when the input Q samples,
        #     sign and normalization are matched. FFT itself is not more accurate.
        #     Accuracy changes here concern low-Q treatment and Q-grid alignment.
        #     For this SAME odd array, an equivalent IFFT expression is:
        #     gr_fine = (M * dq / np.pi) * np.fft.ifft(odd).imag[:M // 2]
        # KO: 이전 코드는 양의 Q 데이터에 IFFT를 사용했습니다.
        #     현재는 홀함수 확장 배열에 FFT를 사용하는 Pavol 방식입니다.
        #     입력 Q 표본, 부호, 정규화를 맞추면 같은 사인 변환을 계산합니다.
        #     FFT 자체가 더 정확한 것은 아닙니다. 이번 정확도 개선은
        #     저Q 처리와 Q 격자 정렬에 관한 것입니다.
        #     위 IFFT 대체식은 동일한 홀함수 배열 odd에만 적용됩니다.
        # ==================================================================
        # OPTIONAL: SAVE THE PADDED F(Q) / 선택 사항: 패딩된 F(Q) 저장
        # EN: Disabled by default. To save, remove ONE leading "# " from
        #     ONLY the 9 code lines between BEGIN/END SAVE_PADDED_FQ below.
        #     Keep the 8-space function indentation and restart the GUI.
        # KO: 기본적으로 저장하지 않습니다. 아래 BEGIN/END 사이 코드 9줄의
        #     앞쪽 "# "만 제거하고 들여쓰기(8칸)를 유지한 뒤 GUI를 재실행하세요.
        #
        # EN: odd is the exact input to BOTH FFT and rFFT, including low-Q
        #     zeros, interpolated measured data, middle high-Q zeros and the
        #     negative-Q odd image. If use_lorch=True, the window is already
        #     included. If the caller passed smoothed F(Q), that is exported.
        # KO: odd는 FFT/rFFT에 실제로 전달하는 배열입니다. 저Q 0, 측정 구간
        #     보간값, 가운데 고Q 0, 음의 Q 홀함수 확장을 모두 포함합니다.
        #     Lorch 또는 스무딩을 적용한 호출이면 그 적용 상태로 저장됩니다.
        #
        # EN: CSV columns are Q (1/Angstrom) and F_padded (1/Angstrom).
        #     Keep FFT order; do NOT sort rows by Q before reusing as FFT input.
        #     The zero Nyquist bin is represented at Q=-M*dq/2.
        # KO: CSV 두 열은 Q와 F_padded입니다. 음의 Q는 배열 뒤에 있습니다.
        #     FFT에 다시 넣을 때 Q로 정렬하지 마십시오.
        #
        # EN: Relative path below saves in the process WORKING DIRECTORY,
        #     not necessarily beside this source file or the input data.
        #     Replace it with an absolute path in an existing folder if desired.
        #     Each transform call OVERWRITES this filename. GUI can calculate
        #     raw, smoothed and Lorch curves in succession: the LAST call wins.
        #     For a specific variant, save only on that call or use distinct names.
        #     Disable the block again after exporting; large CSV writes slow GUI.
        # KO: 아래 상대 경로는 프로그램의 현재 작업 폴더에 저장됩니다.
        #     원하는 기존 폴더의 절대 경로로 바꿀 수 있습니다.
        #     매 변환 호출마다 같은 파일을 덮어씁니다. GUI가 원본/스무딩/Lorch를
        #     연속 계산하면 마지막 호출의 배열이 남습니다. 원하는 종류만 저장하려면
        #     해당 호출에만 활성화하거나 파일명을 구분하십시오.
        #     데이터가 크므로 저장 후 다시 주석 처리하면 반복 저장을 방지합니다.
        #
        # BEGIN SAVE_PADDED_FQ
        # save_path = r"Fq_padded.csv"
        # q_padded = np.fft.fftfreq(M) * M * dq
        # np.savetxt(
        #     save_path,
        #     np.column_stack((q_padded, odd)),
        #     delimiter=",",
        #     header="Q_A^-1,F_padded_A^-1",
        #     comments="", fmt="%.17g"
        # )
        # END SAVE_PADDED_FQ
        # ==================================================================
        # FFT / rFFT comparison: enable ONLY ONE of the following two lines.
        # 비교 방법: 아래 두 줄 중 반드시 한 줄만 활성화하십시오.
        # FFT 테스트: fft 줄의 #을 제거하고 rfft 줄에 #을 붙이십시오.
        # rFFT 테스트: fft 줄에 #을 붙이고 rfft 줄의 #을 제거하십시오.
        # 수정 후 GUI를 완전히 종료하고 다시 실행하여 같은 데이터/설정으로 비교하십시오.
        # Keep the sign, normalization and [:M // 2] below unchanged.
        # transformed = np.fft.fft(odd)   # Original FFT - disabled for comparison
        transformed = np.fft.rfft(odd)   # rFFT - currently enabled
        # NumPy forward FFT has exp(-2*pi*i*j*k/M): minus sign is required.
        gr_fine = -(dq / np.pi) * transformed.imag[:M // 2]
        rfine = np.arange(M // 2) * (2 * np.pi / (M * dq))
        gr = interpolate.CubicSpline(rfine, gr_fine)(r_list)
        return r_list, gr
    except Exception as e:
        print(f"Error in cal_expGr_fft: {e}")
        return None, None


# EN: extrapolate_type="linear" is a legacy, UNUSED compatibility argument.
#     It does NOT enable extrapolation. F(Q)=0 below the first input Q.
#     np.interp performs linear interpolation ONLY within the measured Q range.
# KO: extrapolate_type="linear"는 기존 호출과의 호환성을 위해 남긴 미사용 인자입니다.
#     "linear"가 표시되어 있어도 저Q 외삽은 하지 않습니다.
#     첫 입력 Q(q[0])보다 낮은 구간은 F(Q)=0으로 채웁니다.
#     측정 구간 내부의 선형 보간(np.interp)은 저Q 외삽과 다릅니다.
def cal_expGr_fft_from_Fq(q, Fq, rmin=0, rmax=100, rstep=0.01,
                           extrapolate_type="linear"):
    """Use the same zero-low-Q odd-extension transform for direct F(Q) calls."""
    return cal_expGr_fft(q, Fq, rmin, rmax, rstep,
                         extrapolate_type=extrapolate_type, is_Fq=True)


# Previous implementations preserved for reference (not executed).
# def cal_expGr_fft(q, Sq_or_Fq, rmin=0, rmax=100, rstep=0.01,
#                   extrapolate_type="linear", is_Fq=False, pad_mode="zero",
#                   use_lorch=False):
#     """
#     Calculates G(r) from S(q) or F(q) using FFT with low-q zero filling
#     and high-q padding.
#
#     Low-q zero filling:
#         The unmeasured region below the first input Q is filled with F(Q)=0.
#         The first input F(Q) and all measured values remain unchanged.
#         extrapolate_type is retained for call compatibility; no extrapolation.
#
#     Nyquist check:
#         The IFFT yields total_point values, but only the alias-free first half
#         (r_nyquist = pi / qstep) is physically valid. If rmax exceeds r_nyquist,
#         F(q) is interpolated onto a finer q grid before the IFFT, and only the
#         first half of the output is used.
#
#     High-q padding (pad_mode):
#         'zero'     -> pad with zeros (default; used by the EZPDF GUI)
#         'decay'    -> linear decay from the last F(q) value down to 0
#         'constant' -> extend the last F(q) value
#         The two non-zero modes are provided for completeness / experimentation
#         and are not exposed in the GUI; callers must request them explicitly.
#
#     Returns: (r_list, gr) or (None, None) on error.
#     """
#     try:
#         if len(q) < 2:
#             print(f"Error in cal_expGr_fft: q array is too short (len={len(q)}) to calculate qstep.")
#             return None, None
#
#         qstep_calc = q[1] - q[0]  # Renamed to avoid confusion with parameter if any
#
#         if qstep_calc == 0.0:
#             print(f"Error in cal_expGr_fft: qstep is zero (q[1]==q[0]). Cannot divide by zero.")
#             return None, None
#
#         num_point = int(np.ceil(q[0] / qstep_calc))
#
#         if is_Fq:
#             Fq = Sq_or_Fq
#         else:
#             Fq = (Sq_or_Fq - 1) * q
#
#         # --- Low-q zero filling (no extrapolated signal) ---
#         pad_Fq = np.zeros(num_point)
#
#         # Previous implementation: linear extrapolation below the first Q.
#         # Kept as comments for reference; this block is NOT executed.
#         # if q[0] > 0.0:
#         #     f_inter = interpolate.interp1d(q, Fq, fill_value="extrapolate", kind=extrapolate_type)
#         #     pad_Fq[:num_point] = f_inter(np.arange(num_point) * qstep_calc)
#
#         # Active implementation: zero signal below the first input Q.
#         # Preserve the first measured F(Q) and every subsequent input value.
#         pad_Fq[:] = 0.0
#
#         Fq_vals = np.append(pad_Fq, Fq)
#
#         # Lorch modification function M(q) = sin(pi*q/qmax)/(pi*q/qmax), M(0)=1.
#         # Multiplying F(q) by M(q) before the sine transform damps the sharp
#         # q_max truncation, reducing termination ripple in G(r) at the cost of
#         # slightly broader peaks. qmax is the data's own maximum q so M(qmax)=0.
#         if use_lorch:
#             q_full = np.append(np.arange(num_point) * qstep_calc, q)
#             Fq_vals = Fq_vals * lorch_window(q_full, q[-1])
#
#         r_list = np.arange(rmin, rmax + rstep, rstep)
#
#         # ------------------------------------------------------------------
#         # Nyquist check:
#         #   The IFFT produces total_point values; only the first half
#         #   (indices 0 .. total_point//2) corresponds to physically valid
#         #   positive-r values, giving r_nyquist = pi / qstep.
#         #   If the requested rmax exceeds this limit, the second (aliased)
#         #   half of the IFFT output would be used, introducing spurious
#         #   oscillations at high r.
#         #   Fix: interpolate F(q) onto a finer q grid so that
#         #   pi / qstep_fine >= rmax before running the IFFT.
#         # ------------------------------------------------------------------
#         r_nyquist = np.pi / qstep_calc
#         if rmax > r_nyquist:
#             qstep_needed = np.pi / rmax
#             q_orig = np.arange(len(Fq_vals)) * qstep_calc
#             q_fine = np.arange(q_orig[0], q_orig[-1] + qstep_needed, qstep_needed)
#             Fq_vals = np.interp(q_fine, q_orig, Fq_vals)
#             qstep_calc = qstep_needed
#
#         # ------------------------------------------------------------------
#         # [Previous IFFT method — kept for reference; uncomment to restore]
#         # The IFFT length was tied directly to rstep, so a very small rstep
#         # (e.g. 0.001) made the IFFT very long and slow.
#         #
#         # num_point = len(Fq_vals)
#         # total_point = int(2 * np.pi / (rstep * qstep_calc))
#         # pad_len = total_point - num_point
#         #
#         # # --- High-q padding (pad_mode) ---
#         # if pad_len > 0:
#         #     last_val = Fq_vals[-1]
#         #     if pad_mode == "decay":
#         #         high_q_padding = np.linspace(last_val, 0.0, pad_len)
#         #     elif pad_mode == "constant":
#         #         high_q_padding = np.full(pad_len, last_val)
#         #     else:
#         #         high_q_padding = np.zeros(pad_len)
#         #     Fq_pad = np.concatenate((Fq_vals, high_q_padding))
#         # else:
#         #     Fq_pad = Fq_vals
#         #
#         # norm = total_point * qstep_calc * 2 / np.pi
#         # gr_full = norm * np.imag(np.fft.ifft(Fq_pad))
#         #
#         # half = total_point // 2
#         # rfine = np.arange(half) * rstep
#         # gr_fine = gr_full[:half]
#         # gr = np.interp(r_list, rfine, gr_fine)
#         # ------------------------------------------------------------------
#
#         # ==================================================================
#         # [KR] 빠른 IFFT 방식 — 초보자를 위한 설명
#         # [EN] Faster IFFT method — explanation for beginners
#         # ==================================================================
#         #
#         # ■ 이 블록이 하는 일 / What this block does
#         #   [KR] F(q)를 사인 푸리에 변환해서 G(r)을 만든다.
#         #   [EN] It builds G(r) from F(q) with a sine Fourier transform.
#         #       G(r) = (2/pi) * ∫ F(q) * sin(q*r) dq
#         #   [KR] 이 적분을 IFFT(역 고속 푸리에 변환)로 한 번에 계산한다.
#         #        IFFT 결과는 일정한 간격의 r 격자 위에 G(r) 값을 준다.
#         #        IFFT가 주는 r 간격은 2*pi / (total_point * qstep_calc) 이다.
#         #        즉 IFFT 길이(total_point)가 길수록 r 간격이 촘촘해진다.
#         #   [EN] The integral is evaluated in one step with an IFFT (inverse
#         #        fast Fourier transform), which returns G(r) on an evenly spaced
#         #        r grid. That r spacing is 2*pi / (total_point * qstep_calc),
#         #        so a longer IFFT (larger total_point) gives a finer r grid.
#         #
#         # ■ 이전 방식이 느렸던 이유 / Why the previous method was slow
#         #   [KR] 이전에는 IFFT의 r 간격을 사용자가 준 rstep과 똑같이 맞췄다.
#         #   [EN] The previous method forced the IFFT r spacing to equal the
#         #        user's rstep:
#         #       total_point = 2*pi / (rstep * qstep_calc)
#         #   [KR] 그래서 rstep을 작게 하면 IFFT 길이가 그만큼 길어졌다.
#         #        예) rstep=0.001, qstep=0.02 이면 total_point ≈ 314,159
#         #        또 이 길이가 임의의 정수라서(2의 거듭제곱이 아니라서) FFT가 느렸다.
#         #   [EN] So a small rstep made the IFFT very long.
#         #        e.g. rstep=0.001, qstep=0.02 gives total_point ≈ 314,159.
#         #        That length was also an arbitrary integer (not a power of two),
#         #        which is a slow size for an FFT.
#         #
#         # ■ 핵심 아이디어 / Key idea
#         #   [KR] G(r)에는 pi/qmax보다 세밀한 정보가 없다.
#         #        F(q)는 qmax까지만 있으므로, G(r) 안의 가장 빠른 진동은
#         #        sin(qmax * r) 이고, 그 한 주기의 길이는 2*pi/qmax 이다.
#         #        - pi/qmax 간격 = 이 가장 빠른 진동을 "한 주기당 2점"으로 찍는
#         #          간격(Nyquist 간격). 정보를 잃지 않는 이론적 최소치이지만,
#         #          2점만으로는 점과 점 사이 값을 정확히 보간하기엔 너무 성기다.
#         #        - 그래서 pi/(qmax*20) 간격을 쓴다 → 한 주기당 40점.
#         #          이렇게 충분히 촘촘히 계산한 뒤, 사용자가 원하는 rstep 격자로
#         #          보간하면 결과가 사실상 정확하다.
#         #        - rstep이 이 값보다 크면(성기면) rstep을 그대로 쓴다 → max() 사용.
#         #   [EN] G(r) holds no detail finer than about pi/qmax.
#         #        F(q) stops at qmax, so the fastest oscillation in G(r) is
#         #        sin(qmax * r), whose period is 2*pi/qmax.
#         #        - A spacing of pi/qmax samples that fastest oscillation with
#         #          2 points per period (the Nyquist spacing). It is the minimum
#         #          that loses no information, but 2 points are far too sparse
#         #          to interpolate accurately between them.
#         #        - So a spacing of pi/(qmax*20) is used -> 40 points per period.
#         #          Computing on this fine grid and then interpolating onto the
#         #          user's rstep grid gives an essentially exact result.
#         #        - If rstep is larger (coarser) than this, rstep itself is used
#         #          -> hence max().
#         #
#         # ■ 왜 20을 곱했나 / Why multiply by 20 (accuracy vs. speed trade-off)
#         #   [KR] qmax=20, rstep=0.001 에서 정확한 값과 비교한 결과:
#         #   [EN] Compared with exact values at qmax=20, rstep=0.001:
#         #       factor  pts/period  FFT length   time    max rel. error
#         #        2        4           4096      0.8ms    2.2e-3  (too coarse)
#         #        5       10          16384      1.2ms    6.0e-6
#         #       10       20          32768      1.8ms    3.7e-7
#         #       20       40          65536      3.4ms    2.3e-8  (current value)
#         #       40       80         131072      7.5ms    1.4e-9
#         #       80      160         262144     18.7ms    8.6e-11
#         #   [KR] 20은 오차가 사실상 0이면서 속도 이득을 거의 잃지 않는 값이다.
#         #        이론적으로 정해진 상수는 아니며, 더 빠르게 하려면 10으로
#         #        낮춰도 충분히 정확하다(오차 약 4e-7).
#         #   [EN] 20 gives essentially zero error while keeping almost all of
#         #        the speed gain. It is a chosen design value, not a fixed
#         #        theoretical constant; lowering it to 10 is still accurate
#         #        (error about 4e-7) and a bit faster.
#         #
#         # ■ rmax를 크게 잡을 때 / When rmax is large (500, 1000, 2000 ...)
#         #   [KR] - 20이라는 배수는 qmax(진동의 빠르기)로 정해지는 값이고,
#         #          rmax(얼마나 멀리까지 계산하느냐)와는 관계가 없다.
#         #          따라서 rmax를 키워도 20을 바꿀 필요가 없다.
#         #        - rmax가 커지면 위의 Nyquist 보정에서 q 간격이 pi/rmax로
#         #          촘촘해지므로 FFT가 길어진다. 이것은 이전 방식도 똑같다.
#         #   [EN] - The factor 20 is tied to qmax (how fast G(r) oscillates),
#         #          not to rmax (how far out G(r) is computed), so it does not
#         #          need to change for a large rmax.
#         #        - A larger rmax makes the Nyquist correction above refine the
#         #          q spacing to pi/rmax, which lengthens the FFT. The previous
#         #          method had exactly the same effect.
#         #       previous FFT length ≈ 2 * rmax / rstep
#         #       new      FFT length ≈ 40 * qmax * rmax / pi   (independent of rstep)
#         #   [KR] - 측정 예 / [EN] - Measured (qmax=20, rstep=0.001):
#         #       rmax=1000 : previous 507 ms  -> new 27 ms
#         #       rmax=2000 : previous 2662 ms -> new 55 ms
#         #   [KR] - 주의: 이제 느려지는 원인은 FFT가 아니라 "출력 점 수"이다.
#         #          rmax=1000, rstep=0.001 이면 G(r)이 100만 점이 되어
#         #          그래프 그리기·줌·파일 저장이 느려지고 파일이 커진다.
#         #          G(r)의 실제 분해능은 약 pi/qmax (qmax=20이면 0.157 Å)이므로
#         #          rmax가 클 때는 rstep 0.01 정도면 충분하다.
#         #   [EN] - Note: the bottleneck is now the number of output points, not
#         #          the FFT. rmax=1000 with rstep=0.001 gives 1,000,000 G(r)
#         #          points, which slows plotting, zooming and saving and makes
#         #          large files. The real r resolution of G(r) is about pi/qmax
#         #          (0.157 Å for qmax=20), so rstep ≈ 0.01 is enough for large rmax.
#         #
#         # ■ 순서 요약 / Steps
#         #   [KR] 1) rstep_fft 결정 : max(rstep, pi/(qmax*20))
#         #        2) IFFT 길이를 2의 거듭제곱으로 올림 (FFT가 가장 빠른 길이).
#         #           0을 더 채워 길이를 늘리는 것은 같은 G(r)을 더 촘촘히
#         #           찍을 뿐이라 G(r) 값 자체는 바뀌지 않는다.
#         #        3) IFFT 계산
#         #        4) 결과를 사용자가 원하는 r_list(rmin~rmax, rstep)로 보간
#         #   [EN] 1) Choose rstep_fft = max(rstep, pi/(qmax*20)).
#         #        2) Round the IFFT length up to a power of two (the fastest FFT
#         #           size). Padding with extra zeros only samples the same G(r)
#         #           more finely; the G(r) values themselves do not change.
#         #        3) Run the IFFT.
#         #        4) Interpolate onto the requested r_list (rmin..rmax, rstep).
#         #
#         # ■ 이전 방식으로 되돌리려면 / To go back to the previous method
#         #   [KR] 이 블록(아래 코드 ~ "gr = interpolate.CubicSpline..." 줄)을
#         #        주석 처리하고, 위의 [Previous IFFT method] 블록의 주석(#)을 푼다.
#         #   [EN] Comment out this block (from the code below down to the
#         #        "gr = interpolate.CubicSpline..." line) and uncomment the
#         #        [Previous IFFT method] block above.
#         # ==================================================================
#
#         # [KR] F(q) 점의 개수와, 이 F(q)가 도달하는 최대 q
#         # [EN] Number of F(q) points and the maximum q this F(q) reaches
#         num_point = len(Fq_vals)
#         qmax_fft = num_point * qstep_calc
#
#         # [1] [KR] IFFT에 쓸 r 간격.
#         #          pi/(qmax*20) = 가장 빠른 진동 한 주기당 40점이 되는 간격.
#         #          사용자의 rstep이 이보다 크면(성기면) rstep을 그대로 사용.
#         #     [EN] r spacing used by the IFFT.
#         #          pi/(qmax*20) = 40 points per period of the fastest oscillation.
#         #          If the user's rstep is larger (coarser), rstep is used as is.
#         rstep_fft = max(rstep, np.pi / (qmax_fft * 20.0))
#
#         # [2] [KR] 그 r 간격을 얻기 위한 IFFT 길이, 그리고 2의 거듭제곱으로 올림.
#         #          예) 50,000 → 65,536 (2^16). FFT는 2의 거듭제곱 길이에서 가장 빠르다.
#         #          F(q) 점 수보다 짧아지지 않도록 max(..., num_point) 를 사용.
#         #     [EN] IFFT length needed for that r spacing, rounded up to a power
#         #          of two, e.g. 50,000 -> 65,536 (2^16); FFTs are fastest at
#         #          power-of-two lengths. max(..., num_point) keeps it from being
#         #          shorter than the F(q) data.
#         total_point = int(np.ceil(2 * np.pi / (rstep_fft * qstep_calc)))
#         total_point = 1 << int(np.ceil(np.log2(max(total_point, num_point))))
#
#         # [KR] 고-q 쪽에 덧붙일 0(패딩)의 개수
#         # [EN] Number of zeros (padding) appended on the high-q side
#         pad_len = total_point - num_point
#
#         # --- High-q padding (pad_mode) ---
#         if pad_len > 0:
#             last_val = Fq_vals[-1]
#             if pad_mode == "decay":
#                 # Linear decay from the last value down to 0 across the pad region.
#                 high_q_padding = np.linspace(last_val, 0.0, pad_len)
#             elif pad_mode == "constant":
#                 # Extend the last value across the pad region.
#                 high_q_padding = np.full(pad_len, last_val)
#             else:
#                 # Default "zero" padding (GUI behaviour).
#                 high_q_padding = np.zeros(pad_len)
#             Fq_pad = np.concatenate((Fq_vals, high_q_padding))
#         else:
#             Fq_pad = Fq_vals
#
#         norm = total_point * qstep_calc * 2 / np.pi
#         gr_full = norm * np.imag(np.fft.ifft(Fq_pad))
#
#         # [3] [KR] IFFT 결과 중 앞쪽 절반만 물리적으로 의미가 있다
#         #          (뒤쪽 절반은 음의 r에 해당하는 중복/왜곡된 값).
#         #     [EN] Only the first half of the IFFT output is physically
#         #          meaningful (the second half is the mirrored, aliased part
#         #          that corresponds to negative r).
#         half = total_point // 2
#
#         # [KR] IFFT가 실제로 준 r 격자. 간격은 2*pi / (total_point * qstep_calc).
#         #      이전 방식은 이 간격을 rstep이라고 가정했지만, IFFT 길이를 정수로
#         #      자르는 과정에서 실제 간격이 rstep과 약간 달랐다. 여기서는 실제
#         #      간격을 계산해서 사용하므로 r 위치가 더 정확하다.
#         # [EN] The r grid the IFFT actually produced; its spacing is
#         #      2*pi / (total_point * qstep_calc). The previous method assumed this
#         #      spacing equals rstep, but truncating the IFFT length to an integer
#         #      made it slightly different. Using the true spacing here places
#         #      every r value correctly.
#         rfine = np.arange(half) * (2 * np.pi / (total_point * qstep_calc))
#         gr_fine = gr_full[:half]
#
#         # [4] [KR] 사용자가 원하는 r_list(rmin~rmax, rstep)로 보간.
#         #          np.interp(선형) 대신 CubicSpline(3차 스플라인)을 쓰는 이유:
#         #          - 이전 방식은 IFFT 격자 간격이 rstep과 같아서, 보간이 사실상
#         #            격자점 값을 그대로 읽는 역할이었다(선형이어도 오차 없음).
#         #          - 새 방식은 rstep이 매우 작으면 IFFT를 더 성긴 간격으로
#         #            계산하고 그 사이를 채워야 하므로, 보간 방법이 정확도에
#         #            직접 영향을 준다.
#         #          - 선형 보간은 격자점 사이를 직선으로 이어서 피크 꼭대기처럼
#         #            곡률이 큰 곳을 약간 깎는다(피크 높이가 조금 낮아짐).
#         #     [EN] Interpolate onto the requested r_list (rmin..rmax, rstep).
#         #          Why CubicSpline (cubic) instead of np.interp (linear):
#         #          - In the previous method the IFFT grid spacing equalled rstep,
#         #            so interpolation just read grid values (linear was exact).
#         #          - In the new method, for a very small rstep the IFFT runs on a
#         #            coarser grid and the gaps must be filled, so the
#         #            interpolation method directly affects accuracy.
#         #          - Linear interpolation joins grid points with straight lines,
#         #            slightly cutting off high-curvature regions such as peak
#         #            tops (peaks come out a little too low).
#         #     [KR/EN] 같은 FFT 길이에서 비교 / Same FFT length (qmax=20, rstep=0.001):
#         #           np.interp   : max rel. error ~2.3e-4, interpolation 0.2 ms
#         #           CubicSpline : max rel. error ~2.3e-8, interpolation 2.4 ms
#         #     [KR] 차이는 소수점 넷째 자리 이하라 그래프에서는 구분되지 않지만,
#         #          CubicSpline이 참값에 더 가깝고 추가 시간은 무시할 수준이다.
#         #     [EN] The difference is below the 4th decimal place and invisible on
#         #          a plot, but CubicSpline is closer to the true values and the
#         #          extra time is negligible.
#         #     [KR] np.interp로 바꾸고 싶다면 아래 줄을
#         #              gr = np.interp(r_list, rfine, gr_fine)
#         #          로 바꾸고, 정확도를 유지하려면 위 rstep_fft의 20.0을 80.0으로
#         #          올린다(대신 FFT가 4배 길어져 속도 이득이 줄어든다).
#         #     [EN] To use np.interp instead, change the line below to
#         #              gr = np.interp(r_list, rfine, gr_fine)
#         #          and, to keep the same accuracy, raise 20.0 in rstep_fft above
#         #          to 80.0 (the FFT becomes 4x longer, reducing the speed gain).
#         gr = interpolate.CubicSpline(rfine, gr_fine)(r_list)
#
#         return r_list, gr
#
#     except Exception as e:
#         print(f"Error in cal_expGr_fft: {e}")
#         return None, None
#
#
# def cal_expGr_fft_from_Fq(q, Fq, rmin=0, rmax=100, rstep=0.01,
#                            extrapolate_type="linear"):
#     qstep = q[1] - q[0]
#     num_point = int(np.ceil(q[0] / qstep))
#     pad_Fq = np.zeros(num_point)
#
#     if q[0] > 0.0:
#         f_inter = interpolate.interp1d(q, Fq, fill_value="extrapolate",
#                                        kind=extrapolate_type)
#         pad_Fq[:num_point] = f_inter(np.arange(num_point) * qstep)
#
#     Fq_vals = np.append(pad_Fq, Fq)
#     r_list = np.arange(rmin, rmax + rstep, rstep)
#
#     num_point = len(Fq_vals)
#     total_point = int(2 * np.pi / (rstep * qstep))
#     Fq_pad = np.pad(Fq_vals, (0, total_point - num_point), mode="constant")
#
#     norm = total_point * qstep * 2 / np.pi
#     gr_fine = norm * np.imag(np.fft.ifft(Fq_pad))
#     rfine = np.arange(total_point) * rstep
#     gr = np.interp(r_list, rfine, gr_fine)
#
#     return r_list, gr



def cal_fq(qmin, qmax, Sq, qstep=0.01):
    q_range = np.linspace(qmin, qmax, len(Sq), endpoint=False)
    return (Sq - 1) * q_range, q_range


def get_aff_scattering_factors(atom_names):
    return data.get_aff_scattering_factors(atom_names)


def get_compton_scattering_factors(atom_names):
    return data.get_compton_scattering_factors(atom_names)


def get_compton_parameter_only():
    return data.get_compton_parameter_only()


def detect_header_lines(path, min_cols=2):
    skip = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                skip += 1
                continue
            parts = stripped.split()
            if stripped.startswith("#"):
                skip += 1
                continue
            if len(parts) < min_cols:
                skip += 1
                continue
            try:
                for p in parts[:min_cols]:
                    float(p)
                break
            except ValueError:
                skip += 1
                continue
    return skip


def load_qiq_file(path, min_cols=2, usecols=(0, 1)):
    skip = detect_header_lines(path, min_cols=min_cols)
    with open(path, "rb") as f:
        raw = f.read()
    text = raw.decode("latin1")
    buf = io.StringIO(text)
    data = np.loadtxt(buf, skiprows=skip, usecols=usecols)
    return data


def smooth_whittaker(y, lambda_=1600.0, order=2):
    y = np.asarray(y, dtype=float)
    n = len(y)

    if order >= n:
        raise ValueError(f"Error: order={order} is too large for data length={n}. It must be less than {n}.")
    if lambda_ <= 0:
        raise ValueError(f"Error: lambda_ must be positive. Got {lambda_}.")

    try:
        DTD = make_dT_d(n, order)
        A = np.eye(n) + lambda_ * DTD
        c, low = cho_factor(A)
        return cho_solve((c, low), y)
    except LinAlgError as e:
        raise RuntimeError(f"Numerical error during smoothing: {e}. Consider lowering order or increasing lambda.")
    except Exception as e:
        raise RuntimeError(f"Unexpected error during smoothing: {e}. Check your lambda and order values.")


def make_dT_d(n, d):
    def diff_matrix(k, n):
        D = sp.eye(n, format='csc')
        for _ in range(k):
            D = D[1:] - D[:-1]
        return D

    D = diff_matrix(d, n)
    return D.T @ D


def bandwidth_to_lambda(bandwidth, order):
    return (2 * factorial(order)) ** 2 / (bandwidth ** (2 * order))


def smooth_like_savitzky_golay(y, bandwidth, order):
    lambda_ = bandwidth_to_lambda(bandwidth, order)
    return smooth_whittaker(y, lambda_, order)


def noise_gain_to_lambda(gain, order):
    return (2 * factorial(order)) ** 2 / gain


def smooth_with_noise_gain(y, gain=1e-4, order=2):
    lambda_ = noise_gain_to_lambda(gain, order)
    return smooth_whittaker(y, lambda_, order)


def batch_smooth_whittaker(y_2d, lambda_=1600.0, order=2):
    return np.array([smooth_whittaker(row, lambda_, order) for row in y_2d])