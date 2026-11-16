# ----------------------------------------------------------------------------------
# [EN] Path setup: add the EZPDF_code_version folder (parent of 'examples') to
#      sys.path so 'losa' and 'proc' packages can be imported regardless of the
#      current working directory.
# [KR] 경로 설정: 'examples'의 상위 폴더(EZPDF_code_version)를 sys.path에 추가하여
#      실행 위치와 무관하게 'losa', 'proc' 패키지를 import할 수 있게 합니다.
# ----------------------------------------------------------------------------------
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # [EN] Library for plotting graphs / [KR] 그래프를 그리기 위한 라이브러리
import losa.loadersaver as losa  # [EN] Custom module for loading/saving data / [KR] 데이터 로드 및 저장을 위한 사용자 정의 모듈
import proc.processing as proc   # [EN] Custom module for scientific calculations / [KR] 과학적 계산(S(q), G(r) 등)을 위한 사용자 정의 모듈
import timeit                    # [EN] Tool to measure execution time / [KR] 코드 실행 시간을 측정하는 도구
import numpy as np               # [EN] Library for numerical array processing / [KR] 수치 계산 및 배열 처리를 위한 라이브러리

# EN: Select the Debye engine. Neither option changes FFT/integral G(r).
# KO: Debye 계산 방식 선택. G(r)의 FFT/적분 선택과는 별개입니다.
IQ_METHOD = "original" #"vectorized"  # "original" or "vectorized"

# EN: Smaller blocks reduce temporary memory. Do not allocate all pairs x Q.
# KO: 작은 블록은 임시 메모리를 줄입니다. 모든 원자쌍 x Q를 한 번에 만들지 않습니다.
# Chunk sizes control how much work is processed at once, not the total
# number of atoms, atom pairs, or Q points included in the calculation.
# 청크 크기는 한 번에 처리하는 계산량을 정하며, 계산에 포함되는 전체
# 원자 수, 원자쌍 수 또는 Q점 수를 제한하지 않습니다.
#
# Keep these defaults when the number of atoms in the XYZ file changes.
# All atom pairs and Q points are processed in successive chunks;
# any smaller final chunk is handled automatically.
# XYZ 파일의 원자 수가 달라져도 아래 기본값을 그대로 사용하시면 됩니다.
# 모든 원자쌍과 Q점을 순차적으로 나누어 계산하며,
# 마지막에 남는 작은 묶음도 자동으로 처리합니다.
#
# These settings apply only when IQ_METHOD = "vectorized".
# 아래 설정은 IQ_METHOD = "vectorized"일 때만 적용됩니다.
PAIR_CHUNK_SIZE = 4096  # Maximum atom pairs per chunk / 묶음당 최대 원자쌍 수
Q_CHUNK_SIZE = 64      # Maximum Q points per chunk / 묶음당 최대 Q점 수

# Smaller chunks reduce temporary memory usage but may increase runtime.
# Q_CHUNK_SIZE does not change qstep or the Q range.
# 작은 청크는 임시 메모리 사용량을 줄이지만 계산 시간이 늘어날 수 있습니다.
# Q_CHUNK_SIZE는 qstep이나 Q 범위를 변경하지 않습니다.
#
# The current implementation still stores the full atom-distance matrix;
# chunking does not reduce the memory needed for that matrix.
# 현재 구현은 전체 원자 간 거리 행렬을 메모리에 저장하므로,
# 청크 크기를 줄여도 해당 행렬에 필요한 메모리는 줄어들지 않습니다.

# [EN] Define file paths (String)
# [KR] 파일 경로 설정 (문자열 변수)
input_base = "D:/1-Manuscript_2014/EZPDF_EZPIT/xyz_model_5IrC_r5a-1Ir/"

# [EN] Input files: G(r) experimental data (for comparison) and XYZ structure file
# [KR] 입력 파일: 비교용 G(r) 데이터와 이론적 계산을 위한 구조 파일(.xyz)
input_exp_file = input_base + '/5IrC_r5a-1Ir.calgr'
atom_xyz_file = input_base + "/5IrC_r5a-1Ir.xyz" #5IrC_r5a-1Ir.xyz"  #Ni(OH)2-109391-ICSD-10x10x1.xyz"

# [EN] Database files for atomic scattering factors
# [KR] 원자 산란 인자 정보를 담은 데이터베이스 파일 경로
# Previous paths / 이전 경로 (참고용으로 보존):
# aff_element_file = input_base + '/aff_elementonly.txt'
# aff_parm_file = input_base + 'aff_parmonly.txt'
# EN: Atomic tables are in the code package's data folder, not the XYZ folder.
#     Resolve relative to this script so PyCharm's working directory does not matter.
# KR: 원자 산란인자 테이블은 XYZ 폴더가 아니라 코드의 data 폴더에 있습니다.
#     현재 파일 위치 기준으로 찾으므로 PyCharm 작업 폴더 설정에 영향받지 않습니다.
#     XYZ 입력과 결과 저장용 input_base는 그대로 유지합니다.
database_base = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
aff_element_file = os.path.join(database_base, 'aff_elementonly.txt')
aff_parm_file = os.path.join(database_base, 'aff_parmonly.txt')

# [EN] Load atomic database FIRST (element/ion names + scattering parameters),
#      so the element/ion list can validate the .xyz species (including IONS).
# [KR] 원자 데이터베이스를 먼저 로드합니다 (원소/이온 기호 + 산란 파라미터).
#      이 원소/이온 목록으로 .xyz의 화학종을 검증합니다 (이온 포함).
database_atom_names = losa.load_atom_names(aff_element_file)
database_scat_factors = losa.load_scattering_factors(aff_parm_file)

# [EN] Load atom names and (x,y,z) positions from .xyz file.
#      Passing database_atom_names as valid_symbols lets the loader recognise
#      IONS present in the form-factor table (e.g. 'Fe2+', 'O2-', 'Cl1-'), and
#      auto-skip any header/comment lines.
# [KR] .xyz 파일에서 원자 이름 리스트와 (x,y,z) 좌표 배열을 불러옵니다.
#      database_atom_names를 valid_symbols로 넘기면 form-factor 테이블에 있는
#      이온('Fe2+', 'O2-', 'Cl1-' 등)도 인식하고 헤더/주석 줄은 자동으로 건너뜁니다.
# atom_names: List of strings ['Ir', 'Ir', ...] (or ions like ['Fe2+', ...])
# atom_positions: Numpy array (N rows, 3 columns)
atom_names, atom_positions = losa.load_atom_name_positions(
    atom_xyz_file, database_atom_names)
#print(atom_names)

#import sys
#sys.exit(0)

# print(atom_names)
# print(atom_positions)
# print(len(database_atom_names))
# print(database_scat_factors.shape)

# [EN] Calculation Parameters (Simulation settings)
# [KR] 계산 파라미터 설정 (시뮬레이션 환경 설정)
qmin = 0.0          # [EN] Minimum Q value (1/A) / [KR] Q 최소값
qmax = 30           # [EN] Maximum Q value (1/A) / [KR] Q 최대값
qstep = 0.01       # [EN] Step size for Q / [KR] Q 간격 (작을수록 정밀)
rmin = 0            # [EN] Minimum r value (A) / [KR] 거리 r 최소값
rmax = 30           # [EN] Maximum r value (A) / [KR] 거리 r 최대값
rstep = 0.001       # [EN] Step size for r / [KR] 거리 r 간격
qdamp = 0.0         # [EN] Resolution damping factor / [KR] 기기 해상도 감쇠 인자 (이론 계산시 보통 0)
#method = 'ifft'

# [EN] Start timer to measure total performance
# [KR] 전체 성능 측정을 위해 타이머 시작
t5 = timeit.default_timer()

# [EN] Calculate pairwise distance matrix between all atoms
# [KR] 모든 원자 쌍 사이의 거리 행렬을 계산합니다. (Debye 공식을 위해 필수)
# Output: 2D Numpy array (N x N)
atom_distance_matrix = proc.create_atom_distance_matrix(atom_positions)

# [EN] Group atoms by element type
# [KR] 원자들을 종류별로 분류하고 개수를 셉니다.
atom_unique_names, atom_counts, atom_indices = losa.group_atoms(atom_names)

# [EN] Retrieve scattering factors for the atoms in the structure
# [KR] 구조에 포함된 원자들의 산란 인자 계수를 가져옵니다.
scattering_factors = losa.get_scattering_factors(atom_unique_names, database_atom_names,
                                                database_scat_factors)

# [EN] Calculate theoretical I(q), S(q), F(q) using the Debye scattering equation
# [KR] Debye 산란 공식을 사용하여 이론적인 I(q), S(q), F(q)를 계산합니다.
# q: x-axis, Iq: Scattering Intensity, Sq: Structure Factor
# Previous call / 이전 호출 (참고용으로 보존):
# q, Iq, Sq, Fq, mean_sq_fi, sq_mean_fi = proc.cal_Sq(atom_indices, scattering_factors, atom_distance_matrix,
#                                     qmin=qmin, qmax=qmax, qstep=qstep, return_Iq=True)
# [EN] Time only the selected Debye I(Q), S(Q), F(Q) calculation below.
#      Excludes file loading, distance-matrix construction, G(r), plots and saving.
# [KR] 아래에서 선택한 Debye I(Q), S(Q), F(Q) 계산 구간만 측정합니다.
#      파일 로딩, 거리 행렬 생성, G(r), 그래프, 저장 시간은 포함하지 않습니다.
debye_start = timeit.default_timer()
if IQ_METHOD == "original":
    q, Iq, Sq, Fq, mean_sq_fi, sq_mean_fi = proc.cal_Sq(
        atom_indices, scattering_factors, atom_distance_matrix,
        qmin=qmin, qmax=qmax, qstep=qstep, return_Iq=True)
elif IQ_METHOD == "vectorized":
    q, Iq, Sq, Fq, mean_sq_fi, sq_mean_fi = proc.cal_Sq_vectorized(
        atom_indices, scattering_factors, atom_distance_matrix,
        qmin=qmin, qmax=qmax, qstep=qstep, return_Iq=True,
        pair_chunk_size=PAIR_CHUNK_SIZE, q_chunk_size=Q_CHUNK_SIZE)
else:
    raise ValueError('IQ_METHOD must be "original" or "vectorized"')
debye_elapsed = timeit.default_timer() - debye_start
print(f"Debye I(Q)/S(Q)/F(Q) only [{IQ_METHOD}]: {debye_elapsed:.6f} s")
print("Debye I(Q) method:", IQ_METHOD)


# print(qIq.shape)

# ----------------------------------------------------------------------------------
# Plotting I(q) - Intensity
# ----------------------------------------------------------------------------------
plt.figure(0)
plt.plot(q, Iq, label='I(q)')

# [EN] Save calculated I(q) to a text file
# [KR] 계산된 I(q) 데이터를 텍스트 파일로 저장합니다.
np.savetxt(input_base + '5IrC_r5a-1Ir_integral_qmin0p5.iq', np.column_stack(([q, Iq])))

#plt.plot(q, sq_mean_fi, label='sq_mean_fi')
#plt.plot(q, mean_sq_fi, label='mean_sq_fi')
plt.xlabel('q (1/A)')
plt.ylabel('I(q)')
#plt.xscale("log")
plt.yscale("log") # [EN] Log scale for Y-axis / [KR] Y축을 로그 스케일로 설정
plt.grid()
plt.legend()

# ----------------------------------------------------------------------------------
# Plotting S(q) - Structure Factor
# ----------------------------------------------------------------------------------
plt.figure(1)
plt.plot(q, Sq, label='S(q)')
plt.xlabel('q (1/A)')
plt.ylabel('S(q)')
plt.grid()
plt.legend()

# ----------------------------------------------------------------------------------
# Plotting F(q) - Reduced Structure Function
# ----------------------------------------------------------------------------------
plt.figure(2)
plt.plot(q, Fq, label='F(q)')
plt.xlabel('q (1/A)')
plt.ylabel('F(q)')
plt.grid()
plt.legend()

# ----------------------------------------------------------------------------------
# Method 1: Integral Transform (Direct numerical sum)
# [KR] 방법 1: 적분 변환 (수치 적분, 격자 오차 있음)
# ----------------------------------------------------------------------------------
t0 = timeit.default_timer() # [EN] Start timer for integral method / [KR] 적분 방식 타이머 시작

# [EN] Calculate G(r) using sine integral transform
# [KR] 사인 적분 변환을 사용하여 G(r) 계산
r, Gr = proc.cal_Gr_integral(q, Sq, rmin=rmin, rmax=rmax, rstep=rstep, qdamp=qdamp)

# [EN] Save Integral G(r) result
# [KR] 적분법으로 계산된 G(r) 저장
np.savetxt(input_base + '5IrC_r5a-1Ir_integral_qmin0p5.gr', np.column_stack(([r, Gr])))   # or use "list(zip(r, Gr)))"

t1 = timeit.default_timer() # [EN] Stop timer / [KR] 타이머 종료
print("Time cost real space!!! ", t1-t0)


plt.figure(3)
plt.plot(r, Gr, label='integral') # [EN] Plot Integral result / [KR] 적분 결과 플롯
plt.xlabel('r (A)')
plt.ylabel('G(r)')
plt.legend()

# ----------------------------------------------------------------------------------
# Method 2: Pavol odd FFT / 홀수 대칭 FFT
# [KR] 방법 2: 고속 푸리에 변환 (IFFT) (속도가 빠름)
# ----------------------------------------------------------------------------------
t0 = timeit.default_timer() # [EN] Start timer for IFFT method / [KR] FFT 방식 타이머 시작

# [EN] Calculate G(r) using Pavol FFT. Low-Q is ZERO; extrapolate_type is UNUSED.
# [KR] Pavol FFT: 저Q는 0입니다. extrapolate_type은 호환용 미사용 인자입니다.
r, Gr2 = proc.cal_Gr_fft(q, Sq, rmin=rmin, rmax=rmax, rstep=rstep, qdamp=qdamp)

# [EN] Save IFFT G(r) result
# [KR] IFFT로 계산된 G(r) 저장
np.savetxt(input_base + '5IrC_r5a-1Ir_ifft_qmin0p5.gr', np.column_stack(([r, Gr2])))   # or use "list(zip(r, Gr)))"

t1 = timeit.default_timer()
print("Time cost Fourier space!!! ", t1-t0)
t6 = timeit.default_timer()

print("Total time cost including real and Fourier space!!! ", t6-t5)

plt.plot(r, Gr2, label='Pavol FFT') # [EN] Plot IFFT result / [KR] IFFT 결과 플롯
plt.xlabel('r (A)')
plt.ylabel('G(r)')
plt.grid()
plt.legend()

# ----------------------------------------------------------------------------------
# Comparison: Difference between Integral and IFFT
# [KR] 비교: 적분법과 FFT법의 차이 확인 (0에 가까울수록 좋음)
# ----------------------------------------------------------------------------------
plt.figure(4)
plt.plot(r, Gr-Gr2, label='difference')
plt.xlabel('r (A)')
plt.ylabel('G(r)-G(r)2')
plt.grid()
plt.legend()
plt.show()
# OPTIONAL / 선택: XYZ-derived S(Q) also uses the shared Pavol kernel.
# EN: Uncomment these lines to save the full padded FFT input, BEFORE Qdamp.
# KO: 아래 주석을 해제하면 XYZ의 S(Q)로부터 패딩 배열을 저장합니다.
#     Qdamp는 G(r)에 적용되므로 이 F(Q) 파일에는 포함되지 않습니다.
# _, _, pad = proc.cal_expGr_fft(q, Sq, rmin, rmax, rstep, return_padding=True)
# np.savetxt(input_base + 'xyz_Fq_padded.csv',
#            np.column_stack((pad['q_full_padded'], pad['F_full_padded'])),
#            delimiter=",", header="Q_A^-1,F_padded_A^-1", comments="", fmt="%.17g")
