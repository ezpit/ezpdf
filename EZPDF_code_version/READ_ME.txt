Install / 설치
Replace proc/processing.py and examples/demo_xyz_model_0731_2026.py
at matching paths in your latest EZPDF code installation.
최신 코드 폴더의 동일 경로에 두 파일을 덮어쓰십시오.
This update is based on the cleaned FFT-argument version.
미사용 FFT 인자를 제거한 최신 버전을 기준으로 합니다.

Example settings / 예제 상단 설정
IQ_METHOD = "vectorized"  # new / 새 방식
IQ_METHOD = "original"    # existing / 기존 방식
PAIR_CHUNK_SIZE = 4096
Q_CHUNK_SIZE = 64

Original: loop over Q, form N x N scattering/sine matrices; NumPy already
vectorizes matrix operations within each Q. cal_Sq calculates I(Q) internally.
기존 방식도 각 Q에서 행렬 연산은 벡터화하지만 Q와 원자 할당은 반복문입니다.
New: five-Gaussian form factors evaluated over Q arrays; self term explicitly
summed; unique pairs i<j processed in bounded pair/Q blocks and doubled.
새 방식은 Q 벡터 산란인자, 자기산란항, i<j 쌍 블록 벡터 연산을 사용합니다.
I(Q) = sum_i f_i(Q)^2 + 2 sum_i<j f_i(Q)f_j(Q) sin(Q*dij)/(Q*dij).
S(Q) = 1 + [I(Q)-N*<f^2>]/[N*<f>^2].
F(Q) = Q*[S(Q)-1].
Q=0: I(0)=[sum_i f_i(0)]^2. Self scattering is retained.
No new B factors, cutoffs or histogram approximations.
Q=0과 자기산란을 포함하며 B 인자/거리 cutoff/히스토그램 근사는 추가하지 않습니다.
Coincident distinct atoms use sinc(0)=1; old code may give NaN for these.
동일 위치 원자는 새 방식에서 sinc(0)=1로 처리합니다.

Direct function calls / 직접 함수 호출
q, Iq = proc.cal_Iq_vectorized(atom_indices, scattering_factors,
    atom_distance_matrix, qmin=0, qmax=30, qstep=0.01)
q, Iq, Sq, Fq, mean_sq_fi, sq_mean_fi = proc.cal_Sq_vectorized(
    atom_indices, scattering_factors, atom_distance_matrix,
    qmin=0, qmax=30, qstep=0.01, return_Iq=True)

Original cal_Iq/cal_Sq and all other existing functions/comments retained.
FFT, integral G(r), file paths and output filenames remain unchanged.
기존 함수/주석은 그대로 유지했습니다. 예제의 방식 선택 부분만 추가했습니다.
This is CPU NumPy vectorization, NOT GPU.
All-Q all-pair tensor is avoided, but N x N distances and O(N^2) pair arrays
still occupy memory. Chunk memory grows with PAIR_CHUNK_SIZE*Q_CHUNK_SIZE;
smaller chunks reduce temporary memory, but do not guarantee faster execution.
CPU NumPy 방식이며 GPU가 아닙니다. 거리 행렬 메모리는 여전히 필요합니다.
For less temporary memory, try PAIR_CHUNK_SIZE=1024, Q_CHUNK_SIZE=32.

See VALIDATION.txt: synthetic structures and actual bundled form factors.
Timings are single local measurements, not a universal performance guarantee.
테스트 속도는 이 환경의 단일 측정입니다. 실제 모델/PC 속도는 달라집니다.
Full XYZ example was not run with your Windows XYZ file here.
사용자 Windows XYZ 파일로 예제 전체 실행은 수행하지 않았습니다.
