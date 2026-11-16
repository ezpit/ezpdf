EZPDF code version: Pavol FFT update / 코드 버전 FFT 업데이트

1. Changed / 변경
proc/processing.py:
  cal_Gr_fft, cal_expGr_fft, cal_expGr_fft_from_Fq only.
  Original names and argument signatures retained.
  함수명과 인자 순서를 유지했습니다. 기존 함수 본문은 주석으로 보존했습니다.
  I(Q), S(Q), atomic form factors, integral functions and input paths unchanged.
  I(Q), S(Q), 원자 산란인자, 기존 적분 함수, 입력 경로는 수정하지 않았습니다.

examples/demo_experimental_data_testingcode_0731_2026.py:
  Old decay/constant comparisons commented; zero-Pavol calculation active.
  Updated padding plots and optional full input CSV export.
  Low-Q no longer extrapolated; old explanatory sections retained as LEGACY.
  기존 decay/constant 비교는 주석 보존. 패딩 그래프 및 CSV 예제 수정.

examples/demo_xyz_model_0731_2026.py:
  FFT explanations and optional padded CSV example added.
  Existing data paths and output filenames retained (including legacy "ifft" names).
  설명과 패딩 저장 예제 추가. 경로와 기존 출력 파일명은 그대로입니다.

examples/demo_pavol_fft_padding.py:
  NEW runnable demonstration with synthetic data or your FQ/SQ file.
  새 독립 예제: 합성 데이터 또는 사용자 FQ/SQ 입력.

Other source/data files unchanged. Python cache files omitted from archive.
나머지 소스/테이블은 그대로이며 재생성되는 __pycache__는 ZIP에서 제외했습니다.

2. Quick start / 빠른 시작
Dependencies: numpy scipy; plotting: matplotlib.
From the extracted folder / 압축 해제 폴더에서:
  python examples/demo_pavol_fft_padding.py
  python examples/demo_pavol_fft_padding.py --save-padding --plot
  python examples/demo_pavol_fft_padding.py --input "D:/data/sample.fq" --skiprows 1 --save-padding
  python examples/demo_pavol_fft_padding.py --input "D:/data/sample.sq" --kind sq --skiprows 1
  python examples/demo_pavol_fft_padding.py --rstep 0.001 --out my_output

--skiprows: actual number of header lines / 실제 헤더 줄 수.
--delimiter ",": use for CSV / CSV 입력시 지정.
No input means SYNTHETIC data, not measured or XYZ-derived data.
입력을 생략한 합성 예제는 실험/XYZ 검증 결과로 해석하지 마십시오.
Default output: pavol_example_output/Gr.csv.
--save-padding also writes Fq_padded.csv. Same-name files are overwritten.
--save-padding을 지정하면 패딩 CSV도 저장합니다. 같은 이름은 덮어씁니다.
Existing examples need their original local data paths to exist.
기존 예제는 원래 지정된 데이터 파일이 사용자 PC에 있어야 합니다.

3. Optional export inside processing.py / 함수 내부 선택적 저장
Search "BEGIN SAVE_PADDED_FQ".
Uncomment ONLY the 9 code lines between BEGIN and END, retaining indentation.
BEGIN과 END 사이 코드 9줄의 앞쪽 "# "만 제거하고 들여쓰기는 유지합니다.
Default save_path="Fq_padded.csv" is relative to the process working folder.
Change to an absolute path in an existing folder if needed.
상대 경로는 실행 작업 폴더 기준입니다. 필요시 존재하는 폴더의 절대 경로로 수정합니다.
Each transform call overwrites that file. Raw/smoothed/Lorch calls may follow
one another: the last call wins. Use the example's explicit export to choose
one particular result. Disable export after use to avoid large repeated writes.
매 변환마다 덮어씁니다. 원본/스무딩/Lorch 중 마지막 호출이 남습니다.
특정 결과만 저장하려면 예제의 return_padding 방식으로 저장하십시오.

4. Mathematics / 수학 설명
F(Q) = Q [S(Q)-1].
G(r) = (2/pi) integral F(Q) sin(Q*r) dQ.
Measured Q range is defined by first and last Q input.
Outside that interval the transform input is set to zero.
측정 구간 밖의 입력을 0으로 두는 계산 선택이며 실제 산란이 0이라는 뜻은 아닙니다.

Input is interpolated onto Q[k]=k*dQ, starting exactly at Q=0.
dQ starts at median(diff(Q_input))/4, with the original r-domain guard.
원래 Q 좌표를 반영하여 균일 격자로 보간하며 저Q 외삽은 하지 않습니다.
No new measured information is created by interpolation or zero padding.
보간과 0 패딩은 실험 정보나 물리적 분해능을 추가하지 않습니다.

Odd extension, M total samples:
  odd[0]=0, odd[k]=F(k*dQ), odd[M-k]=-F(k*dQ).
  Measured bands plus low-Q and middle high-Q zeros are included.
F(-Q)=-F(Q). The negative-Q data is an odd image, not measured negative Q.
음의 Q는 측정값이 아니라 FFT를 위한 홀수 대칭 확장입니다.

NumPy forward FFT uses exp(-2*pi*i*j*k/M):
  G_j = -(dQ/pi) * Imag(FFT(odd)[j])
  r_j = j*2*pi/(M*dQ).
This code uses rFFT, which gives the same required bins for real odd input.
An equivalent IFFT expression on the SAME array is:
  G_j = (M*dQ/pi)*Imag(IFFT(odd)[j]).
기존 양의 Q 전용 IFFT 정규화를 홀수 확장 배열에 그대로 적용하면 안 됩니다.

M is the next power of two sufficient for the odd data and requested
internal r sampling. Power-of-two is an implementation choice, not a
mathematical requirement. CubicSpline maps internal r to requested r.
M은 홀수 확장과 내부 r 간격 조건을 만족하는 다음 2의 거듭제곱입니다.
수학적으로 필수는 아니며 이번 업데이트에서는 이 길이 정책을 유지합니다.
출력 r 격자는 내부 FFT 격자와 다를 수 있으며 보간하여 반환합니다.

cal_Gr_fft preserves Qdamp:
  G_out(r) = G_fft(r)*exp(-0.5*(r*qdamp)^2).
qdamp in 1/Angstrom. Default zero disables damping.
Qdamp는 G(r)에만 적용되며 저장 F(Q)에는 포함되지 않습니다.

low_q_mode="anchor" and extrapolate_type="linear" are UNUSED compatibility
arguments; low-Q is always zero. Only pad_mode="zero" is now supported.
두 외삽 인자의 기본 문자열은 호환을 위한 것이며 실제 외삽하지 않습니다.
For Lorch: apply_lorch_function(q,Fq) BEFORE passing Fq to the FFT wrapper.
Lorch가 필요하면 기존 apply_lorch_function으로 먼저 적용합니다.

5. CSV and plots / 저장 및 그래프
Columns: Q (1/Angstrom), F_padded (1/Angstrom).
Signed Q uses FFT order: 0, positive Q, negative Q.
The negative side includes the zero Nyquist bin.
Keep that order for FFT. Sort a COPY by Q for plotting only.
그림용 복사본만 정렬합니다. FFT 입력을 정렬하면 잘못된 결과가 됩니다.
Zoom to around +/- measured Qmax to see the signal; full high-Q zeros can
extend very far. Millions of CSV rows can be slow and exceed Excel limits.
측정 Qmax 주변으로 확대하면 신호를 쉽게 볼 수 있습니다.

6. Validation / 검증
Passed: all source syntax; exact argument signatures; unchanged non-FFT
functions; S(Q)/F(Q) wrappers; Qdamp; odd symmetry; zero bands; FFT vs rFFT;
independent sine sum on the SAME FFT grid; runnable synthetic example and
CSV export.
기존 실험/XYZ 예제 전체 실행은 사용자 PC 전용 원본 데이터 경로가 없어
수행하지 않았습니다. 적분 함수 자체는 기존 코드를 유지하므로 다른
적분 격자/끝점 처리에 따른 차이를 FFT 오류로 단정하면 안 됩니다.

7. Added direct integral / 직접 적분 추가
cal_expGr_integral(q, Sq_or_Fq, rmin=0, rmax=100, rstep=0.01,
                   is_Fq=False, use_lorch=False)
S(Q) input: is_Fq=False; F(Q) input: is_Fq=True.
Original Q trapezoidal weights, vectorized in bounded-memory r chunks.
원본 Q 사다리꼴 가중치와 r 구간별 벡터 연산을 사용합니다.
Supports nonuniform increasing Q. No padding, no Q interpolation.
비균일 증가 Q도 지원하며 패딩/외삽/Q 재보간은 하지 않습니다.
Lorch optional; do not apply twice. No Qdamp in this new function.
새 함수는 Qdamp를 적용하지 않습니다. Lorch 중복 적용은 피하십시오.
Example: python examples/demo_expGr_integral.py
Output: integral_example_output/Gr_integral_vs_fft.csv
Existing FFT and cal_Gr_integral code unchanged by this addition.
이번 추가에서 기존 FFT와 cal_Gr_integral 코드는 변경하지 않았습니다.
