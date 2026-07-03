# Box Perception

Intel RealSense D405로 노란 박스의 중심과 yaw를 추정하기 위한 녹화, replay, offline 분석 코드입니다.

현재 1차 목표는 **D405 RGB-D 녹화 데이터를 replay해서 cropped box에서도 중심과 yaw를 추정**하는 것입니다. 기본 추정 경로는 학습 모델 없이 **rim 평면 위 metric 공간에서 고정 크기 사각형을 3-DOF(center x/y + yaw)로 fitting**하는 `known_size` plane estimator입니다.

## 현재 전제

- 카메라: Intel RealSense D405
- Jetson: Orin AGX
- 해상도: 기본 `1280x720`
- 박스: 크기 고정, 노란/주황색 플라스틱 박스
- 박스 크기: 긴 변 `0.505 m`, 짧은 변 `0.335 m`, 높이 `0.195 m`
- 박스 중심: 화면 안에 들어온다고 가정
- 박스는 완전히 보일 수도 있고, 일부가 잘릴 수도 있고, 양쪽이 잘릴 수도 있음
- 박스 안에는 물건이 있을 수 있지만 박스 위로 튀어나오지는 않음
- 책상 높이와 박스 중심 높이는 고정값으로 둘 수 있음
- `camera-to-T5` transform은 외부 calibration 파라미터로 나중에 제공
- 허용 오차: 중심 `2 cm`, yaw `4 deg`
- 1차 구현 범위: 학습 모델 없이 RGB-D 기하 기반

## 추정 방식

잘린 RGB mask의 bbox/OBB 중심은 실제 박스 중심이 아닙니다. 또한 카메라가 아래로 기울어 있어 가까운 rim과 먼 rim의 픽셀 스케일이 크게 다르므로, 이미지 공간에서 한 개의 depth로 고정 크기 사각형을 맞추는 방식은 center가 앞쪽으로 체계적으로 밀릴 수 있습니다. 그래서 기본 경로는 **rim 평면에 투영한 metric 공간**에서 동작합니다 (`box_pose/plane_fit.py`).

핵심 전제는 **rim 평면이 setup 상수**라는 것입니다. 책상 높이, 박스 높이, 카메라 장착이 모두 고정이고 로봇 베이스는 평면 바닥 위에서만 움직이므로, 박스 상단 rim 평면은 카메라 좌표계에서 프레임과 무관하게 거의 일정합니다.

```text
calibration (박스 top이 잘 보이는 frame, 세션당 1회)
  -> mask+depth RANSAC plane candidates
  -> footprint가 0.335 m short side와 일치하고, 위쪽에 mask 질량이 없는
     최상단 평면만 rim으로 승인 (앞벽/내부 평면 배제)
  -> 여러 frame의 승인 평면을 offset cluster로 합쳐 setup rim plane 확정

per-frame estimation
  -> HSV mask (all evidence) + depth를 calibrated rim plane에 gating
  -> gated pixel의 camera ray를 rim plane과 교차 -> 원근 왜곡 없는 metric 2D
  -> 박스 대각선(0.606 m)을 넘는 connected component 제거
  -> convex hull edge 방향 투표 + 미세 grid refine으로 yaw
  -> image border에 잘리지 않은 edge만 신뢰해 고정 크기(505x335)로 center 복원
  -> 양쪽 다 잘린 축은 *_center_underconstrained reason으로 명시
  -> center_top_camera_m(3D), yaw, short_axis, confidence, reasons 출력
```

Depth는 pose의 주 입력이 아니라 **rim 평면과의 일치 검증 + gating**에만 씁니다. 평면이 주어지면 좌표는 ray-plane 교차에서 나오므로 depth 노이즈가 pose에 직접 들어가지 않습니다. `metric` 결과는 비교용으로 남겨두지만 cropped D405 데이터에서는 `known_size`(plane 경로)가 우선입니다.

퇴화 케이스: 근접해서 **양쪽 짧은 변이 모두 화면 밖**이면 긴 축 방향 위치는 원리적으로 결정 불가입니다. 이때도 yaw와 관측 가능한 축은 출력되며, `long_axis_center_underconstrained` reason이 남으므로 로봇 측에서는 멀리서 확정한 center를 유지하거나 재관측을 요구할 수 있습니다.

`camera-to-T5` transform이 들어오면 `center_top_camera_m`을 T5로 변환해 x/y를 뽑고 z는 고정값을 씁니다. calibration된 rim plane은 T5 기준 table/box-height 상수로 대체할 수도 있습니다.

축 이름은 다음 의미입니다.

- `long_axis`: 박스의 긴 변 방향입니다. 현재 yaw는 이 축 기준으로 출력합니다.
- `short_axis`: `long_axis`에 수직인 짧은 변 방향입니다. 로봇 양손이 박스 양쪽 긴 벽을 잡으려면 end-effector들이 이 축을 따라 박스 중심 쪽으로 모입니다.
- `short_axis`는 잡는 면의 길이 방향이 아니라, 두 end-effector가 서로 접근하는 방향입니다.

frame별 주요 출력:

```json
{
  "known_size": {
    "center_image": [360.0, 720.0],
    "center_top_camera_m": [-0.01, 0.12, 0.43],
    "yaw_mod_180": 3.2,
    "yaw_frame": "image",
    "long_axis_image": [1.0, 0.0],
    "short_axis_image": [0.0, 1.0],
    "box_size_m": {"long": 0.505, "short": 0.335, "height": 0.195},
    "projection_plane": "box_rim_plane",
    "support": {"method": "rim_plane"},
    "confidence": {"ok": true, "score": 0.86, "reasons": []}
  }
}
```

## Repository Layout

```text
box_pose/
  segmentation.py       HSV 기반 노란 박스 후보 mask
  geometry.py           pixel/metric/known-size geometry helpers
  plane_fit.py          rim plane calibration + metric 공간 3-DOF 사각형 fitting (주 경로)
  visualization.py      debug overlay

recording.py            D405 RGB/depth/intrinsics/timestamp 녹화 CLI
replay_recording.py     녹화 session replay 및 frame별 분석
inference.py            D405 live 추론 + RGB overlay + stdout JSONL
demo_offline.py         단일 이미지 offline demo
tests/                  unit tests
```

`recordings/`는 `.gitignore`에 포함되어 있어 대용량 녹화 데이터가 git에 섞이지 않습니다.

## Jetson Installation

아래는 Jetson Orin AGX에서 D405 녹화를 위한 기본 설치 흐름입니다. RealSense 설치 방식은 JetPack/kernel 상태에 따라 달라질 수 있으므로, 막히면 Intel RealSense 공식 librealsense 문서를 기준으로 맞춰야 합니다.

참고한 공식 문서:

- Linux installation: https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md
- Jetson installation: https://github.com/IntelRealSense/librealsense/blob/master/doc/installation_jetson.md
- Python wrapper: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/readme.md
- Depth-to-color alignment example: https://github.com/IntelRealSense/librealsense/blob/master/wrappers/python/examples/align-depth2color.py

### 1. Repo 준비

```bash
cd ~
git clone https://github.com/kaiseong/box_perception.git
cd box_perception
```

이미 받아둔 repo라면:

```bash
cd ~/box_perception
git pull
```

### 2. Python environment

Jetson Ubuntu 22.04 / JetPack 6 계열에서는 apt에 `python3.12`가 없을 수 있습니다. conda를 쓰면 이 문제를 피할 수 있으므로, conda가 있으면 **Python 3.12 conda env**를 권장합니다. `python -V`가 `Python 3.13.x`로 나오면 Jetson aarch64용 `pyrealsense2` wheel이 잡히지 않을 수 있으므로 새 환경을 만듭니다.

Conda 경로:

```bash
cd ~/box_perception
conda create -n box-perception python=3.12 -y
conda activate box-perception
python -V

python -m pip install --upgrade pip
python -m pip install "numpy<2" pyrealsense2

# preview/replay/tests에서 OpenCV가 필요하면 설치
conda install -c conda-forge opencv -y
```

`python -V`는 `Python 3.12.x`여야 합니다. conda 환경에서 `pip install pyrealsense2`가 실패하면 먼저 `python -m pip debug --verbose | grep -m1 aarch64`로 현재 pip가 aarch64 wheel tag를 보는지 확인합니다.

Jetson에서 PyPI가 현재 조합에 맞는 wheel을 못 찾는 경우, 같은 Jetson/aarch64/Python 3.12용 wheel을 직접 설치해도 됩니다. 예를 들어 `pyrealsense2-2.56.5-cp312-cp312-linux_aarch64.whl` 파일이 있다면:

```bash
conda activate box-perception
python -V
python -m pip install /path/to/pyrealsense2-2.56.5-cp312-cp312-linux_aarch64.whl
```

이 wheel은 Python ABI와 플랫폼에 묶여 있으므로 repo에는 커밋하지 않습니다. `cp312` wheel은 Python 3.12에서만 맞고, Python 3.13 환경에서는 import 대상이 아닙니다.

Conda를 쓰지 않는 경우에는 apt에서 구할 수 있는 **Python 3.10 venv**가 가장 안정적입니다.

```bash
cd ~/box_perception
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

python3.10 -m venv .venv --system-site-packages
source .venv/bin/activate
python -V
python -m pip install --upgrade pip
python -m pip install "numpy<2" opencv-python
```

venv 경로에서는 `python -V`가 `Python 3.10.x`여야 합니다. 이미 `.venvs` 같은 다른 이름의 venv를 쓰고 있어도 상관없지만, 그 venv가 Python 3.13이면 새 Python 3.10 venv를 만들어야 합니다.

Jetson에서 apt OpenCV를 쓸 때는 `opencv-python` wheel 대신 system OpenCV가 잡히는 편이 나을 수 있습니다. 이 경우 `--system-site-packages` venv에서 아래 확인만 통과하면 됩니다.

```bash
python - <<'PY'
import sys
import cv2
import numpy as np
print("python", sys.version)
print("opencv", cv2.__version__)
print("numpy", np.__version__)
PY
```

### 3. librealsense / pyrealsense2 설치 확인

먼저 설치 여부를 확인합니다.

```bash
python - <<'PY'
import pyrealsense2 as rs
print("pyrealsense2 ok")
print(rs.context().query_devices())
PY
```

`ModuleNotFoundError: No module named 'pyrealsense2'`가 나오면 librealsense Python binding이 없는 상태입니다.

Jetson에서 apt repo에 RealSense C++ runtime은 보이지만 Python binding 패키지가 없는 경우가 있습니다. 예를 들어 아래처럼 `python3-pyrealsense2`가 보이지 않으면 정상적으로 이 케이스입니다.

```bash
apt-cache search realsense | sort
```

이 경우 apt로는 device/udev/runtime 쪽을 설치하고, venv 안에서는 공식 PyPI wheel을 먼저 설치합니다. 현재 Python 3.10/3.12 aarch64 wheel이 제공되므로 이 경로가 가장 가볍습니다. `python -V`가 `Python 3.13.x`이면 `pip install pyrealsense2`가 `No matching distribution found`로 실패할 수 있습니다.

```bash
sudo apt-get update
sudo apt-get install -y \
  librealsense2 \
  librealsense2-udev-rules \
  librealsense2-utils \
  librealsense2-dev

cd ~/box_perception
# conda 환경이면:
conda activate box-perception

# venv 환경이면:
# source .venv/bin/activate
python -V
python -m pip install pyrealsense2
```

설치 후 카메라를 다시 꽂거나 재부팅한 뒤 확인합니다.

```bash
python - <<'PY'
import pyrealsense2 as rs
ctx = rs.context()
devices = ctx.query_devices()
print("device_count", len(devices))
for dev in devices:
    print(dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number))
PY
```

만약 `pip install pyrealsense2`가 현재 Jetson/Python 조합에서 실패하거나 import는 되지만 shared library 로딩 문제가 나면 공식 문서 기준으로 source build를 사용합니다. venv를 activate한 상태에서 `-DPYTHON_EXECUTABLE="$(which python)"`를 넣어 현재 venv용 binding을 만들도록 합니다.

```bash
cd ~
sudo apt-get update
sudo apt-get install -y \
  git cmake build-essential pkg-config \
  libssl-dev libusb-1.0-0-dev libudev-dev \
  libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
  python3-dev python3-pip

git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
./scripts/setup_udev_rules.sh

mkdir -p build
cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE="$(which python)" \
  -DBUILD_EXAMPLES=ON \
  -DBUILD_GRAPHICAL_EXAMPLES=OFF
make -j"$(nproc)"
sudo make install
sudo ldconfig
```

source build 후 import가 안 보이면 Python path를 확인합니다. 공식 문서도 `/usr/local/lib` 또는 `/usr/local/lib/<python version>/pyrealsense2`를 `PYTHONPATH`에 추가하는 방식을 안내합니다.

```bash
python - <<'PY'
import sys
print("\n".join(sys.path))
PY

find /usr/local/lib -name 'pyrealsense2*.so' -o -path '*/pyrealsense2/__init__.py'
```

예를 들어 `/usr/local/lib/python3.10/pyrealsense2` 아래에 설치되어 있으면:

```bash
export PYTHONPATH="${PYTHONPATH}:/usr/local/lib/python3.10/pyrealsense2"
```

마지막으로 다시 확인합니다.

```bash
cd ~/box_perception
source .venv/bin/activate
python - <<'PY'
import pyrealsense2 as rs
ctx = rs.context()
devices = ctx.query_devices()
print("device_count", len(devices))
for dev in devices:
    print(dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number))
PY
```

GUI가 가능한 환경이면 `realsense-viewer`로 D405가 보이는지 확인합니다.

```bash
realsense-viewer
```

headless/SSH 환경에서는 GUI가 안 뜰 수 있으므로 Python import와 device enumeration만 먼저 확인하면 됩니다.

## D405 Recording

기본 녹화 명령:

```bash
cd ~/box_perception
source .venv/bin/activate

python recording.py \
  --output-root recordings \
  --session-name d405_box_static_001 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --duration-sec 20 \
  --view-rotation cw90
```

현재 D405는 카메라가 90도 돌아간 상태로 장착되어 있으므로, 사람이 보는 분석 기준 이미지는 raw frame을 **시계방향 90도 회전한 좌표계**입니다. `--view-rotation cw90`는 저장된 raw frame을 바꾸지 않고 manifest에 이 분석 방향만 기록합니다. 이미 녹화된 session에 이 metadata가 없으면 replay에서 `--view-rotation cw90`를 지정하면 됩니다.

여러 상황을 모을 때는 session name을 바꿉니다.

```bash
python recording.py --output-root recordings --session-name d405_center_001 --duration-sec 20 --view-rotation cw90
python recording.py --output-root recordings --session-name d405_left_crop_001 --duration-sec 20 --view-rotation cw90
python recording.py --output-root recordings --session-name d405_right_crop_001 --duration-sec 20 --view-rotation cw90
python recording.py --output-root recordings --session-name d405_yaw_plus_001 --duration-sec 20 --view-rotation cw90
python recording.py --output-root recordings --session-name d405_yaw_minus_001 --duration-sec 20 --view-rotation cw90
```

Session 의미:

- `center`: 박스 중심을 화면 중심 근처에 두고 로봇 정면 방향으로 앞뒤 이동
- `left_crop`: 박스를 왼쪽으로 옮긴 뒤 로봇 정면 방향으로 앞뒤 이동
- `right_crop`: 박스를 오른쪽으로 옮긴 뒤 로봇 정면 방향으로 앞뒤 이동
- `yaw_plus`: 박스를 반시계 방향으로 회전시키고, 이후 로봇 정면 방향으로 앞뒤 이동
- `yaw_minus`: 박스를 시계 방향으로 회전시키고, 이후 로봇 정면 방향으로 앞뒤 이동

기본값:

- color stream: `1280x720`, `bgr8`, `30 fps`
- depth stream: `1280x720`, `z16`, `30 fps`
- depth는 color frame에 align
- RGB 저장: `.npy`
- depth 저장: `.npz`, meter 단위 `depth_m`
- IR emitter는 가능한 경우 enabled

저장 속도가 부족하면 depth 압축 비용을 줄입니다.

```bash
python recording.py \
  --output-root recordings \
  --session-name d405_fast_depth_001 \
  --duration-sec 20 \
  --depth-format npy
```

미리보기 창이 필요한 경우:

```bash
python recording.py \
  --output-root recordings \
  --session-name d405_preview_001 \
  --duration-sec 20 \
  --preview
```

`--preview`, `--rgb-format jpg`, `--rgb-format png`는 OpenCV image IO가 필요합니다. 녹화 안정성이 우선이면 기본 `.npy`를 유지합니다.

카메라가 여러 대면 serial number를 지정합니다.

```bash
python recording.py \
  --output-root recordings \
  --session-name d405_serial_test \
  --serial-number 123456789 \
  --duration-sec 20
```

## Recording Output

세션 구조:

```text
recordings/<session_name>/
  manifest.json
  index.jsonl
  rgb/
    frame_000000.npy
    frame_000001.npy
  depth/
    frame_000000.depth.npz
    frame_000001.depth.npz
```

`manifest.json`에는 녹화 전체 metadata가 들어갑니다.

주요 필드:

- `format_version`: `box-perception-recording-v2`
- `config`: recording CLI 설정
- `camera`: RealSense device name, serial, firmware, USB 정보
- `intrinsics` / `color_intrinsics`: color camera intrinsics
- `depth_intrinsics`: depth camera intrinsics
- `depth_scale_m_per_unit`: raw z16 depth scale
- `extrinsics`: color/depth stream 간 extrinsics
- `data_layout.depth_aligned_to_color`: depth가 color frame에 align됐는지 여부
- `data_layout.saved_frame_orientation`: 저장된 frame은 raw camera 방향
- `data_layout.view_rotation_from_raw_to_analysis`: 분석/시각화 때 적용할 회전. 현재 장착은 `cw90`

`index.jsonl`은 frame별 record입니다.

주요 필드:

- `frame_id`
- `wall_time`
- `monotonic_time_sec`
- `rgb_path`
- `depth_path`
- `image_shape`
- `depth_shape`
- `depth_stats`
- `camera_metadata`

`depth_stats`는 depth 품질 판단용으로 저장됩니다.

- `valid_count`
- `total_count`
- `valid_fraction`
- `min_m`
- `max_m`
- `mean_m`
- `median_m`
- `p05_m`
- `p95_m`

## Recording 검증

녹화 후 바로 구조와 depth 값을 확인합니다.

```bash
python - <<'PY'
import json
from pathlib import Path
import numpy as np

session = Path("recordings/d405_box_static_001")
manifest = json.loads((session / "manifest.json").read_text())
records = [json.loads(line) for line in (session / "index.jsonl").read_text().splitlines()]

print("frames", len(records))
print("format", manifest["format_version"])
print("camera", manifest["camera"])
print("intrinsics", manifest["intrinsics"])
print("depth_scale", manifest["depth_scale_m_per_unit"])

first = records[0]
rgb = np.load(session / first["rgb_path"])
depth = np.load(session / first["depth_path"])["depth_m"]
finite = np.isfinite(depth) & (depth > 0)

print("first", first)
print("rgb", rgb.shape, rgb.dtype, int(rgb.min()), int(rgb.max()))
print("depth", depth.shape, depth.dtype, "valid", int(finite.sum()), "/", depth.size)
if finite.any():
    print("depth finite range", float(depth[finite].min()), float(depth[finite].max()))
PY
```

정상 기준:

- `rgb` shape이 `(720, 1280, 3)`
- `depth` shape이 `(720, 1280)`
- `depth` dtype이 `float32`
- `depth_stats.valid_fraction`이 충분히 큼
- `median_m`이 실제 카메라-책상/박스 거리와 대략 맞음

현재 측정된 작업 범위는 대략 다음 정도입니다.

- 가까운 상태: 약 `0.30 m`
- 먼 상태: 약 `0.65 m`

이 값은 로봇과 책상/박스 배치에 따라 바뀌므로 threshold로 하드코딩하지 않습니다. 대신 frame마다 depth 품질을 confidence에 반영합니다.

## Live Inference

로봇이 박스 앞 접근을 끝내고 정지한 뒤 live 추론을 실행합니다. 첫 `--init-frames` 동안 rim plane prior를 초기화하고, 이후에는 매 frame마다 `known_size` plane estimator로 중심/yaw를 추정합니다.

```bash
python inference.py \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --view-rotation cw90 \
  --init-frames 15 \
  --preview
```

기본 동작:

- depth를 color frame에 align합니다.
- raw D405 frame을 분석 좌표계로 `cw90` 회전합니다.
- startup burst에서 rim plane prior를 만듭니다.
- startup plane init이 실패해도 종료하지 않고 per-frame plane discovery로 계속 추론합니다.
- OpenCV window에는 RGB overlay를 표시합니다.
- stdout에는 frame마다 JSON 한 줄을 출력합니다. init/status/warning 로그는 JSON 파싱을 방해하지 않도록 stderr로 보냅니다.

stdout JSONL 예:

```json
{"ok": true, "center_top_camera_m": [0.01, 0.04, 0.4], "yaw_mod_180": 12.3, "long_axis_camera": [1.0, 0.0, 0.0], "short_axis_camera": [0.0, -0.7, 0.7]}
```

주요 옵션:

- `--no-preview`: GUI 없이 stdout JSONL만 출력
- `--max-frames N`: N개 frame 분석 후 종료
- `--serial-number SERIAL`: 여러 RealSense가 있을 때 카메라 지정
- `--disable-emitter`, `--laser-power VALUE`: D405 depth sensor 설정
- `--no-image-fallback`: plane fitting 실패 시 image-space fallback을 끕니다

출력 pose는 아직 camera frame 기준입니다. `camera-to-T5`, ROS, State Machine 직접 연동은 이 스크립트 범위에 포함하지 않았습니다.

## Replay

녹화 세션을 offline 환경에서 다시 분석합니다.

```bash
python replay_recording.py recordings/d405_box_static_001 \
  --max-frames 50 \
  --stride 3 \
  --debug-every 5 \
  --save-mask \
  --view-rotation cw90 \
  --box-long-m 0.505 \
  --box-short-m 0.335 \
  --box-height-m 0.195
```

`--view-rotation auto`가 기본값입니다. 새로 녹화한 session의 manifest에 `view_rotation`이 들어 있으면 자동으로 사용합니다. 기존 녹화처럼 metadata가 없으면 `--view-rotation cw90`를 명시합니다. replay는 RGB, depth, camera intrinsics를 모두 같은 방향으로 회전한 뒤 분석하므로 pixel center/yaw와 metric projection의 좌표계가 일치합니다.

replay는 먼저 세션 전체에서 rim plane calibration을 시도하고 (`Calibrated rim plane from N frames ...` 로그), 성공하면 모든 frame을 고정 평면으로 분석합니다. calibration이 실패하면 frame별 RANSAC discovery로 fallback합니다. `--known-size-method image`로 이전 이미지 공간 estimator를 강제할 수 있습니다.

결과:

```text
recordings/<session_name>/analysis/
  frames.jsonl
  summary.json
  debug/
  mask/
```

`frames.jsonl`에는 세 가지 결과가 들어갑니다.

- `pixel`: 기존 HSV dominant component + `minAreaRect` baseline. cropped frame에서는 자주 실패하는 것이 정상입니다.
- `metric`: 기존 boundary depth 기반 metric OBB baseline. D405 cropped/내용물/벽면 depth에 민감하므로 비교용입니다.
- `known_size`: 현재 주 경로(`method: "plane"`). calibrated rim plane 위 metric 공간에서 고정 크기 사각형을 fitting합니다. plane 경로가 완전히 실패한 frame은 이미지 공간 estimator로 fallback하며 `method: "image_fallback"`으로 표시됩니다.

`summary.json`에서 우선 볼 값:

- `known_size_ok_fraction`
- `known_size_yaw_mod_180`
- `known_size_center_top_camera_m_mean`
- `rim_plane_calibration` (normal/point, frames_used, offset_spread_m)

현재 로컬 D405 녹화 5개 세션을 `--stride 3 --view-rotation cw90`로 replay한 대표 결과:

```text
d405_center_visible  ok=0.65  yaw_spread=1.9 deg   (실패 사유는 대부분 양쪽 crop에 의한 long-axis 미결정)
d405_left_crop       ok=1.00  yaw_spread=2.0 deg
d405_right_crop      ok=1.00  yaw_spread=2.3 deg
d405_yaw_plus        ok=0.62  yaw_spread=21.9 deg  (회전 동작 구간 포함, 근접 구간은 long-axis 미결정)
d405_yaw_minus       ok=0.94  yaw_spread=19.5 deg
```

`yaw_plus` / `yaw_minus`의 spread는 영상 안에서 실제로 박스를 회전시키는 구간이 포함된 값이므로 큰 것이 정상입니다. 정지 yaw 세션 기준 yaw 흔들림은 2 deg 수준입니다. 앞뒤 이동 세션의 center spread는 로봇이 실제로 전후 이동한 궤적을 포함하므로, 최종 `2 cm`, `4 deg` 판정은 camera-to-T5 calibration 이후 T5 기준으로 비교해야 합니다.

debug overlay 색:

- 초록: 기존 `pixel` OBB baseline
- 자홍: `known_size` rectangle model
- 노랑 점: `known_size.center_image`
- 청록 화살표: known long axis
- 자홍 화살표: known short axis

## Offline Demo

단일 이미지 확인:

```bash
python demo_offline.py pallet_box.png
```

생성물:

- `pallet_box_mask.png`
- `pallet_box_debug.png`
- JSON 형태의 pixel OBB 결과

이 demo는 전체 박스가 잘 보이는 정지 이미지 sanity check용입니다. 부분 잘림에서의 최종 중심 추정 기준으로 쓰면 안 됩니다.

## Tests

```bash
python -m py_compile demo_offline.py recording.py replay_recording.py inference.py box_pose/*.py tests/*.py
python -m unittest discover -s tests -v
```

RealSense 카메라 없이도 unit tests는 돌아가야 합니다. 실제 D405 접근은 `recording.py`와 `inference.py` runtime에서만 `pyrealsense2`를 import합니다.

## Troubleshooting

### `ModuleNotFoundError: No module named 'pyrealsense2'`

RealSense Python binding이 현재 venv에서 보이지 않는 상태입니다.

가장 먼저 venv 안에서 PyPI wheel을 설치합니다.

```bash
source .venv/bin/activate
python -m pip install pyrealsense2
```

그리고 RealSense runtime/udev 패키지를 확인합니다.

```bash
sudo apt-get install -y librealsense2 librealsense2-udev-rules librealsense2-utils librealsense2-dev
```

확인:

```bash
which python
python -V
python - <<'PY'
import sys
print(sys.path)
PY
```

source build를 했다면 `/usr/local/lib` 설치 후 `sudo ldconfig`를 실행하고, Python binding `.so`가 설치된 경로가 현재 Python path에 잡히는지 확인합니다.

### 카메라가 안 열림

확인:

```bash
lsusb
python - <<'PY'
import pyrealsense2 as rs
ctx = rs.context()
devices = ctx.query_devices()
print("device_count", len(devices))
for dev in devices:
    print(dev.get_info(rs.camera_info.name), dev.get_info(rs.camera_info.serial_number))
PY
```

대응:

- USB3 포트/케이블 확인
- 다른 process가 카메라를 잡고 있는지 확인
- udev rules 적용 후 재연결 또는 재부팅
- `realsense-viewer`가 떠 있으면 종료

### 요청한 1280x720@30 profile이 안 열림

D405 firmware/USB 상태에 따라 profile이 제한될 수 있습니다. 우선 낮은 profile로 확인합니다.

```bash
python recording.py \
  --output-root recordings \
  --session-name d405_profile_test \
  --width 640 \
  --height 480 \
  --fps 30 \
  --duration-sec 10
```

이게 열리면 1280x720 쪽은 bandwidth/profile 문제일 가능성이 큽니다.

### depth가 너무 sparse하거나 noisy함

확인할 것:

- 박스/책상까지 거리
- IR emitter 설정
- 노출/조명
- 반사면
- 너무 가까운 거리 또는 너무 먼 거리
- depth frame이 color frame에 align되어 있는지

이 프로젝트의 1차 추정 방향은 depth를 주 pose 입력으로 신뢰하지 않고 confidence 검증용으로 쓰는 것입니다. 따라서 depth가 흔들리면 pose를 망가뜨리지 말고 confidence reason을 남기는 쪽으로 처리해야 합니다.

## Next Calibration Step

다음 단계는 perception 결과를 로봇 기준으로 검증하는 것입니다.

1. camera-to-T5 transform을 준비합니다.
2. table plane 또는 box top plane을 T5 기준으로 정의합니다.
3. `known_size.center_top_camera_m`을 T5로 변환하고, z는 작업 정책에 맞는 고정 box-center 높이를 사용합니다.
4. camera-frame long/short axis를 T5 yaw로 변환하는 기준축을 정의합니다.
5. 같은 정지 pose에서 여러 frame을 replay해 center spread가 `2 cm` 이하인지 확인합니다.
6. yaw가 고정 pose에서 `4 deg` 이하로 흔들리는지 확인합니다.

학습 기반 segmentation(YOLO 등)은 geometry path가 실제 책상/조명/장갑 조건에서 계속 실패하는 케이스가 확인된 뒤 fallback으로 추가합니다. 현재 데이터에서는 HSV evidence + calibrated rim-plane metric fitting이 우선 경로입니다.
