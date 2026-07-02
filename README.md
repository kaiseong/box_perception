# Box Perception

Intel RealSense D405로 노란 박스의 중심과 yaw를 추정하기 위한 녹화, replay, offline 분석 코드입니다.

현재 1차 목표는 **다양한 D405 RGB-D 상황을 재현 가능한 dataset으로 녹화**하는 것입니다. 이후 추정기는 이 녹화 데이터에서 RGB 색상/edge, 고정 table/box height, D405 depth 품질 검증을 조합해 `center + yaw + confidence`를 출력하도록 확장합니다.

## 현재 전제

- 카메라: Intel RealSense D405
- Jetson: Orin AGX
- 해상도: 기본 `1280x720`
- 박스: 크기 고정, 노란/주황색 플라스틱 박스
- 박스 중심: 화면 안에 들어온다고 가정
- 박스는 완전히 보일 수도 있고, 일부가 잘릴 수도 있고, 양쪽이 잘릴 수도 있음
- 박스 안에는 물건이 있을 수 있지만 박스 위로 튀어나오지는 않음
- 책상 높이와 박스 중심 높이는 고정값으로 둘 수 있음
- `camera-to-T5` transform은 외부 calibration 파라미터로 나중에 제공
- 허용 오차: 중심 `2 cm`, yaw `4 deg`
- 1차 구현 범위: 학습 모델 없이 RGB-D 기하 기반

## 권장 추정 방향

잘린 RGB mask의 bbox/OBB 중심은 실제 박스 중심이 아닐 수 있습니다. 그래서 최종 추정은 단순 `minAreaRect(mask)`가 아니라 아래 방향으로 가야 합니다.

```text
D405 RGB
  -> yellow/orange mask 및 edge/rim 후보
  -> image ray를 고정 table/box height plane에 투영
  -> 실제 박스 크기의 rectangle model을 부분 관측에 fitting
  -> center, yaw, confidence, reasons 출력

D405 depth
  -> pose를 직접 오염시키는 주 입력으로 쓰지 않음
  -> depth valid ratio, median, spread, 주변 일관성으로 confidence 검증
```

D405 depth는 너무 가깝거나 멀 때, 얇은 테두리/반사/구멍/내부 물품 때문에 흔들릴 수 있습니다. 따라서 1차 추정에서는 **고정 높이 기반 center/yaw를 기본값**으로 두고, raw depth는 `depth_unreliable` 같은 reason과 confidence 조정에 사용합니다.

출력 계약은 항상 best-effort pose를 반환하는 형태가 좋습니다.

```json
{
  "center_image": [640.0, 360.0],
  "center_camera_m": [0.12, -0.03, 0.48],
  "yaw_mod_180_deg": 3.2,
  "confidence": {
    "score": 0.82,
    "meets_grasp_quality": true,
    "reasons": []
  }
}
```

## Repository Layout

```text
box_pose/
  segmentation.py       HSV 기반 노란 박스 후보 mask
  geometry.py           pixel/metric geometry helpers
  visualization.py      debug overlay

recording.py            D405 RGB/depth/intrinsics/timestamp 녹화 CLI
replay_recording.py     녹화 session replay 및 frame별 분석
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

### 2. Python venv

Jetson Ubuntu 22.04 / JetPack 6 계열에서는 apt에 `python3.12`가 없을 수 있습니다. 이 경우 **Python 3.10 venv를 권장**합니다. `python -V`가 `Python 3.13.x`로 나오면 Jetson aarch64용 `pyrealsense2` wheel이 잡히지 않을 수 있으므로 venv를 3.10으로 다시 만듭니다.

```bash
cd ~/box_perception
sudo apt-get install -y python3.10 python3.10-venv python3.10-dev

python3.10 -m venv .venv --system-site-packages
source .venv/bin/activate
python -V
python -m pip install --upgrade pip
python -m pip install "numpy<2" opencv-python
```

`python -V`는 `Python 3.10.x`여야 합니다. 이미 `.venvs` 같은 다른 이름의 venv를 쓰고 있어도 상관없지만, 그 venv가 Python 3.13이면 새 Python 3.10 venv를 만들어야 합니다. Python 3.12를 별도로 설치해 둔 환경이라면 3.12 venv도 사용할 수 있지만, apt에 없으면 3.10으로 진행하는 쪽이 가장 안정적입니다.

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
source .venv/bin/activate
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
  --duration-sec 20
```

여러 상황을 모을 때는 session name을 바꿉니다.

```bash
python recording.py --output-root recordings --session-name d405_center_visible_full_001 --duration-sec 20
python recording.py --output-root recordings --session-name d405_center_visible_left_crop_001 --duration-sec 20
python recording.py --output-root recordings --session-name d405_center_visible_both_crop_001 --duration-sec 20
python recording.py --output-root recordings --session-name d405_near_030cm_001 --duration-sec 20
python recording.py --output-root recordings --session-name d405_far_065cm_001 --duration-sec 20
```

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

## Replay

녹화 세션을 offline 환경에서 다시 분석합니다.

```bash
python replay_recording.py recordings/d405_box_static_001 \
  --max-frames 50 \
  --stride 3 \
  --debug-every 5 \
  --save-mask
```

결과:

```text
recordings/<session_name>/analysis/
  frames.jsonl
  summary.json
  debug/
  mask/
```

현재 replay는 기존 HSV segmentation과 pixel/metric geometry helper를 사용합니다. 부분 잘림까지 robust하게 처리하는 known-size rectangle fitting은 다음 구현 단계입니다.

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
python -m py_compile demo_offline.py recording.py replay_recording.py box_pose/*.py tests/*.py
python -m unittest discover -s tests -v
```

RealSense 카메라 없이도 unit tests는 돌아가야 합니다. 실제 D405 접근은 `recording.py` runtime에서만 `pyrealsense2`를 import합니다.

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

## Next Implementation Step

녹화 데이터가 모이면 다음 순서로 구현합니다.

1. D405 recording session replay에서 RGB/depth sample batch를 읽음
2. HSV mask와 edge/rim 후보를 추출
3. 카메라 intrinsics와 고정 table/box height로 image ray를 plane에 투영
4. 실제 박스 크기 rectangle을 부분 관측에 robust fitting
5. `center`, `yaw_mod_180`, `confidence`, `reasons`를 frame별로 저장
6. `2 cm`, `4 deg` 기준으로 success/fail summary 생성

학습 기반 segmentation은 1차 geometry path가 D405 데이터에서 실패하는 케이스가 확인된 뒤에 fallback으로 추가합니다.
