#!/usr/bin/env bash
# ============================================================
#  CokePi RK3588 — 摄像头检测后端启动脚本
#  用法: ./start_backend.sh [video|detect|all]
# ============================================================
set -e

# ── 默认配置（可通过环境变量覆盖）──────────────────────────
: "${VIDEO_DEVICE:=/dev/video20}"
: "${WS_PORT:=8765}"
: "${DETECTION_PORT:=8766}"
: "${H264_BITRATE:=8000}"           # kbps，路由器内网优先保持画质
: "${H264_GOP:=30}"                 # I帧间隔: 30帧≈1秒，降低剧烈运动时的码率尖峰
: "${RKNN_MODEL:=/home/teamhd/Downloads/ppyoloe_carrace_rk3588_official_split_int8_416.rknn}"
: "${CONF_THRESH:=0.3}"
: "${DETECTION_FPS:=30}"
: "${TELEMETRY_FPS:=20}"
: "${ROS_DOMAIN_ID:=20}"
: "${IMU_TOPIC:=/imu}"
: "${ODOM_TOPIC:=/odom_raw}"
: "${IMU_ACCEL_UNIT:=auto}"        # auto | g | mps2
: "${CLASS_NAMES:=bm,cjl,jsjd,jzt,lu,mtl,nc,tt,ydm,zynsx}"
: "${LOG_DIR:=/tmp/camera_detection_logs}"
: "${VIDEO_SHM_SOCKET:=/tmp/camera_video.shm}"
: "${DETECT_SHM_SOCKET:=/tmp/camera_detect.shm}"
: "${PID_FILE:=/tmp/camera_detection_unified.pid}"

# ── 云端 API 配置（可选，留空则关闭）───────────────────────────────────
# 示例（阿里云）:
#   CLOUD_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1 \
#   CLOUD_API_KEY=sk-xxxx \
#   CLOUD_API_MODEL=qwen-turbo \
#   ./start_backend.sh all
: "${CLOUD_API_URL:=}"
: "${CLOUD_API_KEY:=}"
: "${CLOUD_API_MODEL:=qwen-turbo}"
: "${CLOUD_API_TIMEOUT:=2.5}"

# 千帆视觉理解（二次确认，可选；密钥仅通过运行环境注入）
: "${UNDERSTANDING_API_URL:=https://qianfan.baidubce.com/v2}"
: "${UNDERSTANDING_API_KEY:=}"
: "${UNDERSTANDING_MODEL:=ernie-5.1}"
: "${UNDERSTANDING_TIMEOUT:=3.0}"
: "${UNDERSTANDING_CONF_MIN:=0.30}"
: "${UNDERSTANDING_CONF_MAX:=0.55}"
: "${UNDERSTANDING_COOLDOWN:=30}"

MODE=${1:-all}    # video | detect | all

mkdir -p "$LOG_DIR"

# ── source ROS 2 ────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"   # ros2_ws/

source /opt/ros/humble/setup.bash
if [ -f "$WS_ROOT/install/setup.bash" ]; then
    source "$WS_ROOT/install/setup.bash"
fi

export VIDEO_DEVICE H264_BITRATE H264_GOP WS_PORT DETECTION_PORT RKNN_MODEL CONF_THRESH CLASS_NAMES
export VIDEO_SHM_SOCKET DETECT_SHM_SOCKET
export DETECTION_FPS TELEMETRY_FPS ROS_DOMAIN_ID IMU_TOPIC ODOM_TOPIC IMU_ACCEL_UNIT
export CLOUD_API_URL CLOUD_API_KEY CLOUD_API_MODEL CLOUD_API_TIMEOUT
export UNDERSTANDING_API_URL UNDERSTANDING_API_KEY UNDERSTANDING_MODEL
export UNDERSTANDING_TIMEOUT UNDERSTANDING_CONF_MIN UNDERSTANDING_CONF_MAX
export UNDERSTANDING_COOLDOWN

# ── 检查摄像头 ───────────────────────────────────────────────
if [ ! -e "$VIDEO_DEVICE" ]; then
    echo "[ERROR] 摄像头设备 $VIDEO_DEVICE 不存在，请检查 USB 连接"
    exit 1
fi
echo "[INFO] 摄像头: $VIDEO_DEVICE"

# ── 检查模型文件 ─────────────────────────────────────────────
if [ ! -f "$RKNN_MODEL" ]; then
    echo "[WARN] RKNN 模型文件不存在: $RKNN_MODEL"
    echo "[WARN] 检测服务将无法启动，仅启动视频流服务"
    MODE=video
fi

# ── 启动函数 ─────────────────────────────────────────────────
stop_pid() {
    local pid="${1:-}"
    [ -n "$pid" ] || return 0
    kill -0 "$pid" 2>/dev/null || return 0
    kill "$pid" 2>/dev/null || true
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
        kill -0 "$pid" 2>/dev/null || return 0
        sleep 0.2
    done
    echo "[WARN] PID $pid 未及时退出，执行强制停止"
    kill -KILL "$pid" 2>/dev/null || true
}

stop_existing_unified() {
    [ -f "$PID_FILE" ] || return 0
    local pid
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if [ -n "$pid" ] && [ -r "/proc/$pid/cmdline" ] && \
       tr '\0' ' ' < "/proc/$pid/cmdline" | grep -q 'camera_detection.unified_server'; then
        echo "[INFO] 停止旧统一后端 PID=$pid"
        stop_pid "$pid"
    fi
    rm -f "$PID_FILE"
}

start_all() {
    echo "[INFO] 启动统一后端 (single capture + video + detection)"
    echo "[INFO]   视频流  ws://0.0.0.0:${WS_PORT}"
    echo "[INFO]   检测    ws://0.0.0.0:${DETECTION_PORT}"

    stop_existing_unified
    PYTHONPATH="$SCRIPT_DIR/..:${PYTHONPATH:-}" python3 -m camera_detection.unified_server \
        >> "$LOG_DIR/unified_server.log" 2>&1 &
    UNIFIED_PID=$!
    echo "$UNIFIED_PID" > "$PID_FILE"
    echo "[INFO] unified_server PID=$UNIFIED_PID  log=$LOG_DIR/unified_server.log"
}

start_video() {
    echo "[INFO] 仅启动视频流服务  ws://0.0.0.0:${WS_PORT}"
    VIDEO_SOURCE=v4l2 python3 "$SCRIPT_DIR/../camera_detection/video_server.py" \
        >> "$LOG_DIR/video_server.log" 2>&1 &
    VIDEO_PID=$!
    echo "[INFO] video_server PID=$VIDEO_PID  log=$LOG_DIR/video_server.log"
}

start_detect() {
    echo "[INFO] 仅启动检测服务  ws://0.0.0.0:${DETECTION_PORT}"
    VIDEO_SOURCE=v4l2 python3 "$SCRIPT_DIR/../camera_detection/detection_server.py" \
        >> "$LOG_DIR/detection_server.log" 2>&1 &
    DETECT_PID=$!
    echo "[INFO] detection_server PID=$DETECT_PID  log=$LOG_DIR/detection_server.log"
}

trap_exit() {
    echo ""
    echo "[INFO] 正在停止服务..."
    [ -n "${UNIFIED_PID:-}" ] && stop_pid "$UNIFIED_PID"
    [ -n "${DETECT_PID:-}" ] && stop_pid "$DETECT_PID"
    [ -n "${VIDEO_PID:-}"  ] && stop_pid "$VIDEO_PID"
    [ -n "${RELAY_PID:-}"  ] && stop_pid "$RELAY_PID"
    rm -f "$PID_FILE" "$VIDEO_SHM_SOCKET" "$DETECT_SHM_SOCKET"
    exit 0
}
trap trap_exit INT TERM

# ── 启动 ────────────────────────────────────────────────────
case "$MODE" in
    all)    start_all ;;
    video)  start_video ;;
    detect) start_detect ;;
    *)      echo "用法: $0 [all|video|detect]"; exit 1 ;;
esac

echo ""
echo "======================================================"
echo "  视频流 WebSocket : ws://<板子IP>:${WS_PORT}"
echo "  检测数据 WebSocket: ws://<板子IP>:${DETECTION_PORT}"
echo "======================================================"
echo "  Ctrl+C 停止所有服务"
echo ""

# 等待子进程
wait
