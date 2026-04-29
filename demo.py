# app.py
import numpy as np
import torch
import streamlit as st
from PIL import Image, ImageOps
from model import MnistNetwork

# 需要额外安装：pip install streamlit streamlit-drawable-canvas pillow numpy

st.set_page_config(page_title="MNIST 手写数字识别", layout="centered")
st.title("手写数字识别（MNIST）")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.caption(f"Using device: {device}")


@st.cache_resource
def load_model():
    model = MnistNetwork().to(device)
    state = torch.load("mnist_gpu.pth", map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


model = load_model()

# 画布：白底黑字更接近常见手写板，但 MNIST 是黑底白字
# 所以下面会做一次反色，统一成“黑底白字”再送进模型
canvas_size = st.sidebar.slider("画布大小", 200, 400, 280, step=20)
stroke_width = st.sidebar.slider("笔画粗细", 5, 40, 18, step=1)

try:
    from streamlit_drawable_canvas import st_canvas
except ImportError:
    st.error("缺少依赖：streamlit-drawable-canvas。请先执行：pip install streamlit-drawable-canvas")
    st.stop()

canvas = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=stroke_width,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=canvas_size,
    height=canvas_size,
    drawing_mode="freedraw",
    key="canvas",
)

col1, col2 = st.columns(2)
with col1:
    do_predict = st.button("识别")
with col2:
    clear = st.button("清空（刷新页面也可）")

if clear:
    st.rerun()


def preprocess_rgba_to_tensor(rgba: np.ndarray) -> torch.Tensor:
    # rgba: H x W x 4
    img = Image.fromarray(rgba.astype(np.uint8), mode="RGBA")
    img = img.convert("L")  # 灰度：0(黑)~255(白)

    # 反色：把“白底黑字”变成“黑底白字”，更像 MNIST
    img = ImageOps.invert(img)

    # 缩放到 20x20（MNIST 常见做法），再贴到 28x28 中心
    img = img.resize((20, 20), Image.Resampling.BILINEAR)
    canvas_28 = Image.new("L", (28, 28), color=0)
    canvas_28.paste(img, ((28 - 20) // 2, (28 - 20) // 2))

    # 归一化到 0~1，并形状变为 [1, 1, 28, 28]
    arr = np.array(canvas_28, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return t


if do_predict:
    if canvas.image_data is None:
        st.warning("请先在画布上写一个数字。")
        st.stop()

    x = preprocess_rgba_to_tensor(canvas.image_data).to(device)

    with torch.no_grad():
        logits = model(x)  # [1, 10]
        probs = torch.softmax(logits, dim=1)[0]  # [10]
        pred = int(torch.argmax(probs).item())
        conf = float(probs[pred].item())

    st.subheader(f"预测结果：{pred}")
    st.write(f"置信度：{conf:.4f}")

    st.caption("28x28 输入预览（已反色/居中/缩放）")
    # 显示给用户看的预处理结果
    preview = (x.detach().cpu().numpy()[0, 0] * 255).astype(np.uint8)
    st.image(preview, width=140, clamp=True)

    st.caption("各类别概率")
    st.bar_chart(probs.detach().cpu().numpy())
