from distutils.core import setup

setup(
    name="depth_anything",
    version="1.0.0",
    install_requires=[
        "gradio-imageslider",
        "gradio==4.14.0",
        "torch",
        "torchvision",
        "opencv-python",
        "huggingface-hub"
    ],
    url="none",
    packages=["depth_anything", "metric_depth"],
)
