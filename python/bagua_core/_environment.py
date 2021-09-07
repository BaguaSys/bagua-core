import ctypes
import os


def _preload_libraries():
    cwd = os.path.dirname(os.path.abspath(__file__))
    libnccl_path = os.path.join(cwd, ".data", "lib", "libnccl.so")
    ctypes.CDLL(libnccl_path)

    # Load bagua-net
    if os.environ.get("ENABLE_BAGUA_NET", "0") == "1":
        ctypes.CDLL(os.path.join(cwd, ".data", "lib", "libbagua_net.so"))
        ctypes.CDLL(os.path.join(cwd, ".data", "lib", "libnccl-net.so"))
