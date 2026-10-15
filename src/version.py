"""项目版本信息
所有版本号统一在这里修改
"""

__version__ = "1.30.4"
__version_tuple__ = (1, 30, 4)
__author__ = "twelwel"
__email__ = "2033697674@qq.com"
__license__ = "MIT"

APP_NAME = "LST"
APP_NAME_CN = "实时屏幕翻译"

REPO_URL = "https://github.com/twelwel/LST"
RELEASE_URL = REPO_URL + "/releases"

VERSION_INFO = {
    "1.30.4": "v1.30.4：PP-OCRv6 tiny；通用/英语特化/标题过滤；"
              "位置匹配防堆叠；帧差连通域 + 分块 OCR 修复",
    "1.30.3": "v1.30.3：CUDA 支持多引擎并行；DML 单引擎稳定",
    "1.30.2": "v1.30.2：修复译文位置错乱（段落 key 加入位置信息）",
    "1.30.1": "v1.30.1：修复 API 策略同步 + 单引擎稳定模式 + CPU 引擎数可调",
    "1.30.0": "v1.30：新增小窗口模式；译文颜色全模式通用；优化自适应字号",
    "1.20.0": "v1.20：GPU 引擎分配 + 多 GPU 并行",
    "1.10.0": "v1.10：新增本地词典和快速覆盖优化",
    "1.0.0":  "首个正式版本：实时屏幕 OCR + 翻译",
}


def get_version():
    return __version__


def get_full_name():
    return f"{APP_NAME} v{__version__}"


def get_window_title():
    return f"{APP_NAME_CN} v{__version__}"