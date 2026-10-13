"""项目版本信息
所有版本号统一在这里修改
"""

__version__ = "1.0.0"
__version_tuple__ = (1, 0, 0)
__author__ = "你的名字"
__email__ = "your@email.com"
__license__ = "MIT"

APP_NAME = "LST"
APP_NAME_CN = "实时屏幕翻译"

REPO_URL = "https://github.com/twelwel/LST"
RELEASE_URL = REPO_URL + "/releases"

VERSION_INFO = {
    "1.0.0": "首个正式版本：实时屏幕 OCR + 翻译",
}


def get_version():
    return __version__


def get_full_name():
    return f"{APP_NAME} v{__version__}"


def get_window_title():
    return f"{APP_NAME_CN} v{__version__}"