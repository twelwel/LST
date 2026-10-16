"""翻译模型抽象基类"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple


class BaseTranslationModel(ABC):
    """所有本地翻译模型的统一接口"""

    @property
    @abstractmethod
    def name(self) -> str:
        """模型显示名"""
        pass

    @property
    @abstractmethod
    def size_mb(self) -> float:
        """模型大小（MB）"""
        pass

    @property
    @abstractmethod
    def supported_languages(self) -> List[Tuple[str, str]]:
        """支持的语言对 [(src, tgt), ...]"""
        pass

    @abstractmethod
    def load(self, model_path: Path) -> bool:
        """加载模型，返回是否成功"""
        pass

    @abstractmethod
    def translate(self, text: str,
                  src_lang: str = 'auto',
                  tgt_lang: str = 'zh') -> str:
        """翻译，返回译文"""
        pass

    @abstractmethod
    def unload(self) -> None:
        """卸载模型，释放内存"""
        pass

    @property
    def is_loaded(self) -> bool:
        """是否已加载"""
        return False

    def supports(self, src: str, tgt: str) -> bool:
        """是否支持该语言对"""
        if src == 'auto':
            return any(t == tgt for _, t in self.supported_languages)
        return (src, tgt) in self.supported_languages