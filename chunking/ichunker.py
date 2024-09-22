from abc import ABC, abstractmethod
from typing import List

class IChunker(ABC):

    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        pass

