# pylint: disable = C0114
import abc
from typing import Any


# pylint: disable = C0116, C0103
def with_order(cls: Any) -> Any:
    def __ne__(self: Any, b: Any) -> bool:
        return not self.__eq__(b)

    def __le__(self: Any, b: Any) -> bool:
        return bool(self.__eq__(b) or self.__lt__(b))

    def __ge__(self: Any, b: Any) -> bool:
        return not self.__lt__(b)

    def __gt__(self: Any, b: Any) -> bool:
        return not self.__le__(b)

    setattr(cls, "__ne__", __ne__)
    setattr(cls, "__le__", __le__)
    setattr(cls, "__ge__", __ge__)
    setattr(cls, "__gt__", __gt__)
    return cls


# pylint: disable = C0115
class WithOrderMixin(abc.ABC):
    @abc.abstractmethod
    def __lt__(self, b: Any) -> bool:
        pass

    def __ne__(self, b: Any) -> bool:
        return not self.__eq__(b)

    def __le__(self, b: Any) -> bool:
        return self.__eq__(b) or self.__lt__(b)

    def __ge__(self, b: Any) -> bool:
        return not self.__lt__(b)

    def __gt__(self, b: Any) -> bool:
        return not self.__le__(b)
