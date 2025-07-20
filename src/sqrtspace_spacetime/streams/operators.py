"""
Stream operators for transformation.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Iterable, Iterator, List, TypeVar, Optional

T = TypeVar('T')
U = TypeVar('U')


class StreamOperator(ABC):
    """Base class for stream operators."""
    
    @abstractmethod
    def apply(self, iterator: Iterator[T]) -> Iterator[Any]:
        """Apply operator to iterator."""
        pass


class MapOperator(StreamOperator):
    """Map each element to a new value."""
    
    def __init__(self, func: Callable[[T], U]):
        self.func = func
    
    def apply(self, iterator: Iterator[T]) -> Iterator[U]:
        for item in iterator:
            yield self.func(item)


class FilterOperator(StreamOperator):
    """Filter elements by predicate."""
    
    def __init__(self, predicate: Callable[[T], bool]):
        self.predicate = predicate
    
    def apply(self, iterator: Iterator[T]) -> Iterator[T]:
        for item in iterator:
            if self.predicate(item):
                yield item


class FlatMapOperator(StreamOperator):
    """Map each element to multiple elements."""
    
    def __init__(self, func: Callable[[T], Iterable[U]]):
        self.func = func
    
    def apply(self, iterator: Iterator[T]) -> Iterator[U]:
        for item in iterator:
            result = self.func(item)
            if hasattr(result, '__iter__'):
                yield from result
            else:
                yield result


class ChunkOperator(StreamOperator):
    """Group elements into fixed-size chunks."""
    
    def __init__(self, size: int):
        self.size = max(1, size)
    
    def apply(self, iterator: Iterator[T]) -> Iterator[List[T]]:
        chunk = []
        
        for item in iterator:
            chunk.append(item)
            
            if len(chunk) >= self.size:
                yield chunk
                chunk = []
        
        # Don't forget last chunk
        if chunk:
            yield chunk


class WindowOperator(StreamOperator):
    """Sliding window over stream."""
    
    def __init__(self, size: int, slide: int = 1):
        self.size = max(1, size)
        self.slide = max(1, slide)
    
    def apply(self, iterator: Iterator[T]) -> Iterator[List[T]]:
        window = []
        
        for item in iterator:
            window.append(item)
            
            if len(window) >= self.size:
                yield window.copy()
                
                # Slide window
                for _ in range(min(self.slide, len(window))):
                    window.pop(0)


class TakeWhileOperator(StreamOperator):
    """Take elements while predicate is true."""
    
    def __init__(self, predicate: Callable[[T], bool]):
        self.predicate = predicate
    
    def apply(self, iterator: Iterator[T]) -> Iterator[T]:
        for item in iterator:
            if self.predicate(item):
                yield item
            else:
                break


class DropWhileOperator(StreamOperator):
    """Drop elements while predicate is true."""
    
    def __init__(self, predicate: Callable[[T], bool]):
        self.predicate = predicate
        self.dropping = True
    
    def apply(self, iterator: Iterator[T]) -> Iterator[T]:
        for item in iterator:
            if self.dropping and self.predicate(item):
                continue
            else:
                self.dropping = False
                yield item


class DistinctOperator(StreamOperator):
    """Remove duplicate elements."""
    
    def __init__(self, key_func: Optional[Callable[[T], Any]] = None):
        self.key_func = key_func or (lambda x: x)
    
    def apply(self, iterator: Iterator[T]) -> Iterator[T]:
        seen = set()
        
        for item in iterator:
            key = self.key_func(item)
            if key not in seen:
                seen.add(key)
                yield item


class TakeOperator(StreamOperator):
    """Take first n elements."""
    
    def __init__(self, n: int):
        self.n = n
    
    def apply(self, iterator: Iterator[T]) -> Iterator[T]:
        for i, item in enumerate(iterator):
            if i >= self.n:
                break
            yield item


class SkipOperator(StreamOperator):
    """Skip first n elements."""
    
    def __init__(self, n: int):
        self.n = n
    
    def apply(self, iterator: Iterator[T]) -> Iterator[T]:
        for i, item in enumerate(iterator):
            if i >= self.n:
                yield item