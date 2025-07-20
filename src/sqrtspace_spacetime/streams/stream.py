"""
Memory-efficient streaming operations.
"""

import csv
import json
import asyncio
from pathlib import Path
from typing import (
    Any, Callable, Dict, Iterable, Iterator, List, Optional, 
    TypeVar, Union, AsyncIterator, Tuple
)

from sqrtspace_spacetime.config import config
from sqrtspace_spacetime.streams.operators import (
    MapOperator, FilterOperator, FlatMapOperator, ChunkOperator,
    TakeOperator, SkipOperator
)

T = TypeVar('T')
U = TypeVar('U')


class Stream(Iterable[T]):
    """
    A lazy, memory-efficient stream for processing large datasets.
    """
    
    def __init__(self, source: Union[Iterable[T], Iterator[T], Callable[[], Iterator[T]]]):
        """
        Initialize stream.
        
        Args:
            source: Data source (iterable, iterator, or callable returning iterator)
        """
        if callable(source):
            self._source = source
        elif hasattr(source, '__iter__'):
            self._source = lambda: iter(source)
        else:
            raise TypeError("Source must be iterable or callable")
        
        self._operators: List[Any] = []
    
    def __iter__(self) -> Iterator[T]:
        """Create iterator with all operators applied."""
        iterator = self._source()
        
        # Apply operators in sequence
        for op in self._operators:
            iterator = op.apply(iterator)
        
        return iterator
    
    # Transformation operators
    
    def map(self, func: Callable[[T], U]) -> 'Stream[U]':
        """Apply function to each element."""
        new_stream = Stream(self._source)
        new_stream._operators = self._operators.copy()
        new_stream._operators.append(MapOperator(func))
        return new_stream
    
    def filter(self, predicate: Callable[[T], bool]) -> 'Stream[T]':
        """Keep only elements matching predicate."""
        new_stream = Stream(self._source)
        new_stream._operators = self._operators.copy()
        new_stream._operators.append(FilterOperator(predicate))
        return new_stream
    
    def flat_map(self, func: Callable[[T], Iterable[U]]) -> 'Stream[U]':
        """Map each element to multiple elements."""
        new_stream = Stream(self._source)
        new_stream._operators = self._operators.copy()
        new_stream._operators.append(FlatMapOperator(func))
        return new_stream
    
    def chunk(self, size: Optional[int] = None) -> 'Stream[List[T]]':
        """Group elements into chunks."""
        if size is None:
            # Use √n chunking
            # Since we don't know total size, use a reasonable default
            size = 1000
        
        new_stream = Stream(self._source)
        new_stream._operators = self._operators.copy()
        new_stream._operators.append(ChunkOperator(size))
        return new_stream
    
    def take(self, n: int) -> 'Stream[T]':
        """Take first n elements."""
        new_stream = Stream(self._source)
        new_stream._operators = self._operators.copy()
        new_stream._operators.append(TakeOperator(n))
        return new_stream
    
    def skip(self, n: int) -> 'Stream[T]':
        """Skip first n elements."""
        new_stream = Stream(self._source)
        new_stream._operators = self._operators.copy()
        new_stream._operators.append(SkipOperator(n))
        return new_stream
    
    def distinct(self) -> 'Stream[T]':
        """Remove duplicate elements."""
        def distinct_op(iterator):
            seen = set()
            for item in iterator:
                if item not in seen:
                    seen.add(item)
                    yield item
        
        new_stream = Stream(self._source)
        new_stream._operators = self._operators.copy()
        new_stream._operators.append(lambda it: distinct_op(it))
        return new_stream
    
    # Terminal operators
    
    def collect(self) -> List[T]:
        """Collect all elements into a list."""
        return list(self)
    
    def reduce(self, func: Callable[[U, T], U], initial: U) -> U:
        """Reduce stream to single value."""
        result = initial
        for item in self:
            result = func(result, item)
        return result
    
    def count(self) -> int:
        """Count elements."""
        return sum(1 for _ in self)
    
    def first(self) -> Optional[T]:
        """Get first element."""
        for item in self:
            return item
        return None
    
    def foreach(self, func: Callable[[T], None]) -> None:
        """Apply function to each element."""
        for item in self:
            func(item)
    
    def group_by(self, key_func: Callable[[T], Any]) -> Dict[Any, List[T]]:
        """Group elements by key."""
        from sqrtspace_spacetime.algorithms import external_groupby
        return external_groupby(self, key_func)
    
    def sort(self, key: Optional[Callable[[T], Any]] = None, reverse: bool = False) -> List[T]:
        """Sort elements."""
        from sqrtspace_spacetime.algorithms import external_sort_key, external_sort
        
        if key:
            return external_sort_key(self, key=key, reverse=reverse)
        else:
            return external_sort(self, reverse=reverse)
    
    def to_file(self, path: Union[str, Path], mode: str = 'w') -> None:
        """Write stream to file."""
        path = Path(path)
        
        with open(path, mode) as f:
            for item in self:
                f.write(str(item) + '\n')
    
    def to_csv(self, path: Union[str, Path], headers: Optional[List[str]] = None) -> None:
        """Write stream to CSV file."""
        path = Path(path)
        
        with open(path, 'w', newline='') as f:
            writer = None
            
            for item in self:
                if writer is None:
                    # Initialize writer based on first item
                    if isinstance(item, dict):
                        writer = csv.DictWriter(f, fieldnames=headers or item.keys())
                        if headers or item:
                            writer.writeheader()
                    else:
                        writer = csv.writer(f)
                        if headers:
                            writer.writerow(headers)
                
                if isinstance(item, dict):
                    writer.writerow(item)
                elif isinstance(item, (list, tuple)):
                    writer.writerow(item)
                else:
                    writer.writerow([item])
    
    def to_jsonl(self, path: Union[str, Path]) -> None:
        """Write stream to JSON Lines file."""
        path = Path(path)
        
        with open(path, 'w') as f:
            for item in self:
                f.write(json.dumps(item) + '\n')
    
    # Async support
    
    async def async_foreach(self, func: Callable[[T], Any]) -> None:
        """Apply async function to each element."""
        for item in self:
            if asyncio.iscoroutinefunction(func):
                await func(item)
            else:
                func(item)
    
    # Factory methods
    
    @classmethod
    def from_iterable(cls, iterable: Iterable[T]) -> 'Stream[T]':
        """Create stream from iterable."""
        return cls(iterable)
    
    @classmethod
    def from_file(cls, path: Union[str, Path], mode: str = 'r') -> 'Stream[str]':
        """Create stream from file."""
        return FileStream(path, mode)
    
    @classmethod
    def from_csv(cls, path: Union[str, Path], headers: bool = True, **kwargs) -> 'Stream[Dict[str, Any]]':
        """Create stream from CSV file."""
        return CSVStream(path, headers=headers, **kwargs)
    
    @classmethod
    def from_jsonl(cls, path: Union[str, Path]) -> 'Stream[Any]':
        """Create stream from JSON Lines file."""
        return JSONLStream(path)
    
    @classmethod
    def range(cls, *args) -> 'Stream[int]':
        """Create stream of integers."""
        return cls(lambda: iter(range(*args)))
    
    @classmethod
    def infinite(cls, func: Callable[[], T]) -> 'Stream[T]':
        """Create infinite stream."""
        def generator():
            while True:
                yield func()
        return cls(generator)


class FileStream(Stream[str]):
    """Stream lines from a file."""
    
    def __init__(self, path: Union[str, Path], mode: str = 'r', encoding: str = 'utf-8'):
        self.path = Path(path)
        self.mode = mode
        self.encoding = encoding
        
        def file_iterator():
            with open(self.path, self.mode, encoding=self.encoding) as f:
                for line in f:
                    yield line.rstrip('\n\r')
        
        super().__init__(file_iterator)


class CSVStream(Stream[Dict[str, Any]]):
    """Stream rows from CSV file."""
    
    def __init__(self, path: Union[str, Path], headers: bool = True, **csv_kwargs):
        self.path = Path(path)
        self.headers = headers
        self.csv_kwargs = csv_kwargs
        
        def csv_iterator():
            with open(self.path, 'r', newline='') as f:
                if self.headers:
                    reader = csv.DictReader(f, **self.csv_kwargs)
                else:
                    reader = csv.reader(f, **self.csv_kwargs)
                
                for row in reader:
                    yield row
        
        super().__init__(csv_iterator)


class JSONLStream(Stream[Any]):
    """Stream objects from JSON Lines file."""
    
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        
        def jsonl_iterator():
            with open(self.path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        
        super().__init__(jsonl_iterator)