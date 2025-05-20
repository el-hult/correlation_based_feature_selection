import typing 
import dataclasses
import heapq

T = typing.TypeVar('T')

class PrioQueue(typing.Generic[T]):
    """A simple PrioQueue as per https://docs.python.org/3/library/heapq.html"""

    @dataclasses.dataclass(order=True)
    class PrioritizedItem:
        priority: float
        item: T =dataclasses.field(compare=False)

    def __init__(self):
        self.pq: typing.List[PrioQueue.PrioritizedItem] =[]

    def add_task(self,task,priority):
        heapq.heappush(self.pq, PrioQueue.PrioritizedItem(priority,task))

    def pop_task(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        if len(self.pq) == 0:
            raise KeyError('pop from an empty priority queue')
        
        prioritized_task = heapq.heappop(self.pq)
        return prioritized_task.item
    
    def pop_task_and_prio(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        if len(self.pq) == 0:
            raise KeyError('pop from an empty priority queue')
        
        prioritized_task = heapq.heappop(self.pq)
        return prioritized_task.item, prioritized_task.priority
    
    def __len__(self):
        return len(self.pq)