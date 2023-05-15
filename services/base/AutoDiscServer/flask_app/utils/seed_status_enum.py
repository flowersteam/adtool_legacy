from enum import IntEnum


class SeedStatusEnum(IntEnum):
    DONE = 0
    RUNNING = 1
    CANCELLED = 2
    ERROR = 3
