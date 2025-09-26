class Item:
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)


class Piece(Item):
    def __init__(self, name:int, **kwargs):
        super().__init__(**kwargs)
        self.name = name

    def __str__(self):
        return f"{self.name}"


class Edge(Item):
    def __init__(self, id, **kwargs):
        super().__init__(**kwargs)
        self.id = id


class Solution(Item):
    pass


class LoopingList(list):
    def __getitem__(self, i):
        if isinstance(i, int):
            return super().__getitem__(i % len(self))
        else:
            return super().__getitem__(i)