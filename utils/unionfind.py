from typing import List


class UnionFind:

    def __init__(self, n: int) -> None:
        """
        n       : numbers of root.
        parents : parents[x] is parent of x.
        ranks   : ranks[x] is rank of x.
        """
        self.n = n
        self.parents = [i for i in range(n)]
        self.ranks = [0] * n

    def find(self, x: int) -> int:
        """
        Return root of x.
        """
        if self.parents[x] == x:
            return x
        else:
            self.parents[x] = self.find(self.parents[x])
            return self.parents[x]

    def union(self, x: int, y: int) -> None:
        """
        Union the tree to which the x, y belong
        """
        # get root of x, y.
        x = self.find(x)
        y = self.find(y)

        if x == y:
            return

        if self.ranks[x] > self.ranks[y]:
            x, y = y, x

        if self.ranks[x] == self.ranks[y]:
            self.ranks[x] += 1

        self.parents[y] = x

    def roots(self) -> List[int]:
        """
        Return all roots.
        """
        return [i for i in range(self.n) if self.parents[i] == i]

    def members(self, x: int) -> List[int]:
        """
        Return all members which belong same tree as x.
        """
        root = self.find(x)
        return [i for i in range(self.n) if self.find(i) == root]
