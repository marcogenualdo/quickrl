class SumTree:
    def __init__ (self, size): 

        # number of leaves - effective memory size
        self.leaves = size
        self.tree = [0] * (2 * size - 1)

        # writing pin - gets reset to the first leaf when it writes on the last
        self.pin = size - 1

    
    def total (self):
        """Returns the sum of all the elements in the tree."""
        return self.tree[0]


    def siftup (self, child, change):
        """Adds the childs value to the parents one, preserving the SumTree structure."""

        parent = (child - 1) // 2
        self.tree[parent] += change

        if parent:
            self.siftup(parent, change)


    def update (self, index, priority):
        tree_index = (self.pin - self.leaves + 1 + index) % self.leaves + self.leaves - 1
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self.siftup(tree_index, change)

    
    def push (self, priority):
        """Inserts new value into the leaves"""

        change = priority - self.tree[self.pin]
        self.tree[self.pin] = priority
        self.siftup(self.pin, change)

        # next time write onto the next leaf, if leaves are over, restart from the first
        self.pin += 1
        if self.pin >= len(self.tree): 
            self.pin = self.leaves - 1


    def descend (self, parent, randnum):
        left_child = 2 * parent + 1
        right_child = left_child + 1

        # exit if parent is a leaf (i.e. it does not have a left child)
        if left_child >= len(self.tree):
            return parent - self.pin

        if randnum <= self.tree[left_child]:
            return self.descend(left_child, randnum)
        else:
            return self.descend(right_child, randnum - self.tree[left_child])

    def get (self, randnum):
        """Gets one leaf accorting to priority."""
        return self.descend(0, randnum)


# demosntration / test
if __name__ == '__main__':
    from random import randint

    t = SumTree(2 ** 3)
    for i in range(2 ** 6): 
        t.push(randint(0,10))

    print(t.tree)
    print(t.get(randint(0,t.total())))
