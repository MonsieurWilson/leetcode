#!/usr/bin/python
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

class Interval:
    def __init__(self, s = 0, e = 0):
        self.start = s
        self.end = e
        #------------------------------#
        def createList():
            # @return a ListNode
            head = ListNode(0)
            ptr = head
            while True:
                ele = raw_input()
                if ele == '#':
                    break
            ptr.next = ListNode(ele)
            ptr = ptr.next
            return head.next

def printList(head):
    # @param a ListNode
    if head == None:
        print "It's a empty list."
    while head:
        print head.val,
        if head.next:
            print '->',
        head = head.next
#---------------------------------#

class Solution:
    # @param A a list of integers
    # @return an integer
    def removeDuplicates(self, A):
        count, N = 0, len(A)
        for idx in xrange(N):
            if idx < N - 2 and A[idx] == A[idx + 2]:
                count += 1
            elif count:
                A[idx - count] = A[idx]
        return N - count
if __name__ == '__main__':
    s = Solution()
    A = [1,1,1,2,2,3]
    print s.removeDuplicates(A)
    print A
