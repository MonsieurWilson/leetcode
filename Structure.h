#ifndef STRUCTURE
#define STRUCTURE

struct TreeNode{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x):val(x),left(NULL),right(NULL){}
};

struct ListNode{
    int val;
    ListNode *next;
    ListNode(int x):val(x),next(NULL){}
};

struct TreeLinkNode{
    int val;
    TreeLinkNode *next;
    TreeLinkNode *left, *right;
    TreeLinkNode(int x):val(x),next(NULL),left(NULL),right(NULL){}
};

struct RandomListNode{
    int label;
    RandomListNode *next, *random;
    RandomListNode(int x) : label(x), next(NULL), random(NULL) {}
};

struct UndirectedGraphNode{
    int label;
    vector<UndirectedGraphNode *> neighbors;
    UndirectedGraphNode(int x) : label(x) {};
};

struct Interval {
     int start;
     int end;
     Interval() : start(0), end(0) {}
     Interval(int s, int e) : start(s), end(e) {}
};

/*
 * Method:
 * createTree(const vector<string> &vec)
 */

class Tree{
public:
    static TreeNode *createTree(const vector<string> &vec){
        queue<TreeNode *> q;
        TreeNode *root = NULL;
        if (vec.size() > 0){
            root = new TreeNode(atoi(vec[0].c_str()));
            q.push(root);
        }
        int idx = 1;
        while (idx < vec.size()){
            if (q.front() == NULL){
                q.pop();
                continue;
            }
            if (vec[idx] != string("#")){
                q.front()->left = new TreeNode(atoi(vec[idx].c_str()));
            }
            if (idx != vec.size() - 1 && vec[idx + 1] != string("#")){
                q.front()->right = new TreeNode(atoi(vec[idx + 1].c_str()));
            }
            q.push(q.front()->left);
            q.push(q.front()->right);
            q.pop();
            idx += 2;
        }
        return root;
    }
};

/*
 * Method:
 * createList()
 */

class List{
public:
    static ListNode *createList(){
        string val;
        ListNode *head = NULL, *prep;
        while (cin >> val && val != string("#")){
            ListNode *ptr = new ListNode(atoi(val.c_str()));
            if (head == NULL){
                head = ptr;
                prep = head;
            }
            else{
                prep->next = ptr;
                prep = ptr;
            }
        }
        return head;
    }
};

static ostream & operator << (ostream &out, const ListNode *head) {
    if (head == nullptr) {
        out << "The list is empty.";
    }
    else {
        const ListNode *ptr = head;
        while (ptr) {
            out << ptr->val;
            if (ptr->next) {
                out << "->";
            }
            ptr = ptr->next;
        }
    }
    return out;
}

// Binary Search Tree Iterator
// Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.
// Calling next() will return the next smallest number in the BST.
// Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.
/**
 * Your BSTIterator will be called like this:
 * BSTIterator i = BSTIterator(root);
 * while (i.hasNext()) cout  <<  i.next();
 */
class BSTIterator{
public:
    // Inorder traversal
    BSTIterator(TreeNode *root) {
        ptr = root;
    }

    /** @return whether we have a next smallest number */
    bool hasNext() {
        return !nodeStack.empty() || ptr != NULL;
    }

    /** @return the next smallest number */
    int next() {
        while (!nodeStack.empty() || ptr != NULL){
            if (ptr != NULL){
                nodeStack.push(ptr);
                ptr = ptr->left;
            }
            else{
                ptr = nodeStack.top()->right;
                break;
            }
        }
        int result = nodeStack.top()->val;
        nodeStack.pop();
        return result;
    }
private:
    stack<TreeNode *> nodeStack;
    TreeNode *ptr;
};

// Min Stack
// Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
// push(x) -- Push element x onto stack.
// pop() -- Removes the element on top of the stack.
// top() -- Get the top element.
// getMin() -- Retrieve the minimum element in the stack.
class MinStack {
public:
    void push(int x) {
        eleStack.push(x);
        if (minStack.empty() || (!minStack.empty() && minStack.top() >= x)) {
            minStack.push(x);
        }
    }

    void pop() {
        if (!eleStack.empty()) {
            if (eleStack.top() == minStack.top()) {
                minStack.pop();
            }
            eleStack.pop();
        }
    }

    int top() {
        if (!eleStack.empty()) {
            return eleStack.top();
        }
        return -1;
    }

    int getMin() {
        if (!minStack.empty()) {
            return minStack.top();
        }
        return -1;
    }
private:
    stack<int> eleStack, minStack;
};

// Implement Stack using Queues
// Implement the following operations of a stack using queues.
// push(x) -- Push element x onto stack.
// pop() -- Removes the element on top of the stack.
// top() -- Get the top element.
// empty() -- Return whether the stack is empty.
// Notes:
// You must use only standard operations of a queue -- which means only push to back, peek/pop from front, size, and is empty operations are valid.
// Depending on your language, queue may not be supported natively. You may simulate a queue by using a list or deque (double-ended queue), as long as you use only standard operations of a queue.
// You may assume that all operations are valid (for example, no pop or top operations will be called on an empty stack).
class Stack {
public:
    // Push element x onto stack.
    void push(int x) {
        q.push(x);
        for (int i = 1; i < q.size(); ++i) {
            q.push(q.front());
            q.pop();
        }
    }

    // Removes the element on top of the stack.
    void pop() {
        q.pop();
    }

    // Get the top element.
    int top() {
        return q.front();
    }

    // Return whether the stack is empty.
    bool empty() {
        return q.empty();
    }
private:
    queue<int> q;
};

// Implement Queue using Stacks
// Implement the following operations of a queue using stacks.
// push(x) -- Push element x to the back of queue.
// pop() -- Removes the element from in front of queue.
// peek() -- Get the front element.
// empty() -- Return whether the queue is empty.
// Notes:
// You must use only standard operations of a stack -- which means only push to top, peek/pop from top, size, and is empty operations are valid.
// Depending on your language, stack may not be supported natively. You may simulate a stack by using a list or deque (double-ended queue), as long as you use only standard operations of a stack.
// You may assume that all operations are valid (for example, no pop or peek operations will be called on an empty queue).
class Queue {
public:
    // Push element x to the back of queue.
    void push(int x) {
        inSt.push(x);
    }

    // Removes the element from in front of queue.
    void pop(void) {
        peek();
        outSt.pop();
    }

    // Get the front element.
    int peek(void) {
        if (outSt.empty()) {
            while (!inSt.empty()) {
                outSt.push(inSt.top());
                inSt.pop();
            }
        }
        return outSt.top();
    }

    // Return whether the queue is empty.
    bool empty(void) {
        return inSt.empty() && outSt.empty();
    }
private:
    stack<int> inSt, outSt;
};
#endif
