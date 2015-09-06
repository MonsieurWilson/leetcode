#ifndef SOLUTION
#define SOLUTION

// C Library
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstring>
#include <climits>
// Containers
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <stack>
#include <queue>

using namespace std;
// Data Structure head files and I/O head files
#include "Structure.h"
#include "Vector.h"
#include "Array.h"

#include <sstream>


class Solution{
public:
    // Excel Sheet Column Number
    // Given a column title as appear in an Excel sheet, return its corresponding column number.
    // For example:
    // A -> 1
    // B -> 2
    // C -> 3
    // ...
    // Z -> 26
    // AA -> 27
    // AB -> 28
    int titleToNumber(string s) {
        int bitNumber = s.size();
        int value = 0;
        for (int idx = 0; idx != bitNumber; ++idx) {
            value = value * 26 + (s[idx] - 'A' + 1);
        }
        return value;
    }
    // Excel Sheet Column Title
    // Given a positive integer, return its corresponding column title as appear in an Excel sheet.
    //
    // For example:
    // 1 -> A
    // 2 -> B
    // 3 -> C
    // ...
    // 26 -> Z
    // 27 -> AA
    // 28 -> AB
    string convertToTitle(int n) {
        string ret = "";
        do {
            --n;
            ret = char('A' + n % 26) + ret;
            n /= 26;
        } while (n);
        return ret;
    }
    // Single Number
    // Given an array of integers, every element appears twice except for one. Find that single one.
    // Note:
    // Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
    int singleNumber(int A[], int n) {
        // Bit Manipulation
        int sum = 0;
        for (size_t idx = 0; idx != n; ++idx) {
            sum ^= A[idx];
        }

        return sum;
    }

    int singleNumber_hash(int A[], int n) {
        // Hash Table
        map<int,int> hash_map;
        for (size_t idx = 0; idx != n; ++idx) {
            pair<map<int,int>::iterator,bool> p = hash_map.insert(make_pair(A[idx],1));
            if (p.second == false) {
                ++p.first->second;
            }
        }
        int ret = 0;
        for (map<int,int>::iterator mapit = hash_map.begin(); mapit != hash_map.end(); ++mapit) {
            if (mapit->second == 1) {
                ret = mapit->first;
                break;
            }
        }
        return ret;
    }
    // Maximum Depth of Binary Tree
    // Given a binary tree, find its maximum depth.
    // The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.
    int maxDepth(TreeNode *root) {
        // Recursive
        // Depth-first-Search
        if (root == nullptr) {
            return 0;
        }
        int leftDepth = 1 + maxDepth(root->left),rightDepth = 1 + maxDepth(root->right);

        return leftDepth >= rightDepth ? leftDepth : rightDepth;
    }

    int maxDepth_iteration(TreeNode *root) {
        // Iterative
        if (root == nullptr) {
            return 0;
        }
        stack<TreeNode *> nodeStack;
        stack<int> depthStack;
        // Initialize the depth
        int depth = 0;
        nodeStack.push(root);
        depthStack.push(1);
        while(!nodeStack.empty()) {
            TreeNode *tempNode = nodeStack.top();
            int tempDepth = depthStack.top();
            nodeStack.pop();
            depthStack.pop();
            if (tempNode->left == nullptr && tempNode->right == nullptr) {
                depth = tempDepth >= depth ? tempDepth : depth;
            }
            else {
                if (tempNode->left != nullptr) {
                    nodeStack.push(tempNode->left);
                    depthStack.push(tempDepth + 1);
                }
                if (tempNode->right != nullptr) {
                    nodeStack.push(tempNode->right);
                    depthStack.push(tempDepth + 1);
                }
            }
        }
        return depth;
    }
    // Majority Element
    // Given an array of size n, find the majority element. The majority element is the element that appears more than floor(n/2) times.
    // You may assume that the array is non-empty and the majority element always exist in the array.
    // Solution: Sorting, Hash table, Divide and conquer, Bit manipulation, Moore voting algorithm
    int majorityElement(vector<int> &num) {
        // Sorting method
        sort(num.begin(), num.end());
        vector<int>::size_type mid = (num.size() - 1) / 2;
        return num[mid];
    }
    int majorityElement_voting(vector<int> &num) {
        // Moore voting algorithm
        int candidate = 0, counter = 0;
        for(vector<int>::const_iterator it = num.begin(); it != num.end(); ++it) {
            if (counter == 0) {
                candidate = *it;
                ++counter;
            }
            else {
                if (candidate == *it) {
                    ++counter;
                }
                else {
                    --counter;
                }
            }
        }
        return candidate;
    }
    // Same Tree
    // Given two binary trees, write a function to check if they are equal or not.
    // Two binary trees are considered equal if they are structurally identical and the nodes have the same value.
    bool isSameTree(TreeNode *p, TreeNode *q) {
        if (p == nullptr && q == nullptr) {
            return true;
        }
        else if (p != nullptr && q != nullptr) {
            if (p->val != q->val) {
                return false;
            }
            return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
        }
        else {
            return false;
        }
    }
    bool isSameTree_iteration(TreeNode *p, TreeNode *q) {
        // Using auxiliary stack
        if (p == nullptr && q == nullptr) {
            return true;
        }
        else if (p != nullptr && q != nullptr) {
            stack<TreeNode *> nodeStack_p, nodeStack_q;
            nodeStack_p.push(p);
            nodeStack_q.push(q);
            while (!nodeStack_p.empty() && !nodeStack_q.empty()) {
                TreeNode *tempNode_p = nodeStack_p.top(), *tempNode_q = nodeStack_q.top();
                nodeStack_p.pop();
                nodeStack_q.pop();
                if (tempNode_p == nullptr && tempNode_q == nullptr) {
                    continue;
                }
                else if (tempNode_p != nullptr && tempNode_q != nullptr && tempNode_p->val == tempNode_q->val) {
                    nodeStack_p.push(tempNode_p->left);
                    nodeStack_p.push(tempNode_p->right);
                    nodeStack_q.push(tempNode_q->left);
                    nodeStack_q.push(tempNode_q->right);
                }
                else {
                    return false;
                }
            }
            return true;
        }
        else {
            return false;
        }
    }
    // Reverse Integer
    // Reverse digits of an integer.
    // Example1: x = 123, return 321
    // Example2: x = -123, return -321
    // click to show spoilers.
    // Have you thought about this?
    // Here are some good questions to ask before coding. Bonus points for you if you have already thought through this!
    // If the integer's last digit is 0, what should the output be? ie, cases such as 10, 100.
    // Did you notice that the reversed integer might overflow? Assume the input is a 32-bit integer, then the reverse of 1000000003 overflows. How should you handle such cases?
    // Throw an exception? Good, but what if throwing an exception is not an option? You would then have to re-design the function (ie, add an extra parameter).
    int reverseNumber(int x) {
        // Noticing about the overflow situation
        int symbol = 1;
        long long num = x;
        if (num < 0) {
            symbol = -1;
            num = -num;
        }
        queue<int> tempQueue;
        while (num) {
            tempQueue.push(num%10);
            num /= 10;
        }
        long long ret = 0;
        while (!tempQueue.empty()) {
            ret = 10 * ret + tempQueue.front();
            tempQueue.pop();
        }
        if (ret > INT_MAX) {
            return 0;
        }
        return static_cast<int>(ret * symbol);
    }
    int reverseNumber_improved(int x) {
        long long ret = 0;
        int num = abs(x);
        while (num) {
            ret = ret * 10 + num % 10;
            num /= 10;
        }
        if (ret > INT_MAX || ret < INT_MIN) {
            return 0;
        }
        return ret * (x > 0 ? 1 : -1);
    }
    // Best Time to Buy and Sell Stock II
    // Design an algorithm to find the maximum profit. You may complete as many transactions as you like (ie, buy one and sell one share of the stock multiple times). However, you may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
    int maxProfit(vector<int> &prices) {
        // Allow to make several transactions
        // Greedy
        int profit = 0;
        for (int idx = 1; idx < prices.size(); ++idx) {
            if (prices[idx] > prices[idx - 1]) {
                profit += prices[idx] - prices[idx - 1];
            }
        }
        return profit;
    }
    // Unique Binary Search Trees
    // Given n, how many structurally unique BST's (binary search trees) that store values 1...n?
    // For example,
    // Given n = 3, there are a total of 5 unique BST's.
    // 1         3     3      2      1
    //  \       /     /      / \      \
    //   3     2     1      1   3      2
    //  /     /       \                 \
    // 2     1         2                 3
    int numTrees(int n) {
        if (n == 0) {
            // empty tree
            return 1;
        }
        if (n < 3) {
            return n;
        }

        int ret = 0;
        for (int i = 1; i <= n; ++i) {
            ret += numTrees(i-1) * numTrees(n-i);
        }
        return ret;
    }

    int numtrees_dp(int n) {
        vector<int> dp(n + 1, 0);
        dp[0] = dp[1] = 1;
        for (int idxi = 2; idxi <= n; ++idxi) {
            for (int idxj = 1; idxj <= idxi; ++idxj) {
                dp[idxi] += dp[idxj -1] * dp[idxi - idxj];
            }
        }
        return dp[n];
    }
    // Linked List Cycle
    // Given a linked list, determine if it has a cycle in it.
    // Follow up:
    // Can you solve it without using extra space?
    bool hasCycle(ListNode *head) {
        if (head == nullptr) {
            return false;
        }
        ListNode *p = head, *q = p->next;
        int count = 1;
        while (q != nullptr) {
            if (q == p) {
                return true;
            }
            q = q->next;
            if (count % 2 == 0) {
                p = p->next;
            }
            ++count;
        }
        return false;
    }
    // Binary Tree Preorder Traversal
    // Given a binary tree, return the preorder traversal of its nodes' values.
    // For example:
    // Given binary tree {1,#,2,3},
    //    1
    //     \
    //      2
    //     /
    //    3
    // return [1,2,3].
    // Note: Recursive solution is trivial, could you do it iteratively?
    vector<int> preorderTraversal(TreeNode *root) {
        // Recursive
        vector<int> ret;
        preorderTraversalHelper(root, ret);
        return ret;

    }
    void preorderTraversalHelper(TreeNode *root, vector<int> &vec) {
        // Recursive helper function
        if (root != nullptr) {
            vec.push_back(root->val);
            preorderTraversalHelper(root->left, vec);
            preorderTraversalHelper(root->right, vec);
        }
    }

    vector<int> preorderTraversal_iteration(TreeNode *root) {
        // Iterative
        vector<int> ret;
        if (root == nullptr) {
            return ret;
        }

        stack<TreeNode *> tempStack;
        tempStack.push(root);
        while (!tempStack.empty()) {
            TreeNode *ptr = tempStack.top();
            tempStack.pop();
            ret.push_back(ptr->val);
            if (ptr->right != nullptr) {
                tempStack.push(ptr->right);
            }
            if (ptr->left != nullptr) {
                tempStack.push(ptr->left);
            }
        }

        return ret;
    }
    // Binary Tree Inorder Traversal
    // Given a binary tree, return the inorder traversal of its nodes' values.
    // For example:
    // Given binary tree {1,#,2,3},
    //    1
    //     \
    //      2
    //     /
    //    3
    // return [1,3,2].
    // Note: Recursive solution is trivial, could you do it iteratively?
    // confused what "{1,#,2,3}" means?
    vector<int> inorderTraversal(TreeNode *root) {
        // Recursive
        vector<int> ret;
        inorderTraversalHelper(root,ret);
        return ret;
    }
    void inorderTraversalHelper(TreeNode *root, vector<int> &vec) {
        // Auxiliary function
        if (root != nullptr) {
            inorderTraversalHelper(root->left,vec);
            vec.push_back(root->val);
            inorderTraversalHelper(root->right,vec);
        }
    }

    vector<int> inorderTraversal_iteration(TreeNode *root) {
        // Iterative
        vector<int> ret;
        if (root == nullptr) {
            return ret;
        }
        stack<TreeNode *> tempStack;
        TreeNode *ptr = root;
        while (!tempStack.empty() || ptr != nullptr) {
            if (ptr != nullptr) {
                tempStack.push(ptr);
                ptr = ptr->left;
            }
            else {
                ptr = tempStack.top();
                tempStack.pop();
                ret.push_back(ptr->val);
                ptr = ptr->right;
            }
        }

        return ret;
    }

    vector<int> morrisInorderTraversal(TreeNode *root) {
        // Morris Traversal
        vector<int> order;
        for(TreeNode *now = root, *tmp; now;) {
            if(now->left == nullptr) {
                order.push_back(now->val);
                now=now->right;
            }
            else {
                for(tmp = now->left; tmp->right != nullptr && tmp->right != now;) {
                    tmp = tmp->right;
                }
                if(tmp->right) {
                    order.push_back(now->val);
                    tmp->right = nullptr;
                    now = now->right;
                }
                else {
                    tmp->right = now;
                    now = now->left;
                }
            }
        }
        return order;
    }
    // Binary Tree Postorder Traversal
    // Given a binary tree, return the postorder traversal of its nodes' values.
    // For example:
    // Given binary tree {1,#,2,3},
    //    1
    //     \
    //      2
    //     /
    //    3
    // return [3,2,1].
    // Note: Recursive solution is trivial, could you do it iteratively?
    vector<int> postorderTraversal(TreeNode *root) {
        // Recursive
        vector<int> ret;
        postorderTraversalHelper(root,ret);
        return ret;
    }
    void postorderTraversalHelper(TreeNode *root,vector<int> &ret) {
        if (root != nullptr) {
            postorderTraversalHelper(root->left,ret);
            postorderTraversalHelper(root->right,ret);
            ret.push_back(root->val);
        }
    }

    vector<int> postorderTraversal_iteration(TreeNode *root) {
        // Iterative
        // Save the ret of root->right->left, and return the reverse ret will get the postorderTraversal ret
        vector<int> ret;
        if (root == nullptr) {
            return ret;
        }
        stack<TreeNode *> tempStack;
        tempStack.push(root);
        while (!tempStack.empty()) {
            TreeNode *ptr = tempStack.top();
            tempStack.pop();
            ret.push_back(ptr->val);
            if (ptr->left != nullptr) {
                tempStack.push(ptr->left);
            }
            if (ptr->right != nullptr) {
                tempStack.push(ptr->right);
            }
        }
        reverse(ret.begin(),ret.end());
        return ret;
    }
    // Populating Next Right Pointers in Each Node
    // Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to nullptr.
    // Note:
    // 1)You may only use constant extra space.
    // 2)You may assume that it is a perfect binary tree (ie, all leaves are at the same level, and every parent has two children).
    void connect(TreeLinkNode *root) {
        if (root == nullptr) {
            return;
        }
        // Pointer ptr indicates the first node of every level, using cur to traversal the level ptr points
        TreeLinkNode *ptr = root, *cur = nullptr;
        while (ptr->left != nullptr) {
            cur = ptr;
            while (cur != nullptr) {
                cur->left->next = cur->right;
                if (cur->next != nullptr) {
                    cur->right->next = cur->next->left;
                }
                cur = cur->next;
            }
            ptr = ptr->left;
        }
    }
    // Search Insert Position
    // Given a sorted array and a target value, return the index if the target is found. If not, return the index where it would be if it were inserted in order.
    // You may assume no duplicates in the array.
    int searchInsert(int A[], int n, int target) {
        // Using binary search
        int beg = 0, end = n - 1;
        while (beg <= end) {
            int mid = (beg + end) / 2;
            if (A[mid] == target) {
                beg = mid;
                break;
            }
            else if (A[mid] < target) {
                beg = mid + 1;
            }
            else {
                end = mid - 1;
            }
        }
        return beg;
    }
    // Remove Duplicates from Sorted List
    // Given a sorted linked list, delete all duplicates such that each element appear only once.
    ListNode *deleteDuplicates(ListNode *head) {
        if (head != nullptr) {
            ListNode *prep = head, *ptr = head->next;
            while (ptr != nullptr) {
                if (ptr->val == prep->val) {
                    prep->next = ptr->next;
                    ptr = prep->next;
                }
                else {
                    prep = prep->next;
                    ptr = ptr->next;
                }
            }
        }
        return head;
    }
    // N-Queens II
    // Follow up for N-Queens problem.
    // Now, instead outputting board configurations, return the total number of distinct solutions.
    int totalNQueens(int n) {
        int ret = 0;
        vector<int> sol(n + 1, 0);
        int idx = 1;
        while (idx > 0) {
            ++sol[idx];
            while ((sol[idx] <= n) && (legalPosition(sol,idx) == false)) {
                ++sol[idx];
            }
            if (sol[idx] <= n) {
                if (idx == n) {
                    ++ret;
                }
                else {
                    ++idx;
                    sol[idx] = 0;
                }
            }
            else {
                --idx;
            }
        }
        return ret;
    }
    bool legalPosition(vector<int> sol, int pos) {
        bool ret = true;
        for (int idx = 1; idx != pos; ++idx) {
            if (sol[idx] == sol[pos] || (abs(pos - idx) == abs(sol[pos] - sol[idx]))) {
                ret = false;
                break;
            }
        }
        return ret;
    }
    // Single Number II
    // Given an array of integers, every element appears three times except for one. Find that single one.
    // Note:
    // Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?
    int singleNumberII(int A[], int n) {
        int ret;
        sort(A,A+n);
        for (int idx = 0; idx != n; ) {
            int num1 = A[idx], num2;
            if (idx + 2 < n) {
                num2 = A[idx + 2];
                if (num1 == num2) {
                    idx += 3;
                }
                else {
                    ret = num1;
                    break;
                }
            }
            else {
                ret = num1;
                break;
            }
        }
        return ret;
    }
    // Maximum Subarray
    // Find the contiguous subarray within an array (containing at least one number) which has the largest sum.
    int maxSubArray(int A[], int n) {
        // Dynamic Programming
        int tempSum, sum;
        tempSum = sum = A[0];
        for (int idx = 1; idx != n; ++idx) {
            if (tempSum < 0) {
                tempSum = 0;
            }
            tempSum += A[idx];
            if (tempSum > sum) {
                sum = tempSum;
            }
        }
        return sum;
    }
    // Climbing Stairs
    // You are climbing a stair case. It takes n steps to reach to the top.
    // Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?
    int climbStairs(int n) {
        // Fibonacci sequence
        // Recursive algorithm will get a TLE
        if (n <= 3) {
            return n;
        }
        return climbStairs(n - 1) + climbStairs(n - 2);
    }

    int climbStairs_iteration(int n) {
        if (n <= 3) {
            return n;
        }
        int oneLeft = 2, twoLeft = 3;
        int ret;
        for (int idx = 4; idx <= n; ++idx) {
            ret = oneLeft + twoLeft;
            oneLeft = twoLeft;
            twoLeft = ret;
        }
        return ret;
    }
    // Convert Sorted Array to Binary Search Tree
    // Given an array where elements are sorted in ascending order, convert it to a height balanced BST.
    TreeNode *sortedArrayToBST(vector<int> &num) {
        TreeNode *root = nullptr;
        if (!num.empty()) {
            int beg = 0, end = num.size();
            root = sortHelper(num, beg, end);
        }
        return root;
    }
    TreeNode *sortHelper(vector<int> &num, const int &beg, const int &end) {
        if (beg < end) {
            int mid = (beg + end - 1) / 2;
            TreeNode *root = new TreeNode(num[mid]);
            root->left = sortHelper(num, beg, mid);
            root->right = sortHelper(num, mid + 1, end);
            return root;
        }
        return nullptr;
    }
    // Merge Two Sorted Lists
    // Merge two sorted linked lists and return it as a new list. The new list should be made by splicing together the nodes of the first two lists.
    ListNode *mergeTwoLists(ListNode *l1, ListNode *l2) {
        ListNode *head = new ListNode(0);
        ListNode *ptr = head;
        ListNode *pointer1 = l1, *pointer2 = l2;
        while (pointer1 != nullptr || pointer2 != nullptr) {
            if (pointer2 == nullptr || ((pointer1 != nullptr) && (pointer1->val < pointer2->val))) {
                ptr->next = pointer1;
                ptr = ptr->next;
                pointer1 = pointer1->next;
            }
            else {
                ptr->next = pointer2;
                ptr = ptr->next;
                pointer2 = pointer2->next;
            }
        }
        return head->next;
    }
    // Remove Element
    // Given an array and a value, remove all instances of that value in place and return the new length.
    // The order of elements can be changed. It doesn't matter what you leave beyond the new length.
    int removeElement(int A[], int n, int elem) {
        int count = 0;
        for (int idx = 0; idx !=n; ++idx) {
            if (A[idx] == elem) {
                ++count;
            }
            else {
                A[idx - count] = A[idx];
            }
        }
        return n - count;
    }
    // Find Peak Element
    // A peak element is an element that is greater than its neighbors.
    // Given an input array where num[i] ≠ num[i+1], find a peak element and return its index.
    // The array may contain multiple peaks, in that case return the index to any one of the peaks is fine.
    // You may imagine that num[-1] = num[n] = -∞.
    // Note:
    // Your solution should be in logarithmic complexity.
    int findPeakElement(const vector<int> &num) {
        int first = 0, last = num.size() - 1, mid;
        while (first < last) {
            mid = first + (last - first) / 2;
            int left, right;
            if (mid == 0) {
                left = INT_MIN;
                right = num[mid + 1];
            }
            else if (mid == num.size() - 1) {
                left = num[mid - 1];
                right = INT_MIN;
            }
            else {
                left = num[mid - 1];
                right = num[mid + 1];
            }
            // Noticing about the case:[INT_MIN,INT_MIN + 1], '>=' but not '>'
            if (num[mid] >= left && num[mid] >= right) {
                return mid;
            }
            else if (num[mid] >= left && num[mid] <= right) {
                first = mid + 1;
            }
            else {
                last = mid - 1;
            }
        }
        return first;
    }
    // Swap Nodes in Pairs
    // Given a linked list, swap every two adjacent nodes and return its head.
    // Your algorithm should use only constant space. You may not modify the values in the list, only nodes itself can be changed.
    ListNode *swapPairs(ListNode *head) {
        if (head != nullptr && head->next != nullptr) {
            ListNode *temp = new ListNode(0);
            temp->next = head;
            // Using two pointers to swap nodes
            ListNode *pointer1 = head, *pointer2 = head->next, *ptr = temp;
            while (pointer1 != nullptr && pointer2 != nullptr) {
                ptr->next = pointer2;
                pointer1->next = pointer2->next;
                pointer2->next = pointer1;

                //reset the ptr、pointer1、pointer2
                ptr = pointer1;
                pointer1 = pointer1->next;
                if (pointer1 != nullptr) {
                    pointer2 = pointer1->next;
                }
            }
            head = temp->next;
        }
        return head;
    }
    // Balanced Binary Tree
    // Given a binary tree, determine if it is height-balanced.
    // For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.
    bool isBalanced(TreeNode *root) {
        if (root == nullptr) {
            return true;
        }
        else {
            int leftDepth = maxDepth(root->left), rightDepth = maxDepth(root->right);
            if (abs(leftDepth - rightDepth) < 2) {
                return isBalanced(root->left) && isBalanced(root->right);
            }
            return false;
        }
    }

    static const int UNBALANCE = -1;
    bool isBalancedImproved(TreeNode *root) {
        // Pruning
        if (root == nullptr) {
            return true;
        }
        int ret = isBalancedImprovedHelper(root);
        return (isBalancedImprovedHelper(root) == UNBALANCE) ? false : true;
    }
    int isBalancedImprovedHelper(TreeNode *root) {
        if (root == nullptr) {
            return 0;
        }
        int leftDepth = isBalancedImprovedHelper(root->left), rightDepth = isBalancedImprovedHelper(root->right);
        if (leftDepth == UNBALANCE || rightDepth == UNBALANCE) {
            return UNBALANCE;
        }
        if (abs(leftDepth - rightDepth) < 2) {
            return max(leftDepth,rightDepth) + 1;
        }
        else {
            return UNBALANCE;
        }
    }
    // Sort Colors
    // Given an array with n objects colored red, white or blue, sort them so that objects of the same color are adjacent, with the colors in the order red, white and blue.
    // Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.
    // Note:
    // You are not suppose to use the library's sort function for this problem.
    void sortColors(int A[], int n) {
        int zero = 0, two = n - 1;
        for (int idx = 0; idx <= two; ++idx) {
            while (A[idx] == 2 && idx < two) {
                swap(A[idx],A[two--]);
            }
            while (A[idx] == 0 && idx > zero) {
                swap(A[idx],A[zero++]);
            }
        }
    }
    // Gray Code
    // The gray code is a binary numeral system where two successive values differ in only one bit.
    vector<int> grayCode(int n) {
        vector<int> ret;
        ret.push_back(0);
        for (int idx = 0; idx != n; ++idx) {
            int temp = 1 << idx;
            for (int idx2 = ret.size() - 1; idx2 >= 0; --idx2) {
                ret.push_back(ret[idx2] + temp);
            }
        }
        return ret;
    }
    // Remove Duplicates from Sorted Array
    // Given a sorted array, remove the duplicates in place such that each element appear only once and return the new length.
    // Do not allocate extra space for another array, you must do this in place with constant memory.
    int removeDuplicates(int A[], int n) {
        int count = 0;
        for (int idx = 0; idx != n; ++idx) {
            if (idx != n - 1 && A[idx] == A[idx + 1]) {
                ++count;
            }
            else {
                if (count > 0) {
                    A[idx - count] = A[idx];
                }
            }
        }
        return n - count;
    }
    // Unique Paths
    // A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
    // The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).
    // How many possible unique paths are there?
    int uniquePaths(int m, int n) {
        // Combination
        return combination(m + n -2, n - 1);
    }
    int combination(int m, int n) {
        if (n > m / 2) {
            combination(m, m - n);
        }
        long long ret = 1;
        for (int idx = 1; idx <= n; ++idx) {
            ret *= m - n + idx;
            ret /= idx;
        }
        return static_cast<int>(ret);
    }

    int uniquePaths_dp(int m, int n) {
        // Dynamic Programming
        vector<int> dp(n, 0);
        for (int idxi = 0; idxi < m; ++idxi) {
            for (int idxj = 0; idxj < n; ++idxj) {
                if (idxi == 0 || idxj == 0) {
                    dp[idxj] = 1;
                }
                else {
                    dp[idxj] += dp[idxj - 1];
                }
            }
        }
        return dp[n - 1];
    }
    // Find Minimum in Rotated Sorted Array
    // Suppose a sorted array is rotated at some pivot unknown to you beforehand.
    // Find the minimum element.
    // You may assume no duplicate exists in the array.
    int findMin(vector<int> &num) {
        // Binary Search
        int beg = 0, end = num.size();
        int mid;
        while (beg < end) {
            mid = (beg + end - 1) / 2;
            if (num[mid] < num[(mid - 1 + end) % end] && num[mid] < num[(mid + 1) % end]) {
                break;
            }
            if (num[mid] > num[end - 1]) {
                // Pivot is in the right half
                beg = mid + 1;
            }
            else {
                end = mid;
            }
        }
        return num[mid];
    }
    // Generate Parentheses
    // Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
    // For example, given n = 3, a solution set is:
    // "((()))", "(()())", "(())()", "()(())", "()()()"
    int left, right, n;
    vector<string> generateParenthesis(int n) {
        // Backtracking
        Solution::n = n;
        left = right = n;
        int *array = new int[2 * n + 1];
        vector<string> ret;
        gpHelper(1,ret,array);
        return ret;
    }
    void gpHelper(const int &k, vector<string> &ret, int *array) {
        if (k > 2 * n) {
            string s;
            for (int i = 1; i <= 2 * n; ++i) {
                if (array[i] == 1) {
                    s.append("(");
                }
                else if (array[i] == -1) {
                    s.append(")");
                }
            }
            ret.push_back(s);
        }
        // The left parentheses left shouldn't be more than right ones.
        if (left > 0 && left <= right) {
            --left;
            array[k] = 1;
            gpHelper(k + 1,ret,array);
            ++left;
            array[k] = 0;
        }
        if (left < right) {
            --right;
            array[k] = -1;
            gpHelper(k + 1,ret,array);
            ++right;
            array[k] = 0;
        }
    }

    vector<string> generateParenthesis2(int n) {
        vector<string> ret;
        if (n >= 1) {
            string parens;
            gp_recursionHelper(ret, parens, n, n);
        }
        return ret;
    }
    void gp_recursionHelper(vector<string> &ret, string &parens, const int &left, const int &right) {
        if (left > right || left < 0) {
            return;
        }
        if (left == 0 && right == 0) {
            ret.push_back(parens);
            return;
        }
        parens.append("(");
        gp_recursionHelper(ret,parens,left - 1,right);
        parens.erase(parens.end() - 1);
        parens.append(")");
        gp_recursionHelper(ret,parens,left,right - 1);
        parens.erase(parens.end() - 1);
    }
    // Container With Most Water
    // Given n non-negative integers a1, a2, ..., an, where each represents a point at coordinate (i, ai). n vertical lines are drawn such that the two endpoints of line i is at (i, ai) and (i, 0). Find two lines, which together with x-axis forms a container, such that the container contains the most water.
    int maxArea(vector<int> &height) {
        int beg = 0, end = height.size();
        int ret = 0;
        while (beg < end) {
            ret = max(ret, min(height[beg], height[end - 1]) * (end - beg - 1));
            if (height[beg] < height[end - 1]) {
                ++beg;
            }
            else {
                --end;
            }
        }
        return ret;
    }
    // Rotate Image
    // You are given an n x n 2D matrix representing an image.
    // Rotate the image by 90 degrees (clockwise).
    // Follow up:
    // Could you do this in-place?
    void rotate(vector<vector<int> > &matrix) {
        int size = matrix.size();
        for (int idxj = 0; idxj < size / 2; ++idxj) {
            for (int idxi = idxj; idxi < size - 1 - idxj; ++idxi) {
                int temp = matrix[idxi][idxj];
                matrix[idxi][idxj] = matrix[size - 1 - idxj][idxi];
                matrix[size - 1 - idxj][idxi] = matrix[size - 1 - idxi][size - 1 - idxj];
                matrix[size - 1 - idxi][size - 1 - idxj] = matrix[idxj][size - 1 - idxi];
                matrix[idxj][size - 1 - idxi] = temp;
            }
        }
    }
    // Best Time to Buy and Sell Stock
    // If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.
    int maxProfitII(vector<int> &prices) {
        int maxProfit = 0;
        int min = INT_MAX;
        for (vector<int>::const_iterator it = prices.begin(); it != prices.end(); ++it) {
            min = (*it < min) ? *it : min;
            maxProfit = (*it - min) > maxProfit ? (*it - min) : maxProfit;
        }
        return maxProfit;
    }
    // Permutations
    // Given a collection of numbers, return all possible permutations.
    // For example,
    // [1,2,3] have the following permutations:
    // [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], and [3,2,1].
    vector<vector<int> > permute(vector<int> &num) {
        // Backtracking
        int size = num.size();
        vector<vector<int> > ret;
        vector<int> tmp(size,0);
        permuteBackTrack(0, size, ret, tmp, num);
        return ret;
    }
    void permuteBackTrack(int k, int size, vector<vector<int> > &ret, vector<int> &tmp, vector<int> &num) {
        if (k == size) {
            ret.push_back(tmp);
            return;
        }
        for (int idx = 0; idx != size; ++idx) {
            tmp[k] = num[idx];
            if (legalPermutation(k, tmp)) {
                permuteBackTrack(k + 1, size , ret, tmp, num);
            }
        }
    }
    bool legalPermutation(int k, vector<int> tmp) {
        for (int idx = 0; idx < k; ++idx) {
            if (tmp[k] == tmp[idx]) {
                return false;
            }
        }
        return true;
    }

    vector<vector<int> > permute_improved(vector<int> &num) {
        vector<vector<int> > ret;
        if (!num.empty()) {
            permute_improved(0, num, ret);
        }
        return ret;
    }
    void permute_improved(int idx, vector<int> line, vector<vector<int> > &ret) {
        ret.push_back(line);
        for (int i = idx; i < line.size(); ++i) {
            for (int j = i + 1; j < line.size(); ++j) {
                swap(line[i], line[j]);
                permute_improved(i + 1, line, ret);
                swap(line[i], line[j]);
            }
        }
    }
    // Merge Sorted Array
    // Given two sorted integer arrays A and B, merge B into A as one sorted array.
    void merge(int A[], int m, int B[], int n) {
        for (int idx = m + n - 1; idx >= 0; --idx) {
            int ptrA = m - 1, ptrB = n - 1;
            if (ptrB < 0) {
                return;
            }
            else if (ptrA < 0) {
                A[idx] = B[ptrB--];
            }
            else if (A[ptrA] < B[ptrB]) {
                A[idx] = B[ptrB--];
            }
            else {
                A[idx] = A[ptrA--];
            }
        }
    }
    // Search a 2D Matrix
    // Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
    // 1)Integers in each row are sorted from left to right.
    // 2)The first integer of each row is greater than the last integer of the previous row.
    bool searchMatrix(vector<vector<int> > &matrix, int target) {
        // Binary Search in 2D
        int rowNum = matrix.size();
        if (rowNum <= 0) {
            return false;
        }
        int colNum = matrix[0].size();
        int beg = 0, end = rowNum - 1, line;
        while (beg <= end) {
            int mid = beg + (end - beg) / 2;
            if (matrix[mid][0] == target) {
                return true;
            }
            else if (matrix[mid][0] > target) {
                end = mid - 1;
            }
            else {
                beg = mid + 1;
            }
        }
        if (end < 0) {
            return false;
        }
        line = end; beg = 0; end = colNum - 1;
        while (beg <= end) {
            int mid = beg + (end - beg) / 2;
            if (matrix[line][mid] == target) {
                return true;
            }
            else if (matrix[line][mid] > target) {
                end = mid - 1;
            }
            else {
                beg = mid + 1;
            }
        }
        return false;
    }
    // Search in Rotated Sorted Array(both I and II)
    // Suppose a sorted array is rotated at some pivot unknown to you beforehand.
    // You are given a target value to search. If found in the array return its index, otherwise return -1.
    // What if duplicates are allowed?
    // Would this affect the run-time complexity? How and why?
    int search(int A[], int n, int target) {
        if (n <= 0) {
            return -1;
        }
        int first = 0, last = n - 1, pos = -1;
        while (first <= last) {
            int mid = first + (last - first) / 2;
            if (A[mid] == target) {
                pos = mid;
                break;
            }
            if (A[mid] > A[first]) {
                if (A[first] <= target && target < A[mid]) {
                    last = mid - 1;
                }
                else {
                    first = mid + 1;
                }
            }
            else if (A[mid] < A[first]) {
                if (A[mid] <= target && target < A[first]) {
                    first = mid + 1;
                }
                else {
                    last = mid - 1;
                }
            }
            else {
                // If A[mid] == A[first], let first increase itself to handle the duplicate situation
                ++first;
            }
        }
        return pos;
    }
    // Minimum Path Sum
    // Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right which minimizes the sum of all numbers along its path.
    int minPathSum(vector<vector<int> > &grid) {
        // Dynamic Programming
        int rowNum = grid.size(), colNum = grid[0].size();
        vector<int> dp(colNum,0);
        dp[0] = grid[0][0];
        for (int idx = 1; idx != colNum; ++idx) {
            dp[idx] = grid[0][idx] + dp[idx - 1];
        }
        for (int r = 1; r != rowNum; ++r) {
            dp[0] += grid[r][0];
            for (int c = 1; c != colNum; ++c) {
                int val1 = grid[r][c] + dp[c];
                int val2 = grid[r][c] + dp[c - 1];
                dp[c] = val1 < val2 ? val1 : val2;
            }
        }
        return dp[colNum - 1];
    }
    // Plus One
    // Given a non-negative number represented as an array of digits, plus one to the number.
    // The digits are stored such that the most significant digit is at the head of the list.
    vector<int> plusOne(vector<int> &digits) {
        // Plus one at the digits' back
        int carry = 1;
        for (vector<int>::reverse_iterator it = digits.rbegin(); it != digits.rend(); ++it) {
            *it += carry;
            carry = 0;
            if (*it >= 10) {
                *it %= 10;
                carry = 1;
            }
        }
        if (carry == 1) {
            digits.push_back(0);
            for (vector<int>::reverse_iterator it = digits.rbegin(); it != digits.rend() - 1; ++it) {
                *it = *(it - 1);
            }
            digits[0] = 1;
        }
        return digits;
    }
    // Populating Next Right Pointers in Each Node II
    // Follow up for problem "Populating Next Right Pointers in Each Node".
    // What if the given tree could be any binary tree? Would your previous solution still work?
    // Note: You may only use constant extra space.
    void connectII(TreeLinkNode *root) {
        if (root == nullptr) {
            return;
        }
        TreeLinkNode *head = root;
        // A helper pointer to help build every node's next pointer
        TreeLinkNode *helper = new TreeLinkNode(0);
        // Store the next head's position
        TreeLinkNode *nextHead = helper;
        while (head != nullptr) {
            TreeLinkNode *mov = head;
            while (mov != nullptr) {
                if (mov->left != nullptr) {
                    helper->next = mov->left;
                    helper = helper->next;
                }
                if (mov->right != nullptr) {
                    helper->next = mov->right;
                    helper = helper->next;
                }
                mov = mov->next;
            }
            head = nextHead->next;
            // Reset the helper node and its next pointer
            helper = nextHead;
            helper->next = nullptr;
        }
    }
    // Symmetric Tree
    // Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).
    bool isSymmetric(TreeNode *root) {
        // Recursive
        if (root == nullptr) {
            return true;
        }
        return isSymmetricHelper(root->left, root->right);
    }
    bool isSymmetricHelper(TreeNode *pLeft, TreeNode *pRight) {
        if (pLeft == nullptr && pRight == nullptr) {
            return true;
        }
        else if (pLeft == nullptr || pRight == nullptr) {
            return false;
        }
        else {
            if (pLeft->val != pRight->val) {
                return false;
            }
            else {
                return isSymmetricHelper(pLeft->left,pRight->right) && isSymmetricHelper(pLeft->right,pRight->left);
            }
        }
    }

    bool isSymmetric_iteration(TreeNode *root) {
        // Using two queues to store nodes
        if (root == nullptr) {
            return true;
        }
        stack<TreeNode *> s1,s2;
        s1.push(root->left);
        s2.push(root->right);
        TreeNode *r, *l;
        while (!s1.empty() && !s2.empty()) {
            l = s1.top();
            r = s2.top();
            s1.pop();
            s2.pop();
            if (l == nullptr && r == nullptr) {
                continue;
            }
            else if (l == nullptr || r == nullptr) {
                return false;
            }
            if (l->val != r->val) {
                return false;
            }
            s1.push(l->left);
            s1.push(l->right);
            s2.push(r->right);
            s2.push(r->left);
        }
        return true;
    }
    // Set Matrix Zeros
    // Given a m x n matrix, if an element is 0, set its entire row and column to 0. Do it in place.
    void setZeroes(vector<vector<int> > &matrix) {
        int rowNum = matrix.size(), colNum = matrix[0].size();
        if (rowNum == 0 || colNum == 0) {
            return;
        }
        bool hasZeroFirstRow = false, hasZeroFirstCol = false;
        for (int c = 0; c != colNum; ++c) {
            if (matrix[0][c] == 0) {
                hasZeroFirstRow = true;
                break;
            }
        }
        for (int r = 0; r != rowNum; ++r) {
            if (matrix[r][0] == 0) {
                hasZeroFirstCol = true;
                break;
            }
        }
        for (int r = 1; r != rowNum; ++r) {
            for (int c = 1; c != colNum; ++c) {
                if (matrix[r][c] == 0) {
                    matrix[0][c] = 0;
                    matrix[r][0] = 0;
                }
            }
        }
        for (int r = 1; r != rowNum; ++r) {
            for (int c = 1; c != colNum; ++c) {
                if (matrix[0][c] == 0 || matrix[r][0] == 0) {
                    matrix[r][c] = 0;
                }
            }
        }
        if (hasZeroFirstRow) {
            for (int c = 0; c != colNum; ++c) {
                matrix[0][c] = 0;
            }
        }
        if (hasZeroFirstCol) {
            for (int r = 0; r != rowNum; ++r) {
                matrix[r][0] = 0;
            }
        }
    }
    // Linked List Cycle II
    // Given a linked list, return the node where the cycle begins. If there is no cycle, return null.
    // Follow up:
    // Can you solve it without using extra space?
    ListNode *detectCycle(ListNode *head) {
        if (head == nullptr) {
            return nullptr;
        }
        ListNode *slow = head, *fast = slow->next;
        long long slowVal = 0, fastVal = slowVal + 1;
        while (fast != nullptr) {
            if (slow == fast) {
                break;
            }
            slow = slow->next;
            fast = fast->next;
            ++slowVal;
            ++fastVal;
            if (fast != nullptr) {
                fast = fast->next;
                ++fastVal;
            }
        }
        // No loop
        if (fast == nullptr) {
            return nullptr;
        }
        long long loopSize = fastVal - slowVal;
        slow = fast = head;
        while (loopSize) {
            fast = fast->next;
            --loopSize;
        }
        while (slow != fast) {
            slow = slow->next;
            fast = fast->next;
        }
        return fast;
    }
    ListNode *detectCycle_improved(ListNode *head) {
        ListNode *slow = head, *fast = head;
        while (fast->next != nullptr && fast->next->next != nullptr) {
            fast = fast->next->next;
            slow = slow->next;
            if (fast == slow) {
                fast = head;
                while (fast != slow) {
                    fast = fast->next;
                    slow = slow->next;
                }
                return slow;
            }
        }
        return nullptr;
    }
    // Pascal's Triangle
    // Given numRows, generate the first numRows of Pascal's triangle.
    // For example, given numRows = 5,
    // Return
    // [
    //       [1],
    //      [1,1],
    //     [1,2,1],
    //    [1,3,3,1],
    //   [1,4,6,4,1]
    // ]
    vector<vector<int> > generate(int numRows) {
        vector<vector<int> > table;
        if (numRows > 0) {
            for (int idx = 1; idx <= numRows; ++idx) {
                table.push_back(vector<int>(idx, 1));
                if (idx > 2) {
                    for (int cur = 1, row = idx - 1; cur != idx - 1; ++cur) {
                        table[row][cur] = table[row - 1][cur - 1] + table[row - 1][cur];
                    }
                }
            }
        }
        return table;
    }
    // Binary Tree Level Order Traversal
    // Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).
    vector<vector<int> > levelOrder(TreeNode *root) {
        vector<vector<int> > ret;
        levelOrderHelper(root, 0, ret);
        return ret;
    }
    void levelOrderHelper(TreeNode *root, int level, vector<vector<int> > &ret) {
        if (root == nullptr) {
            return;
        }
        if (ret.size() <= level) {
            vector<int> line;
            ret.push_back(line);
        }
        ret[level].push_back(root->val);
        levelOrderHelper(root->left, level + 1, ret);
        levelOrderHelper(root->right, level + 1, ret);
    }

    vector<vector<int> > levelOrder_iteration(TreeNode *root) {
        // Iteration
        queue<TreeNode *> nodeQueue;
        vector<vector<int> > ret;
        if (root == nullptr) {
            return ret;
        }
        nodeQueue.push(root);
        nodeQueue.push(nullptr);
        vector<int> line;
        ret.push_back(line);
        while (!nodeQueue.empty()) {
            TreeNode *tmp = nodeQueue.front();
            nodeQueue.pop();
            if (tmp == nullptr) {
                if (nodeQueue.front() == nullptr) {
                    break;
                }
                vector<int> line;
                ret.push_back(line);
            }
            else {
                ret[ret.size() - 1].push_back(tmp->val);
                if (tmp->left != nullptr) {
                    nodeQueue.push(tmp->left);
                }
                if (tmp->right != nullptr) {
                    nodeQueue.push(tmp->right);
                }
                if (nodeQueue.front() == nullptr) {
                    nodeQueue.push(nullptr);
                }
            }
        }
        return ret;
    }
    // Binary Tree Level Order Traversal II
    // Given a binary tree, return the bottom-up level order traversal of its nodes' values. (ie, from left to right, level by level from leaf to root).
    vector<vector<int> > levelOrderBottom(TreeNode *root) {
        vector<vector<int> > inverse;
        levelOrderBottomHelper(root, 0, inverse);
        vector<vector<int> > ret(inverse.rbegin(), inverse.rend());
        return ret;
    }
    void levelOrderBottomHelper(TreeNode *root, int level, vector<vector<int> > &inverse) {
        if (root == nullptr) {
            return;
        }
        if (inverse.size() <= level) {
            vector<int> line;
            inverse.push_back(line);
        }
        inverse[level].push_back(root->val);
        levelOrderBottomHelper(root->left, level + 1, inverse);
        levelOrderBottomHelper(root->right, level + 1, inverse);
    }
    // Remove Duplicates from Sorted Array II
    // Follow up for "Remove Duplicates":
    // What if duplicates are allowed at most twice?
    // For example,
    // Given sorted array A = [1,1,1,2,2,3],
    // Your function should return length = 5, and A is now [1,1,2,2,3].
    int removeDuplicatesII(int A[], int n) {
        int count = 0;
        for (int idx = 0; idx != n; ++idx) {
            if (idx < n - 2 && A[idx] == A[idx + 2]) {
                ++count;
            }
            else if (count > 0) {
                A[idx - count] = A[idx];
            }
        }
        return n - count;
    }

    int removeDuplicatesII_improved(int A[], int n) {
        if (n <= 2) return n;
        int lens = 1;
        for (int idx = 2; idx < n; ++idx) {
            if (A[idx] != A[lens - 1]) {
                A[++lens] = A[idx];
            }
        }
        return ++lens;
    }
    // Combinations
    // Given two integers n and k, return all possible combinations of k numbers out of 1 ... n.
    vector<vector<int> > combine(int n, int k) {
        vector<vector<int> > ret;
        if (n < 1 || n < k || k < 1) {
            return ret;
        }
        vector<int> line;
        line.push_back(1);
        bool needPop = false;
        while (!line.empty() && line.front() <= n - k + 1) {
            if (line.size() == k) {
                ret.push_back(line);
            }
            int last = line.back();
            if (last < n) {
                if (line.size() == k || needPop == true ) {
                    line.pop_back();
                }
                needPop = false;
                line.push_back(last + 1);
            }
            else {
                line.pop_back();
                needPop = true;
            }
        }
        return ret;
    }

    vector<vector<int> > combine_fast(int n, int k) {
        vector<vector<int> > ret;
        if (n < 1 || k > n || k < 1) {
            return ret;
        }
        vector<int> line(k, 1);
        int idx = 0;
        bool needBack = false;
        while (idx >= 0 && line[0] <= n - k + 1) {
            if (idx == k - 1) {
                ret.push_back(line);
            }
            int lastEle = line[idx];
            if (lastEle < n) {
                if (idx == k - 1 || needBack == true) {
                    --idx;
                }
                needBack = false;
                line[idx + 1] = lastEle + 1;
                ++idx;
            }
            else {
                --idx;
                needBack = true;
            }
        }
        return ret;
    }
    // Sum Root to Leaf Numbers
    // Given a binary tree containing digits from 0-9 only, each root-to-leaf path could represent a number.
    // An example is the root-to-leaf path 1->2->3 which represents the number 123.
    // Find the total sum of all root-to-leaf numbers.
    int sumNumbers(TreeNode *root) {
        // Recursive
        int curSum = 0, totalSum = 0;
        sumNumbersHelper(root, curSum, totalSum);
        return totalSum;
    }
    void sumNumbersHelper(TreeNode *root, int curSum, int &totalSum) {
        // Here, curSum should not be reference.
        if (root == nullptr) {
            return;
        }
        curSum = 10 * curSum + root->val;
        if (root->left == nullptr && root->right == nullptr) {
            totalSum += curSum;
            return;
        }
        sumNumbersHelper(root->left, curSum, totalSum);
        sumNumbersHelper(root->right, curSum, totalSum);
    }
    // Path Sum
    // Given a binary tree and a sum, determine if the tree has a root-to-leaf path such that adding up all the values along the path equals the given sum.
    bool hasPathSum(TreeNode *root, int sum) {
        // Recursive
        bool found = false;
        int curSum = 0;
        hasPathSumHelper(root, sum, found, curSum);
        return found;
    }
    void hasPathSumHelper(TreeNode *root, int sum, bool &found, int curSum) {
        if (root == nullptr) {
            return;
        }
        curSum += root->val;
        if (found == true) {
            // Prunning
            return;
        }
        if (root->left == nullptr && root->right == nullptr && curSum == sum) {
            found = true;
            return;
        }
        hasPathSumHelper(root->left, sum, found, curSum);
        hasPathSumHelper(root->right, sum, found, curSum);
    }
    // Pascal's Triangle II
    // Given an index k, return the kth row of the Pascal's triangle.
    // For example, given k = 3,
    // Return [1,3,3,1].
    // Note:
    // Could you optimize your algorithm to use only O(k) extra space?
    vector<int> getRow(int rowIndex) {
        int length = rowIndex + 1;
        vector<int> temp(length,1);
        if (length <= 2) {
            return temp;
        }
        int prep = 0, cur = prep + 2,mov = cur;
        int count;
        // Start at 3 Pascal's Triangle
        for (int num = 3; num <= length; ++num) {
            count = 1;
            while (count <= num) {
                int prepNext = (prep + 1) % length;
                int movNext = (mov + 1) % length;
                if (count == 1 || count == num) {
                    temp[mov] = 1;
                }
                else {
                    temp[mov] = temp[prep] + temp[prepNext];
                    prep = prepNext;
                }
                mov = movNext;
                ++count;
            }
            prep = cur;
            cur = mov;
        }
        vector<int> ret;
        count = 1;
        while (count <= length) {
            ret.push_back(temp[prep]);
            prep = (prep + 1) % length;
            ++count;
        }
        return ret;
    }
    // Find Minimum in Rotated Sorted Array II
    // Follow up for "Find Minimum in Rotated Sorted Array":
    // What if duplicates are allowed?
    // Would this affect the run-time complexity? How and why?
    int findMinII(vector<int> &num) {
        int beg = 0, end = num.size();
        int pos;
        while (beg < end) {
            int mid = (beg + end - 1) / 2;
            if (num[beg] < num[end - 1]) {
                // Already sorted, return the beg.
                pos = beg;
                break;
            }
            else if (num[beg] == num[end - 1]) {
                // handle the duplicates
                --end;
            }
            else {
                if (num[mid] >= num[beg]) {
                    beg = mid + 1;
                }
                else if (num[mid] < num[beg]) {
                    end = mid + 1;
                }
            }
            pos = beg;
        }
        return num[pos];
    }
    // Trapping Rain Water
    // Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.
    // For example,
    // Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.
    int trap(int A[], int n) {
        // Recursive
        if (n < 2) {
            return 0;
        }
        int beg = 0, end = n - 1;
        int height = min(A[beg], A[end]);
        int max = -1, index = 0;
        for (int idx = beg + 1; idx < end; ++idx) {
            if (A[idx] > max) {
                max = A[idx];
                index = idx;
            }
        }
        if (height < max) {
            // Ensure the edge height to hold the water
            return trap(A, index + 1) + trap(A + index, n - index);
        }
        int ret = 0;
        for (int idx = beg + 1; idx < end; ++idx) {
            if (height > A[idx]) {
                ret += height - A[idx];
            }
        }
        return ret;
    }
    int trap_iteration(int A[], int n) {
        // Iterative
        int left = 0, right = n - 1;
        int area = 0;
        int maxLeft = A[left], maxRight = A[right];
        while (left < right) {
            if (A[left] <= A[right]) {
                if (A[left] >= maxLeft) {
                    maxLeft = A[left];
                }
                else {
                    area += maxLeft - A[left];
                }
                ++left;
            }
            else {
                if (A[right] >= maxRight) {
                    maxRight = A[right];
                }
                else {
                    area += maxRight - A[right];
                }
                --right;
            }
        }
        return area;
    }
    // Minimum Depth of Binary Tree
    // Given a binary tree, find its minimum depth.
    // The minimum depth is the number of nodes along the shortest path from the root node down to the nearest leaf node.
    int minDepth(TreeNode *root) {
        // Recursive
        if (root == nullptr) {
            // Null tree
            return 0;
        }
        if (root->left == nullptr && root->right == nullptr) {
            // Leaf node
            return 1;
        }
        else if (root->left != nullptr && root->right != nullptr) {
            int leftDepth = 1 + minDepth(root->left), rightDepth = 1 + minDepth(root->right);
            return leftDepth <= rightDepth ? leftDepth : rightDepth;
        }
        else {
            if (root->left != nullptr) {
                return minDepth(root->left) + 1;
            }
            else {
                return minDepth(root->right) + 1;
            }
        }
    }
    int minDepth_iteration(TreeNode *root) {
        // Iterative
        // Two queues
        if (root == nullptr) {
            return 0;
        }
        queue<TreeNode *> nodeQueue;
        queue<int> depthQueue;
        nodeQueue.push(root);
        depthQueue.push(1);
        int depth;
        while (!nodeQueue.empty()) {
            TreeNode *ptr = nodeQueue.front();
            nodeQueue.pop();
            depth = depthQueue.front();
            depthQueue.pop();
            if (ptr->left == nullptr && ptr->right == nullptr) {
                break;
            }
            if (ptr->left != nullptr) {
                nodeQueue.push(ptr->left);
                depthQueue.push(depth + 1);
            }
            if (ptr->right != nullptr) {
                nodeQueue.push(ptr->right);
                depthQueue.push(depth + 1);
            }
        }
        return depth;
    }
    int minDepth_iteration2(TreeNode *root) {
        // Iterative
        // One queue
        if (root == nullptr) {
            return 0;
        }
        queue<TreeNode *> nodeQueue;
        nodeQueue.push(root);
        int depth = 1;
        while (!nodeQueue.empty()) {
            int levelNum = nodeQueue.size();
            for (int idx = 1; idx <= levelNum; ++idx) {
                TreeNode *ptr = nodeQueue.front();
                nodeQueue.pop();
                if (ptr->left == nullptr && ptr->right == nullptr) {
                    // Leaf node
                    return depth;
                }
                if (ptr->left != nullptr) {
                    nodeQueue.push(ptr->left);
                }
                if (ptr->right != nullptr) {
                    nodeQueue.push(ptr->right);
                }
            }
            ++depth;
        }
        return depth;
    }
    // Length of Last Word
    // Given a string s consists of upper/lower-case alphabets and empty space characters ' ', return the length of last word in the string.
    // If the last word does not exist, return 0.
    // Note: A word is defined as a character sequence consists of non-space characters only.
    // For example, Givens "Hello World" return 5.
    int lengthOfLastWord(const char *s) {
        int len = strlen(s);
        int beg = 0, end = -1;
        for (int idx = len - 1; idx >= 0; --idx) {
            if (s[idx] == ' ') {
                if (end < 0) {
                    if (idx != 0) {
                        continue;
                    }
                    else {
                        return 0;
                    }
                }
                else {
                    beg = idx + 1;
                    break;
                }
            }
            else {
                if (end < 0) {
                    end = idx;
                }
            }
        }
        return end - beg + 1;
    }

    int lengthOfLastWord_improved(const char *s) {
        // A more concise version
        int len = strlen(s), ret = 0;
        while (len > 0 && s[len - 1] == ' ') {
            --len;
        }
        while (len > 0 && s[len - 1] != ' ') {
            ++ret;
            --len;
        }
        return ret;
    }
    // Palindrome Number
    // Determine whether an integer is a palindrome. Do this without extra space.
    bool isPalindrome(int x) {
        if (x == 0) {
            return true;
        }
        if (x < 0) {
            return false;
        }
        // Noticing about overflow!
        int bitNumber = 0;
        long long tens = 1;
        int num = x;
        while (num > 0) {
            num /= 10;
            tens *= 10;
            ++bitNumber;
        }
        tens /= 10;
        bitNumber /= 2;
        num = x;
        int tmp = 1;
        while (bitNumber > 0) {
            --bitNumber;
            int firstNum = num / tens % 10;
            int lastNum = num / tmp % 10;
            if (firstNum != lastNum) {
                return false;
            }
            tens /= 10;
            tmp *= 10;
        }
        return true;
    }
    bool isPalindrome_improved(int x) {
        if (x < 0) {
            return false;
        }
        int original = x, reversal = 0;
        while (original >= 10) {
            reversal = reversal * 10 + original % 10;
            original /= 10;
        }
        return reversal == x / 10 && original == x % 10;
    }
    // Remove Nth Node From End of List
    // Given a linked list, remove the nth node from the end of list and return its head.
    // For example, Given linked list: 1->2->3->4->5, and n = 2.
    // After removing the second node from the end, the linked list becomes 1->2->3->5.
    // Note: Given n will always be valid. Try to do this in one pass.
    ListNode *removeNthFromEnd(ListNode *head, int n) {
        ListNode *temp = new ListNode(0);
        temp->next = head;
        if (head != nullptr && n > 0){
            ListNode *prep = temp, *ptr = head, *past = head;
            for (int i = 0; i < n; ++i) {
                past = past->next;
            }
            while (past != nullptr){
                prep = prep->next;
                ptr = ptr->next;
                past = past->next;
            }
            prep->next = ptr->next;
        }
        return temp->next;
    }
    // Factorial Trailing Zeroes
    // Given an integer n, return the number of trailing zeroes in n!.
    // Note: Your solution should be in logarithmic time complexity.
    int trailingZeros(int n) {
        int zeros = 0, num = n;
        if (num <= 0) {
            return 0;
        }
        while (num > 0) {
            int times = num / 5;
            zeros += times;
            num = times;
        }
        return zeros;
    }
    // Longest Consecutive Sequence
    // Given an unsorted array of integers, find the length of the longest consecutive elements sequence.
    // For example, Given [100, 4, 200, 1, 3, 2], The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.
    // Your algorithm should run in O(n) complexity.
    int longestConsecutive(vector<int> &num) {
        int res = 0;
        unordered_map<int, int> tempMap;
        for (int idx = 0; idx != num.size(); ++idx) {
            int ele = num[idx];
            if (!tempMap[ele]) {
                tempMap[ele] = 1 + tempMap[ele + 1] + tempMap[ele - 1];
                if (tempMap[ele + 1]) {
                    // Updating the right edge's value
                    tempMap[ele + tempMap[ele + 1]] = tempMap[ele];
                }
                if (tempMap[ele - 1]) {
                    // Updating the left edge's value
                    tempMap[ele - tempMap[ele - 1]] = tempMap[ele];
                }
            }
            res = max(tempMap[ele], res);
        }
        return res;
    }
    // Flatten Binary Tree to Linked List
    // Given a binary tree, flatten it to a linked list in-place.
    // If you notice carefully in the flattened tree, each node's right child points to the next node of a pre-order traversal.
    void flatten(TreeNode *root) {
        if (root == nullptr) {
            return;
        }
        stack<TreeNode *> nodeStack;
        nodeStack.push(root);
        TreeNode *prep = nullptr;
        while (!nodeStack.empty()) {
            TreeNode *ptr = nodeStack.top();
            nodeStack.pop();
            if (prep != nullptr) {
                prep->left = nullptr;
                prep->right = ptr;
            }
            prep = ptr;
            if (ptr->right != nullptr) {
                nodeStack.push(ptr->right);
            }
            if (ptr->left != nullptr) {
                nodeStack.push(ptr->left);
            }
        }
    }
    // Unique Paths II
    // Follow up for "Unique Paths":
    // Now consider if some obstacles are added to the grids. How many unique paths would there be?
    // An obstacle and empty space is marked as 1 and 0 respectively in the grid.
    // Note: m and n will be at most 100.
    int uniquePathsWithObstacles(vector<vector<int> > &obstacleGrid) {
        // Dynamic Programming
        int col = obstacleGrid[0].size(), row = obstacleGrid.size();
        vector<int> line(col, 0);
        vector<vector<int> > dp(row, line);

        for (int idx = 0; idx != col; ++idx) {
            if (obstacleGrid[0][idx] == 1) {
                break;
            }
            else {
                dp[0][idx] = 1;
            }
        }
        for (int idx = 0; idx != row; ++idx) {
            if (obstacleGrid[idx][0] == 1) {
                break;
            }
            else {
                dp[idx][0] = 1;
            }
        }

        for (int idxi = 1; idxi < row; ++idxi) {
            for (int idxj = 1; idxj < col; ++idxj) {
                if (obstacleGrid[idxi][idxj] != 1) {
                       dp[idxi][idxj] = dp[idxi - 1][idxj] + dp[idxi][idxj - 1];
                }
            }
        }
        return dp[row - 1][col - 1];
    }
    // Valid Parentheses
    // Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
    // The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.
    bool isValid(string s) {
        stack<char> parentheses;
        for (int idx = 0; idx != s.size(); ++idx) {
            if (parentheses.empty() || s[idx] == '(' || s[idx] == '[' || s[idx] == '{') {
                parentheses.push(s[idx]);
            }
            else {
                if (parentheses.empty()) {
                    return false;
                }
                char topEle = parentheses.top();
                parentheses.pop();
                if ((topEle == '(' && s[idx] != ')') || (topEle == '[' && s[idx] != ']') || (topEle == '{' && s[idx] != '}')) {
                    return false;
                }
            }
        }
        return parentheses.empty() ? true : false;
    }
    // Subsets
    // Given a set of distinct integers, S, return all possible subsets.
    // Note:
    // 1.Elements in a subset must be in non-descending order.
    // 2.The solution set must not contain duplicate subsets.
    // For example,
    // If S = [1,2,3], a solution is:
    // [
    //   [3],
    //   [1],
    //   [2],
    //   [1,2,3],
    //   [1,3],
    //   [2,3],
    //   [1,2],
    //   []
    // ]
    vector<vector<int> > subsets(vector<int> &S) {
        sort(S.begin(), S.end());
        vector<vector<int> > ret;
        vector<int> subSet;
        subsetsHelper(S, 0, subSet, ret);
        return ret;
    }
    void subsetsHelper(const vector<int> &S, int idx, vector<int> &subSet, vector<vector<int> > &ret) {
        if (idx == S.size()) {
            ret.push_back(subSet);
            return;
        }
        subSet.push_back(S[idx]);
        subsetsHelper(S, idx + 1, subSet, ret);
        subSet.pop_back();
        subsetsHelper(S, idx + 1, subSet, ret);
    }

    vector<vector<int> > subsets_iteration(vector<int> &S) {
        sort(S.begin(), S.end());
        vector<vector<int> > ret(1, vector<int>());
        for (int idx = 0; idx < S.size(); ++idx) {
            int N = ret.size();
            for (int j = 0; j < N; ++j) {
                ret.push_back(ret[j]);
                ret.back().push_back(S[idx]);
            }
        }
        return ret;
    }
    // Search for a Range
    // Given a sorted array of integers, find the starting and ending position of a given target value.
    // Your algorithm's runtime complexity must be in the order of O(log n).
    // If the target is not found in the array, return [-1, -1].
    // For example,
    // Given [5, 7, 7, 8, 8, 10] and target value 8,
    // return [3, 4].
    vector<int> searchRange(int A[], int n, int target) {
        // The worst case is not O(logN)
        int beg = 0, end = n - 1;
        int firstIdx, lastIdx;
        while (beg < end - 1) {
            int mid = beg + (end - beg) / 2;
            if (A[mid] <= target) {
                beg = mid;
            }
            else {
                end = mid;
            }
        }
        vector<int> ret;
        if (n <= 0 || (A[end] != target && A[beg] != target)) {
            // Not found
            vector<int> ret(2, -1);
            return ret;
        }
        else if (A[end] == target) {
            lastIdx = end;
        }
        else {
            lastIdx = beg;
        }
        firstIdx = lastIdx;
        while (firstIdx >= 0 && A[--firstIdx] == target);
        ret.push_back(++firstIdx);
        ret.push_back(lastIdx);

        return ret;
    }

    vector<int> searchRange_improved(int A[], int n, int target) {
        // A always O(logN) algorithm
        int firstIdx = searchRangeLeft(A, n, target), lastIdx = searchRangeRight(A, n, target);
        vector<int> ret;
        ret.push_back(firstIdx);
        ret.push_back(lastIdx);
        return ret;
    }
    int searchRangeLeft(int A[], int n, int target) {
        int beg = 0, end = n - 1;
        while (beg <= end) {
            int mid = beg + (end - beg) / 2;
            if (A[mid] == target) {
                if (mid == 0 || A[mid - 1] != target) {
                    return mid;
                }
                else {
                    end = mid - 1;
                }
            }
            else if (A[mid] > target) {
                end = mid - 1;
            }
            else {
                beg = mid + 1;
            }
        }
        return -1;
    }
    int searchRangeRight(int A[], int n, int target) {
        int beg = 0, end = n - 1;
        while (beg <= end) {
            int mid = beg + (end - beg) / 2;
            if (A[mid] == target) {
                if (mid == n - 1 || A[mid + 1] != target) {
                    return mid;
                }
                else {
                    beg = mid + 1;
                }
            }
            else if (A[mid] > target) {
                end = mid - 1;
            }
            else {
                beg = mid + 1;
            }
        }
        return -1;
    }
    // Convert Sorted List to Binary Search Tree
    // Given a singly linked list where elements are sorted in ascending order, convert it to a height balanced BST.
    ListNode *current;
    TreeNode *sortedListToBST(ListNode *head) {
        // From leaf to root
        this->current = head;
        return populateTree(lengthOfList(head));
    }
    int lengthOfList(ListNode *head) {
        int ret = 0;
        ListNode *ptr = head;
        while (ptr != nullptr) {
            ++ret;
            ptr = ptr->next;
        }
        return ret;
    }
    TreeNode *populateTree(int n) {
        if (n == 0) {
            return nullptr;
        }
        TreeNode *root = new TreeNode(0);
        root->left = populateTree(n / 2);
        root->val = current->val;
        current = current->next;
        root->right = populateTree(n - n / 2 - 1);
        return root;
    }
    // Unique Binary Search Trees II
    // Given n, generate all structurally unique BST's (binary search trees) that store values 1...n.
    vector<TreeNode *> generateTrees(int n) {
        return generateHelper(1, n);
    }
    vector<TreeNode *> generateHelper(const int &beg, const int &end) {
        vector<TreeNode *> ret;
        TreeNode *root = nullptr;
        if (beg >= end) {
            TreeNode *root = (beg == end) ? new TreeNode(beg) : nullptr;
            ret.push_back(root);
            return ret;
        }
        for (int idx = beg;  idx <= end;  ++idx) {
            vector<TreeNode *> leftSubTree, rightSubTree;
            leftSubTree = generateHelper(beg, idx - 1);
            rightSubTree = generateHelper(idx + 1, end);

            for (int i = 0;  i < leftSubTree.size();  ++i) {
                for (int j = 0;  j< rightSubTree.size();  ++j) {
                    root = new TreeNode(idx);
                    root->left = leftSubTree[i];
                    root->right = rightSubTree[j];
                    ret.push_back(root);
                }
            }
        }
        return ret;
    }
    // Subsets II
    // Given a collection of integers that might contain duplicates, S, return all possible subsets.
    // Note:
    // 1.Elements in a subset must be in non-descending order.
    // 2.The solution set must not contain duplicate subsets.
    // For example,
    // If S = [1,2,2], a solution is:
    // [
    //     [2],
    //     [1],
    //     [1,2,2],
    //     [2,2],
    //     [1,2],
    //     [],
    // ]
    vector<vector<int> > subsetsWithDup(vector<int> &S) {
        vector<vector<int> > ret;
        vector<int> subSet;
        sort(S.begin(), S.end());
        subsetsWithDupHelper(S, 0, subSet, ret);
        return ret;
    }
    void subsetsWithDupHelper(vector<int> &S, int idx, vector<int> &subSet, vector<vector<int> > &ret) {
        if (idx >= S.size()) {
            ret.push_back(subSet);
            return;
        }
        int ele = S[idx];
        subSet.push_back(ele);
        subsetsWithDupHelper(S, idx + 1, subSet, ret);
        subSet.pop_back();
        while (idx != S.size() && S[idx] == ele) {
            ++idx;
        }
        subsetsWithDupHelper(S, idx, subSet, ret);
    }
    // Partition List
    // Given a linked list and a value x, partition it such that all nodes less than x come before nodes greater than or equal to x.
    // You should preserve the original relative order of the nodes in each of the two partitions.
    // For example,
    // Given 1->4->3->2->5->2 and x = 3,
    // return 1->2->2->4->3->5.
    ListNode *partition(ListNode *head, int x) {
        if (head == nullptr) {
            return head;
        }
        ListNode *smallPartBeg = head, *smallPartEnd = head, *curlPtr = head;
        ListNode *bigPartBeg = nullptr, *bigPartEnd = nullptr;
        while (curlPtr != nullptr) {
            if (curlPtr->val >= x) {
                if (bigPartBeg == nullptr) {
                    bigPartBeg = bigPartEnd = curlPtr;
                }
                else {
                    bigPartEnd->next = curlPtr;
                    bigPartEnd = curlPtr;
                }
                if (smallPartBeg == curlPtr) {
                    smallPartBeg = smallPartEnd = curlPtr->next;
                }
            }
            else {
                if (smallPartEnd != curlPtr) {
                    smallPartEnd->next = curlPtr;
                    smallPartEnd = curlPtr;
                }
            }
            curlPtr = curlPtr->next;
        }
        smallPartEnd != nullptr ? smallPartEnd->next = bigPartBeg : NULL;
        bigPartEnd != nullptr ? bigPartEnd->next = nullptr : NULL;

        return smallPartBeg != nullptr ? smallPartBeg : bigPartBeg;
    }
    // Combination Sum
    // Given a set of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.
    // The same repeated number may be chosen from C unlimited number of times.
    // Note:
    // All numbers (including target) will be positive integers.
    // Elements in a combination (a1, a2, … , ak) must be in non-descending order. (ie, a1 ≤ a2 ≤ … ≤ ak).
    // The solution set must not contain duplicate combinations.
    // For example, given candidate set 2,3,6,7 and target 7,
    // A solution set is:
    // [7]
    // [2, 2, 3]
    vector<vector<int> > combinationSum(vector<int> &candidates, int target) {
        vector<vector<int> > ret;
        vector<int> solution;
        sort(candidates.begin(), candidates.end());
        combinationSumHelper(candidates, target, solution, ret, 0, 0);
        return ret;
    }
    void combinationSumHelper(vector<int> &candidates, const int &target, vector<int> &solution, vector<vector<int> > &ret, int idx, int localSum){
        if (idx >= candidates.size() || localSum > target) {
            return;
        }
        int ele = candidates[idx];
        localSum += ele;
        solution.push_back(ele);
        if (localSum == target) {
            ret.push_back(solution);
        }
        else if (localSum < target) {
            combinationSumHelper(candidates, target, solution, ret, idx, localSum);
        }
        localSum -= ele;
        solution.pop_back();

        while (idx != candidates.size() && candidates[idx] == ele) {
            ++idx;
        }
        combinationSumHelper(candidates, target, solution, ret, idx, localSum);
    }

    vector<vector<int> > combinationSum_improved(vector<int> &candidates, int target) {
        // Another easy understanding method.
        vector<vector<int> > ret;
        vector<int> output;
        sort(candidates.begin(), candidates.end());
        combinationSum_improvedHelper(candidates.begin(), candidates.end(), target, output, ret);
        return ret;
    }

    void combinationSum_improvedHelper(vector<int>::iterator begin, vector<int>::iterator end, int target, vector<int> &output, vector<vector<int> > &ret) {
        if (target == 0) {
            ret.push_back(output);
            return;
        }
        // branch 1: use current number
        if (*begin <= target) {
            output.push_back(*begin);
            combinationSum_improvedHelper(begin, end, target - *begin, output, ret);
            output.pop_back();
        }
        // branch 2: don't use
        if (begin + 1 < end)  {
            combinationSum_improvedHelper(begin + 1, end, target, output, ret);
        }
    }
    // Valid Sudoku
    // Determine if a Sudoku is valid, according to: Sudoku Puzzles - The Rules.
    // The Sudoku board could be partially filled, where empty cells are filled with the character '.'.
    bool isValidSudoku(vector<vector<char> > &board) {
        vector<vector<int> > bucket;
        for (int idxi = 0; idxi < 9; ++idxi) {
            bucket = vector<vector<int> >(3, vector<int>(9, 0));
            for (int idxj = 0; idxj < 9; ++idxj) {
                if (board[idxi][idxj] != '.' && ++bucket[0][board[idxi][idxj] - '1'] > 1) {
                    return false;
                }
                if (board[idxj][idxi] != '.' && ++bucket[1][board[idxj][idxi] - '1'] > 1) {
                    return false;
                }
                int row = 3 * (idxi / 3) + idxj / 3, col = 3 * (idxi % 3) + idxj % 3;
                if (board[row][col] != '.' && ++bucket[2][board[row][col] - '1'] > 1) {
                    return false;
                }
            }
        }
        return true;
    }
    // Intersection of Two Linked Lists
    // Write a program to find the node at which the intersection of two singly linked lists begins.
    // A:      a1 → a2
    //                ↘
    //                  c1 → c2 → c3
    //                ↗
    // B: b1 → b2 → b3
    // begin to intersect at node c1.
    // Notes:
    // If the two linked lists have no intersection at all, return null.
    // The linked lists must retain their original structure after the function returns.
    // You may assume there are no cycles anywhere in the entire linked structure.
    // Your code should preferably run in O(n) time and use only O(1) memory.
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        if (headA == nullptr || headB == nullptr) {
            return nullptr;
        }
        ListNode *endA = headA;
        while (endA->next != nullptr) {
            endA = endA->next;
        }
        // Let the tail of the A points to the head of B
        endA->next = headB;

        ListNode *slow = headA, *fast = headA;
        while (fast->next != nullptr && fast->next->next != nullptr) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) {
                fast = headA;
                while (fast != slow) {
                    fast = fast->next;
                    slow = slow->next;
                }
                // Keep the original Linked List
                endA->next = nullptr;
                return slow;
            }
        }
        // Keep the original Linked List
        endA->next = nullptr;
        return nullptr;
    }
    // Jump Game
    // Given an array of non-negative integers, you are initially positioned at the first index of the array.
    // Each element in the array represents your maximum jump length at that position.
    // Determine if you are able to reach the last index.
    // For example:
    // A = [2,3,1,1,4], return true.
    // A = [3,2,1,0,4], return false.
    bool canJump(int A[], int n) {
        if (n <= 1) {
            return true;
        }
        int idx = n - 2, dist = 1;
        while (idx >= 0) {
            (A[idx] >= dist) ? (dist = 1) : ++dist;
            --idx;
        }
        return dist == 1 ? true : false;
    }
    // 3Sum Closest
    // Given an array S of n integers, find three integers in S such that the sum is closest to a given number, target. Return the sum of the three integers. You may assume that each input would have exactly one solution.
    // For example, given array S = {-1 2 1 -4}, and target = 1.
    // The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).
    int threeSumClosest(vector<int> &num, int target) {
        sort(num.begin(), num.end());
        int len = num.size();
        // If the length ≤ 3, then return the sum of the array.
        if (len <= 3) {
            return accumulate(num.begin(), num.end(), 0);
        }
        // Initial the sum.
        int sum = num[0] + num[1] + num[len - 1];
        for (int idx = 0; idx < len - 2; ++idx) {
            int b = idx + 1, e = len - 1;
            while (b < e) {
                int temp = num[b] + num[e] + num[idx];
                if (abs(sum - target) > abs(temp - target)) {
                    sum = temp;
                    // Solution is only one.
                    if (sum == target) {
                        return sum;
                    }
                }
                temp > target ? --e : ++b;
            }
        }
        return sum;
    }
    // Triangle
    // Given a triangle, find the minimum path sum from top to bottom. Each step you may move to adjacent numbers on the row below.
    // For example, given the following triangle
    // [
    //      [2],
    //     [3,4],
    //    [6,5,7],
    //   [4,1,8,3]
    // ]
    // The minimum path sum from top to bottom is 11 (i.e., 2 + 3 + 5 + 1 = 11).
    int minimumTotal(vector<vector<int> > &triangle) {
        int N = triangle.size();
        // Initialize the O(N) extra space.
        vector<int> tempVector(N, 0);
        tempVector[0] = triangle[0][0];
        for (int row = 1; row < N; ++row) {
            int prep = tempVector[0], cur;
            tempVector[0] += triangle[row][0];
            for (int col = 1; col < row; ++col) {
                cur = tempVector[col];
                tempVector[col] = prep > cur ? cur : prep;
                tempVector[col] += triangle[row][col];
                prep = cur;
            }
            tempVector[row] = prep + triangle[row][row];
        }

        // Find the minimum of the tempVector.
        int ret = tempVector[0];
        for (int idx = 1; idx < N; ++idx) {
            ret = ret > tempVector[idx] ? tempVector[idx] : ret;
        }
        return ret;
    }
    // Path Sum II
    // Given a binary tree and a sum, find all root-to-leaf paths where each path's sum equals the given sum.
    // For example:
    // Given the below binary tree and sum = 22,
    //            5
    //           / \
    //          4   8
    //         /   / \
    //        11  13  4
    //       /  \    / \
    //      7    2  5   1
    // return
    // [
    //     [5,4,11,2],
    //     [5,8,4,5]
    // ]
    vector<vector<int> > pathSum(TreeNode *root, int sum) {
        vector<vector<int> > ret;
        vector<int> route;
        int curSum = 0;
        pathSumHelper(root, route, ret, curSum, sum);
        return ret;
    }
    void pathSumHelper(TreeNode *root, vector<int> route, vector<vector<int> > &ret, int curSum, const int &sum) {
        if (root == nullptr) {
            return;
        }
        curSum += root->val;
        route.push_back(root->val);
        if (root->left == nullptr && root->right == nullptr && curSum == sum) {
            ret.push_back(route);
        }
        pathSumHelper(root->left, route, ret, curSum, sum);
        pathSumHelper(root->right, route, ret, curSum, sum);
    }
    // Construct Binary Tree from Inorder and Postorder Traversal
    // Given inorder and postorder traversal of a tree, construct the binary tree.
    // Note:
    // You may assume that duplicates do not exist in the tree.
    TreeNode *buildTree(vector<int> &inorder, vector<int> &postorder) {
        return buildTreeHelper(inorder, 0, inorder.size() - 1, postorder, 0, postorder.size() - 1);
    }
    TreeNode *buildTreeHelper(vector<int> &inorder, const int &inBeg, const int &inEnd, vector<int> &postorder, const int &postBeg, const int &postEnd) {
        TreeNode *root = nullptr;
        if (inBeg <= inEnd) {
            root = new TreeNode(postorder[postEnd]);
            int idx;
            // Devide the inorder sequence into two parts based on root.
            for (idx = inBeg; idx <= inEnd; ++idx) {
                if (inorder[idx] == postorder[postEnd]) {
                    break;
                }
            }
            root->left = buildTreeHelper(inorder, inBeg, idx - 1, postorder, postBeg, postBeg + (idx - inBeg) -1);
            root->right = buildTreeHelper(inorder, idx + 1, inEnd, postorder, postBeg + (idx - inBeg), postEnd - 1);
        }
        return root;
    }
    // Construct Binary Tree from Preorder and Inorder Traversal
    // Given preorder and inorder traversal of a tree, construct the binary tree.
    TreeNode *buildTreeII(vector<int> &preorder, vector<int> &inorder) {
        return buildTreeIIHelper(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1);
    }
    TreeNode *buildTreeIIHelper(vector<int> &preorder, const int &preBeg, const int &preEnd, vector<int> &inorder, const int &inBeg, const int &inEnd) {
        TreeNode *root = nullptr;
        if (inBeg <= inEnd) {
            root = new TreeNode(preorder[preBeg]);
            int idx;
            // Devide the inorder sequence into two parts based on root.
            for (idx = inBeg; idx <= inEnd; ++idx) {
                if (inorder[idx] == preorder[preBeg]) {
                    break;
                }
            }
            root->left = buildTreeIIHelper(preorder, preBeg + 1, preBeg + (idx - inBeg), inorder, inBeg, idx - 1);
            root->right = buildTreeIIHelper(preorder, preBeg + (idx - inBeg) + 1, preEnd, inorder, idx + 1, inEnd);
        }
        return root;
    }
    // Longest Common Prefix
    // Write a function to find the longest common prefix string amongst an array of strings.
    string longestCommonPrefix(vector<string> &strs) {
        string ret;
        if (strs.size() <= 0) {
            return ret;
        }
        ret.assign(strs[0]);
        for (int idx = 1; idx < strs.size(); ++idx) {
            int len = min(ret.length(), strs[idx].length());
            int cur = 0;
            for (; cur < len; ++cur) {
                if (ret[cur] != strs[idx][cur]) {
                    break;
                }
            }
            if (cur == 0) {
                return "";
            }
            else {
                ret.assign(strs[idx], 0, cur);
            }
        }
        return ret;
    }
    // Pow(x, n)
    // Implement pow(x, n).
    double pow(double x, int n) {
        // O(logN)
        // x ^ 9 = x ((x^2)^2)^2
        // x ^ 6 = x^2 (x^2)^2
        double ret = 1.0;
        for (int i = n; i != 0; i /= 2, x *= x) {
            if (i % 2 != 0) {
                ret *= x;
            }
        }
        return n < 0 ? 1.0 / ret : ret;
    }

    double pow_recursion(double x, int n) {
        // Recursive
        // O(logN)
        if (n == 0) {
            return 1;
        }
        else if (n == 1) {
            return x;
        }
        // handle the case 1.0, INT_MIN
        else if (abs(x) == 1) {
            return n % 2 == 0 ? 1 : x;
        }
        if (n < 0) {
            return pow_recursion(1.0 / x, -n);
        }
        else if (n % 2 == 0) {
            return pow_recursion(x * x, n / 2);
        }
        else {
            return x * pow_recursion(x * x, (n - 1) / 2);
        }
    }
    // Letter Combinations of a Phone Number
    // Given a digit string, return all possible letter combinations that the number could represent.
    // A mapping of digit to letters (just like on the telephone buttons) is given below.
    // Input:Digit string "23"
    // Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
    vector<string> letterCombinations(string digits) {
        vector<string> ret;
        if (digits.length() == 0) {
            string letters;
            ret.push_back(letters);
            return ret;
        }
        // Map between numbers and letters.
        string tmp[] = {" ", "ΩΩ", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        vector<string> numToLetterMap(tmp, tmp + 10);
        string letters;
        letterCombinationsHelper(0, digits, numToLetterMap, letters, ret);
        return ret;
    }
    void letterCombinationsHelper(const int &pos, const string &digits, const vector<string> &numToLetterMap, string letters, vector<string> &ret) {
        if (pos >= digits.length()) {
            ret.push_back(letters);
            return;
        }
        for (int idx = 0; idx < numToLetterMap[digits[pos] - '0'].size(); ++idx) {
            letterCombinationsHelper(pos + 1, digits, numToLetterMap, letters + numToLetterMap[digits[pos] - '0'][idx], ret);
        }
    }
    // Binary Tree Zigzag Level Order Traversal
    // Given a binary tree, return the zigzag level order traversal of its nodes' values. (ie, from left to right, then right to left for the next level and alternate between).
    // For example:
    // Given binary tree {3,9,20,#,#,15,7},
    //     3
    //    / \
    //   9  20
    //  / \
    // 15  7
    // return its zigzag level order traversal as:
    // [
    //   [3],
    //   [20,9],
    //   [15,7]
    // ]
    vector<vector<int> > zigzagLevelOrder(TreeNode *root) {
        vector<vector<int> > ret;
        queue<TreeNode *> nodeQueue;
        if (root == nullptr) {
            return ret;
        }
        nodeQueue.push(root);
        nodeQueue.push(nullptr);
        vector<int> tmp;
        ret.push_back(tmp);
        while (!nodeQueue.empty()) {
            TreeNode *tmp = nodeQueue.front();
            nodeQueue.pop();
            if (tmp == nullptr) {
                if (nodeQueue.front() == nullptr) {
                    break;
                }
                vector<int> line;
                ret.push_back(line);
            }
            else {
                ret[ret.size() - 1].push_back(tmp->val);
                if (tmp->left != nullptr) {
                    nodeQueue.push(tmp->left);
                }
                if (tmp->right != nullptr) {
                    nodeQueue.push(tmp->right);
                }
                if (nodeQueue.front() == nullptr) {
                    nodeQueue.push(nullptr);
                }
            }
        }
        // Zigzag
        for (int idx = 1; idx < ret.size(); idx += 2) {
            reverse(ret[idx].begin(), ret[idx].end());
        }
        return ret;
    }
    // Reverse Linked List II
    // Reverse a linked list from position m to n. Do it in-place and in one-pass.
    // For example:
    // Given 1->2->3->4->5->NULL, m = 2 and n = 4,
    // return 1->4->3->2->5->NULL.
    // Note:
    // Given m, n satisfy the following condition:
    // 1 ≤ m ≤ n ≤ length of list.
    ListNode *reverseBetween(ListNode *head, int m, int n) {
        if (head == nullptr) {
            return head;
        }
        int count = 1;
        ListNode *reverseHead = head, *breakpoint = head;
        // Find the breakpoint node and the reverseHead node which points to the begining of the reverse range.
        if (m == 1) {
            breakpoint = nullptr;
        }
        else {
            while (count < m - 1) {
                ++count;
                breakpoint = breakpoint->next;
            }
            reverseHead = breakpoint->next;
            ++count;
        }
        ListNode *reverseTail = reverseHead, *remainHead = reverseHead->next;
        while (count < n) {
            ListNode *tmp = remainHead->next;
            remainHead->next = reverseTail;

            reverseTail = remainHead;
            remainHead = tmp;
            ++count;
        }
        (breakpoint == nullptr) ? (head = reverseTail) : (breakpoint->next = reverseTail);
        reverseHead->next = remainHead;
        return head;
    }
    // Palindrome Partitioning
    // Given a string s, partition s such that every substring of the partition is a palindrome.
    // Return all possible palindrome partitioning of s.
    // For example, given s = "aab",
    // Return
    // [
    //   ["aa","b"],
    //   ["a","a","b"]
    // ]
    vector<vector<string> > partition(string s) {
        vector<vector<string> > ret;
        vector<string> line;
        partitionHelper(s, 0, line, ret);
        return ret;
    }
    void partitionHelper(const string &s, int idx, vector<string> line, vector<vector<string> > &ret) {
        if (idx > s.size() - 1) {
            ret.push_back(line);
            return;
        }
        string str;
        for (int i = idx; i < s.size(); ++i) {
            str += s[i];
            if (isPalindromeString(str)) {
                line.push_back(str);
                partitionHelper(s, i + 1, line, ret);
                line.pop_back();
            }
        }
    }
    bool isPalindromeString(const string &s) {
        int beg = 0, end = s.size() - 1;
        while (beg < end && s[beg] == s[end]) {
            ++beg;
            --end;
        }
        if (beg >= end) {
            return true;
        }
        else {
            return false;
        }
    }
    // N-Queens
    // Given an integer n, return all distinct solutions to the n-queens puzzle.
    // Each solution contains a distinct board configuration of the n-queens' placement, where 'Q' and '.' both indicate a queen and an empty space respectively.
    // For example,
    // There exist two distinct solutions to the 4-queens puzzle:
    // [
    //  [".Q..",  // Solution 1
    //   "...Q",
    //   "Q...",
    //   "..Q."],
    //  ["..Q.",  // Solution 2
    //   "Q...",
    //   "...Q",
    //   ".Q.."]
    // ]
    vector<vector<string> > solveNQueens(int n) {
        vector<vector<string> > ret;
        vector<int> sol(n + 1, 0);
        int idx = 1;
        while (idx > 0) {
            ++sol[idx];
            while (sol[idx] <= n && legalPosition(sol, idx) == false) {
                ++sol[idx];
            }
            if (sol[idx] <= n) {
                if (idx == n) {
                    vector<string> line;
                    for (int i = 1; i <= n; ++i) {
                        string s(sol[i] - 1, '.');
                        s.append("Q");
                        s.append(n - sol[i], '.');
                        line.push_back(s);
                    }
                    ret.push_back(line);
                }
                else {
                    ++idx;
                    sol[idx] = 0;
                }
            }
            else {
                --idx;
            }
        }
        return ret;
    }
    // Gas Station
    // There are N gas stations along a circular route, where the amount of gas at station i is gas[i].
    // You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from station i to its next station (i+1). You begin the journey with an empty tank at one of the gas stations.
    // Return the starting gas station's index if you can travel around the circuit once, otherwise return -1.
    // Note:
    // The solution is guaranteed to be unique.
    int canCompleteCircuit(vector<int> &gas, vector<int> &cost) {
        // Greedy
        // If the ith station can't reach the jth station, then the i+1, i+2...j-1 th station can't reach the jth station neither.
        int sum = 0, index = 0;
        int N = gas.size();
        for (int idx = 0; idx < 2 * N - 1; ++idx) {
            sum += gas[idx % N] - cost[idx % N];
            if (sum < 0) {
                index = idx + 1;
                if (index >= N) {
                    return -1;
                }
                sum = 0;
            }
        }
        return index;
    }
    // Edit Distance
    // Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. (each operation is counted as 1 step.)
    // You have the following 3 operations permitted on a word:
    // a) Insert a character
    // b) Delete a character
    // c) Replace a character
    int minDistance(string word1, string word2) {
        int len1 = word1.size(), len2 = word2.size();
        if (len1 == 0) {
            return len2;
        }
        if (len2 == 0) {
            return len1;
        }
        // bucket[i][j] represents the dist from word1.substr(0, i) to word2.substr(0, j).
        vector<vector<int> >bucket = vector<vector<int> >(len1 + 1, vector<int>(len2 + 1, 0));
        for (int idx = 0; idx <= len1; ++idx) {
            bucket[idx][0] = idx;
        }
        for (int idx = 0; idx <= len2; ++idx) {
            bucket[0][idx] = idx;
        }
        for (int curA = 1; curA <= len1; ++curA) {
            for (int curB = 1; curB <= len2; ++curB) {
                int cost = (word1[curA - 1] == word2[curB - 1] ? 0 : 1);
                bucket[curA][curB] = min(min(bucket[curA][curB - 1] + 1, bucket[curA - 1][curB] + 1), bucket[curA - 1][curB - 1] + cost);
            }
        }
        return bucket[len1][len2];
    }
    // Count and Say
    // The count-and-say sequence is the sequence of integers beginning as follows:
    // 1, 11, 21, 1211, 111221, ...
    // 1 is read off as "one 1" or 11.
    // 11 is read off as "two 1s" or 21.
    // 21 is read off as "one 2, then one 1" or 1211.
    // Given an integer n, generate the nth sequence.
    // Note: The sequence of integers will be represented as a string.
    string countAndSay(int n) {
        string s("1"), ret(s);
        char tmp;
        for (; n > 1; --n) {
            ret = "";
            // Initialize the tmp to be first char of the s.
            tmp = s.front();
            int count = 1;
            for (int idx = 1; idx < s.size(); ++idx) {
                if (tmp != s[idx]) {
                    char num = '1' + count - 1;
                    ret = ret + num + tmp;
                    tmp = s[idx];
                    count = 1;
                }
                else {
                    ++count;
                }
            }
            char num = '1' + count - 1;
            ret = ret + num + tmp;
            s = ret;
        }
        return ret;
    }
    // Insertion Sort List
    // Sort a linked list using insertion sort.
    ListNode *insertionSortList(ListNode *head) {
        if (head == nullptr) {
            return head;
        }
        ListNode *cur = head->next, *sortedTail = head;
        ListNode *tmp = new ListNode(0);
        tmp->next = head;
        while (cur != nullptr) {
            if (cur->val < sortedTail->val) {
                ListNode *ptr = tmp;
                while (ptr->next->val < cur->val) {
                    ptr = ptr->next;
                }
                sortedTail->next = cur->next;
                cur->next = ptr->next;
                ptr->next = cur;
                cur = sortedTail->next;
            }
            else {
                sortedTail = cur;
                cur = cur->next;
            }
        }
        head = tmp->next;
        delete tmp;
        return head;
    }
    // Distinct Subsequences
    // Given a string S and a string T, count the number of distinct subsequences of T in S.
    // A subsequence of a string is a new string which is formed from the original string by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (ie, "ACE" is a subsequence of "ABCDE" while "AEC" is not).
    //
    // Here is an example:
    // S = "rabbbit", T = "rabbit"
    // Return 3.
    int numDistinct(string S, string T) {
        int lenS = S.size(), lenT = T.size();
        vector<vector<int> > bucket = vector<vector<int> >(lenS + 1, vector<int>(lenT + 1, 0));
        for (int curS = 1; curS <= lenS; ++curS) {
            for (int curT = 1; curT <= min(lenT, curS); ++curT) {
                if (S[curS - 1] == T[curT - 1]) {
                    if (curT == 1) {
                        bucket[curS][curT] = bucket[curS - 1][curT] + 1;
                    }
                    else {
                        bucket[curS][curT] = bucket[curS - 1][curT - 1] + bucket[curS - 1][curT];
                    }
                }
                else {
                    bucket[curS][curT] = bucket[curS - 1][curT];
                }
            }
        }
        return bucket[lenS][lenT];
    }
    // Next Permutation
    // Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.
    // If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending order).
    // The replacement must be in-place, do not allocate extra memory.
    // Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the right-hand column.
    // 1,2,3 → 1,3,2
    // 3,2,1 → 1,2,3
    // 1,1,5 → 1,5,1
    void nextPermutation(vector<int> &num) {
        int start = num.size() - 1;
        while (start > 0 && num[start - 1] >= num[start]) {
            --start;
        }
        if (start > 0) {
            int idx = num.size() - 1;
            while (num[idx] <= num[start - 1]) {
                --idx;
            }
            swap(num[idx], num[start - 1]);
        }
        // Noticing about the edge to pass the case when start = num.size() - 2.
        for (int idx = start; idx < (start + num.size()) / 2; ++idx) {
            swap(num[idx], num[num.size() - 1 + start - idx]);
        }
        //reverse(num.begin() + start - 1, num.end());
    }
    // Permutations II
    // Given a collection of numbers that might contain duplicates, return all possible unique permutations.
    // For example,
    // [1,1,2] have the following unique permutations:
    // [1,1,2], [1,2,1], and [2,1,1].
    vector<vector<int> > permuteUnique(vector<int> &num) {
        vector<vector<int> > ret;
        if (num.size() == 0) {
            return ret;
        }
        sort(num.begin(), num.end());
        ret.push_back(num);
        while (next_permutation(num.begin(), num.end())) {
            ret.push_back(num);
        }
        return ret;
    }
    // Reverse Nodes in k-Group
    // Given a linked list, reverse the nodes of a linked list k at a time and return its modified list.
    // If the number of nodes is not a multiple of k then left-out nodes in the end should remain as it is.
    // You may not alter the values in the nodes, only nodes itself may be changed.
    // Only constant memory is allowed.
    // For example,
    // Given this linked list: 1->2->3->4->5
    // For k = 2, you should return: 2->1->4->3->5
    // For k = 3, you should return: 3->2->1->4->5
    ListNode *reverseKGroup(ListNode *head, int k) {
        ListNode *tmp = head;
        int length = 0;
        while (tmp != nullptr) {
            tmp = tmp->next;
            ++length;
        }
        for (int idx = k; idx <= length; idx += k) {
            head = reverseBetween(head, idx - k + 1, idx);
        }
        return head;
    }

    ListNode *reverseKGroup_improved(ListNode *head, int k) {
        ListNode *count = head;
        int length = 0;
        while (count != nullptr) {
            count = count->next;
            ++length;
        }
        if (length == 0 || k <= 1) {
            return head;
        }
        //    reverseHead  reverseTail
        //             |    | -> | -> |
        //  breakpoint |    | -> | -> |     remainHead
        //      \      |    | -> | -> |     /
        // ...-> x ->) a -> b -> c -> d ->( y ->...
        // ...-> x ->) a <- b <- c <- d   ( y ->...
        ListNode *reverseHead = head, *breakpoint = head, *reverseTail = reverseHead, *remainHead = reverseHead->next;
        for (int idx = k; idx <= length; idx += k) {
            int count = 1;
            // Reverse the list.
            while (count < k) {
                ListNode *tmp = remainHead->next;
                remainHead->next = reverseTail;
                reverseTail = remainHead;
                remainHead = tmp;
                ++count;
            }
            (breakpoint == head) ? (head = reverseTail) : (breakpoint->next = reverseTail);
            reverseHead->next = remainHead;
            if (remainHead == nullptr) {
                break;
            }
            // Let breakpoint->next to be the next reversion's first node.
            breakpoint = reverseHead;
            reverseTail = reverseHead = breakpoint->next;
            remainHead = reverseHead->next;
        }
        return head;
    }
    // Remove Duplicates from Sorted List II
    // Given a sorted linked list, delete all nodes that have duplicate numbers, leaving only distinct numbers from the original list.
    // For example,
    // Given 1->2->3->3->4->4->5, return 1->2->5.
    // Given 1->1->1->2->3, return 2->3.
    ListNode *deleteDuplicatesII(ListNode *head) {
        if (head == nullptr) {
            return head;
        }
        ListNode *cur = head;
        ListNode *prep = new ListNode(0);
        prep->next = cur;
        ListNode *newHead = prep;
        while (cur != nullptr) {
            while (cur->next != nullptr && cur->val != cur->next->val) {
                cur = cur->next;
                prep = prep->next;
            }
            if (cur->next == nullptr) {
                return newHead->next;
            }
            int dupValue = cur->val;
            while (cur != nullptr && cur->val == dupValue) {
                cur = cur->next;
            }
            prep->next = cur;
        }
        return newHead->next;
    }
    // Add Binary
    // Given two binary strings, return their sum (also a binary string).
    // For example,
    // a = "11"
    // b = "1"
    // Return "100".
    string addBinary(string a, string b) {
        int idxA = a.length() - 1, idxB = b.length() - 1;
        int carry = 0;
        string sum = "";
        while (idxA >= 0 || idxB >= 0 || carry) {
            int tmp = carry;
            if (idxA >= 0) {
                tmp += a[idxA--] - '0';
            }
            if (idxB >= 0) {
                tmp += b[idxB--] - '0';
            }
            // Notice!
            (tmp > 1) ? (carry = 1, tmp -= 2) : (carry = 0);
            sum = to_string(tmp) + sum;
        }
        return sum;
    }
    // Combination Sum II
    // Given a collection of candidate numbers (C) and a target number (T), find all unique combinations in C where the candidate numbers sums to T.
    // Each number in C may only be used once in the combination.
    // Note:
    // All numbers (including target) will be positive integers.
    // Elements in a combination (a1, a2, … , ak) must be in non-descending order. (ie, a1 ≤ a2 ≤ … ≤ ak).
    // The solution set must not contain duplicate combinations.
    // For example, given candidate set 10,1,2,7,6,1,5 and target 8,
    // A solution set is:
    // [1, 7]
    // [1, 2, 5]
    // [2, 6]
    // [1, 1, 6]
    vector<vector<int> > combinationSum2(vector<int> &num, int target) {
        vector<vector<int> > ret;
        vector<int> line;
        sort(num.begin(), num.end());
        combinationSum2Helper(0, line, ret, num, target);
        return ret;
    }
    void combinationSum2Helper(int idx, vector<int> &line, vector<vector<int> > &ret, const vector<int> &num, int target) {
        if (target == 0) {
            ret.push_back(line);
            return;
        }
        if (idx >= num.size()) {
            return;
        }
        // 1.Use current num.
        if (num[idx] <= target) {
            line.push_back(num[idx]);
            combinationSum2Helper(idx + 1, line, ret, num, target - num[idx]);
            line.pop_back();
        }
        // 2.Don't use current num.
        while (idx + 1 < num.size() && num[idx] == num[idx +  1]) {
            ++idx;
        }
        combinationSum2Helper(idx + 1, line, ret, num, target);
    }
    // Jump Game II
    // Given an array of non-negative integers, you are initially positioned at the first index of the array.
    // Each element in the array represents your maximum jump length at that position.
    // Your goal is to reach the last index in the minimum number of jumps.
    // For example:
    // Given array A = [2,3,1,1,4]
    // The minimum number of jumps to reach the last index is 2. (Jump 1 step from index 0 to 1, then 3 steps to the last index.)
    int jump(int A[], int n) {
        // Every jump, the range we can jump is [low, high].
        int low = 0, high = 0;
        int ret = 0;
        while (high < n - 1) {
            int preHigh = high;
            for (int idx = low; idx <= preHigh; ++idx) {
                high = (idx + A[idx]) > high ? A[idx] + idx : high;
                // (idx + A[idx]) > high ? (high = A[idx] + idx) : NULL;
            }
            low = preHigh + 1;
            ++ret;
        }
        return ret;
    }

    int jump_greedy(int A[], int n) {
        if(n <= 1){
            return 0;
        }
        int maxReachPos = A[0];
        int curMaxReachPos = A[0];
        int curStep = 1;
        for(int i = 1; i <= maxReachPos; i++){
            curMaxReachPos = max(curMaxReachPos, i + A[i]);
            if(i == n - 1){
                return curStep;
            }
            if(i == maxReachPos){
                maxReachPos = curMaxReachPos;
                curStep++;
            }
        }
        return 0;
    }
    // Copy List with Random Pointer
    // A linked list is given such that each node contains an additional random pointer which could point to any node in the list or null.
    // Return a deep copy of the list.
    RandomListNode *copyRandomList(RandomListNode *head) {
        // Using unordered_map
        if (head == nullptr) {
            return head;
        }
        // A head node.
        RandomListNode *copy = new RandomListNode(0);
        RandomListNode *curCopy = copy, *curOld = head;
        unordered_map<RandomListNode *, RandomListNode *> randomPointer;
        // Copy nodes.
        while (curOld != nullptr) {
            RandomListNode *node = new RandomListNode(curOld->label);
            curCopy->next = node;
            randomPointer[curOld] = node;

            curCopy = curCopy->next;
            curOld = curOld->next;
        }
        // Copy nodes' random pointers.
        curCopy = copy->next;
        curOld = head;
        while (curCopy != nullptr) {
            (curOld->random != nullptr) ? (curCopy->random = randomPointer[curOld->random]) : NULL;
            curCopy = curCopy->next;
            curOld = curOld->next;
        }

        return copy->next;
    }
    RandomListNode *copyRandomList_improved(RandomListNode *head) {
        // A constant space method.
        RandomListNode *cur = head, *next;
        // Scanning the List and make a copy for each node.
        while (cur != nullptr) {
            next = cur->next;
            RandomListNode *dupNode = new RandomListNode(cur->label);
            cur->next = dupNode;
            dupNode->next = next;

            cur = next;
        }
        cur = head;
        // Assigning random pointers for each copy nodes.
        while (cur != nullptr) {
            (cur->random != nullptr) ? (cur->next->random = cur->random->next) : NULL;
            cur = cur->next->next;
        }
        // Extract the copy list.
        RandomListNode *copy = new RandomListNode(0), *curCopy = copy;
        cur = head;
        while (cur != nullptr) {
            curCopy->next = cur->next;
            // Restore the original list.
            cur->next = cur->next->next;

            curCopy = curCopy->next;
            cur = cur->next;
        }
        return copy->next;
    }
    // Recover Binary Search Tree
    // Two elements of a binary search tree (BST) are swapped by mistake.
    // Recover the tree without changing its structure.
    // Note:
    // A solution using O(n) space is pretty straight forward. Could you devise a constant space solution?
    void recoverTree(TreeNode *root) {
        // O(logN) space complexity method using inorder traversal.
        TreeNode *first, *second, *prep;
        first = second = prep = nullptr;
        recoverTreeHelper(root, &first, &second, &prep);
        if (first != nullptr) {
            swap(first->val, second->val);
        }
    }
    void recoverTreeHelper(TreeNode *root, TreeNode **first, TreeNode **second, TreeNode **prep) {
        if (root == nullptr) {
            return;
        }
        recoverTreeHelper(root->left, first, second, prep);
        if (*prep != nullptr && (*prep)->val > root->val) {
            if (*first == nullptr) {
                *first = *prep;
                *second = root;
            }
            else {
                *second = root;
            }
        }
        *prep = root;
        recoverTreeHelper(root->right, first, second, prep);
    }
    // Anagrams
    // Given an array of strings, return all groups of strings that are anagrams.
    // Note: All inputs will be in lower-case.
    vector<string> anagrams(vector<string> &strs) {
        vector<string> ret;
        unordered_map<string, vector<string>::iterator> umap;
        for (vector<string>::iterator it = strs.begin(); it != strs.end(); ++it) {
            string tmp(*it);
            sort(tmp.begin(), tmp.end());
            if (umap.count(tmp) > 0) {
                ret.push_back(*it);
                if (umap[tmp] != strs.end()) {
                    ret.push_back(*(umap[tmp]));
                    umap[tmp] = strs.end();
                }
            }
            else {
                umap[tmp] = it;
            }
        }
        return ret;
    }
    // Maximum Gap
    // Given an unsorted array, find the maximum difference between the successive elements in its sorted form.
    // Try to solve it in linear time/space.
    // Return 0 if the array contains less than 2 elements.
    // You may assume all elements in the array are non-negative integers and fit in the 32-bit signed integer range.
    int maximumGap(vector<int> &num) {
        // Bucket sort.
        int N = num.size();
        if (N < 2) {
            return 0;
        }
        // 1.Find the minVal and the maxVal.
        int minVal, maxVal;
        minVal = maxVal = num[0];
        for (vector<int>::const_iterator it = num.begin() + 1; it != num.end(); ++it) {
            minVal = min(minVal, *it);
            maxVal = max(maxVal, *it);
        }

        // 2.Store every bucket's minVal and maxVal.
        double width = ceil((maxVal - minVal) * 1.0 / (N - 1));
        vector<int> bucket(N * 2, -1);
        for (vector<int>::const_iterator it = num.begin(); it != num.end(); ++it) {
            int bucketNum = (*it - minVal) / width;
            (bucket[2 * bucketNum] < 0) ? (bucket[2 * bucketNum] = *it) : (bucket[2 * bucketNum] = min(bucket[2 * bucketNum], *it));
            bucket[2 * bucketNum + 1] = max(bucket[2 * bucketNum + 1], *it);
        }

        // 3.Search the maximum gap between different buckets.
        int ret = INT_MIN;
        int left;
        for (int idx = 1; idx < 2 * N - 1; idx += 2) {
            if (bucket[idx] >= 0) {
                left = bucket[idx];
            }
            if (bucket[idx + 1] >= 0) {
                ret = max(bucket[idx + 1] - left, ret);
            }
        }
        return ret;
    }

    int maximumGap_sort(vector<int> &num) {
        // Sort method.
        if (num.size() < 2) {
            return 0;
        }
        sort(num.begin(), num.end());
        int max = INT_MIN;
        int prep = num[0];
        for (int idx = 1; idx < num.size(); ++idx) {
            max = (num[idx] - prep) > max ? num[idx] - prep : max;
            // max < (num[idx] - prep) ? (max = num[idx] - prep) : NULL;
            prep = num[idx];
        }
        return max;
    }
    // Clone Graph
    // Clone an undirected graph. Each node in the graph contains a label and a list of its neighbors.
    UndirectedGraphNode *cloneGraph(UndirectedGraphNode *node) {
        // BFS copy.
        if (node == nullptr) {
            return nullptr;
        }
        queue<UndirectedGraphNode *> BFSQueue;
        BFSQueue.push(node);
        unordered_map<int, UndirectedGraphNode *> visited;
        // Copy the first node.
        UndirectedGraphNode *copy = new UndirectedGraphNode(node->label);
        // Mark as visited.
        visited[node->label] = copy;
        while (!BFSQueue.empty()) {
            UndirectedGraphNode *oNode = BFSQueue.front();
            BFSQueue.pop();
            // Copy the list of its neighbors.
            vector<UndirectedGraphNode *> oNeighbors = oNode->neighbors;
            for (vector<UndirectedGraphNode *>::const_iterator it = oNeighbors.begin(); it != oNeighbors.end(); ++it) {
                if (visited.count((*it)->label) == 0) {
                    UndirectedGraphNode *neighborNode = new UndirectedGraphNode((*it)->label);
                    visited[(*it)->label] = neighborNode;
                    BFSQueue.push(*it);
                }
                (visited[oNode->label]->neighbors).push_back(visited[(*it)->label]);
            }
        }
        return copy;
    }
    UndirectedGraphNode *cloneGraph_DFS(UndirectedGraphNode *node) {
        // DFS copy
        unordered_map<int, UndirectedGraphNode *> visited;
        return cloneGraph_DFS(node, visited);
    }
    UndirectedGraphNode *cloneGraph_DFS(UndirectedGraphNode *node, unordered_map<int, UndirectedGraphNode *> &visited) {
        if (node == nullptr) {
            return node;
        }
        unordered_map<int, UndirectedGraphNode *>::iterator it = visited.find(node->label);
        if (it == visited.end()) {
            UndirectedGraphNode *copy = new UndirectedGraphNode(node->label);
            visited[node->label] = copy;
            for (vector<UndirectedGraphNode *>::const_iterator neighborsIt = (node->neighbors).begin(); neighborsIt != (node->neighbors).end(); ++neighborsIt) {
                (copy->neighbors).push_back(cloneGraph_DFS(*neighborsIt, visited));
            }
            return copy;
        }
        else {
            return it->second;
        }
    }
    // Scramble String
    // Given a string s1, we may represent it as a binary tree by partitioning it to two non-empty substrings recursively.
    // Below is one possible representation of s1 = "great":
    //     great
    //     /    \
    //    gr    eat
    //   / \    /  \
    //  g   r  e   at
    //             / \
    //            a   t
    // To scramble the string, we may choose any non-leaf node and swap its two children.
    // For example, if we choose the node "gr" and swap its two children, it produces a scrambled string "rgeat".
    //      rgeat
    //      /    \
    //     rg    eat
    //    / \    /  \
    //   r   g  e   at
    //              / \
    //             a   t
    // We say that "rgeat" is a scrambled string of "great".
    // Similarly, if we continue to swap the children of nodes "eat" and "at", it produces a scrambled string "rgtae".
    //     rgtae
    //     /    \
    //    rg    tae
    //   / \    /  \
    //  r   g  ta  e
    //         / \
    //        t   a
    // We say that "rgtae" is a scrambled string of "great".
    // Given two strings s1 and s2 of the same length, determine if s2 is a scrambled string of s1.
    bool isScramble(string s1, string s2) {
        if (s1.size() != s2.size()) {
            return false;
        }
        if (s1 == s2) {
            return true;
        }
        // Check whether s1 and s2 contain the same letter permutations.
        string tmp1 = s1, tmp2 = s2;
        sort(tmp1.begin(), tmp1.end());
        sort(tmp2.begin(), tmp2.end());
        if (tmp1 != tmp2) {
            return false;
        }
        // Split the s1 and s2 into two parts and check separatly.
        int N = s1.size();
        for (int idx = 1; idx != N; ++idx) {
            bool condition1 = isScramble(s1.substr(0, idx), s2.substr(N - idx, idx)) && isScramble(s1.substr(idx, N - idx), s2.substr(0, N - idx));
            bool condition2 = isScramble(s1.substr(0, idx), s2.substr(0, idx)) && isScramble(s1.substr(idx, N - idx), s2.substr(idx, N - idx));
            if (condition1 || condition2) {
                return true;
            }
        }
        return false;
    }
    // Best Time to Buy and Sell Stock III
    // Say you have an array for which the ith element is the price of a given stock on day i.
    // Design an algorithm to find the maximum profit. You may complete at most two transactions.
    // Note:
    // You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
    int maxProfitIII(vector<int> &prices) {
        if (prices.size() < 2) {
            return 0;
        }
        // leftMax[i] means the maxProfit before suffix i, while rightMax[i] means the maxProfit after suffix i.
        vector<int> leftMax(prices.size(), 0), rightMax(prices.size(), 0);
        int minEle = prices[0], maxEle = prices[prices.size() - 1];
        for (int idx = 1; idx < prices.size(); ++idx) {
            if (prices[idx] < minEle) {
                minEle = prices[idx];
            }
            leftMax[idx] = max(leftMax[idx - 1], prices[idx] - minEle);

            int ridx = prices.size() - 1 - idx;
            if (prices[ridx] > maxEle) {
                maxEle = prices[ridx];
            }
            rightMax[ridx] = max(rightMax[ridx + 1], maxEle - prices[ridx]);
        }
        // Combine the leftMax and rightMax.
        int ret = 0;
        for (int idx = 0; idx != leftMax.size(); ++idx) {
            ret = max(ret, leftMax[idx] + rightMax[idx]);
        }
        return ret;
    }
    // First Missing Positive
    // Given an unsorted integer array, find the first missing positive integer.
    // For example,
    // Given [1,2,0] return 3,
    // and [3,4,-1,1] return 2.
    // Your algorithm should run in O(n) time and uses constant space.
    int firstMissingPositive(int A[], int n) {
        // O(n) / O(1)
        // Making array A a hash table which suffix i stores (positive) number i + 1.
        for (int idx = 0; idx < n; ++idx) {
            int digit = A[idx];
            while (0 < digit && digit <= n && A[digit - 1] != digit) {
                swap(A[digit - 1], A[idx]);
                digit = A[idx];
            }
        }
        // Scan the hash table, if the A[i] != i + 1, then number i + 1 misses.
        for (int idx = 0; idx < n; ++idx) {
            if (A[idx] != idx + 1) {
                return idx + 1;
            }
        }

        return n + 1;
    }
    // Sqrt(x)
    // Implement int sqrt(int x).
    // Compute and return the square root of x.
    int sqrt(int x) {
        int beg = 1, end = max(1, x / 2);
        while (beg <= end) {
            long long mid = beg + (end - beg) / 2;
            if (mid * mid == x) {
                return mid;
            }
            else if (mid * mid > x) {
                end = mid - 1;
            }
            else {
                beg = mid + 1;
            }
        }
        return end;
    }
    // Add Two Numbers
    // You are given two linked lists representing two non-negative numbers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
    // Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
    // Output: 7 -> 0 -> 8
    ListNode *addTwoNumbers(ListNode *l1, ListNode *l2) {
        ListNode *sumList = new ListNode(-1);
        ListNode *cur1 = l1, *cur2 = l2, *curSum = sumList;
        int catching = 0;
        while (cur1 != nullptr || cur2 != nullptr) {
            int sum = catching;
            if (cur1 != nullptr) {
                sum += cur1->val;
                cur1 = cur1->next;
            }
            if (cur2 != nullptr) {
                sum += cur2->val;
                cur2 = cur2->next;
            }
            sum >= 10 ? (sum -= 10, catching = 1) : (catching = 0);
            curSum->next = new ListNode(sum);
            curSum = curSum->next;
        }
        if (catching != 0) {
            curSum->next = new ListNode(catching);
        }
        return sumList->next;
    }
    // ZigZag Conversion
    // The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)
    // P   A   H   N
    // A P L S I I G
    // Y   I   R
    // And then read line by line: "PAHNAPLSIIGYIR"
    // Write the code that will take a string and make this conversion given a number of rows:
    // string convert(string text, int nRows);
    // convert("PAYPALISHIRING", 3) should return "PAHNAPLSIIGYIR".
    string convert(string s, int nRows) {
        if (nRows == 1 || s == "") {
            return s;
        }
        string zigZag;
        int step = nRows + (nRows - 2);
        for (int idx = 0; idx < nRows; ++idx) {
            int thisStep = step - 2 * idx;
            // case idx == nRows - 1
            if (!thisStep) thisStep = step;
            for (int cur = idx; cur < s.size();) {
                zigZag += s[cur];
                cur += thisStep;
                if (thisStep != step) thisStep = step - thisStep;
            }
        }
        return zigZag;
    }
    // Permutation Sequence
    // The set [1,2,3,…,n] contains a total of n! unique permutations.
    // By listing and labeling all of the permutations in order,
    // We get the following sequence (ie, for n = 3):
    // "123"
    // "132"
    // "213"
    // "231"
    // "312"
    // "321"
    // Given n and k, return the kth permutation sequence.
    // Note: Given n will be between 1 and 9 inclusive.
    string getPermutation(int n, int k) {
        // Computing the nth factorial.
        int factorial = 1;
        for (int idx = 2; idx <= n; ++idx) {
            factorial *= idx;
        }
        // A hash table for number idx at suffix idx - 1.
        vector<int> numTable;
        for (int idx = 1; idx <= n; ++idx) {
            numTable.push_back(idx);
        }
        // Selecting number from hash table according to k and n-1...th factorial.
        string ret;
        for (--k; n > 0; --n) {
            factorial /= n;
            int num = k / factorial;
            ret += '0' + numTable[num];
            numTable.erase(numTable.begin() + num);
            k %= factorial;
        }
        return ret;
    }
    // Validate Binary Search Tree
    // Given a binary tree, determine if it is a valid binary search tree (BST).
    // Assume a BST is defined as follows:
    // The left subtree of a node contains only nodes with keys less than the node's key.
    // The right subtree of a node contains only nodes with keys greater than the node's key.
    // Both the left and right subtrees must also be binary search trees.
    bool isValidBST(TreeNode *root) {
        // Recursive.
        // Noticing about the overflow.
        return isValidBSTHelper(root, LONG_MAX, LONG_MIN);
    }
    bool isValidBSTHelper(TreeNode *root, long leftMax, long rightMin) {
        if (root == nullptr) {
            return true;
        }
        if (root->val >= leftMax || root->val <= rightMin) {
            return false;
        }
        return isValidBSTHelper(root->left, root->val, rightMin) && isValidBSTHelper(root->right, leftMax, root->val);
    }

    bool isValidBST_iteration(TreeNode *root) {
        // Inorder traversal.
        stack<TreeNode *> tempStack;
        TreeNode *ptr = root;
        TreeNode *prev = nullptr;
        while (!tempStack.empty() || ptr != nullptr) {
            if (ptr != nullptr) {
                tempStack.push(ptr);
                ptr = ptr->left;
            }
            else {
                ptr = tempStack.top();
                tempStack.pop();
                if (prev != nullptr && prev->val >= ptr->val) {
                    return false;
                }
                prev = ptr;
                ptr = ptr->right;
            }
        }
        return true;
    }
    // Largest Rectangle in Histogram
    // Given n non-negative integers representing the histogram's bar height where the width of each bar is 1, find the area of largest rectangle in the histogram.
    // For example,
    // Given height = [2,1,5,6,2,3],
    // return 10.
    int largestRectangleArea(vector<int> &height) {
        // O(logN) Divide and Conquer
        // case : 3 5 5 2 5 5 6 6 4 4 1 1 2 5 5 6 6 4 1 3
        return largestRectangleArea(height, 0, height.size() - 1);
    }
    int largestRectangleArea(const vector<int> &height, const int &beg, const int &end) {
        if (beg > end) {
            return 0;
        }
        else if (beg == end) {
            return height[beg];
        }
        else {
            int mid = beg + (end - beg) / 2;
            // left side.
            int leftMax = largestRectangleArea(height, beg, mid);
            // right side.
            int rightMax = largestRectangleArea(height, mid + 1, end);
            // mid
            int midMax = largestRectangleMidArea(height, beg, mid, end);
            // Return the max.
            return max(max(leftMax, rightMax), midMax);
        }
    }
    int largestRectangleMidArea(const vector<int> &height, const int &beg, const int &mid, const int &end) {
        int left = mid, right = mid + 1;
        int minVal = INT_MAX;
        int area = 0;
        while (left >= beg && right <= end) {
            minVal = min(minVal, min(height[left], height[right]));
            area = max(area, minVal * (right - left + 1));
            if (left == beg) {
                ++right;
            }
            else if (right == end) {
                --left;
            }
            else {
                height[left - 1] < height[right + 1] ? ++right : --left;
            }
        }
        return area;
    }
    int largestRectangleArea_stack(vector<int> &height) {
        // Stack, O(n)
        int maxArea = 0;
        stack<int> idxStack;
        height.push_back(0);
        for (int idx = 0; idx < height.size(); ++idx) {
            while (!idxStack.empty() && height[idxStack.top()] >= height[idx]) {
                int h = height[idxStack.top()];
                idxStack.pop();
                int beg = idxStack.empty() ? -1 : idxStack.top();
                maxArea = max(maxArea, h * (idx - beg - 1));
            }
            idxStack.push(idx);
        }
        return maxArea;
    }
    // Word Break
    // Given a string s and a dictionary of words dict, determine if s can be segmented into a space-separated sequence of one or more dictionary words.
    // For example, given
    // s = "leetcode",
    // dict = ["leet", "code"].
    // Return true because "leetcode" can be segmented as "leet code".
    bool wordBreak(string s, unordered_set<string> &dict) {
        vector<vector<bool> > dp = vector<vector<bool> >(s.size(), vector<bool>(s.size(), false));
        for (int step = 1; step <= s.size(); ++step) {
            for (int idx = 0; idx <= s.size() - step; ++idx) {
                if (dict.find(s.substr(idx, step)) != dict.end()) {
                    dp[idx][idx + step - 1] = true;
                }
                else {
                    bool canBreak = false;
                    for (int cur = 1; cur <= step; ++cur) {
                        canBreak = canBreak || (dp[idx][idx + cur - 1] && dp[idx + cur][idx + step - 1]);
                        if (canBreak == true) {
                            dp[idx][idx + step - 1] = true;
                            break;
                        }
                    }
                }
            }
        }
        return dp[0][s.size() - 1];
    }

    bool wordBreak_improved(string s, unordered_set<string> &dict) {
        vector<bool> dp = vector<bool>(s.size(), false);
        for (int idx = 0; idx < s.size(); ++idx) {
            if (dict.find(s.substr(0, idx + 1)) != dict.end()) {
                dp[idx] = true;
            }
            else {
                for (int cur = idx - 1; cur >= 0; --cur) {
                    if (dp[cur] == true && dict.find(s.substr(cur + 1, idx - cur)) != dict.end()) {
                        dp[idx] = true;
                        break;
                    }
                }
            }
        }
        return dp[s.size() - 1];
    }
    // Implement strStr().
    // Returns the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.
    int strStr(const char *haystack, const char *needle) {
        const char *ptr1 = haystack, *ptr2 = needle;
        while (*ptr1 != '\0' && *ptr2 != '\0') {
            const char *cur = ptr1;
            while (*cur != '\0' && *ptr2 != '\0' && *cur == *ptr2) {
                ++cur;
                ++ptr2;
            }
            if (*ptr2 == '\0') {
                return ptr1 - haystack;
            }
            else if (*cur == '\0') {
                return -1;
            }
            else {
                ptr2 = needle;
                ++ptr1;
            }
        }
        return (*ptr2 == '\0') ? 0 : -1;
    }

    int KMPstrStr(const char *haystack, const char *needle) {
        // KMP algorithm.
        int n = strlen(haystack);
        int m = strlen(needle);
        if (m == 0) return 0;
        // Counting prefix function for needle.
        vector<int> prefix(m, 0);
        for (int i = 1; i < m; i++) {
            prefix[i] = prefix[i - 1];
            while (prefix[i] > 0 && needle[prefix[i]] != needle[i]) {
                prefix[i] = prefix[prefix[i] - 1];
            }
            if (needle[prefix[i]] == needle[i]) {
                prefix[i]++;
            }
        }
        for (int i = 0, pre = 0; i < n; i++) {
            while (pre > 0 && needle[pre] != haystack[i]) {
                pre = prefix[pre - 1];
            }
            if (needle[pre] == haystack[i]) {
                pre++;
            }
            // Having reached the tail of the needle, return the start index.
            if (pre == m) {
                return i - m + 1;
            }
        }
        return -1;
    }
    // Longest Substring Without Repeating Characters
    // Given a string, find the length of the longest substring without repeating characters. For example, the longest substring without repeating letters for "abcabcbb" is "abc", which the length is 3. For "bbbbb" the longest substring is "b", with the length of 1.
    int lengthOfLongestSubstring(string s) {
        int beg = 0, end = 0;
        int maxLength = 0;
        unordered_map<char, int> dupCharIdx;
        while (end < s.size()) {
            if (dupCharIdx.count(s[end]) > 0 && beg <= dupCharIdx[s[end]]) {
                // If beg > duplicates' suffix, it means we don't contain the duplicate char from beg, just updating the suffix in the map.
                maxLength = max(maxLength, end - beg);
                beg = dupCharIdx[s[end]] + 1;
            }
            // Updating the duplicates' suffix.
            dupCharIdx[s[end]] = end;
            ++end;
        }
        return max(maxLength, end - beg);
    }

    int lengthOfLongestSubstring_improved(string s) {
        int beg = 0, end = 0;
        int maxLength = 0;
        vector<int> dupCharIdx(256, -1);
        while (end < s.size()) {
            if (dupCharIdx[s[end]] >= 0 && beg <= dupCharIdx[s[end]]) {
                // If beg > duplicates' suffix, it means we don't contain the duplicate char from beg, just updating the suffix in the map.
                maxLength = max(maxLength, end - beg);
                beg = dupCharIdx[s[end]] + 1;
            }
            // Updating the duplicates' suffix.
            dupCharIdx[s[end]] = end;
            ++end;
        }
        return max(maxLength, end - beg);
    }
    // Valid Palindrome
    // Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.
    // For example,
    // "A man, a plan, a canal: Panama" is a palindrome.
    // "race a car" is not a palindrome.
    // Note:
    // Have you consider that the string might be empty? This is a good question to ask during an interview.
    // For the purpose of this problem, we define empty string as valid palindrome.
    bool isPalindrome(string s) {
        int beg = 0, end = s.size() - 1;
        while (beg < end) {
            if (isAlpha(s[beg]) == false) {
                ++beg;
            }
            else if (isAlpha(s[end]) == false) {
                --end;
            }
            else if (equalIgnoreCase(s[beg], s[end]) == true) {
                ++beg;
                --end;
            }
            else {
                return false;
            }
        }
        return true;
    }
    bool isAlpha(const char ch) {
        return ('A' <= ch && ch <= 'Z') || ('a' <= ch && ch <= 'z') || ('0' <= ch && ch <= '9');
    }
    bool equalIgnoreCase(const char a, const char b) {
        int aval = a <= 'Z' ? a - 'A' : a - 'a', bval = b <= 'Z' ? b - 'A' : b - 'a';
        return aval == bval;
    }
    // Rotate List
    // Given a list, rotate the list to the right by k places, where k is non-negative.
    // For example:
    // Given 1->2->3->4->5->NULL and k = 2,
    // return 4->5->1->2->3->NULL.
    ListNode *rotateRight(ListNode *head, int k) {
        if (head == nullptr || k == 0) {
            return head;
        }
        ListNode *prePivot = head, *fast = head, *tail = head;
        // Calculate the lengh of the list and find the tail node.
        long long len = 0;
        while (tail->next != nullptr) {
            tail = tail->next;
            ++len;
        }
        ++len;
        // Make the list a cycle.
        tail->next = head;
        // Finding the pivot node's prev node.
        for (int num = 0; num <= k % len; ++num) {
            fast = fast->next;
        }
        while (fast != tail->next) {
            fast = fast->next;
            prePivot = prePivot->next;
        }
        // Rotate the list.
        tail->next = head;
        head = prePivot->next;
        prePivot->next = nullptr;

        return head;
    }
    ListNode *rotateRight_improved(ListNode *head, int k) {
        if (head == nullptr || k == 0) {
            return head;
        }
        // Make list to be a circle.
        ListNode *ptr = head;
        long long len = 1;
        while (ptr->next != nullptr) {
            ptr = ptr->next;
            ++len;
        }
        ptr->next = head;
        // Find the pivot node's prev node.
        for(int count = 1; count <= len - k % len; ++count) {
            ptr = ptr->next;
        }
        // Rotate
        head = ptr->next;
        ptr->next = nullptr;

        return head;
    }
    // Maximal Rectangle
    // Given a 2D binary matrix filled with 0's and 1's, find the largest rectangle containing all ones and return its area.
    int maximalRectangle(vector<vector<char> > &matrix) {
        if (matrix.size() == 0) {
            return 0;
        }
        int rowNum = matrix.size(), colNum = matrix[0].size();
        int ret = 0;
        vector<int> height(colNum, 0);
        for (int r = 0; r < rowNum; ++r) {
            for (int c = 0; c < colNum; ++c) {
                if (matrix[r][c] == '0') {
                    height[c] = 0;
                }
                else {
                    ++height[c];
                }
            }
            ret = max(ret, maximalRectangleHelper(height));
        }
        return ret;
    }
    int maximalRectangleHelper(vector<int> &height) {
        stack<int> idxStack;
        int maxArea = 0;
        height.push_back(0);
        for (int idx = 0; idx < height.size(); ++idx) {
            while (!idxStack.empty() && height[idxStack.top()] >= height[idx]) {
                int h = height[idxStack.top()];
                idxStack.pop();
                int beg = idxStack.empty() ? -1 : idxStack.top();
                maxArea = max(maxArea, h * (idx - beg - 1));
            }
            idxStack.push(idx);
        }
        return maxArea;
    }
    // Merge Intervals
    // Given a collection of intervals, merge all overlapping intervals.
    // For example,
    // Given [1,3],[2,6],[8,10],[15,18],
    // return [1,6],[8,10],[15,18].
    vector<Interval> merge(vector<Interval> &intervals) {
        sort(intervals.begin(), intervals.end(), mySortFunction);
        vector<Interval> ret;
        for (int idx = 0; idx < intervals.size(); ++idx) {
            if (ret.size() == 0 || ret.back().end < intervals[idx].start) {
                ret.push_back(intervals[idx]);
            }
            else {
                ret.back().end = max(ret.back().end, intervals[idx].end);
            }
        }
        return ret;
    }
    static bool mySortFunction(const Interval &a, const Interval &b) {
        return a.start < b.start;
    }
    // 4Sum
    // Given an array S of n integers, are there elements a, b, c, and d in S such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.
    // Note:
    // Elements in a quadruplet (a,b,c,d) must be in non-descending order. (ie, a ≤ b ≤ c ≤ d)
    // The solution set must not contain duplicate quadruplets.
    // For example, given array S = {1 0 -1 0 -2 2}, and target = 0.
    // A solution set is:
    //    (-1,  0, 0, 1)
    //    (-2, -1, 1, 2)
    //    (-2,  0, 0, 2)
    vector<vector<int> > fourSum(vector<int> &num, int target) {
        sort(num.begin(), num.end());
        vector<vector<int> > ret;
        for (int fst = 0; fst < num.size(); ++fst) {
            for (int snd = fst + 1; snd < num.size(); ++snd) {
                int thd = snd + 1, fth = num.size() - 1;
                while (thd < fth) {
                    int sum = num[fst] + num[snd] + num[thd] + num[fth];
                    if (sum == target) {
                        int tmp[4] = {num[fst], num[snd], num[thd], num[fth]};
                        ret.push_back(vector<int>(tmp, tmp + 4));
                        while (thd < num.size() - 1 && num[thd + 1] == num[thd]) {
                            ++thd;
                        }
                        while (fth > thd && num[fth - 1] == num[fth]) {
                            --fth;
                        }
                        ++thd;
                        --fth;
                    }
                    else if (sum < target) {
                        ++thd;
                    }
                    else {
                        --fth;
                    }
                }
                while (snd < num.size() - 1 && num[snd + 1] == num[snd]) {
                    ++snd;
                }
            }
            while (fst < num.size() - 1 && num[fst + 1] == num[fst]) {
                ++fst;
            }
        }
        return ret;
    }
    // Sudoku Solver
    // Write a program to solve a Sudoku puzzle by filling the empty cells.
    // Empty cells are indicated by the character '.'.
    // You may assume that there will be only one unique solution.
    void solveSudoku(vector<vector<char> > &board) {
        vector<vector<int> > rowMap, colMap, squareMap;
        bool canBreak = false;
        rowMap = colMap = squareMap = vector<vector<int> >(9, vector<int>(9, 0));
        for (int r = 0; r < 9; ++r) {
            for (int c = 0; c < 9; ++c) {
                if (board[r][c] != '.') ++rowMap[r][board[r][c] - '1'];
                if (board[c][r] != '.') ++colMap[r][board[c][r] - '1'];
                int row = 3 * (r / 3) + c / 3, col = 3 * (r % 3) + c % 3;
                if (board[row][col] !='.') ++squareMap[r][board[row][col] - '1'];
            }
        }
        solveSudoku(0, 0, board, rowMap, colMap, squareMap, canBreak);
    }
    void solveSudoku(int r, int c, vector<vector<char> > &board, vector<vector<int> > &rowMap, vector<vector<int> > &colMap, vector<vector<int> > &squareMap, bool &canBreak) {
        if (r >= 9) {
            canBreak = true;
            return;
        }
        if (board[r][c] == '.') {
            for (int num = 0; num <= 8; ++num) {
                if (!rowMap[r][num] && !colMap[c][num] && !squareMap[3 * (r / 3) + c / 3][num]) {
                    board[r][c] = '1' + num;
                    ++rowMap[r][num];
                    ++colMap[c][num];
                    ++squareMap[3 * (r / 3) + c / 3][num];
                    if (canBreak == true) {
                        return;
                    }
                    rowMap[r][num] = colMap[c][num] = squareMap[3 * (r / 3) + c / 3][num] = 0;
                    board[r][c] = '.';
                }
            }
        }
        else {
            if (c < 8) {
                solveSudoku(r, c + 1, board, rowMap, colMap, squareMap, canBreak);
            }
            else {
                solveSudoku(r + 1, 0, board, rowMap, colMap, squareMap, canBreak);
            }
        }
    }
    // Sort List
    // Sort a linked list in O(n log n) time using constant space complexity.
    ListNode *sortList_quickSort(ListNode *head) {
        // Quick Sort.
        int length = 0;
        ListNode *cur = head;
        while (cur != nullptr) {
            ++length;
            cur = cur->next;
        }
        sortList_quickSort(head, length);
        return head;
    }
    void sortList_quickSort(ListNode *head, int length) {
        if (length <= 1) {
            return;
        }
        int len = sortListPartition(head, length);
        sortList_quickSort(head, len);
        ListNode *tmp = new ListNode(0);
        tmp->next = head;
        ListNode *cur = tmp;
        for (int l = 0; l <= len; ++l) {
            cur = cur->next;
        }
        // Let cur points to the begining of the part2.
        while (cur->next != nullptr && cur->next->val == cur->val) {
            ++len;
            cur = cur->next;
        }
        sortList_quickSort(cur->next, length - len - 1);
    }
    int sortListPartition(ListNode *head, int length) {
        ListNode *tail = head;
        for (int l = 1; l < length; ++l) {
            tail = tail->next;
        }
        int pivot = tail->val;
        ListNode *tmp = new ListNode(0);
        tmp->next = head;
        ListNode *point = tmp, *cur = head;
        for (int l = 1; l < length; ++l, cur = cur->next) {
            if (cur->val < pivot) {
                point = point->next;
                swap(point->val, cur->val);
            }
        }
        point = point->next;
        swap(point->val, tail->val);
        int len = 0;
        // Calculate the lengh of the first part after partition.
        for (cur = tmp; cur->next != point; cur = cur->next) {
            ++len;
        }
        return len;
    }

    ListNode *sortList_mergeSort(ListNode *head) {
        // Merge Sort.
        // Recursive.
        if (head == nullptr || head->next == nullptr) {
            return head;
        }
        // Find the middle node.
        ListNode *fast = head->next->next, *mid = head;
        while (fast != nullptr && fast->next != nullptr) {
            fast = fast->next->next;
            mid = mid->next;
        }
        ListNode *right = sortList_mergeSort(mid->next);
        mid->next = nullptr;
        ListNode *left = sortList_mergeSort(head);
        return merge_mergeSort(left, right);
    }
    ListNode *merge_mergeSort(ListNode *l1, ListNode *l2) {
        ListNode *head = nullptr, *ptr = head;
        ListNode *pointer1 = l1, *pointer2 = l2;
        while (pointer1 != nullptr || pointer2 != nullptr) {
            if (pointer2 == nullptr || ((pointer1 != nullptr) && (pointer1->val < pointer2->val))) {
                if (ptr == nullptr) {
                    head = pointer1;
                    ptr = head;
                }
                else {
                    ptr->next = pointer1;
                    ptr = ptr->next;
                }
                pointer1 = pointer1->next;
            }
            else {
                if (ptr == nullptr) {
                    head = pointer2;
                    ptr = head;
                }
                else {
                    ptr->next = pointer2;
                    ptr = ptr->next;
                }
                pointer2 = pointer2->next;
            }
        }
        return head;
    }

    ListNode *sortList_iteration(ListNode *head) {
        // Bottom to up merge sort.
        ListNode *tmp = new ListNode(0);
        tmp->next = head;
        ListNode *cur = head;
        int lens = 0;
        while (cur != nullptr) {
            cur = cur->next;
            ++lens;
        }
        for (int l = 2; l < 2 * lens; l *= 2) {
            cur = tmp;
            while (cur != nullptr && cur->next != nullptr) {
                cur = sortList_iteration(cur, l);
            }
        }
        return tmp->next;
    }
    ListNode *sortList_iteration(ListNode *beg, int lens) {
        // Merge the list whose length is half of lens.
        // Return the sorted list's tail node.
        ListNode *beg1 = beg->next, *beg2, *end1, *end2;
        ListNode *cur = beg1, *tail;
        beg2 = end1 = end2 = tail = nullptr;
        for (int l = 1; l <= lens; ++l, cur = cur->next) {
            if (l == lens / 2) {
                end1 = cur;
                beg2 = cur->next;
            }
            if (lens == l || cur->next == nullptr) {
                end2 = cur;
                break;
            }
        }
        if (beg2 == nullptr) {
            return nullptr;
        }
        end1->next = nullptr;
        tail = end2->next;
        end2->next = nullptr;
        cur = beg;
        while (beg1 != nullptr || beg2 != nullptr) {
            if (beg2 == nullptr || (beg1 != nullptr && beg1->val < beg2->val)) {
                cur->next = beg1;
                beg1 = beg1->next;
            }
            else {
                cur->next = beg2;
                beg2 = beg2->next;
            }
            cur = cur->next;
        }
        cur->next = tail;
        return cur;
    }
    // Merge k Sorted Lists
    // Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.
    ListNode *mergeKLists(vector<ListNode *> &lists) {
        return mergeKLists(lists, 0, lists.size() - 1);
    }
    ListNode *mergeKLists(vector<ListNode *> &lists, const int &beg, const int &end) {
        if (beg > end) {
            return nullptr;
        }
        else if (beg == end) {
            return lists[beg];
        }
        else {
            int mid  = beg + (end - beg) / 2;
            ListNode *l = mergeKLists(lists, beg, mid), *r = mergeKLists(lists, mid + 1, end);
            return merge_mergeSort(l, r);
        }
    }
    // Insert Interval
    // Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).
    // You may assume that the intervals were initially sorted according to their start times.
    // Example 1:
    // Given intervals [1,3],[6,9], insert and merge [2,5] in as [1,5],[6,9].
    // Example 2:
    // Given [1,2],[3,5],[6,7],[8,10],[12,16], insert and merge [4,9] in as [1,2],[3,10],[12,16].
    // This is because the new interval [4,9] overlaps with [3,5],[6,7],[8,10].
    vector<Interval> insert(vector<Interval> &intervals, Interval newInterval) {
        vector<Interval>::iterator it = intervals.begin();
        for (; it != intervals.end(); ++it) {
            if (it->start > newInterval.start) {
                intervals.insert(it, newInterval);
                break;
            }
            else if (it->start == newInterval.start) {
                it->end = max(it->end, newInterval.end);
                break;
            }
        }
        if (it == intervals.end()) {
            intervals.push_back(newInterval);
        }
        vector<Interval> ret;
        for (int idx = 0; idx < intervals.size(); ++idx) {
            if (ret.size() == 0 || ret.back().end < intervals[idx].start) {
                ret.push_back(intervals[idx]);
            }
            else {
                ret.back().end = max(ret.back().end, intervals[idx].end);
            }
        }
        return ret;
    }
    // Binary Tree Maximum Path Sum
    // Given a binary tree, find the maximum path sum.
    // The path may start and end at any node in the tree.
    // For example:
    // Given the below binary tree,
    //        1
    //       / \
    //      2   3
    // Return 6.
    int maxPathSum(TreeNode *root) {
        int croSum = INT_MIN, sideSum = maxPathSum(root, croSum);
        return max(croSum, sideSum);
    }
    int maxPathSum(TreeNode *root, int &croSum) {
        if (root == nullptr) {
            return INT_MIN;
        }
        int leftSum = maxPathSum(root->left, croSum), rightSum = maxPathSum(root->right, croSum);
        croSum = max(max(leftSum, rightSum), croSum);
        if (leftSum < 0 && rightSum < 0) {
            return root->val;
        }
        else if (leftSum < 0) {
            return root->val + rightSum;
        }
        else if (rightSum < 0) {
            return root->val + leftSum;
        }
        else {
            croSum = max(croSum, root->val + leftSum + rightSum);
            return root->val + max(leftSum, rightSum);
        }
    }

    int maxPathSum_improved(TreeNode *root) {
        // Another better method.
        int maxSum = INT_MIN;
        maxPathSum_improved(root, maxSum);
        return maxSum;
    }
    int maxPathSum_improved(TreeNode *root, int &maxSum) {
        if (root == nullptr) {
            return 0;
        }
        int leftSum = maxPathSum_improved(root->left, maxSum), rightSum = maxPathSum_improved(root->right, maxSum);
        leftSum = max(0, leftSum);
        rightSum = max(0, rightSum);
        maxSum = max(maxSum, root->val + leftSum + rightSum);
        return root->val + max(leftSum, rightSum);
    }
    // Reorder List
    // Given a singly linked list L: L0→L1→…→Ln-1→Ln,
    // reorder it to: L0→Ln→L1→Ln-1→L2→Ln-2→…
    // You must do this in-place without altering the nodes' values.
    // For example,
    // Given {1,2,3,4}, reorder it to {1,4,2,3}.
    void reorderList_TLE(ListNode *head) {
        // TLE
        ListNode *beg = head, *end = nullptr;
        while (true) {
            ListNode *pre = beg;
            if (pre == nullptr || pre->next == nullptr || pre->next->next == nullptr) {
                return;
            }
            // Scan the list to find the pre node of the last node.
            while (pre->next->next != end) {
                pre = pre->next;
            }
            end = pre->next;
            // Delete the last node.
            pre->next = nullptr;
            // Reorder the last node.
            end->next = beg->next;
            beg->next = end;

            beg = end->next;
            end = nullptr;
        }
    }
    void reorderList(ListNode *head) {
        // Split the list into two parts.
        if (head == nullptr) {
            return;
        }
        ListNode *slow = head, *fast = head;
        ListNode *l1 = head, *l2 = nullptr;
        while (fast != nullptr && fast->next != nullptr && fast->next->next != nullptr) {
            slow = slow->next;
            fast = fast->next->next;
        }
        ListNode *cur = slow->next;
        while (cur != nullptr) {
            ListNode *next = cur->next;
            cur->next = l2;
            l2 = cur;
            cur = next;
        }
        slow->next = nullptr;
        // Merge the two lists.
        while (l1 != nullptr && l2 != nullptr) {
            ListNode *l2Next = l2->next;
            l2->next = l1->next;
            l1->next = l2;

            l1 = l2->next;
            l2 = l2Next;
        }
    }
    // Restore IP Addresses
    // Given a string containing only digits, restore it by returning all possible valid IP address combinations.
    // For example:
    // Given "25525511135",
    // return ["255.255.11.135", "255.255.111.35"]. (Order does not matter)
    vector<string> restoreIpAddresses(string s) {
        vector<string> ret;
        string ip;
        restoreIpAddresses(s, 0, 1, ip, ret);
        return ret;
    }
    void restoreIpAddresses(string s, const int &idx, const int& count, string ip, vector<string> &ret) {
        if (count > 4 || idx >= s.size()) {
            return;
        }
        for (int len = 1; len <= 3; ++len) {
            string i = s.substr(idx, len);
            if (isLegalIp(i) == false) {
                return;
            }
            if (count == 1) {
                restoreIpAddresses(s, idx + len, count + 1, i, ret);
            }
            else if (count == 4 && idx + len == s.size()) {
                ret.push_back(ip + '.' + i);
            }
            else {
                restoreIpAddresses(s, idx + len, count + 1, ip + '.' + i, ret);
            }
        }
    }
    bool isLegalIp(const string &s) {
        int ipNum = atoi(s.c_str());
        if (ipNum > 255) {
            return false;
        }
        int len = 0;
        do {
            ipNum /= 10;
            ++len;
        }while (ipNum);
        return len == s.size() ? true : false;
    }
    // Spiral Matrix
    // Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.
    // For example,
    // Given the following matrix:
    // [
    //   [ 1, 2, 3 ],
    //   [ 4, 5, 6 ],
    //   [ 7, 8, 9 ]
    // ]
    // You should return [1,2,3,6,9,8,7,4,5].
    vector<int> spiralOrder(vector<vector<int> > &matrix) {
        vector<int> ret;
        int rowNum = matrix.size();
        if (rowNum != 0 ) {
            int colNum = matrix[0].size();
            int level = min(ceil(rowNum / 2.0), ceil(colNum / 2.0));
            for (int r = 0; r < level; ++r) {
                for (int c = r; c < colNum - r; ++c) {
                    ret.push_back(matrix[r][c]);
                }
                for (int c = r + 1; c < rowNum - r; ++c) {
                    ret.push_back(matrix[c][colNum - 1 - r]);
                }
                if (r == rowNum - r - 1 || r == colNum - r - 1) {
                    // Overcome the case rowNum = 1 or colNum = 1.
                    break;
                }
                for (int c = colNum - r - 2; c >= r; --c) {
                    ret.push_back(matrix[rowNum - r - 1][c]);
                }
                for (int c = rowNum - r - 2; c > r; --c) {
                    ret.push_back(matrix[c][r]);
                }
            }
        }
        return ret;
    }
    // Spiral Matrix II
    // Given an integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.
    //
    // For example,
    // Given n = 3,
    //
    // You should return the following matrix:
    // [
    //   [ 1, 2, 3 ],
    //   [ 8, 9, 4 ],
    //   [ 7, 6, 5 ]
    // ]
    vector<vector<int> > generateMatrix(int n) {
        // Main idea:
        // ->->->->
        // | ->-> |
        // | <- | |
        // <-<-<-<-
        //
        // # # # #
        // %     $
        // %     $
        // & & & $
        vector<vector<int> > matrix(n, vector<int>(n, 0));
        int level = n / 2;
        int num = 1;
        for (int r = 0; r <= level; ++r) {
            // Shown as #.
            for (int c = r; c < n - r; ++c) {
                matrix[r][c] = num++;
            }
            // Shown as $.
            for (int c = r + 1; c < n - r; ++c) {
                matrix[c][n - r - 1] = num++;
            }
            // Shown as &.
            for (int c = n - r - 2; c >= r; --c) {
                matrix[n - r - 1][c] = num++;
            }
            // Shown as %.
            for (int c = n - r - 2; c > r; --c) {
                matrix[c][r] = num++;
            }
        }
        return matrix;
    }
    // Longest Palindromic Substring
    // Given a string S, find the longest palindromic substring in S. You may assume that the maximum length of S is 1000, and there exists one unique longest palindromic substring.
    string longestPalindrome(string s) {
        string ret;
        for (int idx = 0; idx < s.length(); ++idx) {
            string str1 = longestPalindrome(s, idx, idx), str2 = longestPalindrome(s, idx, idx + 1);
            if (ret.length() < str1.length()) {
                ret = str1;
            }
            if (ret.length() < str2.length()) {
                ret = str2;
            }
        }
        return ret;
    }
    string longestPalindrome(const string &s, const int &beg, const int &end) {
        int l = beg, r = end;
        while (l >= 0 && r < s.length() && s[l] == s[r]) {
            --l;
            ++r;
        }
        return s.substr(l + 1, r - l - 1);
    }

    string preProcess(const string &s) {
        if (s.size() == 0) return "^$";
        string ret = "^";
        for (int idx = 0; idx < s.size(); ++idx) {
            ret += "#";
            ret += s.substr(idx, 1);
        }
        ret += "#$";
        return ret;
    }
    string longestPalindrome_mancher(string s) {
        string T = preProcess(s);
        int n = T.size(), C = 0, R = 0;
        vector<int> P(n, 0);
        for (int idx = 1; idx < n - 1; ++idx) {
            P[idx] = (R > idx) ? min(R - idx, P[2 * C - idx]) : 0;
            // elegent!
            while (T[idx - 1 - P[idx]] == T[idx + 1 + P[idx]]) ++P[idx];
            if (idx + P[idx] > R) {
                C = idx;
                R = idx + P[idx];
            }
        }

        int center = 0, maxLen = 0;
        for (int idx = 1; idx < n - 1; ++idx) {
            if (P[idx] > maxLen) {
                maxLen = P[idx];
                center = idx;
            }
        }
        return s.substr((center - 1 - maxLen) / 2, maxLen);
    }
    // Multiply Strings
    // Given two numbers represented as strings, return multiplication of the numbers as a string.
    // Note: The numbers can be arbitrarily large and are non-negative.
    string multiply(string num1, string num2) {
        string ret(num1.length() + num2.length(), '0');
        for (int idx1 = num1.length() - 1; idx1 >= 0; --idx1) {
            int carry = 0;
            for (int idx2 = num2.length() - 1; idx2 >= 0; --idx2) {
                int sum = (ret[idx1 + idx2 + 1] - '0') + (num1[idx1] - '0') * (num2[idx2] - '0') + carry;
                ret[idx1 + idx2 + 1] = sum % 10 + '0';
                carry = sum / 10;
            }
            ret[idx1] += carry;
        }
        size_t found = ret.find_first_not_of("0");
        if (found != string::npos) {
            return ret.substr(found);
        }
        else {
            return "0";
        }
    }
    // Regular Expression Matching
    // Implement regular expression matching with support for '.' and '*'.
    // '.' Matches any single character.
    // '*' Matches zero or more of the preceding element.
    // The matching should cover the entire input string (not partial).
    // The function prototype should be:
    // bool isMatch(const char *s, const char *p)
    // Some examples:
    // isMatch("aa","a") → false
    // isMatch("aa","aa") → true
    // isMatch("aaa","aa") → false
    // isMatch("aa", "a*") → true
    // isMatch("aa", ".*") → true
    // isMatch("ab", ".*") → true
    // isMatch("aab", "c*a*b") → true
    bool isMatch(const char *s, const char *p) {
        if (*p == '\0') {
            return *s == '\0';
        }
        if (p[1] == '*') {
            return matchStar(p[0], s, p + 2);
        }
        if (*s != '\0' && (*p == '.' || *p == *s)) {
            return isMatch(s + 1, p + 1);
        }
        return false;
    }
    bool matchStar(char c, const char *s, const char *p) {
        do {
            if (isMatch(s, p)) {
                return true;
            }
        } while (*s != '\0' && (*s++ == c || c == '.'));

        return false;
    }

    bool isMatch_dp(const char *s, const char *p) {
        // DP
        int lens = strlen(s), lenp = strlen(p);
        vector<vector<bool> > dp = vector<vector<bool> >(lens + 1, vector<bool>(lenp + 1, false));
        // Base case.
        bool valid = false;
        dp[0][0] = true;
        for (int idx = 2; idx <= lenp; idx += 2) {
            if (p[idx - 1] == '*') {
                valid = true;
                dp[0][idx] = true;
            }
            else {
                valid = false;
            }
            if (valid == false) {
                break;
            }
        }
        // DP
        for (int idxs = 1; idxs <= lens; ++idxs) {
            for (int idxp = 1; idxp <= lenp; ++idxp) {
                if (s[idxs - 1] == p[idxp - 1] || p[idxp - 1] == '.') {
                    dp[idxs][idxp] = dp[idxs - 1][idxp - 1];
                }
                else if (p[idxp - 1] == '*') {
                    if (s[idxs - 1] == p[idxp - 2] || p[idxp - 2] == '.') {
                        dp[idxs][idxp] = dp[idxs - 1][idxp] || dp[idxs][idxp - 2];
                    }
                    else {
                        dp[idxs][idxp] = dp[idxs][idxp - 2];
                    }
                }
            }
        }
        return dp[lens][lenp];
    }

    bool isMatch_dp_improved(const char *s, const char *p) {
        int lens = strlen(s), lenp = strlen(p);
        vector<bool> dp(lenp + 1, false), pre(dp);
        pre[0] = true;
        bool valid = false;
        for (int idx = 2; idx <= lenp; idx += 2) {
            if (p[idx - 1] == '*') {
                valid = true;
                pre[idx] = true;
            }
            else {
                valid = false;
            }
            if (valid == false) {
                break;
            }
        }

        for (int idxs = 1; idxs <= lens; ++idxs) {
            for (int idxp = 1; idxp <= lenp; ++idxp) {
                dp[idxp] = false;
                if (s[idxs - 1] == p[idxp - 1] || p[idxp - 1] == '.') {
                    dp[idxp] = pre[idxp - 1];
                }
                else if (p[idxp - 1] == '*') {
                    if (s[idxs - 1] == p[idxp - 2] || p[idxp - 2] == '.') {
                        dp[idxp] = pre[idxp] || dp[idxp - 2];
                    }
                    else {
                        // discard the p[idxp - 2].
                        dp[idxp] = dp[idxp - 2];
                    }
                }
            }
            pre = dp;
        }
        return pre[lenp];
    }
    // Evaluate Reverse Polish Notation
    // Evaluate the value of an arithmetic expression in Reverse Polish Notation.
    // Valid operators are +, -, *, /. Each operand may be an integer or another expression.
    // Some examples:
    //   ["2", "1", "+", "3", "*"] -> ((2 + 1) * 3) -> 9
    //   ["4", "13", "5", "/", "+"] -> (4 + (13 / 5)) -> 6
    int evalRPN(vector<string> &tokens) {
        stack<int> store;
        for (vector<string>::const_iterator it = tokens.begin(); it != tokens.end(); ++it) {
            if (*it == "*" || *it == "+" || *it == "-" || *it == "/") {
                int op2 = store.top();
                store.pop();
                int op1 = store.top();
                store.pop();
                int ret = doOperation(op1, op2, (*it)[0]);
                store.push(ret);
            }
            else {
                store.push(stoi(*it));
            }
        }
        return store.top();
    }
    int doOperation(const int &op1, const int &op2, const char &op) {
        switch (op) {
            case '*' : return op1 * op2;
            case '-' : return op1 - op2;
            case '/' : return op1 / op2;
            case '+' : return op1 + op2;
        }
        return 0;
    }
    // Longest Valid Parentheses
    // Given a string containing just the characters '(' and ')', find the length of the longest valid (well-formed) parentheses substring.
    // For "(()", the longest valid parentheses substring is "()", which has length = 2.
    // Another example is ")()())", where the longest valid parentheses substring is "()()", which has length = 4.
    int longestValidParentheses(string s) {
        stack<int> idxStack;
        vector<bool> mem(s.length(), false);
        int temp = 0;
        for (int idx = 0; idx < s.length(); ++idx) {
            if (s[idx] == '(') {
                idxStack.push(idx);
            }
            else if (!idxStack.empty()) {
                temp = idxStack.top();
                idxStack.pop();
                for (int i = temp; i <= idx; ++i) {
                    mem[i] = true;
                }
            }
        }
        temp = 0;
        int longest = 0;
        for (int idx = 0; idx < mem.size(); ++idx) {
            if (mem[idx] == true) {
                ++temp;
            }
            else {
                temp = 0;
            }
            longest = temp > longest ? temp : longest;
        }
        return longest;
    }

    int longestValidParentheses_improved(string s) {
        if (s.size() < 2) {
            return 0;
        }
        int longest = 0, tmpLongest = 0;
        stack<int> lenStack;
        char pre = s[0], cur;
        for (int idx = 1; idx < s.size(); ++idx) {
            cur = s[idx];
            if (pre == '(' && cur == '(') {
                lenStack.push(tmpLongest);
                tmpLongest = 0;
            }
            else if (pre == ')' && cur == ')') {
                if (!lenStack.empty()) {
                    tmpLongest += lenStack.top() + 2;
                    lenStack.pop();
                }
                else {
                    // Invalid parentheses.
                    tmpLongest = 0;
                }
            }
            else if (pre == '(' && cur ==')') {
                tmpLongest += 2;
            }
            longest = tmpLongest > longest ? tmpLongest : longest;
            pre = cur;
        }
        return longest;
    }

    int longestValidParentheses_dp(string s) {
        int n = s.size();
        vector<int> dp(n, 0);
        int max = 0;
        for(int i = 0; i < n; ++i) {
            if(s[i] == ')') {
                if(i - 1 >= 0 && s[i - 1] == '(') {
                    dp[i] = 2;
                    if(i - 2 >= 0 && dp[i - 2] > 0) {
                        dp[i] += dp[i - 2];
                    }
                }
                else if(i - 1 >= 0 && dp[i - 1] > 0 && i - 1 - dp[i - 1] >= 0 && s[i - 1 - dp[i - 1]] == '(') {
                    dp[i] = dp[i - 1] + 2;
                    if(i - 2 -dp[i - 1] >= 0 && dp[i - 2 - dp[i - 1]] > 0) {
                        dp[i] += dp[i - 2 - dp[i - 1]];
                    }
                }
                max = dp[i] > max ? dp[i] : max;
            }
        }
        return max;
    }
    // Interleaving String
    // Given s1, s2, s3, find whether s3 is formed by the interleaving of s1 and s2.
    // For example,
    // Given:
    // s1 = "aabcc",
    // s2 = "dbbca",
    // When s3 = "aadbbcbcac", return true.
    // When s3 = "aadbbbaccc", return false.
    bool isInterleave(string s1, string s2, string s3) {
        if (s1.size() + s2.size() != s3.size()) {
            return false;
        }
        vector<vector<bool> > dp(s1.size() + 1, vector<bool>(s2.size() + 1, false));
        dp[0][0] = true;
        for (int idx = 1; idx <= s1.size(); ++idx) {
            if (dp[idx - 1][0] == true) {
                dp[idx][0] = (s3[idx - 1] == s1[idx - 1]) ? true : false;
            }
            else {
                break;
            }
        }
        for (int idx = 1; idx <= s2.size(); ++idx) {
            if (dp[0][idx - 1] == true) {
                dp[0][idx] = (s3[idx - 1] == s2[idx - 1]) ? true : false;
            }
            else {
                break;
            }
        }
        for (int i = 1; i <= s1.size(); ++i) {
            for (int j = 1; j <= s2.size(); ++j) {
                if (s1[i - 1] != s3[i + j - 1] && s2[j - 1] != s3[i + j - 1]) {
                    continue;
                }
                else if (s1[i - 1] == s3[i + j - 1] && s2[j - 1] != s3[i + j - 1]) {
                    dp[i][j] = dp[i - 1][j];
                }
                else if (s2[j - 1] == s3[i + j - 1] && s1[i - 1] != s3[i + j -1]) {
                    dp[i][j] = dp[i][j - 1];
                }
                else {
                    dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
                }
            }
        }
        return dp[s1.size()][s2.size()];
    }
    // Simplify Path
    // Given an absolute path for a file (Unix-style), simplify it.
    // For example,
    // path = "/home/", => "/home"
    // path = "/a/./b/../../c/", => "/c"
    // click to show corner cases.
    //
    // Corner Cases:
    // Did you consider the case where path = "/../"?
    // In this case, you should return "/".
    // Another corner case is the path might contain multiple slashes '/' together, such as "/home//foo/".
    // In this case, you should ignore redundant slashes and return "/home/foo".
    string simplifyPath(string path) {
        stack<string> pathStack;
        int idx = 0;
        while (idx < path.size()) {
            // Delete the redundant '/'
            while (idx < path.size() && path[idx] == '/') {
                ++idx;
            }
            // Count the number of '.'
            int count = 0;
            while (idx < path.size() && path[idx] == '.') {
                ++idx;
                ++count;
            }
            // Combine the '.' and the words except '/'
            string ele(count, '.');
            while (idx < path.size() && path[idx] != '/') {
                ele += path[idx++];
            }
            // Only '.'s between two '/'
            if (count == ele.size()) {
                if (count == 2) {
                    if (!pathStack.empty()) {
                        pathStack.pop();
                    }
                    continue;
                }
                else if (count == 1) {
                    continue;
                }
            }
            pathStack.push(ele);
        }
        // Form the paths in the ordered format
        string ret = "";
        while (!pathStack.empty()) {
            if (ret == "") {
                ret = pathStack.top() + ret;
            }
            else {
                ret = pathStack.top() + '/' + ret;
            }
            pathStack.pop();
        }
        return '/' + ret;
    }

    string simplifyPath_improved(string path) {
        string ret, tmp;
        vector<string> pathVector;
        stringstream ss(path);
        while (getline(ss, tmp, '/')) {
            if (tmp == "" || tmp == ".") {
                continue;
            }
            if (tmp == ".." && !pathVector.empty()) {
                pathVector.pop_back();
            }
            else if (tmp != "..") {
                pathVector.push_back(tmp);
            }
        }
        for (vector<string>::const_iterator it = pathVector.begin(); it != pathVector.end(); ++it) {
            ret += '/' + *it;
        }
        return pathVector.empty() ? "/" : ret;
    }
    // Candy
    // There are N children standing in a line. Each child is assigned a rating value.
    // You are giving candies to these children subjected to the following requirements:
    // Each child must have at least one candy.
    // Children with a higher rating get more candies than their neighbors.
    // What is the minimum candies you must give?
    int candy(vector<int> &ratings) {
        int N = ratings.size();
        vector<int> candies(N, 1);
        for (int idx = 1; idx < N; ++idx) {
            if (ratings[idx] > ratings[idx - 1]) {
                candies[idx] = candies[idx - 1] + 1;
            }
        }
        for (int idx = N - 2; idx >= 0;--idx) {
            if (ratings[idx] > ratings[idx + 1]) {
                candies[idx] = max(candies[idx], candies[idx + 1] + 1);
            }
        }
        return accumulate(candies.begin(), candies.end(), 0);
    }
    // Word Search
    // Given a 2D board and a word, find if the word exists in the grid.
    // The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.
    // For example,
    // Given board =
    // [
    //   ["ABCE"],
    //   ["SFCS"],
    //   ["ADEE"]
    // ]
    // word = "ABCCED", -> returns true,
    // word = "SEE", -> returns true,
    // word = "ABCB", -> returns false.
    bool exist(vector<vector<char> > &board, string word) {
        bool found = false;
        vector<vector<bool> > visited(board.size(), vector<bool>(board[0].size(), false));
        for (int r = 0; r < board.size(); ++r) {
            for (int c = 0; c < board[0].size(); ++c) {
                if (exist(board, visited, r, c, word)) {
                    return true;
                }
            }
        }
        return found;
    }
    bool exist(const vector<vector<char> > &board, vector<vector<bool> > &visited, const int &r, const int &c, const string &word) {
        // DFS.
        if (board[r][c] != word[0] || visited[r][c]) {
            return false;
        }
        if (word.size() <= 1) {
            return true;
        }
        visited[r][c] = true;
        if (r > 0 && exist(board, visited, r - 1, c, word.substr(1))) {
            return true;
        }
        if (r < board.size() - 1 && exist(board, visited, r + 1, c, word.substr(1))) {
            return true;
        }
        if (c > 0 && exist(board, visited, r, c - 1, word.substr(1))) {
            return true;
        }
        if (c < board[0].size() && exist(board, visited, r, c + 1, word.substr(1))) {
            return true;
        }
        visited[r][c] = false;
        return false;
    }
    // Palindrome Partitioning II
    // Given a string s, partition s such that every substring of the partition is a palindrome.
    // Return the minimum cuts needed for a palindrome partitioning of s.
    // For example, given s = "aab",
    // Return 1 since the palindrome partitioning ["aa","b"] could be produced using 1 cut.
    int minCut_fault(string s) {
        // TLE
        int N = s.size();
        vector<vector<bool> > palindrome(N, vector<bool>(N, false));
        for (int idx = 0; idx < N; ++idx) {
            palindrome[idx][idx] = true;
        }
        for (int len = 2; len <= N; ++len) {
            for (int idx = 0; idx + len <= N; ++idx) {
                bool shorter = len == 2 ? true : palindrome[idx + 1][idx + len - 2];
                palindrome[idx][idx + len - 1] = s[idx] == s[idx + len - 1] && shorter;
            }
        }

        vector<int> cut(N, 0);
        for (int idx = 1; idx < N; ++idx) {
            if (!palindrome[0][idx]) {
                cut[idx] = idx;
                for (int j = 1; j <= idx; ++j) {
                    if (palindrome[j][idx]) {
                        cut[idx] = min(cut[idx], cut[j - 1] + 1);
                    }
                }
            }
        }
        return cut[N - 1];
    }

    int minCut(string s) {
        // TLE
        // Java version will pass the OJ, while C++ won't. Why?
        int N = s.size();
        if (N == 0) {
            return 0;
        }
        vector<vector<bool> > palindrome(N, vector<bool>(N, false));
        vector<int> cut(N, 0);
        for (int start = N - 1; start >= 0; --start) {
            cut[start] = N - start - 1;
            for (int end = start; end < N; ++end) {
                if (s[start] == s[end]) {
                    palindrome[start][end] = (end - start < 2) ? true : palindrome[start + 1][end - 1];
                }
                if (palindrome[start][end]) {
                    if (end == N - 1) {
                        cut[start] = 0;
                    }
                    else {
                        cut[start] = min(cut[start], cut[end + 1] + 1);
                    }
                }
            }
        }
        return cut[0];
    }
    // Substring with Concatenation of All Words
    // You are given a string, S, and a list of words, L, that are all of the same length. Find all starting indices of substring(s) in S that is a concatenation of each word in L exactly once and without any intervening characters.
    // For example, given:
    // S: "barfoothefoobarman"
    // L: ["foo", "bar"]
    // You should return the indices: [0,9].
    // (order does not matter).
    vector<int> findSubstring_fault(string S, vector<string> &L) {
        // TLE
        vector<int> indices;
        if (L.size() == 0 || S.size() == 0 || S.size() < L[0].length()) {
            return indices;
        }
        int wordLength = L[0].length();
        unordered_multimap<string, bool> words;
        for (vector<string>::const_iterator it = L.begin(); it != L.end(); ++it) {
            words.insert(pair<string, bool>(*it, false));
        }
        for (int idx = 0; idx < S.length(); ++idx) {
            if (words.count(S.substr(idx, wordLength)) > 0) {
                for (int j = idx; j <= S.size() - wordLength; j += wordLength) {
                    pair<unordered_multimap<string, bool>::iterator, unordered_multimap<string, bool>::iterator> found = words.equal_range(S.substr(j, wordLength));
                    while (found.first != found.second && found.first->second) {
                        ++found.first;
                    }
                    if (found.first != found.second) {
                        found.first->second = true;
                    }
                    else {
                        break;
                    }
                }
                bool valid = true;
                for (unordered_multimap<string, bool>::iterator it = words.begin(); it != words.end(); ++it) {
                    if (it->second) {
                        it->second = false;
                    }
                    else {
                        valid = false;
                    }
                }
                if (valid) {
                    indices.push_back(idx);
                }
            }
        }
        return indices;
    }

    vector<int> findSubstring(string S, vector<string> &L) {
        // Window moving method.
        vector<int> indices;
        if (L.size() == 0 || S.size() == 0 || S.size() < L[0].length()) {
            return indices;
        }
        int wordLength = L[0].length();
        unordered_map<string, int> words;
        for (int i = 0; i < L.size(); ++i) {
            ++words[L[i]];
        }
        for (int i = 0; i < wordLength; ++i) {
            int start = i, wordCount = 0;
            unordered_map<string, int> temp;
            for (int j = i; j <= S.size() - wordLength; j += wordLength) {
                string str = S.substr(j, wordLength);
                if (words.count(str) > 0) {
                    ++temp[str];
                    if (temp[str] <= words[str]) {
                        ++wordCount;
                    }
                    else {
                        while (temp[str] > words[str]) {
                            string str1 = S.substr(start, wordLength);
                            --temp[str1];
                            if (temp[str1] < words[str1]) {
                                --wordCount;
                            }
                            start += wordLength;
                        }
                    }
                    // valid substring.
                    if (wordCount == L.size()) {
                        indices.push_back(start);
                        // move advance for a wordLength step.
                        --temp[S.substr(start, wordLength)];
                        --wordCount;
                        start += wordLength;
                    }
                }
                else {
                    temp.clear();
                    wordCount = 0;
                    start = j + wordLength;
                }
            }
        }
        return indices;
    }
    // Reverse Bits
    // Reverse bits of a given 32 bits unsigned integer.
    // For example, given input 43261596 (represented in binary as 00000010100101000001111010011100), return 964176192 (represented in binary as 00111001011110000010100101000000).
    // Follow up:
    // If this function is called many times, how would you optimize it?
    // Related problem: Reverse Integer
    uint32_t reverseBits(uint32_t n) {
        uint32_t ret = 0;
        int bits = 1;
        while (bits <= 32) {
            ret <<= 1;
            ret |= (n & 0x01);
            n >>= 1;
            ++bits;
        }
        return ret;
    }
    // Word Ladder
    // Given two words (start and end), and a dictionary, find the length of shortest transformation sequence from start to end, such that:
    // Only one letter can be changed at a time
    // Each intermediate word must exist in the dictionary
    // For example,
    // Given:
    // start = "hit"
    // end = "cog"
    // dict = ["hot","dot","dog","lot","log"]
    // As one shortest transformation is "hit" -> "hot" -> "dot" -> "dog" -> "cog",
    // return its length 5.
    // Note:
    // Return 0 if there is no such transformation sequence.
    // All words have the same length.
    // All words contain only lowercase alphabetic characters.
    int ladderLength(string start, string end, unordered_set<string> &dict) {
        unordered_map<string, int> visited;
        visited[start] = 1;
        queue<string> trans;
        trans.push(start);
        int wordLen = start.length();
        while (!trans.empty()) {
            string word = trans.front();
            trans.pop();
            for (int i = 0; i < wordLen; ++i) {
                for (char v = 'a'; v <= 'z'; ++v) {
                    string tmp = word.substr(0, i) + v;
                    if (i < wordLen - 1) {
                        tmp += word.substr(i + 1);
                    }
                    if (tmp == end) {
                        return visited[word] + 1;
                    }
                    else if (dict.count(tmp) > 0 && visited.count(tmp) == 0) {
                        visited[tmp] = visited[word] + 1;
                        trans.push(tmp);
                    }
                }
            }
        }
        return 0;
    }
    // Repeated DNA Sequences
    // All DNA is composed of a series of nucleotides abbreviated as A, C, G, and T, for example: "ACGAATTCCG". When studying DNA, it is sometimes useful to identify repeated sequences within the DNA.
    // Write a function to find all the 10-letter-long sequences (substrings) that occur more than once in a DNA molecule.
    // For example,
    // Given s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT",
    // geturn:
    // ["AAAAACCCCC", "CCCCCAAAAA"].
    vector<string> findRepeatedDnaSequences(string s) {
        vector<string> ret;
        unordered_set<int> map;
        int len = s.length();
        for (int idx = 0; idx <= len - 10; ++idx) {
            string tmp = s.substr(idx, 10);
            int hashValue = getHash(tmp);
            if (map.count(hashValue) > 0 && find(ret.begin(), ret.end(), tmp) == ret.end()) {
                ret.push_back(tmp);
            }
            else {
                map.insert(hashValue);
            }
        }
        return ret;
    }
    int getHash(const string &str) {
        int ret = 0;
        for (int idx = 0; idx < str.size(); ++idx) {
            ret = ret << 2 | getHashCode(str[idx]);
        }
        return ret;
    }
    int getHashCode(const char &c) {
        switch (c) {
            case 'A' : return 0;
            case 'C' : return 1;
            case 'G' : return 2;
            case 'T' : return 3;
        }
        // Handle the warning.
        return 0;
    }

    vector<string> findRepeatedDnaSequences_improved(string s) {
        vector<string> ret;
        unordered_map<int ,bool> m;
        for (int tmp = 0, idx = 0; idx <= s.size() - 10; ++idx) {
            tmp = tmp << 3 & (0x3FFFFFFF | (s[idx] & 7));
            if (m.find(tmp) != m.end()) {
                if (m[tmp]) {
                    ret.push_back(s.substr(idx, 10));
                    m[tmp] = false;
                }
            }
            else {
                m[tmp] = true;
            }
        }
        return ret;
    }
    // Maximum Product
    // Find the contiguous subarray within an array (containing at least one number) which has the largest product.
    // For example, given the array [2,3,-2,4],
    // the contiguous subarray [2,3] has the largest product = 6.
    int maxProduct(int A[], int n) {
        vector<int> maxP(n, 0), minP(n, 0);
        maxP[0] = minP[0] = A[0];
        for (int idx = 1; idx < n; ++idx) {
            if (A[idx] > 0) {
                maxP[idx] = max(maxP[idx - 1] * A[idx], A[idx]);
                minP[idx] = min(minP[idx - 1] * A[idx], A[idx]);
            }
            else {
                maxP[idx] = max(minP[idx - 1] * A[idx], A[idx]);
                minP[idx] = min(maxP[idx - 1] * A[idx], A[idx]);
            }
        }
        return *max_element(maxP.begin(), maxP.end());
    }

    int maxProduct_improved(int A[], int n) {
        // Save space.
        int ret, maxP, minP;
        ret = maxP = minP = A[0];
        for (int idx = 1; idx < n; ++idx) {
            if (A[idx] > 0) {
                maxP = max(maxP * A[idx], A[idx]);
                minP = min(minP * A[idx], A[idx]);
            }
            else {
                int temp = maxP;
                maxP = max(minP * A[idx], A[idx]);
                minP = min(temp * A[idx], A[idx]);
            }
            if (maxP > ret) {
                ret = maxP;
            }
        }
        return ret;
    }
    // Number of 1 Bits
    // Write a function that takes an unsigned integer and returns the number of ’1' bits it has (also known as the Hamming weight).
    // For example, the 32-bit integer ’11' has binary representation 00000000000000000000000000001011, so the function should return 3.
    int hammingWeight(uint32_t n) {
        int ret = 0;
        while (n) {
            ret += n & 0x1;
            n >>= 1;
        }
        return ret;
    }

    int hammingWeight_improved(uint32_t n) {
        int ret = 0;
        while (n) {
            n &= n - 1;
            ++ret;
        }
        return ret;
    }
    // Minimum Window Substring
    // Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).
    // For example,
    // S = "ADOBECODEBANC"
    // T = "ABC"
    // Minimum window is "BANC".
    // Note:
    // If there is no such window in S that covers all characters in T, return the emtpy string "".
    // If there are multiple such windows, you are guaranteed that there will always be only one unique minimum window in S.
    string minWindow(string S, string T) {
        string ret;
        if (!S.size() || !T.size()) {
            return ret;
        }
        int len = T.size();
        vector<int> map(256, -1);
        vector<int> tmp(256, -1);
        for (int i = 0; i < len; ++i) {
            ++map[T[i]];
        }
        int start = 0, winLen = 1, matchLen = 0;
        for (int idx = 0; idx < S.size(); ++idx, ++winLen) {
            if (map[S[idx]] >= 0) {
                if (tmp[S[idx]] < map[S[idx]]) {
                    ++matchLen;
                }
                ++tmp[S[idx]];
            }
            if (matchLen == len) {
                // Remove the redundant.
                while (start <= idx && (map[S[start]] < 0 || tmp[S[start]] > map[S[start]])) {
                    --winLen;
                    --tmp[S[start]];
                    ++start;
                }
                if (ret == "" || ret.size() > winLen) {
                    ret = S.substr(start, winLen);
                }
                // Move the window.
                --tmp[S[start]];
                --matchLen;
                --winLen;
                ++start;
            }
        }
        return ret;
    }
    // Two Sum
    // Given an array of integers, find two numbers such that they add up to a specific target number.
    // The function twoSum should return indices of the two numbers such that they add up to the target, where index1 must be less than index2. Please note that your returned answers (both index1 and index2) are not zero-based.
    // You may assume that each input would have exactly one solution.
    // Input: numbers={2, 7, 11, 15}, target=9
    // Output: index1=1, index2=2
    vector<int> twoSum(vector<int> &numbers, int target) {
        unordered_map<int, int> hashTable;
        vector<int> ret;
        for (int idx = 0; idx < numbers.size(); ++idx) {
            if (hashTable.count(numbers[idx]) > 0) {
                ret.push_back(hashTable[numbers[idx]] + 1);
                ret.push_back(idx + 1);
                return ret;
            }
            hashTable[target - numbers[idx]] = idx;
        }
        return ret;
    }
    // Rotate Array
    // Rotate an array of n elements to the right by k steps.
    // For example, with n = 7 and k = 3, the array [1,2,3,4,5,6,7] is rotated to [5,6,7,1,2,3,4].
    // Note:
    // Try to come up as many solutions as you can, there are at least 3 different ways to solve this problem.
    void rotate(int nums[], int n, int k) {
        if (n <= 1 || k % n == 0) {
            return;
        }
        k %= n;
        // Keep rotating until we have rotated n elements.
        for (int idx = 0, count = 0; count < n; ++idx) {
            int cur = idx;
            int temp, next = nums[cur];
            do {
                temp = nums[(cur + k) % n];
                nums[(cur + k) % n] = next;
                next = temp;
                cur = (cur + k) % n;
                ++count;
            } while (cur != idx);
        }
    }

    void rotate2(int nums[], int n, int k) {
        k %= n;
        reverse(nums, nums + n - k);
        reverse(nums + n - k, nums + n);
        reverse(nums, nums + n);
    }
    // 3Sum
    // Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.
    // Note:
    // Elements in a triplet (a,b,c) must be in non-descending order. (ie, a ≤ b ≤ c)
    // The solution set must not contain duplicate triplets.
    // For example, given array S = {-1 0 1 2 -1 -4},
    // A solution set is:
    //   (-1, 0, 1)
    //   (-1, -1, 2)
    vector<vector<int> > threeSum(vector<int> &num) {
        vector<vector<int> > ret;
        if (num.size() < 3) {
            return ret;
        }
        sort(num.begin(), num.end());
        for (int idx = 0; idx <= num.size() - 3;) {
            int beg = idx + 1, end = num.size() - 1;
            while (beg < end) {
                int sum = num[idx] + num[beg] + num[end];
                if (sum <= 0) {
                    if (sum == 0) {
                        vector<int> line;
                        line.push_back(num[idx]);
                        line.push_back(num[beg]);
                        line.push_back(num[end]);
                        ret.push_back(line);
                    }
                    do {
                        ++beg;
                    } while (beg < end && num[beg] == num[beg - 1]);
                }
                else {
                    do {
                        --end;
                    } while (end > beg && num[end] == num[end + 1]);
                }
            }
            do {
                ++idx;
            } while (idx <= num.size() - 3 && num[idx] == num[idx - 1]);
        }
        return ret;
    }
    // Decode Ways
    // A message containing letters from A-Z is being encoded to numbers using the following mapping:
    // 'A' -> 1
    // 'B' -> 2
    // ...
    // 'Z' -> 26
    // Given an encoded message containing digits, determine the total number of ways to decode it.
    // For example,
    // Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12).
    // The number of ways decoding "12" is 2.
    int numDecodings(string s) {
        int N = s.size();
        if (N == 0) {
            return 0;
        }
        vector<int> num;
        for (int idx = 0; idx < N; ++idx) {
            if (s[idx] == '0') {
                if (idx == 0) {
                    return 0;
                }
                else {
                    int n = stoi(s.substr(idx - 1, 2));
                    if (n > 0 && n <= 26) {
                        num.back() = n;
                    }
                    else {
                        return 0;
                    }
                }
            }
            else {
                num.push_back(s[idx] - '0');
            }
        }
        vector<int> dp(num.size() + 1, 1);
        dp[1] = 1;
        for (int idx = 1; idx < num.size(); ++idx) {
            dp[idx + 1] = dp[idx];
            if (num[idx] > 9) {
                continue;
            }
            int n = num[idx - 1] * 10 + num[idx];
            if (n > 0 && n <= 26) {
                dp[idx + 1] += dp[idx - 1];
            }
        }
        return dp[num.size()];
    }

    int numDecodings1(string s) {
        if (s.size() == 0) {
            return 0;
        }
        if (s.size() == 1) {
            return s[0] == '0' ? 0 : 1;
        }

        vector<int> dp(s.size() + 1, 0);
        dp[0] = 1;
        dp[1] = s[0] == '0' ? 0 : 1;
        for (int idx = 2; idx <= s.size(); ++idx) {
            if (s[idx - 1] != '0') {
                dp[idx] = dp[idx - 1];
            }
            if (stoi(s.substr(idx - 2, 2)) <= 26 && s[idx - 2] != '0') {
                dp[idx] += dp[idx - 2];
            }
        }
        return dp[s.size()];
    }

    int numDecodings2(string s) {
        int N = s.size();
        if (N == 0) {
            return 0;
        }
        vector<int> dp(N + 1, 0);
        dp[N] = 1;
        dp[N - 1] = s[N - 1] == '0' ? 0 : 1;
        for (int idx = N - 2; idx >= 0; --idx) {
            if (s[idx] != '0') {
                if (stoi(s.substr(idx, 2)) <= 26) {
                    dp[idx] = dp[idx + 1] + dp[idx + 2];
                }
                else {
                    dp[idx] = dp[idx + 1];
                }
            }
        }
        return dp[0];
    }
    // Divide two integers without using multiplication, division and mod operator.
    // If it is overflow, return MAX_INT.
    int divide(int dividend, int divisor) {
        if (divisor == 0 || (dividend == INT_MIN && divisor == -1)) {
            return INT_MAX;
        }
        bool symbol = (dividend > 0) ^ (divisor > 0);
        long lDividend = labs(dividend), lDivisor = labs(divisor);
        int quot = 0, i = 0;
        while (lDividend >= lDivisor) {
            i = 0;
            while (lDividend >= lDivisor << i) {
                lDividend -= lDivisor << i;
                quot += 1 << i;
                ++i;
            }
        }
        return symbol ? -quot : quot;
    }
    // Reverse Words in a String
    // Given an input string, reverse the string word by word.
    // For example,
    // Given s = "the sky is blue",
    // return "blue is sky the".
    void reverseWords(string &s) {
        string ret;
        for (int idx = s.size() - 1; idx >= 0; --idx) {
            while (idx >= 0 && s[idx] == ' ') {
                --idx;
            }
            int count = 0;
            while (idx >= 0 && s[idx] != ' ') {
                ++count;
                --idx;
            }
            if (count == 0) {
                break;
            }
            else if (ret == "") {
                ret = s.substr(idx + 1, count);
            }
            else {
                ret += ' ' + s.substr(idx + 1, count);
            }
        }
        s = ret;
    }
    // Largest Number
    // Given a list of non negative integers, arrange them such that they form the largest number.
    // For example, given [3, 30, 34, 5, 9], the largest formed number is 9534330.
    // Note: The ret may be very large, so you need to return a string instead of an integer.
    string largestNumber(vector<int> &num) {
        sort(num.begin(), num.end(), mySort);
        if (num[0] == 0) {
            return "0";
        }
        string ret = "";
        for (int idx = 0; idx < num.size(); ++idx) {
            ret.append(to_string(num[idx]));
        }
        return ret;
    }
    static bool mySort(const int &a, const int &b) {
        string num1 = to_string(a) + to_string(b), num2 = to_string(b) + to_string(a);
        return num1 > num2;
    }
    // Compare Version Numbers
    // Compare two version numbers version1 and version2.
    // If version1 > version2 return 1, if version1 < version2 return -1, otherwise return 0.
    // You may assume that the version strings are non-empty and contain only digits and the . character.
    // The . character does not represent a decimal point and is used to separate number sequences.
    // For instance, 2.5 is not "two and a half" or "half way to version three", it is the fifth second-level revision of the second first-level revision.
    // Here is an example of version numbers ordering:
    // 0.1 < 1.1 < 1.2 < 13.37
    int compareVersion(string version1, string version2) {
        int len1 = version1.length(), len2 = version2.length();
        int idx1 = 0, idx2 = 0;
        while (idx1 < len1 || idx2 < len2) {
            int num1 = findVersion(version1, idx1), num2 = findVersion(version2, idx2);
            if (num1 > num2) {
                return 1;
            }
            else if (num1 < num2) {
                return -1;
            }
        }
        return 0;
    }
    int findVersion(const string &version, int &idx) {
        int ret = 0;
        for (;idx < version.length(); ++idx) {
            if (version[idx] == '.') {
                break;
            }
            ret = ret * 10 + version[idx] - '0';
        }
        ++idx;
        return ret;
    }
    // Surrounded Regions
    // Given a 2D board containing 'X' and 'O', capture all regions surrounded by 'X'.
    // A region is captured by flipping all 'O's into 'X's in that surrounded region.
    // For example,
    // X X X X
    // X O O X
    // X X O X
    // X O X X
    // After running your function, the board should be:
    // X X X X
    // X X X X
    // X X X X
    // X O X X
    void solve(vector<vector<char> > &board) {
        if (board.size() < 3 || board[0].size() < 3) {
            return;
        }
        int rowNum = board.size(), colNum = board[0].size();
        for (int r = 0; r < rowNum; ++r) {
            if (board[r][0] == 'O') {
                board[r][0] = '+';
                BFS(board, r, 0);
            }
            if (board[r][colNum - 1] == 'O') {
                board[r][colNum - 1] = '+';
                BFS(board, r, colNum - 1);
            }
        }
        for (int c = 0; c < colNum; ++c) {
            if (board[0][c] == 'O') {
                board[0][c] = '+';
                BFS(board, 0, c);
            }
            if (board[rowNum - 1][c] == 'O') {
                board[rowNum - 1][c] = '+';
                BFS(board, rowNum - 1, c);
            }
        }

        for (int r = 0; r < rowNum; ++r) {
            for (int c = 0; c < colNum; ++c) {
                if (board[r][c] == 'O') {
                    board[r][c] = 'X';
                }
                else if (board[r][c] == '+') {
                    board[r][c] = 'O';
                }
            }
        }
    }
    void BFS(vector<vector<char> > &board, const int &r, const int &c) {
        queue<pair<int, int> > q;
        q.push(pair<int, int>(r, c));
        while (!q.empty()) {
            int row = q.front().first, col = q.front().second;
            q.pop();
            if (row - 1 >= 0 && board[row - 1][col] == 'O') {
                q.push(pair<int, int>(row - 1, col));
                board[row - 1][col] = '+';
            }
            if (row + 1 < board.size() && board[row + 1][col] == 'O') {
                q.push(pair<int, int>(row + 1, col));
                board[row + 1][col] = '+';
            }
            if (col - 1 >= 0 && board[row][col - 1] == 'O') {
                q.push(pair<int, int>(row, col - 1));
                board[row][col - 1] = '+';
            }
            if (col + 1 < board[0].size() && board[row][col + 1] == 'O') {
                q.push(pair<int, int>(row, col + 1));
                board[row][col + 1] = '+';
            }
        }
    }
    // String to Integer (atoi)
    // Implement atoi to convert a string to an integer.
    // Hint: Carefully consider all possible input cases. If you want a challenge, please do not see below and ask yourself what are the possible input cases.
    // Notes: It is intended for this problem to be specified vaguely (ie, no given input specs). You are responsible to gather all the input requirements up front.
    // The function first discards as many whitespace characters as necessary until the first non-whitespace character is found. Then, starting from this character, takes an optional initial plus or minus sign followed by as many numerical digits as possible, and interprets them as a numerical value.
    // The string can contain additional characters after those that form the integral number, which are ignored and have no effect on the behavior of this function.
    // If the first sequence of non-whitespace characters in str is not a valid integral number, or if no such sequence exists because either str is empty or it contains only whitespace characters, no conversion is performed.
    // If no valid conversion could be performed, a zero value is returned. If the correct value is out of the range of representable values, INT_MAX (2147483647) or INT_MIN (-2147483648) is returned.
    int atoi(string str) {
        int idx = 0, N = str.size();
        int symbol = 1;
        // Discard the white spaces.
        while (idx < N && str[idx] == ' ') {
            ++idx;
        }
        // Extract the symbol.
        if (idx == N) {
            return 0;
        }
        else if (str[idx] == '+' || str[idx] == '-') {
            symbol = str[idx] == '+' ? 1 : -1;
            ++idx;
        }
        long num = 0;
        for (; idx < N; ++idx) {
            if (str[idx] > '9' || str[idx] < '0') {
                break;
            }
            num = num * 10 + str[idx] - '0';
            if (num > INT_MAX) {
                // Overflowed.
                return symbol == 1 ? INT_MAX : INT_MIN;
            }
        }
        return symbol * num;
    }
    // Fraction to Recurring Decimal
    // Given two integers representing the numerator and denominator of a fraction, return the fraction in string format.
    // If the fractional part is repeating, enclose the repeating part in parentheses.
    // For example,
    // Given numerator = 1, denominator = 2, return "0.5".
    // Given numerator = 2, denominator = 1, return "2".
    // Given numerator = 2, denominator = 3, return "0.(6)".
    string fractionToDecimal(int numerator, int denominator) {
        if (numerator == 0) {
            return "0";
        }
        string symbol = (numerator > 0) ^ (denominator > 0) ? "-" : "";
        long num = labs(numerator), den = labs(denominator);
        string integer, decimal;
        // Calculate the integer part.
        if (num >= den) {
            integer += to_string(num / den);
            num %= den;
        }
        // Check whether the integer part is 0.
        if (integer == "") {
            integer = "0";
        }
        // Calculate the decimal part.
        unordered_map<int ,int> map;
        map[num] = 0;
        int idx = 0;
        while (num) {
            num *= 10;
            int quotient = num / den, remainder = num % den;
            decimal += to_string(quotient);
            ++idx;
            num = remainder;
            // Check whether the next position will cause a loop.
            if (map.count(remainder) == 0) {
                map[remainder] = idx;
            }
            else {
                string loop = "(" + decimal.substr(map[remainder], idx - map[remainder]) + ")";
                decimal.erase(map[remainder]);
                decimal.append(loop);
                break;
            }
        }
        return symbol.append(decimal == "" ? integer : integer + '.' + decimal);
    }
    // Median of Two Sorted Arrays
    // There are two sorted arrays A and B of size m and n respectively. Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
    double findMedianSortedArrays_lazy(int A[], int m, int B[], int n) {
        // A lazy solution.
        // O(m + n)
        if (m == 0 && n == 0) {
            return 0;
        }
        vector<int> temp(m + n, 0);
        std::merge(A, A + m, B, B + n, temp.begin());
        return (m + n) % 2 == 0 ? (temp[(m + n - 1) / 2] + temp[(m + n + 1) / 2]) / 2.0 : temp[(m + n) / 2];
    }

    double findMedianSortedArrays(int A[], int m, int B[], int n) {
        // O(log(m + n))
        double ret = 0;
        int K;
        if (m > 0 || n > 0) {
            K = (m + n + 1) / 2;
            ret = findMedianSortedArrays(A, m, B, n, K);
            // If the m + n is even, we have to return the average of
            // the k-th element and the k+1-th element.
            if ((m + n) % 2 == 0) {
                K = (m + n + 1) / 2 + 1;
                ret = (ret + findMedianSortedArrays(A, m, B, n, K)) / 2.0;
            }
        }
        return ret;
    }
    int findMedianSortedArrays(int A[], int m, int B[], int n, int K) {
        int mida = m / 2, midb = n / 2;
        if (m <= 0) {
            return B[K - 1];
        }
        else if (n <= 0) {
            return A[K - 1];
        }

        if (B[midb] >= A[mida]) {
            if (midb + mida + 1 >= K) {
                return findMedianSortedArrays(A, m, B, midb, K);
            }
            else {
                return findMedianSortedArrays(A + mida + 1, m - mida - 1, B, n, K - mida - 1);
            }
        }
        else {
            if (mida + midb + 1 >= K) {
                return findMedianSortedArrays(A, mida, B, n, K);
            }
            else {
                return findMedianSortedArrays(A, m, B + midb + 1, n - midb - 1, K - midb - 1);
            }
        }
    }

    double findMedianSortedArrays_twoheap(int A[], int m, int B[], int n) {
        // create a min heap and a max heap
        priority_queue<int> maxheap;
        priority_queue<int, vector<int>, greater<int> > minheap;
        int pA, pB;
        pA = pB = 0;
        if (m > 0 && n > 0) {
            minheap.push(max(A[pA], B[pB]));
            maxheap.push(min(A[pA], B[pB]));
        }
        else if (m == 0 && n > 0) {
            minheap.push(B[pB]);
        }
        else if (m > 0 && n == 0) {
            minheap.push(A[pA]);
        }
        ++pA;
        ++pB;

        while (pA < m || pB < n) {
            if (pA < m) {
                if (maxheap.empty() || A[pA] <= maxheap.top()) {

                    maxheap.push(A[pA]);
                }
                else {
                    minheap.push(A[pA]);
                }
            }
            if (pB < n) {
                if (maxheap.empty() || B[pB] <= maxheap.top()) {
                    maxheap.push(B[pB]);
                }
                else {
                    minheap.push(B[pB]);
                }
            }

            if (minheap.size() > maxheap.size() + 1) {
                maxheap.push(minheap.top());
                minheap.pop();
            }
            else if (maxheap.size() > minheap.size() + 1) {
                minheap.push(maxheap.top());
                maxheap.pop();
            }
            ++pA;
            ++pB;
        }
        if (minheap.size() == maxheap.size()) {
            return (minheap.top() + maxheap.top()) / 2.0;
        }
        else if (minheap.size() > maxheap.size()) {
            return minheap.top();
        }
        else {
            return maxheap.top();
        }

    }
    // Word Break II
    // Given a string s and a dictionary of words dict, add spaces in s to construct a sentence where each word is a valid dictionary word.
    // Return all such possible sentences.
    // For example, given
    // s = "catsanddog",
    // dict = ["cat", "cats", "and", "sand", "dog"].
    // A solution is ["cats and dog", "cat sand dog"].
    vector<string> wordBreakII(string s, unordered_set<string> &dict) {
        vector<string> ret;
        for (int idx = s.size() - 1; idx >= 0; --idx) {
            if (dict.count(s.substr(idx)) != 0) {
                break;
            }
            else if (idx == 0) {
                return ret;
            }
        }

        for (int idx = 0; idx < s.size(); ++idx) {
            if (dict.count(s.substr(0, idx + 1)) != 0) {
                vector<string> tmp = wordBreakII(s.substr(idx + 1), dict);
                if (!tmp.empty()) {
                    for (vector<string>::iterator it = tmp.begin(); it != tmp.end(); ++it) {
                        ret.push_back(s.substr(0, idx + 1) + " " + *it);
                    }
                }
            }
        }
        if (dict.count(s) != 0) {
            ret.push_back(s);
        }
        return ret;
    }
    // Dungeon Game
    // The demons had captured the princess (P) and imprisoned her in the bottom-right corner of a dungeon. The dungeon consists of M x N rooms laid out in a 2D grid. Our valiant knight (K) was initially positioned in the top-left room and must fight his way through the dungeon to rescue the princess.
    // The knight has an initial health point represented by a positive integer. If at any point his health point drops to 0 or below, he dies immediately.
    //
    // _________________________
    // | -2 (K) |  -3  |   3    |
    // | -5     |  -10 |   1    |
    // | 10     |  30  |  -5 (P)|
    // _________________________
    int calculateMinimumHP(vector<vector<int> > &dungeon) {
        if (dungeon.size() == 0) {
            return 0;
        }
        vector<vector<int> > dp(dungeon.size(), vector<int>(dungeon[0].size(), INT_MAX));
        int rowNum = dungeon.size(), colNum = dungeon[0].size();
        for (int r = rowNum - 1; r >= 0; --r) {
            for (int c = colNum - 1; c >= 0; --c) {
                if (r == rowNum - 1 && c == colNum - 1) {
                    dp[r][c] = dungeon[r][c] >= 0 ? 1 : 1 - dungeon[r][c];
                    continue;
                }
                if (r < rowNum - 1) {
                    dp[r][c] = dp[r + 1][c] - dungeon[r][c];
                    if (dp[r][c] <= 0) dp[r][c] = 1;
                }
                if (c < colNum - 1) {
                    int temp = dp[r][c + 1] - dungeon[r][c];
                    if (temp <= 0) temp = 1;
                    dp[r][c] = min(dp[r][c], temp);
                }
            }
        }
        return dp[0][0];
    }
    // Best Time to Buy and Sell Stock IV
    // Say you have an array for which the ith element is the price of a given stock on day i.
    // Design an algorithm to find the maximum profit. You may complete at most k transactions.
    // Note:
    // You may not engage in multiple transactions at the same time (ie, you must sell the stock before you buy again).
    int maxProfitIV(int k, vector<int> &prices) {
        int ret = 0;
        int N = prices.size();
        int v = 0, p = 0;
        vector<int> profits;
        stack<pair<int, int> > vpPairs;
        while (p < N) {
            for (v = p; (v < N - 1) && prices[v] >= prices[v + 1]; ++v);
            for (p = v + 1; (p < N) && prices[p] >= prices[p - 1]; ++p);
            while (!vpPairs.empty() && prices[vpPairs.top().first] > prices[v]) {
                profits.push_back(prices[vpPairs.top().second] - prices[vpPairs.top().first]);
                vpPairs.pop();
            }
            while (!vpPairs.empty() && prices[vpPairs.top().second] <= prices[p - 1]) {
                profits.push_back(prices[vpPairs.top().second] - prices[v]);
                v = vpPairs.top().first;
                vpPairs.pop();
            }
            vpPairs.push(pair<int, int>(v, p - 1));
        }
        while (!vpPairs.empty()) {
            profits.push_back(prices[vpPairs.top().second] - prices[vpPairs.top().first]);
            vpPairs.pop();
        }
        make_heap(profits.begin(), profits.end());
        for (int i = 0; (i < k) && !profits.empty(); ++i) {
            pop_heap(profits.begin(), profits.end());
            ret += profits.back();
            profits.pop_back();
        }
        return ret;
    }
    // Wildcard Matching
    // Implement wildcard pattern matching with support for '?' and '*'.
    // '?' Matches any single character.
    // '*' Matches any sequence of characters (including the empty sequence).
    // The matching should cover the entire input string (not partial).
    // The function prototype should be:
    // bool isMatch(const char *s, const char *p)
    // Some examples:
    // isMatch("aa","a") → false
    // isMatch("aa","aa") → true
    // isMatch("aaa","aa") → false
    // isMatch("aa", "*") → true
    // isMatch("aa", "a*") → true
    // isMatch("ab", "?*") → true
    // isMatch("aab", "c*a*b") → false
    bool isMatchII(const char *s, const char *p) {
        // TLE for large tests.
        int lens = strlen(s), lenp = strlen(p);
        int starNum = count(p, p + lenp, '*');
        if (lenp - starNum > lens) {
            return false;
        }
        vector<bool> dp(lenp + 1, false), pre(dp);
        pre[0] = true;
        for (int idx = 1; idx <= lenp; ++idx) {
            pre[idx] = pre[idx - 1] && p[idx - 1] == '*';
        }
        for (int idxs = 1; idxs <= lens; ++idxs) {
            for (int idxp = 1; idxp <= lenp; ++idxp) {
                dp[idxp] = false;
                if (s[idxs - 1] == p[idxp - 1] || p[idxp - 1] == '?') {
                    dp[idxp] = pre[idxp - 1];
                }
                else if (p[idxp - 1] == '*') {
                    dp[idxp] = pre[idxp] || dp[idxp - 1];
                }
            }
            pre = dp;
        }
        return pre[lenp];
    }
    // Integer to Roman
    // Given an integer, convert it to a roman numeral.
    // Input is guaranteed to be within the range from 1 to 3999.
    string intToRoman(int num) {
        string table[4][10] = {{"", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"},
            {"", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"},
            {"", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"},
            {"", "M", "MM", "MMM"}
        };
        string ret;
        for (int count = 0; num > 0; ++count) {
            ret = table[count][num % 10] + ret;
            num /= 10;
        }
        return ret;
    }
    // Roman to Integer
    // Given a roman numeral, convert it to an integer.
    // Input is guaranteed to be within the range from 1 to 3999.
    int romanToInt(string s) {
        /*
        unordered_map<char, int> table = {
            {'I', 1}, {'V', 5}, {'X', 10}, {'L', 50}, {'C', 100}, {'D', 500}, {'M', 1000}
        };
        */
        unordered_map<char, int> table;
        table['I'] = 1; table['V'] = 5; table['X'] = 10; table['L'] = 50; table['C'] = 100; table['D'] = 500; table['M'] = 1000;
        int ret = 0;
        for (int idx = 0; idx < s.length(); ++idx) {
            int temp = table[s[idx]];
            if (idx != s.length() - 1 && table[s[idx]] < table[s[idx + 1]]) {
                temp = -temp;
            }
            ret += temp;
        }
        return ret;
    }
    // House Robber
    // You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.
    // Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.
    int rob(vector<int> &num) {
        int N = num.size();
        vector<int> dp(N + 2, 0);
        for (int idx = 2; idx < N + 2; ++idx) {
            dp[idx] = max(dp[idx - 1], dp[idx - 2] + num[idx - 2]);
        }
        return dp[N + 1];
    }
    // Binary Tree Right Side View
    // Given a binary tree, imagine yourself standing on the right side of it, return the values of the nodes you can see ordered from top to bottom.
    //
    // For example:
    // Given the following binary tree,
    //     1            <---
    //   /   \
    //  2     3         <---
    //   \     \
    //    5     4       <---
    // You should return [1, 3, 4].
    vector<int> rightSideView(TreeNode *root) {
        vector<int> ret;
        if (root == nullptr) {
            return ret;
        }
        queue<TreeNode *> nodeQueue;
        nodeQueue.push(root);
        nodeQueue.push(nullptr);
        while (!nodeQueue.empty()) {
            TreeNode *ptr = nodeQueue.front();
            ret.push_back(ptr->val);
            while (ptr) {
                if (ptr->right != nullptr) nodeQueue.push(ptr->right);
                if (ptr->left != nullptr) nodeQueue.push(ptr->left);
                nodeQueue.pop();
                ptr = nodeQueue.front();
            }
            nodeQueue.pop();
            if (!nodeQueue.empty()) nodeQueue.push(nullptr);
        }
        return ret;
    }
    // Number of Islands
    // Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.
    // Example 1:
    // 11110
    // 11010
    // 11000
    // 00000
    // Answer: 1
    //
    // Example 2:
    // 11000
    // 11000
    // 00100
    // 00011
    // Answer: 3
    int numIslands(vector<vector<char> > &grid) {
        int ret = 0;
        if (grid.size() == 0) {
            return ret;
        }
        int rowNum = grid.size(), colNum = grid[0].size();
        vector<vector<bool> > visit(rowNum, vector<bool>(colNum, false));
        for (int r = 0; r < rowNum; ++r) {
            for (int c = 0; c < colNum; ++c) {
                if (grid[r][c] == '1' && !visit[r][c]) {
                    numIslands(r, c, grid, visit);
                    ++ret;
                }
            }
        }
        return ret;
    }
    bool numIslands(const int &r, const int &c, vector<vector<char> > &grid, vector<vector<bool> > &visit) {
        bool left, right, up ,down;
        left = right = up = down = true;
        int rowNum = grid.size(), colNum = grid[0].size();
        if (r < 0 || c < 0 || r >= rowNum || c >= colNum || grid[r][c] == '0') {
            return true;
        }
        visit[r][c] = true;
        if (r > 0 && !visit[r - 1][c]) {
            visit[r - 1][c] = true;
            left = numIslands(r - 1, c, grid, visit);
        }
        if (r < rowNum - 1 && !visit[r + 1][c]) {
            visit[r + 1][c] = true;
            right = numIslands(r + 1, c, grid, visit);
        }
        if (c > 0 && !visit[r][c - 1]) {
            visit[r][c - 1] = true;
            up = numIslands(r, c - 1, grid, visit);
        }
        if (c < colNum - 1 && !visit[r][c + 1]) {
            visit[r][c + 1] = true;
            down = numIslands(r, c + 1, grid, visit);
        }
        return left && right && up && down;
    }
    // Valid Number
    // Validate if a given string is numeric.
    // Some examples:
    // "0" => true
    // " 0.1 " => true
    // "abc" => false
    // "1 a" => false
    // "2e10" => true
    // Note: It is intended for the problem statement to be ambiguous. You should gather all requirements up front before implementing one.
    bool isNumber(string s) {
        int idx = 0, len = s.size();
        // Escape the spaces
        while (idx < len && s[idx] == ' ') {
            ++idx;
        }
        while (idx < len && s[len - 1] == ' ') {
            --len;
        }
        int count = 0;
        bool preNum = false;
        for (; idx < len; ++idx) {
            while (idx < len && validNumber(s[idx])) {
                preNum = true;
                ++idx;
            }
            if (idx >= len) {
                break;
            }
            if (count == 0 && preNum && (s[idx] == '/' || s[idx] == 'e')) {
                ++count;
            }
            else if (s[idx] == '.') {
                continue;
            }
            else {
                return false;
            }
        }
        return preNum;
    }
    bool validNumber(const char &ch) {
        return '0' <= ch && ch <= '9';
    }
    // Text Justification
    // Given an array of words and a length L, format the text such that each line has exactly L characters and is fully (left and right) justified.
    // You should pack your words in a greedy approach; that is, pack as many words as you can in each line. Pad extra spaces ' ' when necessary so that each line has exactly L characters.
    // Extra spaces between words should be distributed as evenly as possible. If the number of spaces on a line do not divide evenly between words, the empty slots on the left will be assigned more spaces than the slots on the right.
    // For the last line of text, it should be left justified and no extra space is inserted between words.
    // For example,
    // words: ["This", "is", "an", "example", "of", "text", "justification."]
    // L: 16.
    // Return the formatted lines as:
    // [
    //    "This    is    an",
    //    "example  of text",
    //    "justification.  "
    // ]
    // Note: Each word is guaranteed not to exceed L in length.
    vector<string> fullJustify(vector<string> &words, int L) {
        vector<string> ret;
        int idx = 0;
        while (idx < words.size()) {
            // Find all the words in the line.
            int beg = idx, lineLen = words[idx++].size();
            while (idx < words.size() && lineLen + 1 + words[idx].size() <= L) {
                lineLen += words[idx++].size() + 1;
            }
            // Calculate the spaces between words and extra spaces.
            int spaces = 1, extra = 0;
            if (idx < words.size() && idx != beg + 1) {
                spaces += (L - lineLen) / (idx - beg - 1);
                extra = (L - lineLen) % (idx - beg - 1);
            }
            // Push the lines into the vector.
            ret.push_back(words[beg++]);
            while (extra--) {
                ret.back().append(spaces + 1, ' ');
                ret.back().append(words[beg++]);
            }
            while (beg < idx) {
                ret.back().append(spaces, ' ');
                ret.back().append(words[beg++]);
            }
            // Only have one word in a line.
            ret.back().append(L - ret.back().size(), ' ');
        }
        return ret;
    }
    // Bitwise AND of Numbers Range
    // Given a range [m, n] where 0 <= m <= n <= 2147483647, return the bitwise AND of all numbers in this range, inclusive.
    // For example, given the range [5, 7], you should return 4.
    int rangeBitwiseAnd(int m, int n) {
        int num = m, bits = 0;
        while (num) {
            ++bits;
            num >>= 1;
        }
        if (m == 0 || 1L << bits <= n) {
            return 0;
        }
        else if (m == 1 << bits - 1 || m == n) {
            return m;
        }
        else {
            // '-' manipulation has more priority than '<<' manuipulation.
            m &= (1 << bits - 1) - 1;
            n &= (1 << bits - 1) - 1;
            return 1 << bits - 1 | rangeBitwiseAnd(m, n);
        }
    }
    int rangeBitwiseAnd_improved(int m, int n) {
        int num = 0;
        while (m != n) {
            m >>= 1;
            n >>= 1;
            ++num;
        }
        return m << num;
    }
    // Happy Number
    // Write an algorithm to determine if a number is "happy".
    //
    // A happy number is a number defined by the following process: Starting with any positive integer, replace the number by the sum of the squares of its digits, and repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1. Those numbers for which this process ends in 1 are happy numbers.
    //
    // Example: 19 is a happy number
    //
    // 1^2 + 9^2 = 82
    // 8^2 + 2^2 = 68
    // 6^2 + 8^2 = 100
    // 1^2 + 0^2 + 0^2 = 1
    bool isHappy(int n) {
        if (n < 0) {
            return false;
        }
        unordered_set<int> uset;
        while (!uset.count(n)) {
            vector<int> bits = getBits(n);
            uset.insert(n);
            int ret = 0;
            for (int idx = 0; idx < bits.size(); ++idx) {
                ret += bits[idx] * bits[idx];
            }
            n = ret;
            if (ret == 1) {
                return true;
            }
        }
        return false;
    }
    vector<int> getBits(int n) {
        vector<int> ret;
        while (n) {
            ret.push_back(n % 10);
            n /= 10;
        }
        return ret;
    }
    // Remove Linked List Elements
    // Remove all elements from a linked list of integers that have value val.
    // Example
    // Given: 1 --> 2 --> 6 --> 3 --> 4 --> 5 --> 6, val = 6
    // Return: 1 --> 2 --> 3 --> 4 --> 5
    ListNode* removeElements(ListNode* head, int val) {
        ListNode *node = new ListNode(0);
        node->next = head;
        ListNode *ptr = node;
        while (ptr && ptr->next) {
            if (ptr->next->val == val) {
                ptr->next = ptr->next->next;
            }
            else {
                ptr = ptr->next;
            }
        }
        return node->next;
    }
    // Count Primes
    // Description:
    // Count the number of prime numbers less than a non-negative number, n
    int countPrimes(int n) {
        if (n <= 1) {
            return 0;
        }
        int ret = 0, root = sqrt(n);
        vector<bool> num(n + 1, true);
        num[1] = false;
        for (int i = 2; i <= root; ++i) {
            if (num[i]) {
                ++ret;
                for (int j = i * i; j < n; j += i) {
                    num[j] = false;
                }
            }
        }
        for (int i = root + 1; i < n; ++i) {
            if (num[i]) ++ret;
        }
        return ret;
    }
    // Isomorphic Strings
    // Given two strings s and t, determine if they are isomorphic.
    // Two strings are isomorphic if the characters in s can be replaced to get t.
    // All occurrences of a character must be replaced with another character while preserving the order of characters. No two characters may map to the same character but a character may map to itself.
    // For example,
    // Given "egg", "add", return true.
    // Given "foo", "bar", return false.
    // Given "paper", "title", return true.
    // Note:
    // You may assume both s and t have the same length.
    bool isIsomorphic(string s, string t) {
        unordered_map<char, char> smap, tmap;
        for (int idx = 0; idx < s.size(); ++idx) {
            if (smap.count(s[idx]) && smap[s[idx]] != t[idx]) return false;
            if (tmap.count(t[idx]) && tmap[t[idx]] != s[idx]) return false;
            smap[s[idx]] = t[idx];
            tmap[t[idx]] = s[idx];
        }
        return true;
    }
    // Reverse Linked List
    // Reverse a singly linked list.
    ListNode* reverseList(ListNode* head) {
        if (head == nullptr) {
            return nullptr;
        }
        ListNode *prev = nullptr, *mem = nullptr;
        while (head) {
            mem = head->next;
            head->next = prev;
            prev = head;
            head = mem;
        }
        return prev;
    }
    // Course Schedule
    // There are a total of n courses you have to take, labeled from 0 to n - 1.
    // Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]
    // Given the total number of courses and a list of prerequisite pairs, is it possible for you to finish all courses?
    //
    // For example:
    // 2, [[1,0]]
    // There are a total of 2 courses to take. To take course 1 you should have finished course 0. So it is possible.
    // 2, [[1,0],[0,1]]
    // There are a total of 2 courses to take. To take course 1 you should have finished course 0, and to take course 0 you should also have finished course 1. So it is impossible.
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        // Kahn
        vector<int> indegree(numCourses, 0);
        vector<vector<int> > edges(numCourses, vector<int>());
        int edgeNum = prerequisites.size();
        if (!edgeNum) {
            return true;
        }
        for (int idx = 0; idx < edgeNum; ++idx) {
            int from = prerequisites[idx][0], to = prerequisites[idx][1];
            ++indegree[to];
            edges[from].push_back(to);
        }
        queue<int> zeroDegree;
        for (int idx = 0; idx < numCourses; ++idx) {
            if (indegree[idx] == 0) {
                zeroDegree.push(idx);
            }
        }
        while (!zeroDegree.empty()) {
            int index = zeroDegree.front();
            zeroDegree.pop();
            for (int idx = 0; idx < edges[index].size(); ++idx) {
                --edgeNum;
                int node = edges[index][idx];
                if (--indegree[node] == 0) {
                    zeroDegree.push(node);
                }
            }
        }
        if (edgeNum) {
            return false;
        }
        return true;
    }
    // Course Schedule II
    // There are a total of n courses you have to take, labeled from 0 to n - 1.
    // Some courses may have prerequisites, for example to take course 0 you have to first take course 1, which is expressed as a pair: [0,1]
    // Given the total number of courses and a list of prerequisite pairs, return the ordering of courses you should take to finish all courses.
    // There may be multiple correct orders, you just need to return one of them. If it is impossible to finish all courses, return an empty array.
    // For example:
    // 2, [[1,0]]
    // There are a total of 2 courses to take. To take course 1 you should have finished course 0. So the correct course order is [0,1]
    // 4, [[1,0],[2,0],[3,1],[3,2]]
    // There are a total of 4 courses to take. To take course 3 you should have finished both courses 1 and 2. Both courses 1 and 2 should be taken after you finished course 0. So one correct course order is [0,1,2,3]. Another correct ordering is[0,2,1,3].
    vector<int> findOrder(int numCourses, vector<pair<int, int>>& prerequisites) {
        vector<int> ret, indegree(numCourses, 0);
        vector<vector<int> > edges(numCourses, vector<int>());
        int edgeNum = prerequisites.size();
        for (int idx = 0; idx < edgeNum; ++idx) {
            int to = prerequisites[idx].first, from = prerequisites[idx].second;
            ++indegree[to];
            edges[from].push_back(to);
        }
        queue<int> zeroDegree;
        for (int idx = 0; idx < numCourses; ++idx) {
            if (indegree[idx] == 0) {
                zeroDegree.push(idx);
            }
        }
        while (!zeroDegree.empty()) {
            int index = zeroDegree.front();
            zeroDegree.pop();
            ret.push_back(index);
            for (int idx = 0; idx < edges[index].size(); ++idx) {
                int node = edges[index][idx];
                --edgeNum;
                if (--indegree[node] == 0) {
                    zeroDegree.push(node);
                }
            }
        }
        if (edgeNum) {
            return vector<int>();
        }
        return ret;
    }

    // Minimum Size Subarray Sum
    // Given an array of n positive integers and a positive integer s, find the minimal length of a subarray of which the sum ≥ s. If there isn't one, return 0 instead.
    // For example, given the array [2,3,1,2,4,3] and s = 7,
    // the subarray [4,3] has the minimal length under the problem constraint.
    int minSubArrayLen(int s, vector<int>& nums) {
        int ret = INT_MAX, start = 0;
        int localSum = 0;
        for (int idx = 0; idx < nums.size(); ++idx) {
            localSum += nums[idx];
            while (start < idx && localSum - nums[start] >= s) {
                localSum -= nums[start++];
            }
            if (localSum >= s) {
                ret = min(ret, idx - start + 1);
            }
        }
        return ret == INT_MAX ? 0 : ret;
    }
    // Kth Largest Element in an Array
    // Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.
    // For example,
    // Given [3,2,1,5,6,4] and k = 2, return 5.
    // Note:
    // You may assume k is always valid, 1 ≤ k ≤ array's length.
    int findKthLargest(vector<int>& nums, int k) {
        int N = nums.size(), beg = 0, end = N - 1, pivot = -1;
        while (pivot != N - k && beg <= end) {
            pivot = part(nums, beg, end);
            if (pivot > N - k) {
                end = pivot - 1;
            }
            else if (pivot < N - k) {
                beg = pivot + 1;
            }
        }
        return nums[pivot];
    }
    int part(vector<int> &nums, const int &beg, const int &end) {
        int pivotVal = nums[end], i = beg - 1;
        for (int j = beg; j <= end; ++j) {
            if (nums[j] < pivotVal) {
                swap(nums[++i], nums[j]);
            }
        }
        swap(nums[++i], nums[end]);
        return i;
    }
    // House Robber II
    // After robbing those houses on that street, the thief has found himself a new place for his thievery so that he will not get too much attention. This time, all houses at this place are arranged in a circle. That means the first house is the neighbor of the last one. Meanwhile, the security system for these houses remain the same as for those in the previous street.
    // Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.
    int robII(vector<int>& nums) {
        int lens = nums.size();
        if (!lens) {
            return 0;
        }
        else if (lens <= 3) {
            return *max_element(nums.begin(), nums.end());
        }
        vector<int> dp1(lens, 0), dp2(lens, 0);
        dp1[0] = nums[0];
        dp1[1] = max(nums[0], nums[1]);
        dp2[0] = 0;
        dp2[1] = nums[1];
        for (int idx = 2; idx < lens; ++idx) {
            dp1[idx] = max(dp1[idx - 1], dp1[idx - 2] + nums[idx]);
            dp2[idx] = max(dp2[idx - 1], dp2[idx - 2] + nums[idx]);
        }
        dp1[lens - 1] = dp1[lens - 2];
        return max(dp1[lens - 1], dp2[lens - 1]);
    }
    // Contains Duplicate
    // Given an array of integers, find if the array contains any diplicates. Your function should return true if any value appears at least twice in the array, and it should return false if every element is distinct.
    bool containDuplicate(vector<int> &nums) {
        unordered_set<int> uset;
        for (int idx = 0; idx < nums.size(); ++idx) {
            if (uset.count(nums[idx])) {
                return true;
            }
            uset.insert(nums[idx]);
        }
        return false;
    }
    // Contains Duplicate II
    // Given an array of integers and an integer k, return true if and only if there are two distinct indices i and j in the array such that nums[i] = nums[j] and the difference between i and j is at most k.
    bool containsNearbyDuplicate(vector<int> &nums, int k) {
        unordered_map<int, int> umap;
        for (int idx = 0; idx < nums.size(); ++idx) {
            if (umap.count(nums[idx]) == 0) {
                umap[nums[idx]] = idx;
            }
            else if (idx - umap[nums[idx]] <= k) {
                return true;
            }
        }
        return false;
    }
    // Combination Sum III
    // Find all possible combinations of k numbers that add up to a number n, given that only numbers from 1 to 9 can be used and each combination should be a unique set of numbers.
    // Ensure that numbers within the set are sorted in ascending order.
    // Example 1:
    // Input: k = 3, n = 7
    // Output:
    // [[1,2,4]]
    // Example 2:
    // Input: k = 3, n = 9
    // Output:
    // [[1,2,6], [1,3,5], [2,3,4]]
    vector<vector<int>> combinationSum3(int k, int n) {
        vector<vector<int> > ret;
        vector<int> line;
        combinationSum3Helper(1, k, n, line, ret);
        return ret;
    }
    void combinationSum3Helper(const int beg, const int &k, const int &n, vector<int> &line, vector<vector<int> > &ret) {
        if (beg > 9) {
            return;
        }
        if (line.size() < k) {
            int localSum = accumulate(line.begin(), line.end(), 0) + beg;
            line.push_back(beg);
            if (line.size() == k && localSum == n) {
                ret.push_back(line);
            }
            else if (localSum < n) {
                combinationSum3Helper(beg + 1, k, n, line, ret);
            }
            line.pop_back();
        }
        combinationSum3Helper(beg + 1, k, n, line, ret);
    }
    // Ugly Number
    // Write a program to check whether a given number is an ugly number.
    // Ugly numbers are positive numbers whose prime factors only include 2, 3, 5. For example, 6, 8 are ugly while 14 is not ugly since it includes another prime factor 7.
    // Note that 1 is typically treated as an ugly number.
    bool isUgly(int num) {
        int div[] = {2, 3, 5};
        for (int idx = 0; idx < 3; ++idx) {
            while (num && num % div[idx] == 0) {
                num /= div[idx];
            }
        }
        return num == 1;
    }
    // Ugly Number II
    // Write a program to find the n-th ugly number.
    // Ugly numbers are positive numbers whose prime factors only include 2, 3, 5. For example, 1, 2, 3, 4, 5, 6, 8, 9, 10, 12 is the sequence of the first 10 ugly numbers.
    // Note that 1 is typically treated as an ugly number.
    int nthUglyNumber(int n) {
        vector<int> ret(1, 1);
        int base[] = {2, 3, 5}, idx[] = {0, 0, 0};
        for (int i = 2; i <= n; ++i) {
            int minVal = base[0] * ret[idx[0]];
            for (int j = 1; j < 3; ++j) {
                minVal = min(minVal, base[j] * ret[idx[j]]);
            }

            for (int j = 0; j < 3; ++j) {
                if (base[j] * ret[idx[j]] == minVal) {
                    ++idx[j];
                }
            }
            ret.push_back(minVal);
        }

        return ret[ret.size() - 1];
    }
    // Add Digits
    // Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.
    // For example:
    // Given num = 38, the process is like: 3 + 8 = 11, 1 + 1 = 2. Since 2 has only one digit, return it.
    int addDigits(int num) {
        int ret;
        if (num == 0) {
            ret = 0;
        }
        else {
            ret = num % 9 == 0 ? 9 : num % 9;
        }
        return ret;
    }
    // Binary Tree Paths
    // Given a binary tree, return all root-to-leaf paths.
    // For example, given the following binary tree:
    //    1
    //  /   \
    // 2     3
    //  \
    //   5
    // All root-to-leaf paths are:
    // ["1->2->5", "1->3"]
    vector<string> binaryTreePaths(TreeNode* root) {
        vector<vector<string> > paths;
        if (root) {
            binaryTreePaths(root, vector<string>(), paths);
        }
        vector<string> ret;
        for (int i = 0; i < paths.size(); ++i) {
            string path;
            for (int l = 0; l < paths[i].size(); ++l) {
                path += paths[i][l];
                if (l != paths[i].size() - 1) {
                    path += "->";
                }
            }
            ret.push_back(path);
        }
        return ret;
    }
    void binaryTreePaths(TreeNode *root, vector<string> path, vector<vector<string> > &paths) {
        path.push_back(to_string(root->val));

        if (root->left == nullptr && root->right == nullptr && path.size() != 0) {
            paths.push_back(path);
            return;
        }
        if (root->left != nullptr) {
            binaryTreePaths(root->left, path, paths);
        }
        if (root->right != nullptr) {
            binaryTreePaths(root->right, path, paths);
        }
    }
    // Single Number III
    // Given an array of numbers nums, in which exactly two elements appear only once and all the other elements appear exactly twice. Find the two elements that appear only once.
    // For example:
    // Given nums = [1, 2, 1, 3, 2, 5], return [3, 5].
    // Note:
    // The order of the result is not important. So in the above example, [5, 3] is also correct.
    // Your algorithm should run in linear runtime complexity. Could you implement it using only constant space complexity?
    vector<int> singleNumberIII(vector<int>& nums) {
        int xorVal = 0;
        for (int idx = 0; idx < nums.size(); ++idx) {
            xorVal ^= nums[idx];
        }

        vector<int> ret;
        if (xorVal) {
            int mask = 1;
            while (!(xorVal & mask)) {
                mask <<= 1;
            }
            vector<int> class1, class2;
            for (int idx = 0; idx < nums.size(); ++idx) {
                if (nums[idx] & mask) {
                    class1.push_back(nums[idx]);
                }
                else {
                    class2.push_back(nums[idx]);
                }
            }
            int unqNum = 0;
            for (int idx = 0; idx < class1.size(); ++idx) {
                unqNum ^= class1[idx];
            }
            ret.push_back(unqNum);
            unqNum = 0;
            for (int idx = 0; idx < class2.size(); ++idx) {
                unqNum ^= class2[idx];
            }
            ret.push_back(unqNum);
        }
        return ret;
    }
    // Valid Anagram
    // Given two strings s and t, write a function to determine if t is an anagram of s.
    // For example,
    // s = "anagram", t = "nagaram", return true.
    // s = "rat", t = "car", return false.
    bool isAnagram(string s, string t) {
        int lens = s.size(), lent = t.size();
        if (lens != lent) {
            return false;
        }

        vector<int> hashmap(26, 0);
        for (int idx = 0; idx < lens; ++idx) {
            ++hashmap[s[idx] - 'a'];
        }
        for (int idx = 0; idx < lent; ++idx) {
            if (--hashmap[t[idx] - 'a'] < 0) {
                return false;
            }
        }
        return true;
    }
    // Missing Number
    // Given an array containing n distinct numbers taken from 0, 1, 2, ..., n, find the one that is missing from the array.
    // For example,
    // Given nums = [0, 1, 3] return 2.
    // Note:
    // Your algorithm should run in linear runtime complexity. Could you implement it using only constant extra space complexity?
    int missingNumber(vector<int>& nums) {
        int N = nums.size();
        int sum = N * (N + 1) / 2;
        for (int idx = 0; idx < N; ++idx) {
            sum -= nums[idx];
        }
        return sum;
    }
    // Rectangle Area
    // Find the total area covered by two rectilinear rectangles in a 2D plane.
    // Each rectangle is defined by its bottom left corner and top right corner as shown in the figure.
    // Assume that the total area is never beyond the maximum possible value of int.
    int computeArea(int A, int B, int C, int D, int E, int F, int G, int H) {
        int sqSum = (C - A) * (D - B) + (G - E) * (H - F);
        if (H > B && F < D) {
            if (G > A && E < C) {
                int coinY = min(H, D) - max(F, B);
                int coinX = min(G, C) - max(E, A);
                sqSum -= coinY * coinX;
            }
        }
        return sqSum;
    }
    // Invert Binary Tree
    // Invert a binary tree.
    //      4
    //    /   \
    //   2     7
    //  / \   / \
    // 1   3 6   9
    // to
    //      4
    //    /   \
    //   7     2
    //  / \   / \
    // 9   6 3   1
    TreeNode* invertTree(TreeNode* root) {
        if (root == nullptr) {
            return root;
        }
        if (root->left || root->right ) {
            swap(root->left, root->right);
            invertTree(root->left);
            invertTree(root->right);
        }
        return root;
    }
    TreeNode* invertTree_iteration(TreeNode* root) {
        if (root == nullptr) {
            return root;
        }
        
        stack<TreeNode *> nodeStack;
        nodeStack.push(root);
        while (!nodeStack.empty()) {
            TreeNode *topVal = nodeStack.top();
            swap(topVal->left, topVal->right);
            nodeStack.pop();
            if (topVal->left) {
                nodeStack.push(topVal->left);
            }
            if (topVal->right) {
                nodeStack.push(topVal->right);
            }
        }

        return root;
    }
    // Summary Ranges
    // Given a sorted integer array without duplicates, return the summary of its ranges.
    // For example, given [0,1,2,4,5,7], return ["0->2","4->5","7"].
    vector<string> summaryRanges(vector<int>& nums) {
        vector<string> ret;
        int N = nums.size();
        if (N == 0) {
            return ret;
        }
        int beg = nums[0], cnt = 1;
        for (int idx = 1; idx < N; ++idx) {
            if (nums[idx] != nums[idx - 1] + 1) {
                string range = to_string(beg);
                if (cnt > 1) {
                    range += "->" + to_string(nums[idx - 1]);
                }
                ret.push_back(range);
                cnt = 1;
                beg = nums[idx];
            }
            else {
                ++cnt;
            }
        }
        string range = to_string(beg);
        if (cnt > 1) {
            range += "->" + to_string(nums[N - 1]);
        }
        ret.push_back(range);

        return ret;
    }
    // Power of Two
    // Given an integer, write a function to determine if it is a power of two.
    bool isPowerOfTwo(int n) {
        if (n < 1) {
            return false;
        }
        while (n % 2 == 0) {
            n /= 2;
        }
        return n == 1 ? true : false;
    }
    // Palindrome Linked List
    // Given a singly linked list, determine if it is a palindrome.
    // Follow up:
    // Could you do it in O(n) time and O(1) space?
    bool isPalindrome(ListNode* head) {
        if (head == nullptr) {
            return true;
        }
        /* find the mid node of the list */
        ListNode *slow, *fast;
        slow = fast = head;
        while (fast && fast->next) {
            fast = fast->next->next;
            slow = slow->next;
        }
        if (fast) {
            slow = slow->next;
        }
        /* reverse the last half of the list */
        ListNode *prev, *mem;
        prev = mem = nullptr;
        while (slow) {
            mem = slow->next;
            slow->next = prev;
            prev = slow;
            slow = mem;
        }

        while (prev && prev->val == head->val) {
            prev = prev->next;
            head = head->next;
        }
        return prev == nullptr;
    }
    // Lowest Common Ancestor of a Binary Search Tree
    // Given a binary search tree (BST), find the lowest common ancestor (LCA) of two given nodes in the BST.
    // According to the definition of LCA on Wikipedia: "The lowest common ancestor is defined between two nodes v and w as the lowest node in T that has both v and w as descendants (where we allow a node to be a descendant of itself)."
    //
    //         _______6______
    //        /              \
    //     ___2__          ___8__
    //    /      \        /      \
    //    0      _4       7       9
    //          /  \
    //          3   5
    // For example, the lowest common ancestor (LCA) of nodes 2 and 8 is 6. Another example is LCA of nodes 2 and 4 is 2, since a node can be a descendant of itself according to the LCA definition.
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!covers(root, p) || !covers(root, q)) {
            return nullptr;
        }
        while (root) {
            int val1 = root->val - p->val, val2 = root->val - q->val;
            if (val1 * val2 <= 0) {
                return root;
            }
            else {
                bool isLeft = val1 > 0 ? true : false;
                root = isLeft ? root->left : root->right;
            }
        }
        
        return nullptr;
    }
    // Lowest Common Ancestor of a Binary Tree
    TreeNode* lowestCommonAncestorII(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!covers(root, p) || !covers(root, q)) {
            return nullptr;
        }
        return lowestCommonAncestorHelper(root, p, q);
    }
    bool covers(TreeNode *root, TreeNode *p) {
        if (root == nullptr) {
            return false;
        }
        if (root == p) {
            return true;
        }
        return covers(root->left, p) || covers(root->right, p);
    }
    TreeNode *lowestCommonAncestorHelper(TreeNode *root, TreeNode *p, TreeNode *q) {
        if (root == nullptr) {
            return nullptr;
        }
        if (root == p || root == q) {
            return root;
        }
        bool is_p_on_left = covers(root->left, p), is_q_on_left = covers(root->left, q);
        if (is_p_on_left != is_q_on_left) {
            return root;
        }
        TreeNode *side = is_p_on_left ? root->left : root->right;
        return lowestCommonAncestorHelper(side, p, q);
    }

    TreeNode* lowestCommonAncestorII_improved(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (root == nullptr) {
            return nullptr;
        }
        if (root == p && root == q) {
            return root;
        }
        TreeNode *x = lowestCommonAncestorII_improved(root->left, p, q);
        if (x && x != p && x != q) {
            return x;
        }
        TreeNode *y = lowestCommonAncestorII_improved(root->right, p, q);
        if (y && y != p && y != q) {
            return y;
        }

        if (x && y) {
            return root;
        }
        else if (root == p || root == q) {
            return root;
        }
        else {
            return y ? y : x;
        }
    }
    // Delete Node in a Linked List
    // Write a function to delete a node (except the tail) in a singly linked list, given only access to that node.
    // Supposed the linked list is 1 -> 2 -> 3 -> 4 and you are given the third node with value 3, the linked list should become 1 -> 2 -> 4 after calling your function.
    void deleteNode(ListNode* node) {
        node->val = node->next->val;
        node->next = node->next->next;
    }
    // Contains Duplicate III
    // Given an array of integers, find out whether there are two distinct indices i and j in the array such that the difference between nums[i] and nums[j] is at most t and the difference between i and j is at most k.
    bool containsNearbyAlmostDuplicate(vector<int>& nums, int k, int t) {
        set<int> hashmap;
        for (int i = 0; i < nums.size(); ++i) {
            if (i > k) {
                hashmap.erase(nums[i - k - 1]);
            }
            auto ptr = hashmap.lower_bound(nums[i] - t);
            if (ptr != hashmap.end() && *ptr - nums[i] <= t) {
                return true;
            }
            hashmap.insert(nums[i]);
        }
        return false;
    }
    // Maximal Square
    // Given a 2D binary matrix filled with 0's and 1's, find the largest square containing all 1's and return its area.
    // For example, given the following matrix:
    // 1 0 1 0 0
    // 1 0 1 1 1
    // 1 1 1 1 1
    // 1 0 0 1 0
    // Return 4.
    int maximalSquare(vector<vector<char>>& matrix) {
        int row = matrix.size();
        if (row == 0) {
            return 0;
        }
        int col = matrix[0].size(), maxSq = 0;
        vector<vector<int> > dp(row, vector<int>(col, 0));

        for (int c = 0; c < col; ++c) {
            if (matrix[0][c] == '1') {
                dp[0][c] = 1;
                maxSq = max(maxSq, dp[0][c]);
            }
        }
        for (int r = 0; r < row; ++r) {
            if (matrix[r][0] == '1') {
                dp[r][0] = 1;
                maxSq = max(maxSq, dp[r][0]);
            }
        }
        for (int r = 1; r < row; ++r) {
            for (int c = 1; c < col; ++c) {
                if (matrix[r][c] == '0') {
                    dp[r][c] = 0;
                }
                else {
                    dp[r][c] = 1;
                    if (matrix[r - 1][c - 1] == '1' && matrix[r][c - 1] == '1' && matrix[r - 1][c] == '1') {
                        dp[r][c] += min(dp[r - 1][c - 1], min(dp[r - 1][c], dp[r][c - 1]));
                    }
                    maxSq = max(maxSq, dp[r][c]);
                }
            }
        }
        return maxSq * maxSq;
    }
    // Count Complete Tree Nodes
    // Given a complete binary tree, count the number of nodes.
    // Definition of a complete binary tree from Wikipedia:
    // In a complete binary tree every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2^h nodes inclusive at the last level h.
    int countNodes(TreeNode* root) {
        int depth = 0;
        TreeNode *ptr = root;
        while (ptr) {
            ptr = ptr->left;
            ++depth;
        }
        if (depth <= 1) {
            return depth;
        }

        int d = 1;
        ptr = root;
        int lastNum = 0;
        while (d <= depth - 1) {
            TreeNode *inLeft = inorderLeft(ptr, d, depth - 1);
            if (inLeft->left && inLeft->right) {
                ptr = ptr->right;
                ++d;
                if (d == depth) {
                    lastNum += 2;
                }
                else {
                    lastNum += 1 << (depth - d);
                }
            }
            else if (!inLeft->left && !inLeft->right) {
                ptr = ptr->left;
                ++d;
            }
            else {
                ++lastNum;
                break;
            }
        }
        return (1 << (depth - 1)) + lastNum - 1;
    }
    TreeNode *inorderLeft(TreeNode *root, int cur, const int &dst) {
        if (cur < dst) {
            root = root->left;
            ++cur;
            while (cur != dst) {
                root = root->right;
                ++cur;
            }
        }
        return root;
    }
    // Basic Calculator && Basic Calculator II
    // Implement a basic calculator to evaluate a simple expression string.
    // The expression string may contain open ( and closing parentheses ), the plus + or minus sign -, non-negative integers and empty spaces .
    // You may assume that the given expression is always valid.
    int calculate(string s) {
        vector<string> RPN = convertToRPN(s);
        Vector::printVector(RPN, 0, RPN.size());
        cout << endl;
        return evalRPN(RPN);
    }
    vector<string> convertToRPN(const string &s) {
        vector<string> RPN;
        stack<char> opSt;
        int lens = s.size();
        int idx = 0;
        while (idx < lens) {
            // omit the spaces
            while (idx < lens && s[idx] == ' ') {
                ++idx;
            }
            // parse number or operation
            if (idx >= lens) {
                break;
            }
            else if (s[idx] >= '0' && s[idx] <= '9') {
                int numStart = idx;
                while (idx < lens && s[idx] >= '0' && s[idx] <= '9') {
                    ++idx;
                }
                RPN.push_back(s.substr(numStart, idx - numStart));
            }
            else {
                char topOp;
                if (s[idx] == '(') {
                    opSt.push(s[idx]);
                }
                else if (s[idx] == ')') {
                    while ((topOp = opSt.top()) != '(') {
                        opSt.pop();
                        RPN.push_back(ch2str(topOp));
                    }
                    opSt.pop();
                }
                else if (s[idx] == '*' || s[idx] == '/') {
                    while (!opSt.empty()) {
                        topOp = opSt.top();
                        if (topOp == '*' || topOp == '/') {
                            opSt.pop();
                            RPN.push_back(ch2str(topOp));
                        }
                        else {
                            break;
                        }
                    }
                    opSt.push(s[idx]);
                }
                else {
                    while (!opSt.empty() && (topOp = opSt.top()) != '(') {
                        opSt.pop();
                        RPN.push_back(ch2str(topOp));
                    }
                    opSt.push(s[idx]);
                }
                ++idx;
            }
        }
        while (!opSt.empty()) {
            RPN.push_back(ch2str(opSt.top()));
            opSt.pop();
        }
        return RPN;
    }
    string ch2str(const char &ch) {
        char tmp[1];
        tmp[0] = ch;
        return string(tmp, 1);
    }
    // Majority Element II
    // Given an integer array of size n, find all elements that appear more than n/3 times. The algorithm should run in linear time and in O(1) space.
    vector<int> majorityElement(vector<int>& nums) {
    }
};
#endif
