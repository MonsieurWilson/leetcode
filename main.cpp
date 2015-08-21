#include <iostream>
#include <ctime>
#include <limits>
#include <cstring>

#include "Solution.h"

#define LL long long



/*
vector<string> subsets(const string &s) {
    vector<string> ret(1, "");
    int N = s.size();
    for (int idx = 0; idx < N; ++idx) {
        int M = ret.size();
        for (int j = 0; j < M; ++j) {
            ret.push_back(ret[j]);
            ret.back() += s[idx];
        }
    }
    return ret;
}
*/

bool checkBST(TreeNode *root) {
    int minVal = INT_MIN, maxVal = INT_MAX;
    return checkBST(root, minVal, maxVal);
}

bool checkBST(TreeNode *root, cosnt int &minVal, const int &maxVal) {
    if (root == nullptr) {
        return true;
    }
    if (root->val < minVal || root->val > maxVal) {
        return false;
    }
    if (!checkBST(root->left, minVal, root->val) || !checkBST(root->right, root->val, maxVal)) {
        return false;
    }
    return true;
}


int main(){
    // close the sync.
    ios::sync_with_stdio(false); 
    clock_t startTime, finishTime;
    startTime = clock();
    Solution s;

    /**
     * Get and print the array
     */
    /*
    cout << "Input the array size:" << endl;
    int size;
    cin >> size;
    cout << "Input the array:" << endl;
    int *array = new int[size];
    Array::getArray(array, 0, size);
    cout << "The array is:" << endl;
    // Array::printArray(array, 0, size);
    cout << endl;
    */

    /**
     * Get and print the vector
     */
    cout << "Input the vector size" << endl;
    int size;
    cin >> size;
    cout << "Input the vector:" << endl;
    vector<string> vec;
    Vector::createVector(vec, size);
    cout << "The vector is:" << endl;
    Vector::printVector(vec, 0, size);
    cout << endl;

    /**
     * Create and traversal the tree
     */
    TreeNode *root = Tree::createTree(vec);
    vector<int> traversal = s.preorderTraversal(root);
    cout << "The preorderTraversal sequence is:" << endl;
    Vector::printVector(traversal, 0, traversal.size());
    cout << endl;
    traversal = s.inorderTraversal(root);
    cout << "The inorderTraversal sequence is:" << endl;
    Vector::printVector(traversal, 0, traversal.size());
    cout << endl;
    traversal = s.postorderTraversal(root);
    cout << "The postorderTraversal sequence is:" << endl;
    Vector::printVector(traversal, 0, traversal.size());
    cout << endl;




    // Time
    cout << "-------------------Time Cost-----------------------" << endl; 
    finishTime = clock();
    cout << "The time cost is : " << 1000.0 * (finishTime - startTime) / CLOCKS_PER_SEC << " ms." << endl;
    return 0;
}