using namespace std;

#include "Solution.h"

#define LL long long

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
    vector<int> vec;
    Vector::crtVec(vec, size);

    /**
     * Create and traversal the tree
     */
    /*
    TreeNode *root = Tree::createTree(vec);
    vector<int> traversal = s.inorderTraversal(root);
    cout << "The inorderTraversal sequence is:" << endl;
    cout << traversal << endl;
    */

    vector<int> ret = s.productExceptSelf(vec);
    cout << "The result is :" << endl;
    cout << ret << endl;


    // Time
    cout << "-------------------Time Cost-----------------------" << endl; 
    finishTime = clock();
    cout << "The time cost is : " << 1000.0 * (finishTime - startTime) / CLOCKS_PER_SEC << " ms." << endl;
    return 0;
}
