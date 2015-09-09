using namespace std;

#include "Solution.h"

#define LL long long

int main(){
    // close the sync.
    ios::sync_with_stdio(false); 
    clock_t stime, ftime;
    stime = clock();
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
    Array::crtArr(array, 0, size);
    cout << "The array is:" << endl;
    // Array::prtArr(array, 0, size);
    cout << endl;
    */

    /**
     * Get and print the vector
     */
    /*
    cout << "Input the vector size" << endl;
    int size;
    cin >> size;
    cout << "Input the vector:" << endl;
    vector<int> vec;
    Vector::crtVec(vec, size);
    */

    /**
     * Create and traversal the tree
     */
    /*
    TreeNode *root = Tree::createTree(vec);
    vector<int> traversal = s.inorderTraversal(root);
    cout << "The inorderTraversal sequence is:" << endl;
    cout << traversal << endl;
    */

    vector<vector<int> > mat;
    Matrix::crtMat(mat);
    int target;
    cout << "Input the target:" << endl;
    cin >> target;
    bool ret = s.searchMatrixII(mat, target);
    cout << "The result is :" << endl;
    cout << (ret ? "True" : "False") << endl;


    // Time
    cout << "-------------------Time Cost-----------------------" << endl; 
    ftime = clock();
    cout << "The time cost is : " << 1000.0 * (ftime - stime) / CLOCKS_PER_SEC << " ms." << endl;
    return 0;
}
