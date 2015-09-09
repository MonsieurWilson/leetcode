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
    cout << "Input the array size:" << endl;
    int size;
    cin >> size;
    cout << "Input the array:" << endl;
    int *arr = new int[size];
    Array::crtArr(arr, 0, size);

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

    cout << "The single number in the array is: " << s.singleNumber_hash(arr, size) << endl;


    // Time
    cout << "-------------------Time Cost-----------------------" << endl; 
    ftime = clock();
    cout << "The time cost is : " << 1000.0 * (ftime - stime) / CLOCKS_PER_SEC << " ms." << endl;
    return 0;
}
