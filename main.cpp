#include "Solution.h"

void prtTimeCost(const clock_t stime, const clock_t ftime) {
    cout << "The time cost is : " << 1000.0 * (ftime - stime) / CLOCKS_PER_SEC << " ms.";
}


int main(){
    // close the sync.
    ios::sync_with_stdio(false); 
    clock_t stime, ftime;
    Solution s;

    /**
     * Get and print the array
     */
    /*
    cout << "Input the array size:" << endl;
    int size;
    cin >> size;
    cout << "Input the array:" << endl;
    int *arr = new int[size];
    Array::crtArr(arr, 0, size);
    */

    /**
     * Get and print the vector
     */
    cout << "Input the vector size" << endl;
    int size;
    cin >> size;
    cout << "Input the vector:" << endl;
    vector<string> vec;
    Vector::crtVec(vec, size);

    /**
     * Create and traversal the tree
     */
    TreeNode *root = Tree::crtTree(vec);
    cout << "The inorderTraversal sequence is:" << endl;
    cout << s.postorderTraversal(root) << endl;
    cout << s.postorderTraversal_another(root) << endl;

    /*
    cout << "-------------------Time Cost-----------------------" << endl; 
    */
    stime = clock();

    cout << s.zigzagLevelOrder(root) << endl;



    ftime = clock();
    prtTimeCost(stime, ftime);
    cout << endl;

    return 0;
}
