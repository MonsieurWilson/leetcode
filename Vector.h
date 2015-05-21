#ifndef VECTOR
#define VECTOR

#include <vector>
/*
 * Method:
 * getVector(vector<T> &vec, int size)
 * printVector(const vector<T> &vec, const int &beg, const int &end)
 * getVector2D(vector<vector<T> > &vec);
 * printVector2D(const vector<vector<T> > &vec);
 */
class Vector{
public:
    template <typename T>
    static void createVector(vector<T> &vec, int size){
        for (int idx = 0; idx != size; ++idx){
            T ele;
            cin >> ele;
            vec.push_back(ele);
        }
    }
    template <typename T>
    static void printVector(const vector<T> &vec, const int &beg, const int &end){
        cout << "{";
        for (int idx = beg; idx != end; ++idx){
            cout << vec[idx];
            if (idx != end - 1){
                cout << ",";
            }
        }
        cout << "}";
    }
    // 2D vector
    template <typename T>
    static void createMatrix(vector<vector<T> > &vec){
        cout << "Input the Matrix's row number:" << endl;
        int row;
        cin >> row;
        for (int r = 0; r != row; ++r){
            cout << "Input the " << r << " line's col number:" << endl;
            int col;
            cin >> col;
            cout << "Input the " << r << " line's elements:" << endl;
            vector<T> line;
            for (int c = 0; c != col; ++c){
                T ele;
                cin >> ele;
                line.push_back(ele);
            }
            vec.push_back(line);
        }
    }
    template <typename T>
    static void printMatrix(const vector<vector<T> > &vec){
        int row = vec.size();
        cout << "[" << endl;
        for (int r = 0; r != row; ++r){
            int col = vec[r].size();
            cout << "    ";
            cout << "[";
            for (int c = 0; c != col; ++c){
                cout << vec[r][c];
                if (c != col - 1){
                    cout << ",";
                }
            }
            cout << "]";
            if (r != row - 1){
                cout << "," << endl;
            }
        }
        cout << endl; 
        cout << "]";
    }
};
#endif
