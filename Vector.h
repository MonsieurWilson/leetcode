#ifndef VECTOR
#define VECTOR

/*
 * Method:
 * crtVec(vector<T> &vec, int size)
 * prtVec(const vector<T> &vec, const int &beg, const int &end)
 * crtMat(vector<vector<T> > &vec);
 * prtMat(const vector<vector<T> > &vec);
 */
class Vector{
public:
    template <typename T>
    static void crtVec(vector<T> &vec, int size){
        for (int idx = 0; idx != size; ++idx){
            T ele;
            cin >> ele;
            vec.push_back(ele);
        }
    }
    template <typename T>
    static void prtVec(const vector<T> &vec, const int &beg, const int &end){
        cout << "{";
        for (int idx = beg; idx != end; ++idx){
            cout << vec[idx];
            if (idx != end - 1){
                cout << ",";
            }
        }
        cout << "}";
    }
    template <typename T>
    static void prtVec(const vector<T> &vec){
        prtVec(vec, 0, vec.size());
    }
};

template <typename T>
ostream &operator << (ostream &o, const vector<T> &vec) {
    Vector::prtVec(vec);
    return o;
}

class Matrix {
public:
    template <typename T>
    static void crtMat(vector<vector<T> > &vec){
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
    static void prtMat(const vector<vector<T> > &vec){
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
