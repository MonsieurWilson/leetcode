#ifndef ARRAY
#define ARRAY

/*
 * Method:
 * crtArr(T *arr, const int &beg, const int &end)
 * prtArr(T *arr, const int &beg, const int &end)
 */
class Array{
public:
    template <typename T>
    static void crtArr(T *arr, const int &beg, const int &end){
        for (int idx = beg;idx != end;++idx){
            cin >> array[idx];
        }
    }
    template <typename T>
    static void prtArr(const T *arr, const int &beg, const int &end){
        cout << "{";
        for (int idx = beg;idx != end;++idx){
            cout << array[idx];
            if (idx != end - 1) {
                cout << ",";
            }
        }
        cout << "}";
    }
};
#endif
