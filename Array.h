#ifndef ARRAY
#define ARRAY

/*
 * Method:
 * getArray(T *array, const int &beg, const int &end)
 * printArray(T *array, const int &beg, const int &end)
 */
class Array{
public:
    template <typename T>
    static void createArray(T *array, const int &beg, const int &end){
        for (int idx = beg;idx != end;++idx){
            cin >> array[idx];
        }
    }
    template <typename T>
    static void printArray(const T *array, const int &beg, const int &end){
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
