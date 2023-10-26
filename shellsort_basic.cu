#include <iostream>
#include <vector>

using namespace std;

void shellSort(vector<int> &arr)
{
    int n = arr.size();
    int gap = n / 2;

    while (gap > 0)
    {
        for (int i = gap; i < n; ++i)
        {
            int temp = arr[i];
            int j = i;

            while (j >= gap && arr[j - gap] > temp)
            {
                arr[j] = arr[j - gap];
                j -= gap;
            }

            arr[j] = temp;
        }

        gap /= 2;
    }
}

int main()
{
    vector<int> testArray = {64, 34, 25, 12, 22, 11, 90};
    shellSort(testArray);

    cout << "Sorted Array: ";
    for (int num : testArray)
    {
        cout << num << " ";
    }
    cout << endl;

    return 0;
}
