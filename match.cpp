#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;

// 定义对象结构体，用于表示物体的属性
struct obj {
    int x, y; // 位置
    int width, height; // 尺寸
    vector<float> outlook; // 外观特征向量

    // 构造函数初始化外观特征向量的大小
    obj(int w, int h) : width(w), height(h), outlook(max(w, h), 0.0f) {}
};

// 计算特征向量
vector<float> cal_outlook(vector<vector<int>> &image, obj a) {
    if (a.width > a.height) {
        vector<float> outlook(a.width, 0.0f);
        for (int i = 0; i < a.width; i++) {
            for (int j = 0; j < a.height; j++) {
                outlook[i] += image[a.x - a.width / 2 + i][a.y - a.height / 2 + j];
            }
            outlook[i] /= a.height;
        }
        return outlook;
    } else {
        vector<float> outlook(a.height, 0.0f);
        for (int j = 0; j < a.height; j++) {
            for (int i = 0; i < a.width; i++) {
                outlook[j] += image[a.x - a.width / 2 + i][a.y - a.height / 2 + j];
            }
            outlook[j] /= a.width;
        }
        return outlook;
    }
}

// 重设向量大小
vector<float> resize(const vector<float> &a, int n) {
    int m = a.size();
    vector<float> ans(n, 0.0f);
    for (int i = 0; i < n; i++) {
        int k = i * m / n;
        float b = 1.0f * i * m / n - k;
        ans[i] = a[k] * (1 - b) + (k + 1 < m ? a[k + 1] * b : 0);
    }
    return ans;
}

// 计算两个向量的和
float sum(const vector<float> &v) {
    float s = 0.0f;
    for (float i : v) {
        s += i * i;
    }
    return s;
}

// 计算两个物体的相似度
float match(const obj &a, const obj &b) {
    float sim_size = static_cast<float>(a.width * b.height) / (a.height * b.width); // 尺寸相似度
    if (sim_size < 0.5f || sim_size > 2.0f) return 0.0f;

    float sim_dist = abs(a.x - b.x) + abs(a.y - b.y); // 距离相似度
    if (sim_dist > (a.width + a.height) && sim_dist > (b.width + b.height)) return 0.0f;

    float sim_look = 0.0f; // 外观相似度
    int n = max(a.outlook.size(), b.outlook.size());
    vector<float> a_outlook = a.outlook.size() < n ? resize(a.outlook, n) : a.outlook;
    vector<float> b_outlook = b.outlook.size() < n ? resize(b.outlook, n) : b.outlook;

    float da = sqrt(sum(a_outlook));
    float db = sqrt(sum(b_outlook));
    for (int i = 0; i < n; i++) {
        sim_look += a_outlook[i] * b_outlook[i];
    }
    return sim_look / (da * db); // 返回归一化的点积
}
