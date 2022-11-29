#include <iostream>
#include <vector>
#include <algorithm> 
#include <queue>
#include <functional>

#include "../src/Vec3.h"

struct kd_tree_node{
    int ind;

    float x;
    float y;
    float z;

    int axis;

    int left;
    int right;
};

std::vector<kd_tree_node> make_kd_tree(std::vector<Vec3> dots);
void * send_kd_tree(std::vector<kd_tree_node> kd_tree);

int cudaMain();