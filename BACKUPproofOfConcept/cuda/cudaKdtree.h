#ifndef CUDAKDTREE_H
#define CUDAKDTREE_H

#include <iostream>
#include <vector>
#include <algorithm> 
#include <queue>
#include <functional>

#include "../src/Vec3.h"



/*
    cudaMalloc(void **devPtr, size_t count);
    cudaFree(void *devPtr);

    cudaMemcpy(void *dst, void *src, size_t count, cudaMemcpyKind kind)

    kind -> cudaMemcpyHostToDevice
    kind -> cudaMemcpyDeviceToHost
*/

struct kd_tree_node{
    int ind;

    float x;
    float y;
    float z;

    int axis;

    int left;
    int right;
    kd_tree_node(int a, float b,float c,float d, int e, int f, int g){
        ind = a;
        x = b;
        y = c;
        z = d;

        axis = e;
        left = f;
        
        right = g;
    }    
};

std::vector<kd_tree_node> make_kd_tree(std::vector<Vec3> dots);
kd_tree_node* send_kd_tree(std::vector<kd_tree_node> kd_tree);

void printKnn(kd_tree_node * kd_tree, int nb, float x, float y, float z);
void getKnn(kd_tree_node * kd_tree, int nb, Vec3 point, int * ind, float * SqDist);

void testCuda();

#endif