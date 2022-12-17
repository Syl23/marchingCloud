#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <queue>
#include <algorithm>

#include <random>
#include <memory>
#include <functional>
#include <cuda_runtime.h>

#include "../src/Vec3.h"

extern "C" struct cVec3 {
   float mVals[3];
   

   __device__ cVec3( float x , float y , float z ) {
      mVals[0] = x; mVals[1] = y; mVals[2] = z;
   }

   __device__ cVec3() {
      mVals[0] = 0.0;
      mVals[1] = 0.0;
      mVals[2] = 0.0;
   }

   float __device__ operator [] (unsigned int c) { return mVals[c]; }
   float __device__ operator [] (unsigned int c) const { return mVals[c]; }


   float __device__ squareLength() const {
      return mVals[0]*mVals[0] + mVals[1]*mVals[1] + mVals[2]*mVals[2];
   }

   float __device__ length() const { return sqrt( squareLength() ); }

   void __device__ operator += (cVec3 const & other) {
      mVals[0] += other[0];
      mVals[1] += other[1];
      mVals[2] += other[2];
   }

   void __device__ normalize() { float L = length(); mVals[0] /= L; mVals[1] /= L; mVals[2] /= L; }

   friend std::ostream & operator<<(std::ostream &out, const cVec3 &vec)
    {
        out << "(";
        for (int i = 0; i < 3; i++) {
        out << vec.mVals[i];
        if (i < 2) {
        out << ", ";
        }
        }
        out << ")";
        return out;
    }

   

};

float __device__  dot( cVec3 const & a , cVec3 const & b ) {
   return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

cVec3 __device__ operator + (cVec3 const & a , cVec3 const & b) {
   return cVec3(a[0]+b[0] , a[1]+b[1] , a[2]+b[2]);
}

cVec3 __device__ operator - (cVec3 const & a , cVec3 const & b) {
   return cVec3(a[0]-b[0] , a[1]-b[1] , a[2]-b[2]);
}

cVec3 __device__ operator * (float a , cVec3 const & b) {
   return cVec3(a*b[0] , a*b[1] , a*b[2]);
}
cVec3 __device__ operator * (cVec3 const & b, float a ) {
   return cVec3(a*b[0] , a*b[1] , a*b[2]);
}
cVec3 __device__ operator * (cVec3 const & a , cVec3 const & b) {
   return cVec3(a[0]*b[0] , a[1]*b[1] , a[2]*b[2]);
}

cVec3 __device__ project(cVec3 point,cVec3 normalePlan,cVec3 pointPlan){
    return(point - dot(point - pointPlan, normalePlan)*normalePlan);
}

cVec3 __device__ operator / (cVec3 const &  a , float b) {
   return cVec3(a[0]/b , a[1]/b , a[2]/b);
}

/*


   cVec3(){
   }

   cVec3( float x , float y , float z ) {
      mVals[0] = x; mVals[1] = y; mVals[2] = z;
   }

   float __device__ operator [] (unsigned int c) { return mVals[c]; }

   float __device__ operator [] (unsigned int c) const { return mVals[c]; }

   void __device__ operator = (cVec3 const & other) {
      mVals[0] = other[0] ; mVals[1] = other[1]; mVals[2] = other[2];
   }
   
   void __device__ normalize() { float L = length(); mVals[0] /= L; mVals[1] /= L; mVals[2] /= L; }


   static float __device__ dot( cVec3 const & a , cVec3 const & b ) {
      return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
   }

   static cVec3 __device__ cross( cVec3 const & a , cVec3 const & b ) {
      return cVec3(
         a[1]*b[2] - a[2]*b[1] ,
         a[2]*b[0] - a[0]*b[2] ,
         a[0]*b[1] - a[1]*b[0]
         );
   }

   void __device__ operator += (cVec3 const & other) {
      mVals[0] += other[0];
      mVals[1] += other[1];
      mVals[2] += other[2];
   }

   void __device__ operator -= (cVec3 const & other) {
      mVals[0] -= other[0];
      mVals[1] -= other[1];
      mVals[2] -= other[2];
   }

   void __device__ operator *= (float s) {
      mVals[0] *= s;
      mVals[1] *= s;
      mVals[2] *= s;
   }

   void __device__ operator /= (float s) {
      mVals[0] /= s;
      mVals[1] /= s;
      mVals[2] /= s;
   }
};

static inline cVec3 __device__ operator + (cVec3 const & a , cVec3 const & b) {
   return cVec3(a[0]+b[0] , a[1]+b[1] , a[2]+b[2]);
}
static inline cVec3 __device__ operator - (cVec3 const & a , cVec3 const & b) {
   return cVec3(a[0]-b[0] , a[1]-b[1] , a[2]-b[2]);
}
static inline cVec3 __device__ operator * (float a , cVec3 const & b) {
   return cVec3(a*b[0] , a*b[1] , a*b[2]);
}
static inline cVec3 __device__ operator * (cVec3 const & a , cVec3 const & b) {
   return cVec3(a[0]*b[0] , a[1]*b[1] , a[2]*b[2]);
}
static inline cVec3 __device__ operator / (cVec3 const &  a , float b) {
   return cVec3(a[0]/b , a[1]/b , a[2]/b);
}
static inline std::ostream & __device__ operator << (std::ostream & s , cVec3 const & p) {
    s << p[0] << " " << p[1] << " " << p[2];
    return s;
}
static inline std::istream & __device__ operator >> (std::istream & s , cVec3 & p) {
    s >> p[0] >> p[1] >> p[2];
    return s;
}
*/


extern "C" struct kd_tree_node{
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

/*
    cudaMalloc(void **devPtr, size_t count);
    cudaFree(void *devPtr);

    cudaMemcpy(void *dst, void *src, size_t count, cudaMemcpyKind kind)

    kind -> cudaMemcpyHostToDevice
    kind -> cudaMemcpyDeviceToHost
*/

std::ostream& operator<<(std::ostream& os, const kd_tree_node& kdn){
    os <<"node : \n"<< "\tpoint index : "<<kdn.ind<<"\n\tpoint : ("<<kdn.x<<" "<<kdn.y<<" "<<kdn.z<<")\n\taxis : "<<kdn.axis<<"\n\tleft : "<<kdn.left<<" right : "<<kdn.right;
    return os;
}

bool compareIndVec3 (int axis, std::pair<int,Vec3> i,std::pair<int,Vec3> j) {
    return (i.second[axis]<j.second[axis]); 
}

extern "C" std::vector<kd_tree_node> make_kd_tree(std::vector<Vec3> dots){
    auto res = std::vector<kd_tree_node>(); // le kd-tree
    auto indVec = std::vector<std::pair<int,Vec3>>(dots.size()); //les points à mettre dans le kd tree, on tri la partie qu'on veut au moment qu'on veut

    for(int i = 0 ; i < dots.size() ; i ++){
        indVec[i] = std::make_pair(i,dots[i]);
    }  

    auto fileTraitement = std::queue<std::pair<int, std::pair<int,int>>>() ; // Axe et Indice de début et de fin du tableau indVec
    fileTraitement.push(std::make_pair(0,std::make_pair(0,indVec.size())));

    int maxSize = 0;

    while(!fileTraitement.empty()){
        auto plage = fileTraitement.front().second;
        auto axis = fileTraitement.front().first;

        fileTraitement.pop();

        std::sort(indVec.begin()+plage.first, indVec.begin()+plage.second, std::bind(compareIndVec3,axis,std::placeholders::_1,std::placeholders::_2));

        int med = (plage.first + plage.second)/2;

        int left;
        int right;

        if(plage.second - plage.first > 1){
            left = ++maxSize;
            right = ++maxSize;
        }else if(plage.second - plage.first == 1){
            left = -1;
            right = ++maxSize;
        }else{
            left = -1;
            right = -1;
        }

        //std::cout<<"med : "<<med<<std::endl;

        res.push_back(kd_tree_node(
            indVec[med].first,

            indVec[med].second[0],
            indVec[med].second[1],
            indVec[med].second[2],

            axis,

            left,
            right
        ));

        if(left != -1 && right != -1){
            fileTraitement.push(std::make_pair((axis+1) % 3, std::make_pair(plage.first,med)));
            fileTraitement.push(std::make_pair((axis+1) % 3,std::make_pair(med,plage.second)));
        }else if(right != -1 && left == -1){
            fileTraitement.push(std::make_pair((axis+1) % 3,std::make_pair(med,med)));
        }
    }
    return res;
}

extern "C" kd_tree_node* send_kd_tree(std::vector<kd_tree_node> kd_tree){
    kd_tree_node* res = NULL;

    // for(int i = 0 ; i < 50 ; i++){
    //     std::cout<<i<<" ";
    //     std::cout<<kd_tree[i]<<std::endl;
    // }

    cudaMalloc(&res, kd_tree.size()*sizeof(kd_tree_node));
    cudaMemcpy(res, (void *)kd_tree.data(), kd_tree.size()*sizeof(kd_tree_node), cudaMemcpyHostToDevice);

    return res;
}

    // __device__ pointQueue(int nb){
    //     size = nb;
    //     nbInQueue = 0;

    //     ind = (int *) malloc(nb * sizeof(int));
    //     dist = (float *) malloc(nb * sizeof(float));

    //     for(int i = 0 ; i < nb ; i ++){
    //         dist[i] = -1.0;
    //         ind[i] = 0;
    //     }
    // }
    // __device__ pointQueue(){
    //     int nb = 10;
    //     size = nb;
    //     nbInQueue = 0;

    //     ind = (int *) malloc(nb * sizeof(int));
    //     dist = (float *) malloc(nb * sizeof(float));

    //     for(int i = 0 ; i < nb ; i ++){
    //         dist[i] = -1.0;
    //         ind[i] = 0;
    //     }
    // }

struct pointQueue{
    int size;
    int nbInQueue;

    int * ind;
    float * dist;
};

__device__ void initPointQueue(pointQueue * ptc, int nb){
    ptc->size = nb;
    ptc->nbInQueue = 0;

    ptc->ind  = (int *) malloc(nb * sizeof(int));
    ptc->dist = (float * )malloc(nb * sizeof(float));

    for(int i = 0 ; i < nb ; i ++){
        ptc->dist[i] = -1.0;
        ptc->ind[i] = 0;
    }
}

__device__ void freePointQueue(pointQueue * ptc){
    free(ptc->ind);
    free(ptc->dist);
}


float __device__ getThresholdDist(pointQueue* queue){
    if(queue->nbInQueue == queue->size){
        return queue->dist[queue->size -1];
    }
    return -1;
}


// c'est bien moi qui ai écrit cette fonction, elle a seulement été commentée par chatGPT

// Ajoute un élément (un indice et une distance) à une file d'attente structurée en utilisant un algorithme de tri par insertion.
//
// queue: pointeur vers la file d'attente à laquelle ajouter l'élément
// index: indice de l'élément à ajouter à la file d'attente
// distance: distance de l'élément à ajouter à la file d'attente
void __device__ addToPointQueue(pointQueue* queue, int index, float distance) {
    // variables temporaires pour stocker l'indice et la distance passés en paramètre
    int currentIndex = index;
    float currentDistance = distance;

    // itère à travers tous les éléments de la file d'attente
    for (int i = 0; i < queue->nbInQueue; i++) {
        // si la distance de l'élément de la file d'attente est négative ou si la distance passée en paramètre est inférieure à celle de l'élément de la file d'attente
        if (queue->dist[i] < 0 || currentDistance < queue->dist[i]) {
            // échange l'élément de la file d'attente avec l'indice et la distance temporaires
            int tempIndex = queue->ind[i];
            float tempDistance = queue->dist[i];

            queue->ind[i] = currentIndex;
            queue->dist[i] = currentDistance;

            currentIndex = tempIndex;
            currentDistance = tempDistance;
        }
    }

    // si la file d'attente n'est pas pleine, ajoute l'indice et la distance temporaires à la fin de la file d'attente et incrémente nbInQueue
    if (queue->nbInQueue < queue->size) {
        queue->ind[queue->nbInQueue] = currentIndex;
        queue->dist[queue->nbInQueue] = currentDistance;

        queue->nbInQueue++;
    }
}

float __device__ sq(float f){
    return f*f;
}

const int MAX_STACK_SIZE = 1000;

void __device__ fillQueue(kd_tree_node * kd_tree, pointQueue* queue, int currentInd, int currentAxis, float pointX, float pointY, float pointZ){
  int stack[MAX_STACK_SIZE];
  int stackPointer = 0;
  stack[stackPointer++] = 0;
  stack[stackPointer++] = 0;

  while (stackPointer > 0) {
    int currentAxis = stack[--stackPointer];
    int currentInd = stack[--stackPointer];

    float currentSqDist = sq(kd_tree[currentInd].x - pointX) + sq(kd_tree[currentInd].y - pointY) + sq(kd_tree[currentInd].z - pointZ);
    auto threshold = getThresholdDist(queue);

    int pointIndex = kd_tree[currentInd].ind;
    if (threshold < 0 || currentSqDist < threshold) {
      addToPointQueue(queue, pointIndex, currentSqDist);
    }

    int bestSide = 0;
    int otherSide = 0;

    if (kd_tree[currentInd].left == -1 || (currentAxis == 0 && kd_tree[currentInd].x < pointX) ||
        (currentAxis == 1 && kd_tree[currentInd].y < pointY) || (currentAxis == 2 && kd_tree[currentInd].z < pointZ)) {
      bestSide = kd_tree[currentInd].left;
      otherSide = kd_tree[currentInd].right;
    } else {
      bestSide = kd_tree[currentInd].right;
      otherSide = kd_tree[currentInd].left;
    }

    if (bestSide != -1) {
      stack[stackPointer++] = bestSide;
      stack[stackPointer++] = (currentAxis + 1) % 3;
    }

    threshold = getThresholdDist(queue);

    if (otherSide != -1 &&
        ((currentAxis == 0 && (threshold < 0 || sq(pointX - kd_tree[otherSide].x) < threshold)) ||
         (currentAxis == 1 && (threshold < 0 || sq(pointY - kd_tree[otherSide].y) < threshold)) ||
         (currentAxis == 2 && (threshold < 0 || sq(pointZ - kd_tree[otherSide].z) < threshold)))) {
      stack[stackPointer++] = otherSide;
      stack[stackPointer++] = (currentAxis + 1) % 3;
    }
  }
}

pointQueue* __device__ knearest(
    kd_tree_node * kd_tree,
    float pointX, float pointY, float pointZ,
    int nbNeighbors
    ){
    pointQueue* queue = (pointQueue*) malloc(sizeof(pointQueue));
    initPointQueue(queue, nbNeighbors);


    int currentInd = 0; // indice du noeud du kd_tree
    int currentAxis = 0;

    fillQueue(kd_tree, queue, currentInd, currentAxis, pointX, pointY, pointZ);

    return queue;
}

__device__ void computeKnn(int * indTab, float * sqDistTab, kd_tree_node * kd_tree, int nb, float x, float y, float z){
    auto resQueue = knearest(kd_tree,x,y,z,nb);

    
    for(int i = 0 ; i < nb ; i ++){
        indTab[i] = resQueue->ind[i];
        sqDistTab[i] = resQueue->dist[i];
    }

    freePointQueue(resQueue);
    free(resQueue);
}

__device__ int findNearest(kd_tree_node* tree, cVec3 point) {
    int currNode = 0;
    int bestNode = 0;
    float bestDistance = FLT_MAX;

    while (currNode >= 0) {
        kd_tree_node node = tree[currNode];
        cVec3 nodePoint = cVec3(node.x, node.y, node.z);

        float distance = (nodePoint - point).length();

        if (distance < bestDistance) {
            bestDistance = distance;
            bestNode = node.ind;
        }

        int nextNode;
        if (point[node.axis] < nodePoint[node.axis]) {
            nextNode = node.left;
        } else {
            nextNode = node.right;
        }

        if (nextNode >= 0) {
            kd_tree_node next = tree[nextNode];
            cVec3 nextNodePoint = cVec3(next.x, next.y, next.z);

            float axisDistance = nextNodePoint[node.axis] - point[node.axis];
            float axisDistanceSq = axisDistance * axisDistance;
            if (axisDistanceSq < bestDistance) {
                currNode = nextNode;
            } else {
                currNode = -1;
            }
        } else {
            currNode = -1;
        }
    }

    return bestNode;
}

struct Material{
    cVec3 AMBIANT_COLOR = cVec3(0,0,0);
    cVec3 DIFFUSE_COLOR= cVec3(0.5,0.5,0.5);
    cVec3 SPECULAR_COLOR= cVec3(0.5,0.5,0.5);

    int SPECULAR_EXPONENT = 32;
    float transparency = 0.0;
    float refractionIndex = 1.0;
};

struct cIntersection {
    bool intersected;
    cVec3 position;
    float convTime;
};

struct PointCloudData{
    kd_tree_node* kdTree;
    char* materialIndex;
    Material* materialList;
    cVec3 * positions;
    cVec3 * normals;
};

__device__ float HPSSDist(
    cVec3 inputPoint,
    PointCloudData pcd)
{
    int kerneltype = 0;
    float h = 100;
    unsigned int nbIterations = 4;
    const unsigned int knn= 10;

    int * id_nearest_neighbors = (int *) malloc(knn * sizeof(int));
    float * square_distances_to_neighbors = (float *) malloc(knn * sizeof(float));

    cVec3 precPoint = inputPoint;

    cVec3 nextPoint;
    cVec3 nextNormal;

    for(int itt = 0 ; itt < nbIterations ; itt++){
        //printf("itt");

        //kdtree.knearest(precPoint, knn,id_nearest_neighbors,square_distances_to_neighbors);

        //computeKnn(pcd->kdTree, knn, precPoint[0],precPoint[1],precPoint[2], id_nearest_neighbors, (float *)square_distances_to_neighbors);

        computeKnn(id_nearest_neighbors, (float *)square_distances_to_neighbors, pcd.kdTree, knn, precPoint[0],precPoint[1],precPoint[2]);


        nextPoint  = cVec3(0,0,0);
        nextNormal = cVec3(0,0,0);

        float totWeight = 0.0;

        for(int i = 0 ; i<knn ; i ++){

            auto proj = project(precPoint,pcd.normals[id_nearest_neighbors[i]],pcd.positions[id_nearest_neighbors[i]]);
            float weight = 0.0;
            float r = sqrt(square_distances_to_neighbors[i])/h;
            switch (kerneltype){
            case 0:
                
                weight = exp((-(r*r))/(h*h));
                break;
            case 1:
                weight = 0;
                break;
            case 2:
                weight = 0;
                break;
            }
            totWeight  += weight;
            nextPoint  += weight*proj;
            nextNormal += weight*pcd.normals[id_nearest_neighbors[i]];
        }
        nextPoint = nextPoint / totWeight;
        nextNormal.normalize();
        precPoint = nextPoint;
    }
    
    free(id_nearest_neighbors);
    free(square_distances_to_neighbors);

    
    //signedDist(pos,positions,normals,kdtree,0,100.0,5,10);
    //HPSS(inputPoint,projPoint,projNormal,positions,normals,kdtree,kerneltype,h,nbIterations,knn);

    return(dot(inputPoint-nextPoint,nextNormal));

} 

__device__ float globalDist(cVec3 pos, PointCloudData pcd){
    //printf("globalDist\n");
    return HPSSDist(pos, pcd);

    // pointQueue* resQueue = new pointQueue();
    // initPointQueue(resQueue, 10);


    // freePointQueue(resQueue);
    // delete resQueue;



    // float sphere = pos.length() - 1.0;

    // float cube = max(abs(pos[0])-1,max(abs(pos[1])-1,abs(pos[0])-1));
    // return sphere;
    // return min(sphere,cube); 
}


__device__ cIntersection intersect(cVec3 pos, cVec3 dir, PointCloudData pcd){
    float seuilMin = 0.01;
    float seuilMax = 10;

    int maxItt = 50;

    bool conv = false;

    int i = 0;
    for(int i = 0 ; i < maxItt ; i ++){
        float dist = abs(globalDist(pos, pcd));

        if(dist > seuilMax){
            break;
        }else if(dist < seuilMin){
            conv = true; 
            break;
        }else{
            pos += dist*dir;
        }        
    }
    return {conv,pos,(float)i/(float)maxItt};
}

__device__ cVec3 normale(cVec3 pos, PointCloudData pcd){
    cVec3 eps1 = cVec3(0.01, 0.  , 0.  );
    cVec3 eps2 = cVec3(0.  , 0.01, 0.  );
    cVec3 eps3 = cVec3(0.  , 0.  , 0.01);

    cVec3 res = cVec3(
        globalDist(pos + eps1, pcd) - globalDist(pos - eps1, pcd),
        globalDist(pos + eps2, pcd) - globalDist(pos - eps2, pcd),
        globalDist(pos + eps3, pcd) - globalDist(pos - eps3, pcd)
    );

    res.normalize();

    return res;
}

__device__ int getGlobalIdx_1D_2D(){
    return blockIdx.x * blockDim.x * blockDim.y
    + threadIdx.y * blockDim.x + threadIdx.x;
}


__device__ cVec3 computeColor(cVec3 positionCamera, cVec3 positionPoint, cVec3 normalePoint, cVec3 positionLumiere, Material& material) {

  // Calcul de la direction de la lumière
  cVec3 directionLumiere = positionLumiere - positionPoint;
  directionLumiere.normalize();

  // Calcul de la direction de la caméra
  cVec3 directionCamera = positionCamera - positionPoint;
  directionCamera.normalize();

  // Calcul de la normale moyenne
  cVec3 normaleMoyenne = directionLumiere + directionCamera;
  normaleMoyenne.normalize();

  // Calcul de l'intensité de la lumière en utilisant l'éclairage de Blinn-Phong
  float intensiteLumiere = fmax(dot(normalePoint, directionLumiere), 0.0f);
  float specular = pow(fmax(dot(normalePoint, normaleMoyenne), 0.0f), material.SPECULAR_EXPONENT);

  // Calcul de la couleur finale en utilisant les composantes ambiante, diffuse et spéculaire
  cVec3 couleurFinale = material.AMBIANT_COLOR + intensiteLumiere * material.DIFFUSE_COLOR + specular * material.SPECULAR_COLOR;

  return couleurFinale;
}

// __global__ void cuda_ray_trace(float* rayPos, float * rayDir, float * image, int imgSize, PointCloudData pcd){
//     //printf("cuda_ray_trace\n");

//     int index = getGlobalIdx_1D_2D();
    
//     if(index < imgSize){
//         cVec3 pos = cVec3(rayPos[0],rayPos[1],rayPos[2]);
//         cVec3 dir = cVec3(rayDir[index*3+0], rayDir[index*3+1], rayDir[index*3+2]);

//         auto it = intersect(pos, dir, pcd);

//         //cIntersection it = {true,cVec3(0,0,0),10.0};

//         if(it.intersected){
//             int nearestPoint = findNearest(pcd.kdTree, it.position);
//             //int nearestPoint = 0;

//             auto norm = normale(it.position, pcd);

//             auto c = computeColor(pos, it.position, norm, cVec3(1,1,1), pcd.materialList[pcd.materialIndex[nearestPoint]]);

//             image[index*3+0] = c[0] > 1.0 ? 1.0 : c[0] < 0.0 ? 0.0 : c[0] ;
//             image[index*3+1] = c[1] > 1.0 ? 1.0 : c[1] < 0.0 ? 0.0 : c[1] ;
//             image[index*3+2] = c[2] > 1.0 ? 1.0 : c[2] < 0.0 ? 0.0 : c[2] ;
//         }else{
//             image[index*3+0] = 0.1;
//             image[index*3+1] = 0.1;
//             image[index*3+2] = 0.1;
//         }
//     }
// }

__device__ cVec3 computeTransmission(const cVec3& ray, const cVec3& normal, const cVec3& intersection, float refractionIndex) {
    // Calculer la composante parallèle et perpendiculaire du rayon par rapport à la normale
    float dotProduct = dot(ray, normal);
    cVec3 parallel = normal * dotProduct;
    cVec3 perpendicular = ray - parallel;

    // Calculer l'indice de réfraction du milieu d'où vient le rayon (ici, l'air avec un indice de réfraction de 1)
    float n1 = 1.0;
    // Calculer l'indice de réfraction du matériau au point d'intersection
    float n2 = refractionIndex;

    // Calculer le coefficient de réflexion et de transmission
    float reflectionCoefficient = (n1 - n2) / (n1 + n2);
    reflectionCoefficient *= reflectionCoefficient;
    //float transmissionCoefficient = 1.0 - reflectionCoefficient;

    // Calculer le vecteur de transmission
    cVec3 transmission = (perpendicular * n1 - normal * sqrt(1.0 - reflectionCoefficient)) * n2 + parallel;
    transmission.normalize();
    return transmission;
}

__device__ cVec3 mix(cVec3 color1, cVec3 color2, float mixFactor) {
    return color1 * (1.0 - mixFactor) + color2 * mixFactor;
}

__global__ void cuda_ray_trace(float* rayPos, float * rayDir, float * image, int imgSize, PointCloudData pcd, int maxTransparencyIterations) {
    int index = getGlobalIdx_1D_2D();
    
    if (index < imgSize) {
        cVec3 pos = cVec3(rayPos[0], rayPos[1], rayPos[2]);
        cVec3 dir = cVec3(rayDir[index*3+0], rayDir[index*3+1], rayDir[index*3+2]);

        auto it = intersect(pos, dir, pcd);

        if (it.intersected) {
            int nearestPoint = findNearest(pcd.kdTree, it.position);

            auto norm = normale(it.position, pcd);

            cVec3 color = computeColor(pos, it.position, norm, cVec3(1,1,1), pcd.materialList[pcd.materialIndex[nearestPoint]]);

            // Transparence
            float transparency = pcd.materialList[pcd.materialIndex[nearestPoint]].transparency;

            // Nombre d'itérations de transparence
            int transparencyIterations = 0;

            while (transparency > 0.0 && transparencyIterations < maxTransparencyIterations) {
                // Calculer le vecteur de transmission
                dir = computeTransmission(dir, norm, it.position, pcd.materialList[pcd.materialIndex[nearestPoint]].refractionIndex);
                // Lancer un nouveau rayon de transmission
                auto it2 = intersect(it.position+dir*0.01, dir, pcd);
                if (it2.intersected) {
                    // Si le nouveau rayon de transmission intersecte un autre objet, combiner les couleurs
                    int nearestPoint2 = findNearest(pcd.kdTree, it2.position);
                    norm = normale(it2.position, pcd);
                    cVec3 color2 = computeColor(it.position, it2.position, norm, cVec3(1,1,1), pcd.materialList[pcd.materialIndex[nearestPoint2]]);
                    color = mix(color, color2, transparency);
                    // Mettre à jour la transparence et le nombre d'itérations
                    transparency = pcd.materialList[pcd.materialIndex[nearestPoint2]].transparency;
                    transparencyIterations++;
                } else {
                    // Sinon, utiliser la couleur du fond
                    color = mix(color, cVec3(0.1, 0.1, 0.1), transparency);
                    // Mettre à jour la transparence et le nombre d'itérations
                    transparency = 0.0;
                    transparencyIterations++;
                }
            }

            image[index*3+0] = color[0] > 1.0 ? 1.0 : color[0] < 0.0 ? 0.0 : color[0] ;
            image[index*3+1] = color[1] > 1.0 ? 1.0 : color[1] < 0.0 ? 0.0 : color[1] ;
            image[index*3+2] = color[2] > 1.0 ? 1.0 : color[2] < 0.0 ? 0.0 : color[2] ;
        }else{
            image[index*3+0] = 0.1;
            image[index*3+1] = 0.1;
            image[index*3+2] = 0.1;
        }
    }
}


extern "C" PointCloudData getGPUpcd(std::vector<Vec3> positions, std::vector<Vec3> normals, std::vector<char> materialIndex, std::vector<Material> materialList){
    PointCloudData res;

    // Allouer de la mémoire sur le GPU pour les champs positions, normals, materialIndex et materialList de la structure PointCloudData
    cudaMalloc(&(res.positions), positions.size()*sizeof(cVec3));
    cudaMalloc(&(res.normals), normals.size()*sizeof(cVec3));
    cudaMalloc(&(res.materialIndex), materialIndex.size()*sizeof(char));
    cudaMalloc(&(res.materialList), materialList.size()*sizeof(Material));

    // Copier les données depuis le CPU vers le GPU
    cudaMemcpy(res.positions, (void *)positions.data(), positions.size()*sizeof(cVec3), cudaMemcpyHostToDevice);
    cudaMemcpy(res.normals, (void *)normals.data(), normals.size()*sizeof(cVec3), cudaMemcpyHostToDevice);
    cudaMemcpy(res.materialIndex, (void *)materialIndex.data(), materialIndex.size()*sizeof(char), cudaMemcpyHostToDevice);
    cudaMemcpy(res.materialList, (void *)materialList.data(), materialList.size()*sizeof(Material), cudaMemcpyHostToDevice);

    std::cout<<"Start kd-tree building"<<std::endl;
        // Construire l'arbre kd à partir des positions sur le CPU et l'envoyer sur le GPU
        auto my_kd_tree = make_kd_tree(positions);
        res.kdTree = send_kd_tree(my_kd_tree);
    std::cout<<"End kd-tree building"<<std::endl;

    return res;
}


//calcul la couleur des pixels, par ray marching

extern "C" void cuda_ray_trace_from_camera(int w, int h, Vec3 (*cameraSpaceToWorldSpace)(const Vec3&), Vec3 (*screen_space_to_worldSpace)(float, float), PointCloudData pcd){

    std::vector<float> image(3*w*h, 0.5f);   
    std::vector<float> rayDir(3*w*h);

    std::cout << "Ray tracing a " << w << " x " << h << " image" << std::endl;

    // Init
    auto pos = cameraSpaceToWorldSpace(Vec3(0, 0, 0));

    for (int y = 0; y < h; y++){
        for (int x = 0; x < w; x++){
            float u = ((float)(x) + (float)(rand()) / (float)(RAND_MAX)) / w;
            float v = ((float)(y) + (float)(rand()) / (float)(RAND_MAX)) / h;
            Vec3 dir = screen_space_to_worldSpace(u, v) - pos;
            dir.normalize();

            rayDir[3*(y*w+x) + 0] = dir[0];
            rayDir[3*(y*w+x) + 1] = dir[1];
            rayDir[3*(y*w+x) + 2] = dir[2];

        }
    }

    float * cudaDirTab;
    cudaMalloc(&cudaDirTab, 3*rayDir.size()*sizeof(float));

    float * cudaImage;
    cudaMalloc(&cudaImage, 3*image.size()*sizeof(float));

    float * cudaPos;
    cudaMalloc(&cudaPos, 3*sizeof(float));

    cudaMemcpy(cudaDirTab, (void *)rayDir.data(), rayDir.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaImage, (void *)image.data(), image.size()*sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(cudaPos, &pos, 3*sizeof(float), cudaMemcpyHostToDevice);

    std::cout<<"w : "<<w<<" h : "<<h<<" w*h : "<<w*h<<std::endl;

    int nbth = 5;

    //cuda_ray_trace<<<h,w>>>(cudaPos, cudaDirTab, cudaImage, w);

    int nbBlock = std::ceil((w*h) / (nbth*nbth));
    //nbBlock = 5;

    std::cout<<"Nb block : "<<nbBlock<<std::endl;

    dim3 threadsPerBlock(nbth, nbth);
    dim3 numBlocks(nbBlock, 1);

    cuda_ray_trace<<<numBlocks,threadsPerBlock>>>(cudaPos, cudaDirTab, cudaImage, h*w, pcd, 0);

    std::cout<<"GPU termined"<<std::endl;

    cudaMemcpy((void *)image.data(), (void *)cudaImage, image.size()*sizeof(float), cudaMemcpyDeviceToHost);

    // for(int i = 0 ; i < image.size()/3 ; i ++){
    //     std::cout<<image[i*3+0]<<" "<<image[i*3+1]<<" "<<image[i*3+2]<<std::endl;
    // }


    std::string filename = "./rendu.ppm";

    std::ofstream f(filename.c_str(), std::ios::binary);

    if (f.fail())
    {
        std::cout << "Could not open file: " << filename << std::endl;
        return;
    }
    f << "P3" << std::endl
      << w << " " << h << std::endl
      << 255 << std::endl;
    for (int i = 0; i < w * h; i++)
        f << (int)(255.f * std::min<float>(1.f, image[i*3+0])) << " " << (int)(255.f * std::min<float>(1.f, image[i*3+1])) << " " << (int)(255.f * std::min<float>(1.f, image[i*3+2])) << " ";
    f << std::endl;
    f.close();

    // Reset img
    image.clear();
    image.resize(w * h * 3);
    fill(image.begin(), image.end(), 0.0f);    
}

//rempli la mémoire du GPU avec les position et les direction de chaque rayon
//lance une fonction kernel pour calculer le ray marching
//récupère l'image de la mémoire du GPU et en fait une image