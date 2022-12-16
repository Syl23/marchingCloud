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

struct pointQueue{
    int size;
    int nbInQueue;

    int * ind;
    float * dist;

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
};

__device__ void initPointQueue(pointQueue * ptc, int nb){
    *ptc = {0,0,NULL,NULL};

    ptc->size = nb;
    ptc->nbInQueue = 0;

    ptc->ind  = (int *) malloc(nb * sizeof(int));
    ptc->dist = (float * )malloc(nb * sizeof(float));

    for(int i = 0 ; i < nb ; i ++){
        ptc->dist[i] = -1.0;
        ptc->ind[i] = 0;
    }
}


float __device__ getThresholdDist(pointQueue* queue){
    if(queue->nbInQueue == queue->size){
        return queue->dist[queue->size -1];
    }
    return -1;
}

void __device__ addToPointQueue(pointQueue* queue, int index, float dist){
    int bagi   = index;
    float bagd = dist;

    for(int i = 0 ; i < queue->nbInQueue ; i ++){
        if(queue->dist[i] < 0 || bagd < queue->dist[i]){

            int tmpi = queue->ind[i];
            float tmpd = queue->dist[i];

            queue->ind[i] = bagi;
            queue->dist[i] = bagd;

            bagi = tmpi;
            bagd = tmpd;
        }
    }

    if(queue->nbInQueue < queue->size){
        queue->ind [queue->nbInQueue] = bagi;
        queue->dist[queue->nbInQueue] = bagd;

        queue->nbInQueue ++;
    }
}

float __device__ sq(float f){
    return f*f;
}

void __device__ fillQueue(kd_tree_node * kd_tree, pointQueue* queue, int currentInd, int currentAxis, float pointX, float pointY, float pointZ){
    //trouver le meilleur coté
    //appelle récursif sur le meilleur coté

    //si la pire des meilleur distance est superieur a la distance avec le plan de séparatrion on fais un appelle récursif de l'autre coté



    float curentSqDist = //distance carrée du noeud courrant
        sq(kd_tree[currentInd].x - pointX) +
        sq(kd_tree[currentInd].y - pointY) +
        sq(kd_tree[currentInd].z - pointZ);

    auto threshold = getThresholdDist(queue);


    int pointIndex = kd_tree[currentInd].ind;

    if(threshold < 0 || curentSqDist < threshold)
    addToPointQueue(queue, pointIndex, curentSqDist);


    int bestSide = 0;
    int otherSide = 0; 

    if(
        kd_tree[currentInd].left == -1 ||
        (currentAxis == 0 && kd_tree[currentInd].x < pointX)||
        (currentAxis == 1 && kd_tree[currentInd].y < pointY)||
        (currentAxis == 2 && kd_tree[currentInd].z < pointZ)
    ){
        bestSide = kd_tree[currentInd].left;
        otherSide = kd_tree[currentInd].right;            
    }else{
        bestSide = kd_tree[currentInd].right;
        otherSide = kd_tree[currentInd].left;
    }


    if(bestSide != -1){
        fillQueue(kd_tree, queue, bestSide, (currentAxis+1 )% 3, pointX, pointY, pointZ);
    }

    threshold = getThresholdDist(queue);

    if(otherSide != -1 && (
        (currentAxis == 0 && (threshold < 0 || sq(pointX-kd_tree[otherSide].x) < threshold) )||
        (currentAxis == 1 && (threshold < 0 || sq(pointY-kd_tree[otherSide].y) < threshold) )||
        (currentAxis == 2 && (threshold < 0 || sq(pointZ-kd_tree[otherSide].z) < threshold) ))
        ){
            fillQueue(kd_tree,queue, otherSide, (currentAxis+1 )% 3, pointX, pointY, pointZ);           
        }
}

pointQueue* __device__ knearest(
    kd_tree_node * kd_tree,
    float pointX, float pointY, float pointZ,
    int nbNeighbors
    ){    

    //pointQueue* queue = new pointQueue(nbNeighbors); //les points les plus proches
    //pointQueue* queue = new pointQueue(); //les points les plus proches

    pointQueue* queue;// = new pointQueue();
    initPointQueue(queue, nbNeighbors);


    int currentInd = 0; // indice du noeud du kd_tree
    int currentAxis = 0;

    fillQueue(kd_tree, queue, currentInd, currentAxis, pointX, pointY, pointZ);

    return queue;
}

__device__ void computeKnn(int * indTab, float * sqDistTab, kd_tree_node * kd_tree, int nb, float x, float y, float z){
    //auto resQueue = knearest(kd_tree,x,y,z,nb);

    pointQueue* resQueue;// = new pointQueue();
    initPointQueue(resQueue, nb);
    
    for(int i = 0 ; i < nb ; i ++){
        indTab[i] = resQueue->ind[i];
        sqDistTab[i] = resQueue->dist[i];
    }
}

//for debug purpuse do not delete
/* 
__global__ void computeKnn(int * indTab, float * sqDistTab, kd_tree_node * kd_tree, int nb, float x, float y, float z){
    auto resQueue = knearest(kd_tree,x,y,z,nb);
    
    for(int i = 0 ; i < nb ; i ++){
        indTab[i] = resQueue->ind[i];
        sqDistTab[i] = resQueue->dist[i];
    }
}

void printKnn(kd_tree_node * kd_tree, int nb, float x, float y, float z){
    int * cudaInd;
    float * cudaSqDist;

    int * ind = new int[nb];
    float * SqDist = new float[nb];

    cudaMalloc(&cudaInd,    nb*sizeof(int));
    cudaMalloc(&cudaSqDist, nb*sizeof(float));

    computeKnn<<<1,1>>>(cudaInd,cudaSqDist,kd_tree,nb,x,y,z);

    cudaMemcpy(ind, cudaInd, nb*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(SqDist, cudaSqDist, nb*sizeof(float), cudaMemcpyDeviceToHost);


    for(int i = 0 ; i < nb ; i ++){
        std::cout<<"index : "<<ind[i]<<" dist : "<<SqDist[i]<<std::endl;
    }
}

void getKnn(kd_tree_node * kd_tree, int nb, float x, float y, float z, int * ind, float * SqDist){
    //std::cout<<"knn call"<<std::endl;

    int * cudaInd;
    float * cudaSqDist;

    cudaMalloc(&cudaInd,    nb*sizeof(int));
    cudaMalloc(&cudaSqDist, nb*sizeof(float));

    computeKnn<<<1,1>>>(cudaInd,cudaSqDist,kd_tree,nb,x,y,z);

    cudaMemcpy(ind, cudaInd, nb*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(SqDist, cudaSqDist, nb*sizeof(float), cudaMemcpyDeviceToHost);
}

void getKnn(kd_tree_node * kd_tree, int nb, Vec3 point, int * ind, float * SqDist){
    getKnn(kd_tree,nb, point[0],point[1],point[2], ind,SqDist);
}
*/

struct cIntersection {
    bool intersected;
    cVec3 position;
    float convTime;
};

struct PointCloudData{
    kd_tree_node* kdTree;
    cVec3 * positions;
    cVec3 * normals;
};

__device__ float HPSSDist(
    cVec3 inputPoint,
    PointCloudData pcd)
{
    int kerneltype = 0;
    float h = 100;
    unsigned int nbIterations = 5;
    const unsigned int knn= 10;

    int id_nearest_neighbors[knn];//           = int[knn];//new int[knn];
    float square_distances_to_neighbors[knn];// = float[knn];//new float[knn];

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
    
    // free(id_nearest_neighbors);
    // free(square_distances_to_neighbors);

    
    //signedDist(pos,positions,normals,kdtree,0,100.0,5,10);
    //HPSS(inputPoint,projPoint,projNormal,positions,normals,kdtree,kerneltype,h,nbIterations,knn);

    return(dot(inputPoint-nextPoint,nextNormal));

} 
// __device__ float max(float a, float b, float c){
//     return (a>b && a>c ? a : b>a && b>c ? b : c);
// }


__device__ float globalDist(cVec3 pos, PointCloudData pcd){
    //printf("globalDist\n");
    //return HPSSDist(pos, pcd);

    float sphere = pos.length() - 1.0;

    float cube = max(abs(pos[0])-1,max(abs(pos[1])-1,abs(pos[0])-1));
    return sphere;
    return min(sphere,cube); 
}

__device__ cIntersection intersect(cVec3 pos, cVec3 dir, PointCloudData pcd){
    double seuilMin = 0.005;
    double seuilMax = 10;

    int maxItt = 10;

    bool conv = false;
    bool div  = false;

    int i = 0;
    while(!conv && !div){
        //printf("int\n");
        double dist = globalDist(pos, pcd);

        if(dist > seuilMax || i > maxItt){
            div = true;
            break;
        }else if(dist < seuilMin){
            conv = true; 
            break;
        }else{
            pos += dist*dir;
            i++;
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

__global__ void cuda_ray_trace(float* rayPos, float * rayDir, float * image, int imgSize, PointCloudData pcd){
    //printf("cuda_ray_trace\n");

    int index = getGlobalIdx_1D_2D();
    
    if(index < imgSize){
        cVec3 pos = cVec3(rayPos[0],rayPos[1],rayPos[2]);
        cVec3 dir = cVec3(rayDir[index*3+0], rayDir[index*3+1], rayDir[index*3+2]);

        auto it = intersect(pos, dir, pcd);

        //cIntersection it = {true,cVec3(0,0,0),10.0};

        if(it.intersected){
            image[index*3+0] = 0.1;
            image[index*3+1] = 0.9;
            image[index*3+2] = 0.1;

            // auto c = normale(it.position, pcd);

            // image[index*3+0] = c[0] > 1.0 ? 1.0 : c[0] < 0.0 ? 0.0 : c[0] ;
            // image[index*3+1] = c[1] > 1.0 ? 1.0 : c[1] < 0.0 ? 0.0 : c[1] ;
            // image[index*3+2] = c[2] > 1.0 ? 1.0 : c[2] < 0.0 ? 0.0 : c[2] ;
        }else{
            image[index*3+0] = 0.9;
            image[index*3+1] = 0.1;
            image[index*3+2] = 0.1;
        }

    }

}


    // struct PointCloudData{
    //     kd_tree_node* kdTree;
    //     cVec3 * positions;
    //     cVec3 * normals;
    // };


extern "C" PointCloudData getGPUpcd(std::vector<Vec3> positions, std::vector<Vec3> normals){
    PointCloudData res;
    auto tmpPos = std::vector<cVec3>(positions.size());
    auto tmpNorm = std::vector<cVec3>(normals.size());

    for(int i = 0 ; i < positions.size() ; i ++){
        tmpPos[i] = {positions[i][0],positions[i][1],positions[i][2]};
    }

    for(int i = 0 ; i < normals.size() ; i ++){
        tmpNorm[i] = {normals[i][0],normals[i][1],normals[i][2]};
    }

    
    cudaMalloc(&(res.positions), tmpPos.size()*sizeof(cVec3));
    cudaMalloc(&(res.normals), tmpNorm.size()*sizeof(cVec3));

    cudaMemcpy(res.positions, (void *)tmpPos.data(), tmpPos.size()*sizeof(cVec3), cudaMemcpyHostToDevice);
    cudaMemcpy(res.normals, (void *)tmpNorm.data(), tmpNorm.size()*sizeof(cVec3), cudaMemcpyHostToDevice);
   

    

    std::cout<<"Start kd-tree building"<<std::endl;
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

    //cuda_ray_trace<<<h,w>>>(cudaPos, cudaDirTab, cudaImage, w);

    int nbBlock = std::ceil((w*h) / (32.0*32.0));

    std::cout<<"Nb block : "<<nbBlock<<std::endl;

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(nbBlock, 1);

    cuda_ray_trace<<<numBlocks,threadsPerBlock>>>(cudaPos, cudaDirTab, cudaImage, h*w, pcd);

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