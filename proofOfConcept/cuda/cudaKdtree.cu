#include "cudaKdtree.h"

std::ostream& operator<<(std::ostream& os, const kd_tree_node& kdn){
    os <<"node : \n"<< "\tpoint index : "<<kdn.ind<<"\n\tpoint : ("<<kdn.x<<" "<<kdn.y<<" "<<kdn.z<<")\n\taxis : "<<kdn.axis<<"\n\tleft : "<<kdn.left<<" right : "<<kdn.right;
    return os;
}

bool compareIndVec3 (int axis, std::pair<int,Vec3> i,std::pair<int,Vec3> j) {
    return (i.second[axis]<j.second[axis]); 
}

std::vector<kd_tree_node> make_kd_tree(std::vector<Vec3> dots){
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

        std::sort (indVec.begin()+plage.first, indVec.begin()+plage.second, std::bind(compareIndVec3,axis,std::placeholders::_1,std::placeholders::_2));

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

kd_tree_node* send_kd_tree(std::vector<kd_tree_node> kd_tree){
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

    __device__ pointQueue(int nb){
        size = nb;
        nbInQueue = 0;

        ind = new int[nb];
        dist = new float[nb];

        for(int i = 0 ; i < nb ; i ++){
            dist[i] = -1.0;
        }
    }
};

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

    pointQueue* queue = new pointQueue(nbNeighbors); //les points les plus proches

    int currentInd = 0; // indice du noeud du kd_tree
    int currentAxis = 0;

    fillQueue(kd_tree, queue, currentInd, currentAxis, pointX, pointY, pointZ);

    return queue;
}

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