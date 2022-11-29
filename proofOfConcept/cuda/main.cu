#include "cuda.h"


/*
cudaMalloc(void **devPtr, size_t count);
cudaFree(void *devPtr);

cudaMemcpy(void *dst, void *src, size_t count, cudaMemcpyKind kind)

kind -> cudaMemcpyHostToDevice
kind -> cudaMemcpyDeviceToHost

*/
bool compareIndVec3 (int axis, std::pair<int,Vec3> i,std::pair<int,Vec3> j) {
    return (i.second[axis]<j.second[axis]); 
}

std::vector<kd_tree_node> make_kd_tree(std::vector<Vec3> dots){
    auto res = std::vector<kd_tree_node>();
    auto indVec = std::vector<std::pair<int,Vec3>>(dots.size());

    for(int i = 0 ; i < dots.size() ; i ++){
        indVec.push_back(std::make_pair(i,dots[i]));
    }  

    auto fileTraitement = std::queue<std::pair<int,int>>() ; // Indice de début et de fin du tableau indVec
    fileTraitement.push(std::make_pair(0,indVec.size()));

    int currentAxis = 0;

    while(!fileTraitement.empty()){
        auto plage = fileTraitement.front();
        fileTraitement.pop();

        std::sort (indVec.begin()+plage.first, indVec.begin()+plage.second, std::bind(compareIndVec3,currentAxis,std::placeholders::_1,std::placeholders::_2));

        int med = (plage.first + plage.second)/2;

        int left;
        int right;

        if(plage.second - plage.first > 1){
            left = res.size();
            right = res.size()+1;
        }else{
            left = -1;
            right = -1;
            if(plage.second - plage.first == 0){
                std::cout<<"Petit souci"<<std::endl;
            }
        }

        res.push_back({
            indVec[med].first,

            indVec[med].second[0],
            indVec[med].second[1],
            indVec[med].second[2],

            currentAxis,

            left,
            right,
        });

        if(left != -1){
            fileTraitement.push(std::make_pair(plage.first,med));
            fileTraitement.push(std::make_pair(med,plage.second));
        }

        currentAxis = (currentAxis+1) % 3;
    }
    return res;
}

void * send_kd_tree(std::vector<kd_tree_node> kd_tree){
    void * res = NULL;

    cudaMemcpy(res, (void *)kd_tree.data(), kd_tree.size()*sizeof(kd_tree_node), cudaMemcpyHostToDevice);

    return res;
}

//precPoint, knn,id_nearest_neighbors,square_distances_to_neighbors


/*
struct kd_tree_node{
    int ind;

    float x;
    float y;
    float z;

    int axis;

    int left;
    int right;
};
*/

struct pointToQueue{
    int ind; //indice d'origine du point (pas du noeud dans le kd_tree)
    float dist;
};

void __device__ addToPointQueue(pointToQueue* queue,int* nbPointsInQueue,int maxPoint, pointToQueue pt){
    pointToQueue bag = pt;

    for(int i = 0 ; i < *nbPointsInQueue ; i ++){
        if(bag.dist < queue[i].dist){
            pointToQueue tmp = queue[i];
            queue[i] = bag;
            bag = tmp;
        }        
    }
    
    if(*nbPointsInQueue < maxPoint){
        queue[(*nbPointsInQueue)++] = bag;
    }
    
}

struct Stack{
    int * data;
    int nbItem;
    int maxItem;
};


void __device__ pushToStack(Stack * pile, int node){
    if(pile->nbItem < pile->maxItem){
        pile->data[pile->nbItem++] = node;
    }else{
        int * tmpData = pile->data;
        if(pile->maxItem == 0){
            pile->maxItem = 2;
        }else{
            pile->maxItem *= 2;
        }
        pile->data = (int *) malloc(pile->maxItem * sizeof(int));

        for(int i = 0 ; i < pile->nbItem ; i ++){
            pile->data[i] = tmpData[i];
        }
        free(tmpData);
    }
}

int __device__ popFromStack(Stack * pile){
    return(pile->data[pile->nbItem--]);
}

void __device__ knearest(
    kd_tree_node * kd_tree,
    float pointX, float pointY, float pointZ,
    int nbNeighbors,
    int ** id_nearest_neighbors,
    float ** square_distances_to_neighbors
    ){

    // aller à la feuille la plus proche
    // remonter en ajoutant tout les points
    // une fois que ne nombre de voisin est attein on ajoute que les points plus proches que le nb'eme le plus loins
    // on renvoi que les nb plus proches

    //quand on decende dans l'arbre on ajoute les noeuds à une pile (imo)
    //une fois qu'on a atein uine feuille, il faut remonter,
    //pour chaque noeud, on test si le point est inferieur au seuil (soit la distance avec le point le plus loin, soit pas de seuil avant d'avoir remplie la queue)
        //si c'est le cas on met le point dans la queue
    //si le noeud a des fils, on regarde si le fils oposé au coté qu'on vient de regarder (comment on sait ça ?) peut contenir des points inferieur au seuil 
        //(intersection de de la sphere avec le plan)
        //concrètement on regarde la distance en 1 dimension entre la coo du point et la co du fils (donc juste sur l'axe qui nous interesse)
        //s'il peut être interessant on le mets dans la pile




    int nbPointsInQueue = 0;
    pointToQueue* queue = new pointToQueue[nbNeighbors]; //les points les plus proches

    Stack *nodeStack = new Stack();



    int currentInd = 0; // indice du noeud du kd_tree
    int currentAxis = 0;

    while(kd_tree[currentInd].left != -1){ //trouver la meilleur feuille
        pushToStack(nodeStack,currentInd);

        if(
            (currentAxis == 0 && kd_tree[currentInd].x < pointX)||
            (currentAxis == 1 && kd_tree[currentInd].y < pointY)||
            (currentAxis == 2 && kd_tree[currentInd].z < pointZ)
        ){
            currentInd = kd_tree[currentInd].left;
        }else{
            currentInd = kd_tree[currentInd].right;
        }

        currentAxis = (currentAxis + 1) % 3;
    }

    pushToStack(nodeStack,currentInd);

    float curentSqDist = 
        (kd_tree[currentInd].x - pointX) * (kd_tree[currentInd].x - pointX) +
        (kd_tree[currentInd].y - pointY) * (kd_tree[currentInd].y - pointY) +
        (kd_tree[currentInd].z - pointZ) * (kd_tree[currentInd].z - pointZ);


    addToPointQueue(queue,&nbPointsInQueue,nbNeighbors,{kd_tree[currentInd].ind,curentSqDist});
    

    while(Stack->nbItem > 0){
        int currentNodeInd = popFromStack(Stack);

        //calculer la distance, push dans la file
        //redarger le fils oposé si oui push

    }




}

__device__ int addi(int a, int b){
    return a+b;
}

__global__ void add(int * res, int * a, int * b){
    //threadIdx.x
    //blockDim.x
    int n = threadIdx.x;
    res[n] = addi(a[n], b[n]);
}

int cudaMain() {
    int nb =10;

    auto a = std::vector<int>();
    auto b = std::vector<int>();

    auto res = std::vector<int>();

    for(int i = 0; i < nb ; i ++){
        a.push_back(i);
        b.push_back(i);
    }

    int *cudaA;
    int *cudaB;
    int *cudaRes;

    cudaMalloc(&cudaA   , nb*sizeof(int));
    cudaMalloc(&cudaB   , nb*sizeof(int));
    cudaMalloc(&cudaRes , nb*sizeof(int));

    cudaMemcpy(cudaA, (void *)a.data(), nb*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(cudaB, (void *)b.data(), nb*sizeof(int), cudaMemcpyHostToDevice);

    add<<<1,nb>>>(cudaRes, cudaA, cudaB);

    res.resize(nb);
    cudaMemcpy(res.data(), cudaRes, nb*sizeof(int), cudaMemcpyDeviceToHost);

    for(int i : res){
        std::cout<<i<<std::endl;
    }

    return 0;
}