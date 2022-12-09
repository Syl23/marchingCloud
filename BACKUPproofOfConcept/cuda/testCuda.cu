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

void __device__ freeStack(Stack * pile){
    free(pile->data);
    free(pile);
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

void testCuda() {
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
}