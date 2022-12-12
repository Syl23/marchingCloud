#include <fstream>
#include <string>
#include <vector>
#include <queue>
#include <algorithm>

#include "cudaKdtree.h"
#include "cudaVec3.h"

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
    PointCloudData * pcd)
{
    int kerneltype = 0;
    float h = 100;
    unsigned int nbIterations = 5;
    unsigned int knn= 10;

    int* id_nearest_neighbors           = new int[knn];
    float* square_distances_to_neighbors = new float[knn];

    cVec3 precPoint = inputPoint;

    cVec3 nextPoint;
    cVec3 nextNormal;

    for(int itt = 0 ; itt < nbIterations ; itt++){

        //kdtree.knearest(precPoint, knn,id_nearest_neighbors,square_distances_to_neighbors);

        //computeKnn(pcd->kdTree, knn, precPoint[0],precPoint[1],precPoint[2], id_nearest_neighbors, (float *)square_distances_to_neighbors);
        computeKnn(id_nearest_neighbors, (float *)square_distances_to_neighbors, pcd->kdTree, knn, precPoint[0],precPoint[1],precPoint[2]);


        nextPoint  = cVec3(0,0,0);
        nextNormal = cVec3(0,0,0);

        float totWeight = 0.0;

        for(int i = 0 ; i<knn ; i ++){

            auto proj = project(precPoint,pcd->normals[id_nearest_neighbors[i]],pcd->positions[id_nearest_neighbors[i]]);
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
            nextNormal += weight*pcd->normals[id_nearest_neighbors[i]];
        }
        nextPoint = nextPoint / totWeight;
        nextNormal.normalize();
        precPoint = nextPoint;
    }
    
    //signedDist(pos,positions,normals,kdtree,0,100.0,5,10);
    //HPSS(inputPoint,projPoint,projNormal,positions,normals,kdtree,kerneltype,h,nbIterations,knn);

    return(dot(inputPoint-nextPoint,nextNormal));

} 

__device__ float globalDist(cVec3 pos, PointCloudData * pcd){
    return HPSSDist(pos, pcd);

    //return pos.length() - 1.0; 
}

__device__ cIntersection intersect(cVec3 pos, cVec3 dir, PointCloudData * pcd){
    double seuilMin = 0.005;
    double seuilMax = 10;

    int maxItt = 10;

    bool conv = false;
    bool div  = false;

    int i = 0;
    while(!conv && !div){
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

__device__ cVec3 normale(cVec3 pos, PointCloudData * pcd){
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

__global__ void cuda_ray_trace(float* rayPos, float * rayDir, float * image, int imgSize, PointCloudData * pcd){

    int index = getGlobalIdx_1D_2D();
    
    if(index < imgSize){
        cVec3 pos = cVec3(rayPos[0],rayPos[1],rayPos[2]);
        cVec3 dir = cVec3(rayDir[index*3+0], rayDir[index*3+1], rayDir[index*3+2]);

        auto it = intersect(pos, dir, pcd);

        if(it.intersected){
            image[index*3+0] = 1.0;
            image[index*3+1] = 1.0;
            image[index*3+2] = 1.0;
        }else{
            image[index*3+0] = 0.3;
            image[index*3+1] = 0.3;
            image[index*3+2] = 0.3;
        }




        // image[index*3+0] = rayDir[index*3+0] > 1.0 ? 1.0 : rayDir[index*3+0] < 0.0 ? 0.0 : rayDir[index*3+0] ;
        // image[index*3+1] = rayDir[index*3+1] > 1.0 ? 1.0 : rayDir[index*3+1] < 0.0 ? 0.0 : rayDir[index*3+1] ;
        // image[index*3+2] = rayDir[index*3+2] > 1.0 ? 1.0 : rayDir[index*3+2] < 0.0 ? 0.0 : rayDir[index*3+2] ;
    }

}
//calcul la couleur des pixels, par ray marching

void cuda_ray_trace_from_camera(int w, int h, Vec3 (*cameraSpaceToWorldSpace)(const Vec3&), Vec3 (*screen_space_to_worldSpace)(float, float), PointCloudData * pcd){

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