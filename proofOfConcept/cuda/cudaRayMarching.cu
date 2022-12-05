#include "cudaRayMarching.h"
#include "cudaVec3.h"

#include <fstream>
#include <string>
#include <vector>

struct cIntersection {
    bool intersected;
    cVec3 position;
    float convTime;
};

__device__ float globalDist(cVec3 pos){
    return pos.length() - 1.0; 
}

__device__ cIntersection intersect(cVec3 pos, cVec3 dir){
    double seuilMin = 0.005;
    double seuilMax = 10;

    int maxItt = 10;

    bool conv = false;
    bool div  = false;

    int i = 0;
    while(!conv && !div){
        double dist = globalDist(pos);

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

__device__ cVec3 normale(cVec3 pos){
    cVec3 eps1 = cVec3(0.01, 0.  , 0.  );
    cVec3 eps2 = cVec3(0.  , 0.01, 0.  );
    cVec3 eps3 = cVec3(0.  , 0.  , 0.01);

    cVec3 res = cVec3(
    globalDist(pos + eps1) - globalDist(pos - eps1),
    globalDist(pos + eps2) - globalDist(pos - eps2),
    globalDist(pos + eps3) - globalDist(pos - eps3)
    );

    res.normalize();

    return res;
}

__global__ void cuda_ray_trace(float* rayPos, float * rayDir, float * image, int imgSize){
    int threadLineLen = 32;

    int x = threadIdx.x;
    int y = threadIdx.y;

    int blockInd = blockIdx.x; // block index

    int ind = y*threadLineLen + x; //thread index

    int pixelInd = blockInd * 1024 + ind;
    
    if(pixelInd < imgSize){
        image[ind*3+0] = 1.0f;
        image[ind*3+1] = 0.0f;
        image[ind*3+2] = 0.0f;
    }

}
//calcul la couleur des pixels, par ray marching

void cuda_ray_trace_from_camera(int w, int h, Vec3 (*cameraSpaceToWorldSpace)(const Vec3&), Vec3 (*screen_space_to_worldSpace)(float, float)){

    std::vector<float> image(3*w*h, 1.0f);   
    std::vector<float> rayDir(3*w*h);

    std::cout << "Ray tracing a " << w << " x " << h << " image" << std::endl;

    // Init
    auto pos = cameraSpaceToWorldSpace(Vec3(0, 0, 0));

    for (int y = 0; y < h; y++){
        for (int x = 0; x < w; x++){
            float u = ((float)(x) + (float)(rand()) / (float)(RAND_MAX)) / w;
            float v = ((float)(y) + (float)(rand()) / (float)(RAND_MAX)) / h;
            Vec3 dir = screen_space_to_worldSpace(u, v) - pos;

            rayDir[y*w+x + 0] = dir[0];
            rayDir[y*w+x + 1] = dir[1];
            rayDir[y*w+x + 2] = dir[2];

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

    int nbBlock = std::ceil((w*h) / (32*32));

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(nbBlock, 1);

    cuda_ray_trace<<<numBlocks,threadsPerBlock>>>(cudaPos, cudaDirTab, cudaImage, h*w);

    std::cout<<"GPU termined"<<std::endl;

    cudaMemcpy((void *)image.data(), (void *)cudaImage, image.size()*sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0 ; i < image.size()/3 ; i ++){
        std::cout<<image[i*3+0]<<" "<<image[i*3+1]<<" "<<image[i*3+2]<<std::endl;
    }


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