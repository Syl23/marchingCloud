#define PARAM_NUMBER_SAMPLES 1

struct Intersection {
    bool intersected;
    Vec3 position;
    float convTime;
};

Vec3 normale(Vec3 pos){
    Vec3 eps1 = Vec3(0.01, 0.  , 0.  );
    Vec3 eps2 = Vec3(0.  , 0.01, 0.  );
    Vec3 eps3 = Vec3(0.  , 0.  , 0.01);

    Vec3 res = Vec3(
        globalSignedDist(pos + eps1) - globalSignedDist(pos - eps1),
        globalSignedDist(pos + eps2) - globalSignedDist(pos - eps2),
        globalSignedDist(pos + eps3) - globalSignedDist(pos - eps3)
    );

    res.normalize();

    return res;
}

Intersection intersect(Vec3 pos, Vec3 dir);

void ray_trace_from_camera(){

    int w = glutGet(GLUT_WINDOW_WIDTH);
    int h = glutGet(GLUT_WINDOW_HEIGHT);

    std::vector<Vec3> image(w *h, Vec3(0, 0, 0));
    std::cout << "Ray tracing a " << w << " x " << h << " image" << std::endl;


    camera.apply();
    time_t tmpTime = time(NULL);
    float avgTime = 0.0;
    time_t totTime = 0;



    // Init
    auto pos = cameraSpaceToWorldSpace(Vec3(0, 0, 0));
    //thread everyThread[h];
    auto percent = 0;

    for (int y = 0; y < h; y++)
    {
        // Précalc thread
        for (int x = 0; x < w; x++)
        {
            std::cout<<"pixel"<<std::endl;
            for (unsigned int s = 0; s < PARAM_NUMBER_SAMPLES; ++s)
            {
                float u = ((float)(x) + (float)(rand()) / (float)(RAND_MAX)) / w;
                float v = ((float)(y) + (float)(rand()) / (float)(RAND_MAX)) / h;
                Vec3 dir = screen_space_to_worldSpace(u, v) - pos;
                
                auto inter = intersect(pos,dir);

                if(inter.intersected){
                    image[y*w+x] = 0.5*Vec3(0.5,0.5,0.5)+0.5*normale(inter.position);   
                }else{
                    image[y*w+x] = Vec3(0,0,0);
                }

                inter = intersect(inter.position+0.1*dir,dir);

                if(inter.intersected){
                    image[y*w+x] += 0.5 * Vec3(0.5,0.5,0.5)+0.5*normale(inter.position);   
                }else{
                    image[y*w+x] += Vec3(0,0,0);
                }
            }
        }
        totTime=time(NULL) - tmpTime;

        avgTime = (float)totTime / (y+1);
        time_t remaining = (h-y) * avgTime;
        std::cout<<"Ligne "<<y<<"/"<<h<<" Remaining time : "<<remaining/60<<"m"<<remaining%60<<"s"<<std::endl;
    }

    std::cout << "\tDone : " << time(NULL) - tmpTime << "s" << std::endl;

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
        f << (int)(255.f * std::min<float>(1.f, image[i][0])) << " " << (int)(255.f * std::min<float>(1.f, image[i][1])) << " " << (int)(255.f * std::min<float>(1.f, image[i][2])) << " ";
    f << std::endl;
    f.close();

    // Reset img
    image.clear();
    image.resize(w * h);
    fill(image.begin(), image.end(), Vec3(0, 0, 0));
}

Intersection intersect(Vec3 pos, Vec3 dir){

    double seuilMin = 0.005;
    double seuilMax = 10;

    int maxItt = 10;

    bool conv = false;
    bool div  = false;

    int i = 0;
    while(!conv && !div){
        double dist = globalSignedDist(pos);

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