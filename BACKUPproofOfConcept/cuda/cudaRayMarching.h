#include "../src/Vec3.h"
void cuda_ray_trace_from_camera(int w, int h, Vec3 (*cameraSpaceToWorldSpace)(const Vec3&), Vec3 (*screen_space_to_worldSpace)(float, float));

//rempli la mémoire du GPU avec les position et les direction de chaque rayon
//lance une fonction kernel pour calculer le ray marching
//récupère l'image de la mémoire du GPU et en fait une image