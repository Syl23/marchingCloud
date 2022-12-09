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

//void cuda_ray_trace_from_camera(int w, int h, Vec3 (*cameraSpaceToWorldSpace)(const Vec3&), Vec3 (*screen_space_to_worldSpace)(float, float));
void cuda_ray_trace_from_camera(int, int, Vec3 (*)(Vec3 const&), Vec3 (*)(float, float));

std::vector<kd_tree_node> make_kd_tree(std::vector<Vec3> dots);
kd_tree_node* send_kd_tree(std::vector<kd_tree_node> kd_tree);

void printKnn(kd_tree_node * kd_tree, int nb, float x, float y, float z);
void getKnn(kd_tree_node * kd_tree, int nb, Vec3 point, int * ind, float * SqDist);

void testCuda();