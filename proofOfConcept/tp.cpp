// -------------------------------------------
// gMini : a minimal OpenGL/GLUT application
// for 3D graphics.
// Copyright (C) 2006-2008 Tamy Boubekeur
// All rights reserved.
// -------------------------------------------

// -------------------------------------------
// Disclaimer: this code is dirty in the
// meaning that there is no attention paid to
// proper class attribute access, memory
// management or optimisation of any kind. It
// is designed for quick-and-dirty testing
// purpose.
// -------------------------------------------

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include <GL/glut.h>
#include <float.h>
#include "src/Vec3.h"
#include "src/Camera.h"
//#include "src/jmkdtree.h"

#include "src/matrixUtilities.h"


extern "C" struct Material{
    Vec3 AMBIANT_COLOR = Vec3(0,0,0);
    Vec3 DIFFUSE_COLOR= Vec3(0.5,0.5,0.5);
    Vec3 SPECULAR_COLOR= Vec3(0.5,0.5,0.5);

    int SPECULAR_EXPONENT = 32;
    float transparency = 0.0;
    float refractionIndex = 1.0;
};

extern "C" struct kd_tree_node{
    int ind;

    float x;
    float y;
    float z;

    int axis;

    int left;
    int right;
};

extern "C" struct cVec3 {
   float mVals[3];
};

extern "C" struct PointCloudData{
    kd_tree_node* kdTree;
    char* materialIndex;
    Material* materialList;
    cVec3 * positions;
    cVec3 * normals;
};

extern "C" PointCloudData getGPUpcd(std::vector<Vec3> positions, std::vector<Vec3> normals, std::vector<char> materialIndex, std::vector<Material> materialList);

extern "C" void testCuda();
extern "C" std::vector<kd_tree_node> make_kd_tree(std::vector<Vec3> dots);
extern "C" kd_tree_node* send_kd_tree(std::vector<kd_tree_node> kd_tree);

extern "C" void cuda_ray_trace_from_camera(int w, int h, Vec3 (*cameraSpaceToWorldSpace)(const Vec3&), Vec3 (*screen_space_to_worldSpace)(float, float), PointCloudData pcd);



void ray_trace_from_camera();

std::pair<Vec3,Vec3> AABB;

std::vector< Vec3 > positions;
std::vector< Vec3 > normals;

std::vector< Vec3 > positions2;
std::vector< Vec3 > normals2;

std::vector< Vec3 > mesh_pos;
std::vector< unsigned int > mesh_triangles;

std::vector< Vec3 > gridou;
std::vector< Vec3 > gridouNormals;

PointCloudData pcd;


//BasicANNkdTree kdtree;
//kd_tree_node * cudaKd_tree = NULL;



// -------------------------------------------
// OpenGL/GLUT application code.
// -------------------------------------------

static GLint window;
static unsigned int SCREENWIDTH = 116;
static unsigned int SCREENHEIGHT = 126;
static Camera camera;
static bool mouseRotatePressed = false;
static bool mouseMovePressed = false;
static bool mouseZoomPressed = false;
static int lastX=0, lastY=0, lastZoom=0;
static bool fullScreen = false;




// ------------------------------------------------------------------------------------------------------------
// i/o and some stuff
// ------------------------------------------------------------------------------------------------------------
void loadPN (const std::string & filename , std::vector< Vec3 > & o_positions , std::vector< Vec3 > & o_normals ) {
    unsigned int surfelSize = 6;
    FILE * in = fopen (filename.c_str (), "rb");
    if (in == NULL) {
        std::cout << filename << " is not a valid PN file." << std::endl;
        return;
    }
    size_t READ_BUFFER_SIZE = 1000; // for example...
    float * pn = new float[surfelSize*READ_BUFFER_SIZE];
    o_positions.clear ();
    o_normals.clear ();
    while (!feof (in)) {
        unsigned numOfPoints = fread (pn, 4, surfelSize*READ_BUFFER_SIZE, in);
        for (unsigned int i = 0; i < numOfPoints; i += surfelSize) {
            o_positions.push_back (Vec3 (pn[i], pn[i+1], pn[i+2]));
            o_normals.push_back (Vec3 (pn[i+3], pn[i+4], pn[i+5]));
        }

        if (numOfPoints < surfelSize*READ_BUFFER_SIZE) break;
    }
    fclose (in);
    delete [] pn;
}
void savePN (const std::string & filename , std::vector< Vec3 > const & o_positions , std::vector< Vec3 > const & o_normals ) {
    if ( o_positions.size() != o_normals.size() ) {
        std::cout << "The pointset you are trying to save does not contain the same number of points and normals." << std::endl;
        return;
    }
    FILE * outfile = fopen (filename.c_str (), "wb");
    if (outfile == NULL) {
        std::cout << filename << " is not a valid PN file." << std::endl;
        return;
    }
    for(unsigned int pIt = 0 ; pIt < o_positions.size() ; ++pIt) {
        fwrite (&(o_positions[pIt]) , sizeof(float), 3, outfile);
        fwrite (&(o_normals[pIt]) , sizeof(float), 3, outfile);
    }
    fclose (outfile);
}
void scaleAndCenter( std::vector< Vec3 > & io_positions ) {
    Vec3 bboxMin( FLT_MAX , FLT_MAX , FLT_MAX );
    Vec3 bboxMax( FLT_MIN , FLT_MIN , FLT_MIN );
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        for( unsigned int coord = 0 ; coord < 3 ; ++coord ) {
            bboxMin[coord] = std::min<float>( bboxMin[coord] , io_positions[pIt][coord] );
            bboxMax[coord] = std::max<float>( bboxMax[coord] , io_positions[pIt][coord] );
        }
    }
    Vec3 bboxCenter = (bboxMin + bboxMax) / 2.f;
    float bboxLongestAxis = std::max<float>( bboxMax[0]-bboxMin[0] , std::max<float>( bboxMax[1]-bboxMin[1] , bboxMax[2]-bboxMin[2] ) );
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        io_positions[pIt] = (io_positions[pIt] - bboxCenter) / bboxLongestAxis;
    }
}

void applyRandomRigidTransformation( std::vector< Vec3 > & io_positions , std::vector< Vec3 > & io_normals ) {
    srand(time(NULL));
    Mat3 R = Mat3::RandRotation();
    Vec3 t = Vec3::Rand(1.f);
    for(unsigned int pIt = 0 ; pIt < io_positions.size() ; ++pIt) {
        io_positions[pIt] = R * io_positions[pIt] + t;
        io_normals[pIt] = R * io_normals[pIt];
    }
}

void subsample( std::vector< Vec3 > & i_positions , std::vector< Vec3 > & i_normals , float minimumAmount = 0.1f , float maximumAmount = 0.2f ) {
    std::vector< Vec3 > newPos , newNormals;
    std::vector< unsigned int > indices(i_positions.size());
    for( unsigned int i = 0 ; i < indices.size() ; ++i ) indices[i] = i;
    srand(time(NULL));
    std::random_shuffle(indices.begin() , indices.end());
    unsigned int newSize = indices.size() * (minimumAmount + (maximumAmount-minimumAmount)*(float)(rand()) / (float)(RAND_MAX));
    newPos.resize( newSize );
    newNormals.resize( newSize );
    for( unsigned int i = 0 ; i < newPos.size() ; ++i ) {
        newPos[i] = i_positions[ indices[i] ];
        newNormals[i] = i_normals[ indices[i] ];
    }
    i_positions = newPos;
    i_normals = newNormals;
}

bool save( const std::string & filename , std::vector< Vec3 > & vertices , std::vector< unsigned int > & triangles ) {
    std::ofstream myfile;
    myfile.open(filename.c_str());
    if (!myfile.is_open()) {
        std::cout << filename << " cannot be opened" << std::endl;
        return false;
    } 

    myfile << "OFF" << std::endl;

    unsigned int n_vertices = vertices.size() , n_triangles = triangles.size()/3;
    myfile << n_vertices << " " << n_triangles << " 0" << std::endl;

    for( unsigned int v = 0 ; v < n_vertices ; ++v ) {
        myfile << vertices[v][0] << " " << vertices[v][1] << " " << vertices[v][2] << std::endl;
    }
    for( unsigned int f = 0 ; f < n_triangles ; ++f ) {
        myfile << 3 << " " << triangles[3*f] << " " << triangles[3*f+1] << " " << triangles[3*f+2];
        myfile << std::endl;
    }
    myfile.close();
    return true;
}



// ------------------------------------------------------------------------------------------------------------
// rendering.
// ------------------------------------------------------------------------------------------------------------

void initLight () {
    GLfloat light_position1[4] = {22.0f, 16.0f, 50.0f, 0.0f};
    GLfloat direction1[3] = {-52.0f,-16.0f,-50.0f};
    GLfloat color1[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    GLfloat ambient[4] = {0.3f, 0.3f, 0.3f, 0.5f};

    glLightfv (GL_LIGHT1, GL_POSITION, light_position1);
    glLightfv (GL_LIGHT1, GL_SPOT_DIRECTION, direction1);
    glLightfv (GL_LIGHT1, GL_DIFFUSE, color1);
    glLightfv (GL_LIGHT1, GL_SPECULAR, color1);
    glLightModelfv (GL_LIGHT_MODEL_AMBIENT, ambient);
    glEnable (GL_LIGHT1);
    glEnable (GL_LIGHTING);
}

void init () {
    camera.resize (SCREENWIDTH, SCREENHEIGHT);
    initLight ();
    //glCullFace (GL_BACK);
    //glEnable (GL_CULL_FACE);
    glDepthFunc (GL_LESS);
    glEnable (GL_DEPTH_TEST);
    glClearColor (0.2f, 0.2f, 0.3f, 1.0f);
    glEnable(GL_COLOR_MATERIAL);
}



void drawTriangleMesh( std::vector< Vec3 > const & i_positions , std::vector< unsigned int > const & i_triangles ) {
    glBegin(GL_TRIANGLES);
    for(unsigned int tIt = 0 ; tIt < i_triangles.size() / 3 ; ++tIt) {
        Vec3 p0 = i_positions[i_triangles[3*tIt]];
        Vec3 p1 = i_positions[i_triangles[3*tIt+1]];
        Vec3 p2 = i_positions[i_triangles[3*tIt+2]];
        Vec3 n = Vec3::cross(p1-p0 , p2-p0);
        n.normalize();
        glNormal3f( n[0] , n[1] , n[2] );
        glVertex3f( p0[0] , p0[1] , p0[2] );
        glVertex3f( p1[0] , p1[1] , p1[2] );
        glVertex3f( p2[0] , p2[1] , p2[2] );
    }
    glEnd();
}

void drawPointSet( std::vector< Vec3 > const & i_positions , std::vector< Vec3 > const & i_normals ) {
    glBegin(GL_POINTS);
    for(unsigned int pIt = 0 ; pIt < i_positions.size() ; ++pIt) {
        glNormal3f( i_normals[pIt][0] , i_normals[pIt][1] , i_normals[pIt][2] );
        glVertex3f( i_positions[pIt][0] , i_positions[pIt][1] , i_positions[pIt][2] );
    }
    glEnd();
}

void draw () {
    glPointSize(2); // for example...

    glColor3f(0.8,0.8,1);
    drawPointSet(positions , normals);

    //glColor3f(1,0.5,0.5);
    //drawPointSet(positions2 , normals2);

    //glColor3f(0.5,1.0,0.5);
    //drawPointSet(gridou, gridouNormals);

    //drawTriangleMesh(mesh_pos , mesh_triangles );
}








void display () {
    glLoadIdentity ();
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    camera.apply ();
    draw ();
    glFlush ();
    glutSwapBuffers ();
}

void idle () {
    glutPostRedisplay ();
}

void key (unsigned char keyPressed, int x, int y) {
    switch (keyPressed) {
    case 'f':
        if (fullScreen == true) {
            glutReshapeWindow (SCREENWIDTH, SCREENHEIGHT);
            fullScreen = false;
        } else {
            glutFullScreen ();
            fullScreen = true;
        }
        break;

    case 'w':
        GLint polygonMode[2];
        glGetIntegerv(GL_POLYGON_MODE, polygonMode);
        if(polygonMode[0] != GL_FILL)
            glPolygonMode (GL_FRONT_AND_BACK, GL_FILL);
        else
            glPolygonMode (GL_FRONT_AND_BACK, GL_LINE);
        break;

    case 'r':
        long unsigned int used, free;
        camera.apply();

        cudaMemGetInfo(&used, &free);
        std::cout << "Avant\n"<<"Mémoire GPU utilisée : " << used / 1073741824.0 << " Go" << std::endl;
        std::cout << "Mémoire GPU disponible : " << free / 1073741824.0 << " Go" << std::endl;

        cuda_ray_trace_from_camera(glutGet(GLUT_WINDOW_WIDTH),glutGet(GLUT_WINDOW_HEIGHT),&cameraSpaceToWorldSpace, &screen_space_to_worldSpace, pcd);
        
        cudaMemGetInfo(&used, &free);
        std::cout << "Après\n"<<"Mémoire GPU utilisée : " << used / 1073741824.0 << " Go" << std::endl;
        std::cout << "Mémoire GPU disponible : " << free / 1073741824.0 << " Go" << std::endl;

        //ray_trace_from_camera();
        break;

    default:
        break;
    }
    idle ();
}

void mouse (int button, int state, int x, int y) {
    if (state == GLUT_UP) {
        mouseMovePressed = false;
        mouseRotatePressed = false;
        mouseZoomPressed = false;
    } else {
        if (button == GLUT_LEFT_BUTTON) {
            camera.beginRotate (x, y);
            mouseMovePressed = false;
            mouseRotatePressed = true;
            mouseZoomPressed = false;
        } else if (button == GLUT_RIGHT_BUTTON) {
            lastX = x;
            lastY = y;
            mouseMovePressed = true;
            mouseRotatePressed = false;
            mouseZoomPressed = false;
        } else if (button == GLUT_MIDDLE_BUTTON) {
            if (mouseZoomPressed == false) {
                lastZoom = y;
                mouseMovePressed = false;
                mouseRotatePressed = false;
                mouseZoomPressed = true;
            }
        }
    }
    idle ();
}

void motion (int x, int y) {
    if (mouseRotatePressed == true) {
        //std::cout<<"x : "<<x<<" y : "<<y<<std::endl;
        camera.rotate (x, y);
    }
    else if (mouseMovePressed == true) {
        camera.move ((x-lastX)/static_cast<float>(SCREENWIDTH), (lastY-y)/static_cast<float>(SCREENHEIGHT), 0.0);
        lastX = x;
        lastY = y;
    }
    else if (mouseZoomPressed == true) {
        camera.zoom (float (y-lastZoom)/SCREENHEIGHT);
        lastZoom = y;
    }
}


void reshape(int w, int h) {
    camera.resize (w, h);
}

Vec3 project(Vec3 point,Vec3 normalePlan,Vec3 pointPlan){
    return(point - Vec3::dot(point - pointPlan, normalePlan)*normalePlan);
} 

// void HPSS(
//     Vec3 inputPoint,
//     Vec3 & outputPoint,
//     Vec3 & outputNormal,
//     std::vector<Vec3> const & positions,
//     std::vector<Vec3> const & normals,
//     BasicANNkdTree const & kdtree,
//     int kerneltype,
//     float h,
//     unsigned int nbIterations=10,
//     unsigned int knn= 20)
// {
//     ANNidxArray id_nearest_neighbors           = new ANNidx[knn];
//     ANNdistArray square_distances_to_neighbors = new ANNdist[knn];

//     Vec3 precPoint = inputPoint;

//     Vec3 nextPoint;
//     Vec3 nextNormal;

//     for(int itt = 0 ; itt < nbIterations ; itt++){

//         kdtree.knearest(precPoint, knn,id_nearest_neighbors,square_distances_to_neighbors);

//         //getKnn(cudaKd_tree, knn, precPoint[0],precPoint[1],precPoint[2], id_nearest_neighbors, (float *)square_distances_to_neighbors);


//         nextPoint  = Vec3(0,0,0);
//         nextNormal = Vec3(0,0,0);

//         float totWeight = 0.0;

//         for(int i = 0 ; i<knn ; i ++){

//             auto proj = project(precPoint,normals[id_nearest_neighbors[i]],positions[id_nearest_neighbors[i]]);
//             float weight = 0.0;
//             float r = sqrt(square_distances_to_neighbors[i])/h;
//             switch (kerneltype){
//             case 0:
                
//                 weight = exp((-(r*r))/(h*h));
//                 break;
//             case 1:
//                 weight = 0;
//                 break;
//             case 2:
//                 weight = 0;
//                 break;
//             }
//             totWeight  += weight;
//             nextPoint  += weight*proj;
//             nextNormal += weight*normals[id_nearest_neighbors[i]];
//         }
//         nextPoint = nextPoint / totWeight;
//         nextNormal.normalize();
//         precPoint = nextPoint;
//     }

//     outputPoint = nextPoint;
//     outputNormal = nextNormal;
    


// } 

float randFloat(){
    return(((float)random())/((float)RAND_MAX));
}

void makeSomeNormaleNoise(std::vector<Vec3> & pos, std::vector<Vec3> & norm,float strength){
    for(int i = 0 ; i < pos.size() ; i ++){
        auto rand = (randFloat()*2-1)*strength;
        //std::cout<<"rand : "<<rand<<std::endl;
        pos[i] = pos[i]+rand*norm[i];
        //tab[i].normalize();
    }
}

double mapVal(double x, double minA, double maxA, double minB, double maxB){
    return(((x-minA)/(maxA-minA)) * (maxB-minB) + minB);
}

// double signedDist(
//     Vec3 inputPoint,
//     std::vector<Vec3> const & positions,
//     std::vector<Vec3> const & normals,
//     BasicANNkdTree const & kdtree,
//     int kerneltype,
//     float h = 100.0,
//     unsigned int nbIterations=10,
//     unsigned int knn= 20)
// {
//     Vec3 projPoint = Vec3(0,0,0);
//     Vec3 projNormal= Vec3(0,0,0);
    
    
//     HPSS(inputPoint,projPoint,projNormal,positions,normals,kdtree,kerneltype,h,nbIterations,knn);


    

//     // std::cout<<"inputPoint : "<<inputPoint<<std::endl;
//     // std::cout<<"projPoint : "<<projPoint<<std::endl;
//     // std::cout<<"projNormal : "<<projNormal<<std::endl;

//     return(Vec3::dot(inputPoint-projPoint,projNormal));
// }

// double globalSignedDist(Vec3 pos){
//     return signedDist(pos,positions,normals,kdtree,0,100.0,5,10);
// }

// void flipNormales(

//     std::vector< Vec3 > & mesh_positions ,
//     std::vector< unsigned int > & triangles,
//     std::vector<Vec3> const &  positions,
//     std::vector<Vec3> const & normals,
//     BasicANNkdTree const & kdtree,
//     int res = 32,
//     int kerneltype = 0,    
//     float h = 100.0){

//     for(int i = 0 ; i < triangles.size()/3 ; i ++){
//         Vec3 p0 = mesh_positions[triangles[3*i]];
//         Vec3 p1 = mesh_positions[triangles[3*i+1]];
//         Vec3 p2 = mesh_positions[triangles[3*i+2]];
//         Vec3 n = Vec3::cross(p1-p0 , p2-p0);
//         n.normalize();

//         Vec3 pm = (p0+p1+p2)/3.0 + (p1-p0).length()*0.5*n;

//         auto tmpSgn = signedDist(pm, positions,normals,kdtree, kerneltype)>0.0;

//         if(!tmpSgn){
//             auto tmp = triangles[3*i +0];

//             triangles[3*i +0] = triangles[3*i +1];
//             triangles[3*i +1] = tmp;
//         }
//     }
//     }

void exportOFF(std::vector< Vec3 > & mesh_positions ,std::vector< unsigned int > & triangles){
    std::ofstream myfile;
    myfile.open ("object.off");
    myfile <<"OFF\n";
    myfile <<mesh_positions.size()<<" "<<mesh_triangles.size()/3<<" 0\n";

    for(auto p : mesh_positions){
        myfile<<p[0]<<" "<<p[1]<<" "<<p[2]<<"\n";
    }
    for(int i = 0 ; i < mesh_triangles.size()/3 ; i ++){
        myfile<<"3 "<<mesh_triangles[3*i +0]<<" "<<mesh_triangles[3*i +1]<<" "<<mesh_triangles[3*i +2]<<"\n";
    }
    myfile<<"\n";
    myfile.close();
}


// void setNormales(
//     std::vector< Vec3 > & mesh_positions ,
//     std::vector< unsigned int > & triangles,
//     std::vector< Vec3> & meshNormals,
//     std::vector<Vec3> const &  positions,
//     std::vector<Vec3> const & normals,
//     BasicANNkdTree const & kdtree,
//     int res = 32,
//     int kerneltype = 0,    
//     float h = 100.0){

//     for(int i = 0 ; i < triangles.size()/3 ; i ++){
//         Vec3 p0 = mesh_positions[triangles[3*i]];
//         Vec3 p1 = mesh_positions[triangles[3*i+1]];
//         Vec3 p2 = mesh_positions[triangles[3*i+2]];

//         Vec3 n = Vec3::cross(p1-p0 , p2-p0);
//         n.normalize();

//         Vec3 projNorm = meshNormals[triangles[3*i]] + mesh_positions[triangles[3*i+1]] + mesh_positions[triangles[3*i+2]];
//         projNorm.normalize();

//         bool flip = Vec3::dot(n,projNorm) < 0.0 ;

//         if(flip){
//             auto tmp = triangles[3*i +0];

//             triangles[3*i +0] = triangles[3*i +1];
//             triangles[3*i +1] = tmp;
//         }
//     }

// }

void subdivide(std::vector< Vec3 > & positions ,std::vector< unsigned int > & triangles){
    auto tmpTri = std::vector<unsigned int>();
    for(int i = 0 ; i < triangles.size()/3 ; i ++){
        int pi0 = triangles[3*i];
        int pi1 = triangles[3*i+1];
        int pi2 = triangles[3*i+2];

        Vec3 p3 = (1.0/2.0) * (positions[pi0]+positions[pi1]);
        Vec3 p4 = (1.0/2.0) * (positions[pi1]+positions[pi2]);
        Vec3 p5 = (1.0/2.0) * (positions[pi2]+positions[pi0]);

        int pi3 = positions.size();
        positions.push_back(p3);

        int pi4 = positions.size();
        positions.push_back(p4);

        int pi5 = positions.size();
        positions.push_back(p5);

        tmpTri.push_back(pi0);
        tmpTri.push_back(pi3);
        tmpTri.push_back(pi5);

        tmpTri.push_back(pi3);
        tmpTri.push_back(pi1);
        tmpTri.push_back(pi4);

        tmpTri.push_back(pi5);
        tmpTri.push_back(pi4);
        tmpTri.push_back(pi2);

        tmpTri.push_back(pi3);
        tmpTri.push_back(pi4);
        tmpTri.push_back(pi5);
    }

    triangles.resize(tmpTri.size());
    for(int i = 0 ; i < tmpTri.size() ; i ++){
        triangles[i] = tmpTri[i];
    }

}

// void dualContouring(
//     std::vector< Vec3 > & mesh_positions ,
//     std::vector< unsigned int > & triangles,
//     std::vector<Vec3> const &  positions,
//     std::vector<Vec3> const & normals,
//     BasicANNkdTree const & kdtree,
//     int res = 32,
//     int kerneltype = 0,    
//     float h = 100.0){


//     auto hmap = std::vector<std::vector<std::vector<int>>>(res,std::vector<std::vector<int>>(res,std::vector<int>(res,-1)));

    
//     Vec3 maxVec = positions[0];
//     Vec3 minVec = positions[0];

//     for(auto v : positions){
//         for(int i = 0 ; i<3 ; i ++){
//             maxVec[i] = std::max(maxVec[i],v[i]);
//             minVec[i] = std::min(minVec[i],v[i]);
//         }
//     }

//     Vec3 cellSize = (maxVec-minVec) / res;


//     int vertInd = 0;
//     for(int ix = 0 ; ix < res ; ix ++){
//         for(int iy = 0 ; iy < res ; iy ++){
//             for(int iz = 0 ; iz < res ; iz ++){

//                 double x = mapVal(ix,0,res,minVec[0],maxVec[0]);
//                 double y = mapVal(iy,0,res,minVec[1],maxVec[1]);
//                 double z = mapVal(iz,0,res,minVec[2],maxVec[2]);

//                 Vec3 point = Vec3(x, y, z);

//                 std::vector<Vec3> cubePoints = std::vector<Vec3>();

//                 for(int dx = 0 ; dx <= 1 ; dx ++)for(int dy = 0 ; dy <= 1 ; dy ++)for(int dz = 0 ; dz <= 1 ; dz ++){
//                     cubePoints.push_back(point + cellSize*Vec3(dx,dy,dz));
//                 }

//                 bool cachange = false;
                
//                 bool sgn = signedDist(cubePoints[0], positions,normals,kdtree, kerneltype)>0.0;
//                 for(int i = 1 ; i < cubePoints.size() ; i ++){
//                     auto dist = signedDist(cubePoints[i], positions,normals,kdtree, kerneltype);
//                     bool tmpSgn = dist>0.0;

//                     if(sgn != tmpSgn){
//                         cachange = true;
//                     }else{
//                     }
//                 } 

//                 if(cachange){
//                     hmap[ix][iy][iz] = vertInd;
//                     vertInd++;

//                     mesh_positions.push_back(point+0.5*cellSize);

//                     gridou.push_back(point+0.5*cellSize);
//                     gridouNormals.push_back(Vec3(1,1,1));
//                 }
//             }
//         }
//     }

//     for(int ix = 0 ; ix < res ; ix ++){
//         for(int iy = 0 ; iy < res ; iy ++){
//             for(int iz = 0 ; iz < res ; iz ++){
//                 if(ix + 1 < res && iy + 1 < res && iz < res &&
//                 hmap[ix][iy][iz]!=-1 && hmap[ix+1][iy][iz]!=-1 && hmap[ix+1][iy+1][iz]!=-1 && hmap[ix][iy+1][iz] != -1){

//                     triangles.push_back(hmap[ix][iy][iz]);
//                     triangles.push_back(hmap[ix+1][iy][iz]);
//                     triangles.push_back(hmap[ix+1][iy+1][iz]);

//                     triangles.push_back(hmap[ix][iy][iz]);
//                     triangles.push_back(hmap[ix+1][iy+1][iz]);
//                     triangles.push_back(hmap[ix][iy+1][iz]);
//                 }

//                 if(ix + 1 < res && iy < res && iz+1 < res &&
//                 hmap[ix][iy][iz]!=-1 && hmap[ix+1][iy][iz]!=-1 && hmap[ix+1][iy][iz+1]!=-1 && hmap[ix][iy][iz+1] != -1){

//                     triangles.push_back(hmap[ix][iy][iz]);
//                     triangles.push_back(hmap[ix+1][iy][iz]);
//                     triangles.push_back(hmap[ix+1][iy][iz+1]);

//                     triangles.push_back(hmap[ix][iy][iz]);
//                     triangles.push_back(hmap[ix+1][iy][iz+1]);
//                     triangles.push_back(hmap[ix][iy][iz+1]);
//                 }

//                 if(ix < res && iy +1 < res && iz+1 < res &&
//                 hmap[ix][iy][iz]!=-1 && hmap[ix][iy+1][iz]!=-1 && hmap[ix][iy+1][iz+1]!=-1 && hmap[ix][iy][iz+1] != -1){

//                     triangles.push_back(hmap[ix][iy][iz]);
//                     triangles.push_back(hmap[ix][iy+1][iz]);
//                     triangles.push_back(hmap[ix][iy+1][iz+1]);

//                     triangles.push_back(hmap[ix][iy][iz]);
//                     triangles.push_back(hmap[ix][iy+1][iz+1]);
//                     triangles.push_back(hmap[ix][iy][iz+1]);
//                 }
//             }
//         }
//     }
// }


//#include "src/rayTracing.h"


int main (int argc, char ** argv) {
    if (argc > 2) {
        exit (EXIT_FAILURE);
    }
    glutInit (&argc, argv);
    glutInitDisplayMode (GLUT_RGBA | GLUT_DEPTH | GLUT_DOUBLE);
    glutInitWindowSize (SCREENWIDTH, SCREENHEIGHT);
    window = glutCreateWindow ("tp point processing");

    init ();
    glutIdleFunc (idle);
    glutDisplayFunc (display);
    glutKeyboardFunc (key);
    glutMotionFunc(motion);
    glutMouseFunc(mouse);
    glutReshapeFunc (reshape);// TODO
    key ('?', 0, 0);


    {
        // Load a first pointset, and build a kd-tree:
        //loadPN("pointsets/igea_subsampled_extreme.pn" , positions , normals);


        loadPN("pointsets/igea.pn" , positions , normals);

        auto materialIndex = std::vector<char>(positions.size(),0);
        auto materialList = std::vector<Material>();


        int size = positions.size();
        for(int i = 0 ; i < size ; i++){
            positions.push_back(positions[i] + Vec3(0.8,0,0));
            normals.push_back(normals[i]);
            materialIndex.push_back(1);
        }


        Material gold;
        gold.AMBIANT_COLOR = Vec3(0.24725, 0.1995, 0.0745);
        gold.DIFFUSE_COLOR = Vec3(0.75164, 0.60648, 0.22648);
        gold.SPECULAR_COLOR = Vec3(0.628281, 0.555802, 0.366065);
        gold.SPECULAR_EXPONENT = 51.2;
        gold.transparency = 0.0;
        gold.refractionIndex = 0.0;

        Material silver;
        silver.AMBIANT_COLOR = Vec3(0.19225, 0.19225, 0.19225);
        silver.DIFFUSE_COLOR = Vec3(0.50754, 0.50754, 0.50754);
        silver.SPECULAR_COLOR = Vec3(0.508273, 0.508273, 0.508273);
        silver.SPECULAR_EXPONENT = 51.2;
        silver.transparency = 0.0;
        silver.refractionIndex = 0.0;

        Material emerald;
        emerald.AMBIANT_COLOR = Vec3(0.0215, 0.1745, 0.0215);
        emerald.DIFFUSE_COLOR = Vec3(0.07568, 0.61424, 0.07568);
        emerald.SPECULAR_COLOR = Vec3(0.633, 0.727811, 0.633);
        emerald.SPECULAR_EXPONENT = 76.8;
        emerald.transparency = 0.0;
        emerald.refractionIndex = 0.0;

        Material obsidian;
        obsidian.AMBIANT_COLOR = Vec3(0.05375, 0.05, 0.06625);
        obsidian.DIFFUSE_COLOR = Vec3(0.18275, 0.17, 0.22525);
        obsidian.SPECULAR_COLOR = Vec3(0.332741, 0.328634, 0.346435);
        obsidian.SPECULAR_EXPONENT = 38.4;
        obsidian.transparency = 0.0;
        obsidian.refractionIndex = 0.0;

        Material metal;
        metal.AMBIANT_COLOR = Vec3(0.3, 0.3, 0.3);
        metal.DIFFUSE_COLOR = Vec3(0.4, 0.4, 0.4);
        metal.SPECULAR_COLOR = Vec3(0.8, 0.8, 0.8);
        metal.SPECULAR_EXPONENT = 128;
        metal.transparency = 0.0;
        metal.refractionIndex = 0.0;

        Material rubber;
        rubber.AMBIANT_COLOR = Vec3(0.05, 0.05, 0.05);
        rubber.DIFFUSE_COLOR = Vec3(0.5, 0.5, 0.5);
        rubber.SPECULAR_COLOR = Vec3(0.1, 0.1, 0.1);
        rubber.SPECULAR_EXPONENT = 4;
        rubber.transparency = 0.0;
        rubber.refractionIndex = 0.0;

        Material marble;
        marble.AMBIANT_COLOR = Vec3(0.1, 0.1, 0.1);
        marble.DIFFUSE_COLOR = Vec3(0.4, 0.4, 0.4);
        marble.SPECULAR_COLOR = Vec3(0.8, 0.8, 0.8);
        marble.SPECULAR_EXPONENT = 32;
        marble.transparency = 0.0;
        marble.refractionIndex = 0.0;

        Material chalk;
        chalk.AMBIANT_COLOR = Vec3(0.1, 0.1, 0.1);
        chalk.DIFFUSE_COLOR = Vec3(0.9, 0.9, 0.9);
        chalk.SPECULAR_COLOR = Vec3(0.2, 0.2, 0.2);
        chalk.SPECULAR_EXPONENT = 8;
        chalk.transparency = 0.0;
        chalk.refractionIndex = 0.0;

        Material glass;
        glass.AMBIANT_COLOR = Vec3(0.05, 0.05, 0.05);
        glass.DIFFUSE_COLOR = Vec3(0.1, 0.1, 0.1);
        glass.SPECULAR_COLOR = Vec3(0.9, 0.9, 0.9);
        glass.SPECULAR_EXPONENT = 128;
        glass.transparency = 0.9;
        glass.refractionIndex = 1.5;

        Material plastic;
        plastic.AMBIANT_COLOR = Vec3(0.1, 0.1, 0.1);
        plastic.DIFFUSE_COLOR = Vec3(0.6, 0.6, 0.6);
        plastic.SPECULAR_COLOR = Vec3(0.9, 0.9, 0.9);
        plastic.SPECULAR_EXPONENT = 64;
        plastic.transparency = 0.7;
        plastic.refractionIndex = 1.3;

        Material water;
        water.AMBIANT_COLOR = Vec3(0.1, 0.1, 0.1);
        water.DIFFUSE_COLOR = Vec3(0.4, 0.4, 0.4);
        water.SPECULAR_COLOR = Vec3(0.8, 0.8, 0.8);
        water.SPECULAR_EXPONENT = 32;
        water.transparency = 0.9;
        water.refractionIndex = 1.333;



        materialList.push_back(gold);
        materialList.push_back(emerald);


        //cudaMain();
        //testCuda();

        /*
        for(auto p : positions){
            AABB.first[0] = min(p[0],AABB.first[0]);
            AABB.first[1] = min(p[1],AABB.first[1]);
            AABB.first[2] = min(p[2],AABB.first[2]);

            AABB.second[0] = min(p[0],AABB.second[0]);
            AABB.second[1] = min(p[1],AABB.second[1]);
            AABB.second[2] = min(p[2],AABB.second[2]);
        }
        */

        Vec3 testPoint = Vec3(0,0,1);
        int nbnn = 20;


        //loadPN("pointsets/dino_subsampled_extreme.pn" , positions2 , normals2);
//        kdtree.build(positions);

        pcd = getGPUpcd(positions,normals,materialIndex,materialList);

        // std::cout<<"Start kd-tree building"<<std::endl;
        // auto my_kd_tree = make_kd_tree(positions);
        // auto cudaKd_tree = send_kd_tree(my_kd_tree);
        // std::cout<<"End kd-tree building"<<std::endl;




        /* test

            ANNidxArray id_nearest_neighbors           = new ANNidx[nbnn];
            ANNdistArray square_distances_to_neighbors = new ANNdist[nbnn];

            int* id_nearest_neighborsGPU           = new int[nbnn];
            float* square_distances_to_neighborsGPU = new float[nbnn];

            kdtree.knearest(testPoint, nbnn,id_nearest_neighbors,square_distances_to_neighbors);
            std::cout<<"CPU : "<<std::endl<<std::endl;
            for(int i = 0 ; i < nbnn ; i ++){
                std::cout<<"index : "<<id_nearest_neighbors[i]<<" dist : "<<square_distances_to_neighbors[i]<<std::endl;
            }

            std::cout<<std::endl<<"GPU : "<<std::endl<<std::endl;

            getKnn(cudaKd_tree, nbnn, testPoint[0],testPoint[1],testPoint[2], id_nearest_neighborsGPU, square_distances_to_neighborsGPU);

            for(int i = 0 ; i < nbnn ; i ++){
                std::cout<<"index : "<<id_nearest_neighborsGPU[i]<<" dist : "<<square_distances_to_neighborsGPU[i]<<std::endl;
            }
        */


        // Create a second pointset that is artificial, and project it on pointset1 using MLS techniques:
        positions2.resize( 20000 );
        normals2.resize(positions2.size());
        for( unsigned int pIt = 0 ; pIt < positions2.size() ; ++pIt ) {
            positions2[pIt] = Vec3(
                        -0.6 + 1.2 * (double)(rand())/(double)(RAND_MAX),
                        -0.6 + 1.2 * (double)(rand())/(double)(RAND_MAX),
                        -0.6 + 1.2 * (double)(rand())/(double)(RAND_MAX)
                        );
            positions2[pIt].normalize();
            positions2[pIt] = 0.6 * positions2[pIt];
        }

        // PROJECT USING MLS (HPSS and APSS):

        //makeSomeNormaleNoise(positions, normals,0.05);

        // for(int i = 0 ; i < positions2.size() ; i ++){
        //     HPSS(positions2[i],positions2[i],normals2[i],positions,normals,kdtree,0,1000,10,20);
        // }

        //exportOFF(mesh_pos, mesh_triangles);

    }


    glutMainLoop ();
    return EXIT_SUCCESS;
}

