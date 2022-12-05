
#include <cmath>
#include <iostream>

struct cVec3 {
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