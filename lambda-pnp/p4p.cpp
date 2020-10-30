#include <iostream>
#include <limits>
#include "p4p.h"
#include <lambdatwist/lambdatwist.p3p.h>


using std::cout;using std::endl;
namespace cvl{


PoseD p4p(const std::vector<cvl::Vector3D>& xs,
          const std::vector<cvl::Vector2D>& yns,
          Vector4<uint> indexes)
{
    assert(xs.size()==yns.size());
    assert(indexes.size()==4);
    assert(([&]() -> bool{for(uint i:indexes) if (i>=xs.size()) return false; return true;})());




    Vector<cvl::Matrix<double,3,3>,4> Rs;
    Vector<Vector3<double>,4> Ts;

    int valid = p3p_lambdatwist<double,5>(
                yns[indexes[0]].homogeneous(),yns[indexes[1]].homogeneous(),yns[indexes[2]].homogeneous(),
            xs[ indexes[0]],xs[ indexes[1]],xs[ indexes[2]],
            Rs,Ts);

    // check that all of them look ok...
    /*
    for(int v=0;v<valid;++v)
    {
        PoseD P(Rs[v], Ts[v]);
        for(int i=0;i<4;++i){
            double err=((P*xs[indexes[i]]).dehom() - yns[indexes[i]]).squaredNorm();
            std::cout<<"v: "<<v<<" i: "<<i<<" err: "<<err<<std::endl;
        }
    }
    */


    // pick the minimum, whatever it is
    Vector2d y=yns[indexes[3]];
    Vector3d x=xs[indexes[3]];
    PoseD P=PoseD(); // identity
    double e0=std::numeric_limits<double>::max();


    for(int v=0;v<valid;++v)
    {
        // the lambdatwist rotations have a problem some Rs not quite beeing rotations... ???
        // this is extremely uncommon, except when you have very particular kinds of noise,
        // this gets caught by the benchmark somehow?
        // it never occurs for the correct solution, only ever for possible alternatives.
        // investigate numerical problem later...
        Vector4d q=getRotationQuaternion(Rs[v]);
        q.normalize();
        PoseD tmp(q,Ts[v]);

        Vector3d xr=tmp*x;
        if(xr[2]<0) continue;
        double e=((Rs[v]*x + Ts[v]).dehom() - y).absSum();
        if (std::isnan(e)) continue;
        if (e<e0 ){
            P=tmp;
            assert(P.isnormal());
            e0=e;
        }
    }
    assert(P.isnormal());
    // check that all of them look ok...


    double error=0;
    for(int i=0;i<4;++i){
        double err=((P*xs[indexes[i]]).dehom() - yns[indexes[i]]).squaredNorm();
        error+=err;
    }
    assert(!std::isnan(error));

    return P;
}








}// end namespace cvl
