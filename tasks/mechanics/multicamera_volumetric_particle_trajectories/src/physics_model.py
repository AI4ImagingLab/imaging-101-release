"""Explicit local C++/STL numerical mathematics, compiled without upstream code.

The CPU Windows MSVC exception is documented in validation. Every camera,
image, PSF, stereo matching, coordinate update and trajectory operation is
readable below. No OpenLPT installation, header, binary or runtime download.
"""
from pathlib import Path
import ctypes,hashlib,json,os,subprocess,tempfile,time
import numpy as np
KERNEL_SOURCE = r'''// Explicit tracer-only numerical operators extracted from pyOpenLPT.
// Copyright OpenLPT contributors, MIT; original notices retained in final NOTICE.
// This file is an implementation work product, not an upstream library wrapper.
// Standard C++/STL only; no original headers, classes, TIFF reader or domain API.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#define API extern "C" __declspec(dllexport)
using std::vector;using std::array;
constexpr double SMALL=1e-8,LOG_SMALL=1e-4;
struct P2 {double x=0,y=0;double&operator[](int i){return i?y:x;}double operator[](int i)const{return i?y:x;}};
struct P3 {double x=0,y=0,z=0;double&operator[](int i){return i==0?x:i==1?y:z;}double operator[](int i)const{return i==0?x:i==1?y:z;}};
P2 operator+(P2 a,P2 b){return{a.x+b.x,a.y+b.y};} P2 operator-(P2 a,P2 b){return{a.x-b.x,a.y-b.y};}P2 operator*(P2 a,double s){return{a.x*s,a.y*s};}
P3 operator+(P3 a,P3 b){return{a.x+b.x,a.y+b.y,a.z+b.z};}P3 operator-(P3 a,P3 b){return{a.x-b.x,a.y-b.y,a.z-b.z};}P3 operator*(P3 a,double s){return{a.x*s,a.y*s,a.z*s};}P3 operator/(P3 a,double s){return a*(1/s);}
double dot(P3 a,P3 b){return a.x*b.x+a.y*b.y+a.z*b.z;}double dot(P2 a,P2 b){return a.x*b.x+a.y*b.y;}
double norm(P3 a){return std::sqrt(dot(a,a));}double norm(P2 a){return std::sqrt(dot(a,a));}
P3 cross(P3 a,P3 b){return{a.y*b.z-a.z*b.y,a.z*b.x-a.x*b.z,a.x*b.y-a.y*b.x};}
struct Image {int rows=0,cols=0;vector<double>v;Image()=default;Image(int r,int c,double x=0):rows(r),cols(c),v(size_t(r)*c,x){}double&operator()(int r,int c){return v[size_t(r)*cols+c];}double operator()(int r,int c)const{return v[size_t(r)*cols+c];}};
struct Camera {array<double,9>K,R,Rinv;P3 t,C;array<double,5>d;bool active=true;};
P3 multiply(const array<double,9>&A,P3 p){return{A[0]*p.x+A[1]*p.y+A[2]*p.z,A[3]*p.x+A[4]*p.y+A[5]*p.z,A[6]*p.x+A[7]*p.y+A[8]*p.z};}
Camera decode_camera(const double*p){Camera c;std::copy(p,p+9,c.K.begin());p+=9;std::copy(p,p+9,c.R.begin());p+=9;std::copy(p,p+9,c.Rinv.begin());p+=9;c.t={p[0],p[1],p[2]};p+=3;c.C={p[0],p[1],p[2]};p+=3;std::copy(p,p+5,c.d.begin());return c;}
P2 distort(P2 p,const Camera&c){double x=p.x,y=p.y,r2=x*x+y*y,r4=r2*r2,r6=r4*r2,scale=1+c.d[0]*r2+c.d[1]*r4+c.d[4]*r6;double xd=x*scale+c.d[2]*2*x*y+c.d[3]*(r2+2*x*x),yd=y*scale+c.d[2]*(r2+2*y*y)+c.d[3]*2*x*y;return{xd*c.K[0]+c.K[2],yd*c.K[4]+c.K[5]};}
P2 project(P3 p,const Camera&c){P3 q=multiply(c.R,p)+c.t;return distort({q.z?q.x/q.z:q.x,q.z?q.y/q.z:q.y},c);}
struct Ray {P3 origin,direction;};
Ray ray(P2 p,const Camera&c){double x0=(p.x-c.K[2])/c.K[0],y0=(p.y-c.K[5])/c.K[4],x=x0,y=y0;bool distorted=std::any_of(c.d.begin(),c.d.end(),[](double v){return v!=0;});if(distorted){for(int it=0;it<50;it++){double r2=x*x+y*y,inv=1/(1+((c.d[4]*r2+c.d[1])*r2+c.d[0])*r2);if(inv<0){x=x0;y=y0;break;}double dx=c.d[2]*2*x*y+c.d[3]*(r2+2*x*x),dy=c.d[2]*(r2+2*y*y)+c.d[3]*2*x*y;x=(x0-dx)*inv;y=(y0-dy)*inv;if(norm(distort({x,y},c)-p)<=1e-5)break;}}P3 end=multiply(c.Rinv,{x,y,1})+c.C;P3 direction=end-c.C;return{c.C,direction/norm(direction)};}
P3 solve3(array<double,9>A,P3 b){ // General partial-pivot Gaussian elimination.
 double m[3][4];for(int i=0;i<3;i++){for(int j=0;j<3;j++)m[i][j]=A[i*3+j];m[i][3]=b[i];}
 for(int k=0;k<3;k++){int pivot=k;for(int i=k+1;i<3;i++)if(std::abs(m[i][k])>std::abs(m[pivot][k]))pivot=i;if(std::abs(m[pivot][k])<1e-20)throw std::runtime_error("singular3x3 geometry");for(int j=k;j<4;j++)std::swap(m[k][j],m[pivot][j]);for(int i=k+1;i<3;i++){double scale=m[i][k]/m[k][k];for(int j=k;j<4;j++)m[i][j]-=scale*m[k][j];}}
 P3 x;for(int i=2;i>=0;i--){double sum=m[i][3];for(int j=i+1;j<3;j++)sum-=m[i][j]*x[j];x[i]=sum/m[i][i];}return x;
}
array<double,9> inverse3_original(array<double,9>u){
 array<int,3>rows{0,1,2};array<double,3>scale;array<double,9>b{1,0,0,0,1,0,0,0,1},result=b;
 for(int i=0;i<3;i++)scale[i]=*std::max_element(u.begin()+3*i,u.begin()+3*i+3);
 for(int j=0;j<2;j++){double best=0;int pivot=0;for(int i=j;i<3;i++){double value=std::abs(u[rows[i]*3+j]/scale[rows[i]]);if(value>best){best=value;pivot=i;}}std::swap(rows[pivot],rows[j]);for(int i=j+1;i<3;i++){double m=u[rows[i]*3+j]/u[rows[j]*3+j];u[rows[i]*3+j]=m;for(int k=j+1;k<3;k++)u[rows[i]*3+k]-=m*u[rows[j]*3+k];}}
 for(int col=0;col<3;col++){for(int j=0;j<2;j++)for(int i=j+1;i<3;i++)b[rows[i]*3+col]-=u[rows[i]*3+j]*b[rows[j]*3+col];result[6+col]=b[rows[2]*3+col]/u[rows[2]*3+2];for(int i=1;i>=0;i--){double z=b[rows[i]*3+col];for(int j=i+1;j<3;j++)z-=u[rows[i]*3+j]*result[j*3+col];result[i*3+col]=z/u[rows[i]*3+i];}}
 return result;
}
P3 triangulate(const vector<Ray>&rays,double*error=nullptr){array<double,9>A{};P3 b;for(const auto&r:rays){array<double,9>Q;for(int i=0;i<3;i++)for(int j=0;j<3;j++){Q[i*3+j]=(i==j?1.:0.)-r.direction[i]*r.direction[j];A[i*3+j]+=Q[i*3+j];}b=b+multiply(Q,r.origin);}P3 x=multiply(inverse3_original(A),b);if(error){double e=0;for(const auto&r:rays){P3 diff=x-r.origin;double h=std::pow(norm(diff),2)-std::pow(dot(diff,r.direction),2);if(h<0&&h>=-SMALL)h=0;else if(h< -SMALL)throw std::runtime_error("negative triangulation distance");else h=std::sqrt(h);e+=h*h;}*error=std::sqrt(e/rays.size());}return x;}
vector<P2>detect(const Image&im,double threshold=30.){
 vector<P2>out;out.reserve(size_t(im.rows)*im.cols/128);
 auto ln=[](double v){return std::log(v<LOG_SMALL?LOG_SMALL:v);};
 for(int row=1;row<im.rows-1;row++)for(int col=1;col<im.cols-1;col++){
  // Actual upstream inequalities keep the bottom/right tie, despite its
  // opposite prose comment. Preserve the executable local-max convention.
  double center=im(row,col);if(center<threshold||im(row-1,col)>center||im(row,col-1)>center||im(row+1,col)>=center||im(row,col+1)>=center)continue;
  double l=ln(im(row,col-1)),m=ln(center),r=ln(im(row,col+1));int x1=col-1,x2=col,x3=col+1;
  double numerator=l*(x2*x2-x3*x3)-m*(x1*x1-x3*x3)+r*(x1*x1-x2*x2),denominator=l*(x3-x2)-r*(x1-x2)+m*(x1-x3);if(denominator==0)continue;double x=-.5*(numerator/denominator);if(!std::isfinite(x))continue;
  l=ln(im(row-1,col));r=ln(im(row+1,col));int y1=row-1,y2=row,y3=row+1;numerator=l*(y2*y2-y3*y3)-m*(y1*y1-y3*y3)+r*(y1*y1-y2*y2);denominator=l*(y3-y2)-r*(y1-y2)+m*(y1-y3);if(denominator==0)continue;double y=-.5*(numerator/denominator);if(std::isfinite(y))out.push_back({x,y});
 }return out;
}
// Explicit minstd_rand0 recurrence and paired Fisher-Yates draws preserve the
// original GCC13 default_random_engine/shuffle convention on MSVC as well.
struct UniformSequence {uint64_t state;explicit UniformSequence(uint64_t seed):state(seed?seed:1){}uint64_t next(){state=state*16807%2147483647;return state;}uint64_t bounded(uint64_t maximum){uint64_t scaling=2147483645ULL/(maximum+1),past=(maximum+1)*scaling,x;do{x=next()-1;}while(x>=past);return x/scaling;}};
template<class T>void shuffle_original(vector<T>&x,UniformSequence&rng){size_t n=x.size();if(!n)return;if(2147483645ULL/n>=n){size_t i=1;if(n%2==0){std::swap(x[i],x[rng.bounded(1)]);i++;}while(i<n){uint64_t range=i+1,draw=rng.bounded(range*(range+1)-1);size_t a=draw/(range+1),b=draw%(range+1);std::swap(x[i],x[a]);i++;std::swap(x[i],x[b]);i++;}}else for(size_t i=1;i<n;i++)std::swap(x[i],x[rng.bounded(i)]);}
struct OTF {double a=125,b=1.5,c=1.5;};
struct Particle {P3 xyz;array<P2,4>uv{};bool tracked=false;};
void project_particle(Particle&p,const array<Camera,4>&cams){for(int c=0;c<4;c++)p.uv[c]=project(p.xyz,cams[c]);}
struct Line {P2 origin,direction;};
Line projected_ray(const Ray&r,const Camera&c){P2 a=project(r.origin+r.direction,c),b=project(r.origin+r.direction*2.,c);P2 d=b-a;double length=norm(d);if(length<1e-20)throw std::runtime_error("degenerate epipolar line");return{a,d*(1/length)};}
double line_distance_squared(P2 p,const Line&line){P2 d=p-line.origin;double z=d.y*line.direction.x-d.x*line.direction.y;return z*z;}
bool in_volume(P3 p){return p.x>=-20&&p.x<=20&&p.y>=-20&&p.y<=20&&p.z>=-20&&p.z<=20;}
// Conservative pixel-cell strips followed by the original exact distance test.
// Binning accelerates candidate lookup; it does not crop observations or alter
// geometric tolerances. A cell is retained whenever any point in it can pass.
struct ImageGrid {vector<vector<int>>cells;ImageGrid(const vector<P2>&points):cells(256*256){for(size_t i=0;i<points.size();i++){int x=std::clamp(int(points[i].x/4),0,255),y=std::clamp(int(points[i].y/4),0,255);cells[y*256+x].push_back(int(i));}}};
vector<int> strip_candidates(const vector<Ray>&rays,int target,const vector<int>&chosen_cams,const vector<P2>&chosen_points,const array<Camera,4>&cams,const array<vector<P2>,4>&points,const array<ImageGrid,4>&grids,double tolerance){
 vector<Line>lines;for(const auto&r:rays){Line l=projected_ray(r,cams[target]);l.direction=l.direction*(1/norm(l.direction));lines.push_back(l);}vector<int>out;double tol2=tolerance*tolerance;
 for(int cy=0;cy<256;cy++){
  double low=0,high=1024,y0=4.*cy,y1=std::min(1024.,y0+4.);bool valid=true;
  for(const auto&l:lines){double ux=l.direction.x,uy=l.direction.y;if(std::abs(uy)>1e-14){double a=l.origin.x+((y0-l.origin.y)*ux-tolerance)/uy,b=l.origin.x+((y0-l.origin.y)*ux+tolerance)/uy,c=l.origin.x+((y1-l.origin.y)*ux-tolerance)/uy,d=l.origin.x+((y1-l.origin.y)*ux+tolerance)/uy;low=std::max(low,std::min({a,b,c,d}));high=std::min(high,std::max({a,b,c,d}));}else{double a=(y0-l.origin.y)*ux,b=(y1-l.origin.y)*ux;if(std::min(a,b)>tolerance||std::max(a,b)<-tolerance){valid=false;break;}}}
  if(!valid||low>high)continue;int x0=std::clamp(int(std::floor(low/4)),0,255),x1=std::clamp(int(std::floor(high/4)),0,255);
  for(int cx=x0;cx<=x1;cx++)for(int pid:grids[target].cells[cy*256+cx]){P2 q=points[target][pid];bool accept=true;for(const auto&l:lines)if(line_distance_squared(q,l)>tol2){accept=false;break;}if(!accept)continue;Ray qr=ray(q,cams[target]);for(size_t j=0;j<chosen_cams.size();j++){Line reverse=projected_ray(qr,cams[chosen_cams[j]]);if(line_distance_squared(chosen_points[j],reverse)>tol2){accept=false;break;}}if(accept)out.push_back(pid);}
 }return out;
}
double line_image_length(const Ray&r,const Camera&camera){Line l=projected_ray(r,camera);vector<P2>points;auto add=[&](P2 p){if(p.x>=0&&p.x<=1024&&p.y>=0&&p.y<=1024){for(P2 old:points)if(std::abs(old.x-p.x)<1e-9&&std::abs(old.y-p.y)<1e-9)return;points.push_back(p);}};P2 p=l.origin,u=l.direction;if(std::abs(u.x)>1e-12){double t=-p.x/u.x;add({0,p.y+t*u.y});t=(1023-p.x)/u.x;add({1023,p.y+t*u.y});}if(std::abs(u.y)>1e-12){double t=-p.y/u.y;add({p.x+t*u.x,0});t=(1023-p.y)/u.y;add({p.x+t*u.x,1023});}return points.size()<2?0:norm(points[0]-points[1]);}
int select_camera(const vector<Ray>&rays,const vector<int>&remaining,const array<Camera,4>&cams){
 int best=-1;if(rays.size()==1){double length=std::numeric_limits<double>::infinity();for(int c:remaining){double v=line_image_length(rays[0],cams[c]);if(v<length-1e-9){best=c;length=v;}}}else{double score=-1;for(int c:remaining){vector<Line>lines;for(const auto&r:rays)lines.push_back(projected_ray(r,cams[c]));double value=0;for(size_t i=0;i<lines.size();i++){for(size_t j=i+1;j<lines.size();j++){P2 a=lines[i].direction,b=lines[j].direction;value=std::max(value,std::abs(a.x*b.y-a.y*b.x));if(value>=1-1e-12)break;}if(value>=1-1e-12)break;}if(value>score){score=value;best=c;}}}return best;
}
double quick_tolerance(const vector<Ray>&rays,int total){double degree=180;constexpr double pi=3.14159265358979323846;for(size_t i=0;i<rays.size();i++)for(size_t j=i+1;j<rays.size();j++){double v=std::clamp(dot(rays[i].direction,rays[j].direction),-1.,1.);degree=std::min(degree,std::acos(v)*180/pi);}double angle=std::max(1.,std::min(2.,std::sin(6*pi/180)/std::max(1e-6,std::sin(std::max(degree,1.)*pi/180))));return .04*std::sqrt(double(std::max(total,2))/std::max(int(rays.size()),2))*angle;}
struct StereoCandidate {array<int,4>ids;double error;};
vector<array<int,4>> prune_candidates(const vector<StereoCandidate>&candidates,const array<vector<P2>,4>&points,const array<Camera,4>&cams){
 size_t n=candidates.size();if(!n)return{};array<size_t,5>offset{};for(int c=0;c<4;c++)offset[c+1]=offset[c]+points[c].size();vector<vector<size_t>>flats(n);vector<vector<int>>reverse(offset[4]);
 for(size_t j=0;j<n;j++)for(int c=0;c<4;c++)if(cams[c].active&&candidates[j].ids[c]>=0){size_t f=offset[c]+candidates[j].ids[c];flats[j].push_back(f);reverse[f].push_back(int(j));}
 auto equal=[](double a,double b){return std::abs(a-b)<=std::max(std::abs(a),std::abs(b))*1e-9+1e-12;};vector<double>sum(n,0),score(n,std::numeric_limits<double>::infinity());vector<int>counts(n,0);
 for(const auto&list:reverse){vector<std::pair<double,int>>pool;for(int j:list)if(std::isfinite(candidates[j].error))pool.push_back({candidates[j].error,j});if(pool.empty())continue;if(pool.size()==1){counts[pool[0].second]++;continue;}std::stable_sort(pool.begin(),pool.end(),[&](auto a,auto b){return !equal(a.first,b.first)?a.first<b.first:a.second<b.second;});for(size_t k=0;k<pool.size();){size_t end=k+1;while(end<pool.size()&&equal(pool[end].first,pool[k].first))end++;double pct=.5*double(k+end-1)/(pool.size()-1);for(size_t t=k;t<end;t++){sum[pool[t].second]+=pct;counts[pool[t].second]++;}k=end;}}
 for(size_t j=0;j<n;j++)if(counts[j])score[j]=sum[j]/counts[j];vector<int>order(n);std::iota(order.begin(),order.end(),0);std::stable_sort(order.begin(),order.end(),[&](int a,int b){bool af=std::isfinite(score[a]),bf=std::isfinite(score[b]);if(af&&bf){if(!equal(score[a],score[b]))return score[a]<score[b];}else if(af)return true;else if(bf)return false;double ea=candidates[a].error,eb=candidates[b].error;af=std::isfinite(ea);bf=std::isfinite(eb);if(af&&bf){if(!equal(ea,eb))return ea<eb;}else if(af)return true;else if(bf)return false;return a<b;});
 vector<int>owner(offset[4],-1);vector<char>selected(n,0);auto place=[&](int j){for(size_t f:flats[j])owner[f]=j;selected[j]=1;};
 for(int j:order){if(!std::isfinite(score[j])||!std::isfinite(candidates[j].error)||flats[j].empty())continue;vector<int>conflicts;for(size_t f:flats[j]){int id=owner[f];if(id>=0&&std::find(conflicts.begin(),conflicts.end(),id)==conflicts.end())conflicts.push_back(id);}if(conflicts.empty()){place(j);continue;}double pct=0,error=0;int count=0;for(int id:conflicts)if(std::isfinite(score[id])&&std::isfinite(candidates[id].error)){pct+=score[id];error+=candidates[id].error;count++;}if(!count)continue;pct/=count;error/=count;bool better=(!equal(score[j],pct)&&score[j]+1e-12<pct)||(equal(score[j],pct)&&!equal(candidates[j].error,error)&&candidates[j].error+1e-12<error);if(better){for(int id:conflicts){for(size_t f:flats[id])if(owner[f]==id)owner[f]=-1;selected[id]=0;}place(j);}}
 vector<array<int,4>>out;for(size_t j=0;j<n;j++)if(selected[j])out.push_back(candidates[j].ids);return out;
}
void enumerate_stereo_path(array<int,4>ids,vector<int>chosen,vector<P2>chosen_points,vector<Ray>rays,vector<int>remaining,int total,const array<Camera,4>&cams,const array<vector<P2>,4>&points,const array<ImageGrid,4>&grids,double tolerance,vector<StereoCandidate>&out){
 if(remaining.empty()){vector<Ray>ordered;for(int c=0;c<4;c++)if(ids[c]>=0)ordered.push_back(ray(points[c][ids[c]],cams[c]));double error;P3 p=triangulate(ordered,&error);if(error<=.04&&in_volume(p))out.push_back({ids,error});return;}
 int c=select_camera(rays,remaining,cams);if(c<0)return;auto candidates=strip_candidates(rays,c,chosen,chosen_points,cams,points,grids,tolerance);remaining.erase(std::find(remaining.begin(),remaining.end(),c));
 for(int id:candidates){auto next_rays=rays;next_rays.push_back(ray(points[c][id],cams[c]));double error;try{triangulate(next_rays,&error);}catch(...){continue;}if(error>quick_tolerance(next_rays,total))continue;auto next_ids=ids;next_ids[c]=id;auto next_chosen=chosen;next_chosen.push_back(c);auto next_points=chosen_points;next_points.push_back(points[c][id]);enumerate_stereo_path(next_ids,next_chosen,next_points,next_rays,remaining,total,cams,points,grids,tolerance,out);}
}
vector<Particle> stereo_match(const array<vector<P2>,4>&points,const array<Camera,4>&cams,double tolerance,vector<StereoCandidate>*capture_candidates=nullptr,vector<array<int,4>>*capture_selected=nullptr){
 int ref=-1,total=0;size_t least=std::numeric_limits<size_t>::max();vector<int>active;for(int c=0;c<4;c++)if(cams[c].active){active.push_back(c);total++;if(points[c].size()<least){least=points[c].size();ref=c;}}if(total<2||ref<0)return{};array<ImageGrid,4>grids{ImageGrid(points[0]),ImageGrid(points[1]),ImageGrid(points[2]),ImageGrid(points[3])};vector<int>remaining;for(int c:active)if(c!=ref)remaining.push_back(c);vector<StereoCandidate>all;
 for(size_t id=0;id<points[ref].size();id++){array<int,4>ids{-1,-1,-1,-1};ids[ref]=int(id);enumerate_stereo_path(ids,{ref},{points[ref][id]},{ray(points[ref][id],cams[ref])},remaining,total,cams,points,grids,tolerance,all);}
 std::sort(all.begin(),all.end(),[](const auto&a,const auto&b){return a.ids<b.ids;});auto selected=prune_candidates(all,points,cams);vector<Particle>out;for(const auto&ids:selected){vector<Ray>rays;Particle p;for(int c=0;c<4;c++)if(ids[c]>=0){p.uv[c]=points[c][ids[c]];rays.push_back(ray(p.uv[c],cams[c]));}double error;p.xyz=triangulate(rays,&error);if(error<=.04)out.push_back(p);}if(capture_candidates)*capture_candidates=all;if(capture_selected)*capture_selected=selected;return out;
}
OTF estimate_otf(const vector<Image>&images){vector<double>peaks,obs;vector<P2>design;for(const Image&im:images){auto points=detect(im);if(points.size()>1000){UniformSequence rng(123);shuffle_original(points,rng);points.resize(1000);}if(points.empty())throw std::runtime_error("OTF calibration empty");for(P2 p:points){int cy=std::clamp(int(std::round(p.y)),0,im.rows-1),cx=std::clamp(int(std::round(p.x)),0,im.cols-1);peaks.push_back(im(cy,cx));int loY=std::max(0,int(std::floor(p.y-2))),hiY=std::min(im.rows,int(std::ceil(p.y+3))),loX=std::max(0,int(std::floor(p.x-2))),hiX=std::min(im.cols,int(std::ceil(p.x+3)));for(int y=loY;y<hiY;y++)for(int x=loX;x<hiX;x++){obs.push_back(im(y,x));design.push_back({std::pow(x-p.x,2),std::pow(y-p.y,2)});}}}
 double mean=std::accumulate(peaks.begin(),peaks.end(),0.)/peaks.size(),variance=0;for(double p:peaks)variance+=std::pow(p-mean,2);double amplitude=std::min(mean+2*std::sqrt(variance/peaks.size()),*std::max_element(peaks.begin(),peaks.end()));
 double xx=0,xy=0,yy=0,bx=0,by=0;for(size_t i=0;i<obs.size();i++){double z=-std::log(obs[i]<LOG_SMALL?LOG_SMALL:obs[i])+std::log(amplitude);P2 p=design[i];xx+=p.x*p.x;xy+=p.x*p.y;yy+=p.y*p.y;bx+=p.x*z;by+=p.y*z;}
 double det=xx*yy-xy*xy;return{amplitude,(yy*bx-xy*by)/det,(xx*by-xy*bx)/det};
}
// Fixed observation-coordinate rectangles, with right-open endpoints.
struct Rect {
 int y0=1,y1=0,x0=1,x1=0;
 bool contains(P2 p)const{return x0<=p.x&&p.x<x1&&y0<=p.y&&p.y<y1;}
 bool contains(int y,int x)const{return x0<=x&&x<x1&&y0<=y&&y<y1;}
 int size()const{return std::max(0,y1-y0)*std::max(0,x1-x0);}
};
Rect rectangle(P2 p,double halfsize,int rows=1024,int cols=1024){
 if(p.x<=0||p.y<=0||p.x>=cols||p.y>=rows)return {};
 int width=int(std::round(2*halfsize));
 int x=int(std::round(p.x-(width-1)/2.)),y=int(std::round(p.y-(width-1)/2.));
 return {std::max(0,y),std::min(rows,y+width),std::max(0,x),std::min(cols,x+width)};
}
double interpolate_constant_grid(double value,P3 p){
 // Preserve the author's sequential trilinear arithmetic on the uniform 5^3
 // calibration grid, including its floating-point rounding.
 double weight[3];for(int j=0;j<3;j++){double x=std::clamp(p[j],-20.,20.);int i=std::clamp(int(std::floor((x+20)/10)),0,3);weight[j]=std::clamp((x-(-20+10*i))/10,0.,1.);}
 double x=value*(1-weight[0])+value*weight[0];
 double y=x*(1-weight[1])+x*weight[1];
 return y*(1-weight[2])+y*weight[2];
}
OTF local_otf(OTF p,P3 x){return{interpolate_constant_grid(p.a,x),interpolate_constant_grid(p.b,x),interpolate_constant_grid(p.c,x)};}
double intensity(P2 center,OTF p,int y,int x){double dx=x-center.x,dy=y-center.y;double xx=dx*1.+dy*0.,yy=-dx*0.+dy*1.;return std::max(0.,p.a*std::exp(-(p.b*xx*xx+p.c*yy*yy)));}
array<Image,4> residual_images(const vector<Particle>&particles,const array<Image,4>&original,const array<Camera,4>&cams,const array<OTF,4>&otf,bool ipr=false,const vector<int>*flags=nullptr){
 auto residual=original;
 for(int c=0;c<4;c++){
  if(!cams[c].active&&!ipr)continue;
  for(size_t i=0;i<particles.size();i++){
   if(flags&&(*flags)[i])continue;
   const auto&p=particles[i];Rect r=rectangle(p.uv[c],2,original[c].rows,original[c].cols);OTF model=local_otf(otf[c],p.xyz);
   for(int y=r.y0;y<r.y1;y++)for(int x=r.x0;x<r.x1;x++){double pred=intensity(p.uv[c],model,y,x);if(pred==0)continue;double &v=residual[c](y,x),candidate=original[c](y,x)-pred;if(candidate<v)v=candidate;if(ipr&&v<0)v=0;}
  }
 }
 return residual;
}
struct Augmented {Rect range;Image values;double at(int y,int x)const{return values(y-range.y0,x-range.x0);}};
array<Augmented,4> augment(const Particle&p,const array<Image,4>&res,const array<Camera,4>&cams,const array<OTF,4>&otf){
 array<Augmented,4>out;
 for(int c=0;c<4;c++)if(cams[c].active){
  Rect r=rectangle(p.uv[c],4,res[c].rows,res[c].cols),project_range=rectangle(p.uv[c],2,res[c].rows,res[c].cols);out[c].range=r;
  out[c].values=Image(std::max(0,r.y1-r.y0),std::max(0,r.x1-r.x0));OTF model=local_otf(otf[c],p.xyz);
  for(int y=r.y0;y<r.y1;y++)for(int x=r.x0;x<r.x1;x++){double v=res[c](y,x);if(project_range.contains(y,x))v+=intensity(p.uv[c],model,y,x);out[c].values(y-r.y0,x-r.x0)=std::max(0.,v);}
 }
 return out;
}
double shake_cost(const Particle&p,const array<Augmented,4>&aug,const array<Camera,4>&cams,const array<OTF,4>&otf){
 double sum=0;int used=0;
 for(int c=0;c<4;c++)if(cams[c].active&&aug[c].range.size()>0){
  auto r=aug[c].range;P2 uv=p.uv[c];if(!std::isfinite(uv.x)||!std::isfinite(uv.y))continue;
  int ymin=int(std::floor(uv.y-2)),ymax=int(std::ceil(uv.y+3)),xmin=int(std::floor(uv.x-2)),xmax=int(std::ceil(uv.x+3));double sse=0;OTF model=local_otf(otf[c],p.xyz);
  for(int y=r.y0;y<r.y1;y++)for(int x=r.x0;x<r.x1;x++){double pred=0;if(y>=ymin&&y<ymax&&x>=xmin&&x<xmax)pred=intensity(uv,model,y,x);double d=aug[c].at(y,x)-pred;sse+=d*d;}
  sum+=sse/r.size();used++;
 }
 return used>1?sum/used:std::numeric_limits<double>::infinity();
}
double median(vector<double>v){if(v.empty())return 0;size_t n=v.size(),m=n/2;std::nth_element(v.begin(),v.begin()+m,v.end());double value=v[m];if(n%2==0){std::nth_element(v.begin(),v.begin()+m-1,v.end());value=.5*(value+v[m-1]);}return value;}
array<double,3> quadratic_fit(const vector<double>&x,const vector<double>&y){
 // Preserve (X'X)^-1 X' y multiplication grouping rather than regrouping
 // into a normal-equation RHS; these short polynomial fits are sensitive.
 array<double,9>normal{};vector<array<double,3>>design;
 for(double value:x)design.push_back({std::pow(value,0),std::pow(value,1),std::pow(value,2)});
 for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(const auto&r:design)normal[i*3+j]+=r[i]*r[j];
 auto inverse=inverse3_original(normal);array<double,3>coeff{};
 for(int i=0;i<3;i++)for(size_t j=0;j<x.size();j++){double value=0;for(int k=0;k<3;k++)value+=inverse[i*3+k]*design[j][k];coeff[i]+=value*y[j];}
 return coeff;
}
void shake_coordinate_descent(Particle&p,const array<Augmented,4>&aug,double delta,const array<Camera,4>&cams,const array<OTF,4>&otf){
 for(int axis=0;axis<3;axis++){
  // Qualified correction: evaluate this axis about the accepted full center.
  Particle trial=p;double center=p.xyz[axis];array<array<P2,4>,3>uv;vector<double>xs{-delta,0,delta},cost;
  for(double x:xs){trial.xyz[axis]=center+x;project_particle(trial,cams);uv[cost.size()]=trial.uv;cost.push_back(shake_cost(trial,aug,cams,otf));}
  double left=cost[0],zero=cost[1],right=cost[2],tol=1e-3*std::max(1.,zero);
  bool decreasing=left>zero+tol&&zero>right+tol,increasing=right>zero+tol&&zero>left+tol;
  int direction=decreasing?1:increasing?-1:0;double curvature=left-2*zero+right,slope=(right-left)/std::max(2*delta,1e-12);bool skip=false,allow=true;
  if(curvature>0){double vertex=delta*(left-right)/(2*curvature);if(std::abs(vertex)<=.95*delta){double a=curvature/(delta*delta+1e-12),rho=slope*slope/(2*a*std::max(zero,1.));skip=true;if(rho<8e-4)allow=false;}}
  // The tracer strategy returns four true selection flags even for inactive
  // cameras. Consequently an inactive empty ROI vetoes optional refinements
  // in a reduced-camera pass. Preserve this selected source convention.
  auto inside=[&](const Particle&q){for(int c=0;c<4;c++)if(!aug[c].range.contains(q.uv[c]))return false;return true;};
  auto evaluate=[&](double x){trial.xyz[axis]=center+x;project_particle(trial,cams);if(inside(trial)){xs.push_back(x);cost.push_back(shake_cost(trial,aug,cams,otf));}};
  if(!skip&&(decreasing||increasing)){
   vector<double>jacobians,margins;
   for(int c=0;c<4;c++)if(cams[c].active){Rect r=aug[c].range;if(r.contains(uv[0][c])&&r.contains(uv[2][c]))jacobians.push_back(std::hypot(uv[2][c].x-uv[0][c].x,uv[2][c].y-uv[0][c].y)/std::max(2*delta,1e-12));if(r.contains(uv[1][c]))margins.push_back(std::min(std::min(uv[1][c].x-r.x0,r.x1-1.-uv[1][c].x),std::min(uv[1][c].y-r.y0,r.y1-1.-uv[1][c].y)));}
   double J=median(jacobians),margin=median(margins),cap=J>0&&margin>0?.5*margin/J:0,step=0;
   if(curvature>0){double vertex=delta*(left-right)/(2*curvature);if(std::abs(vertex)>delta){direction=vertex>0?1:-1;step=J>0&&std::abs(vertex)>1/J?1/J:1.15*std::abs(vertex);}}
   if(step<=0&&J>0)step=1/J;double chosen=step>0&&cap>0?std::min(step,cap):0;if(chosen>0)evaluate(direction>=0?chosen:-chosen);
  }
  auto coeff=quadratic_fit(xs,cost);if(allow&&coeff[2]>0){double vertex=-coeff[1]/(2*coeff[2]);if(vertex>=*std::min_element(xs.begin(),xs.end())&&vertex<=*std::max_element(xs.begin(),xs.end()))evaluate(vertex);}
  size_t best=0;for(size_t i=1;i<cost.size();i++)if(cost[i]<cost[best])best=i;p.xyz[axis]=center+xs[best];project_particle(p,cams);
 }
}
double particle_score(const Particle&p,const array<Augmented,4>&aug,const array<Image,4>&res,const array<Camera,4>&cams,const array<OTF,4>&otf){
 array<Rect,4>ranges;array<int,4>use{};array<double,4>peak{};int count=0;
 for(int c=0;c<4;c++)if(cams[c].active){Rect r=rectangle(p.uv[c],2,res[c].rows,res[c].cols);ranges[c]=r;if(r.y1<r.y0||r.x1<r.x0)continue;use[c]=1;count++;for(int y=r.y0;y<r.y1;y++)for(int x=r.x0;x<r.x1;x++){double value=aug[c].range.contains(y,x)?aug[c].at(y,x):res[c](y,x);if(value>peak[c])peak[c]=value;}}
 if(count<2)return 1e-12;
 if(count>2){int best=-1;double v=-std::numeric_limits<double>::infinity();for(int c=0;c<4;c++)if(use[c]&&peak[c]>v){best=c;v=peak[c];}if(best>=0)use[best]=0;}
 double numerator=0,denominator=0;
 for(int c=0;c<4;c++)if(use[c]){double a=0,b=0;OTF model=local_otf(otf[c],p.xyz);Rect r=ranges[c];for(int y=r.y0;y<r.y1;y++)for(int x=r.x0;x<r.x1;x++){double value=aug[c].range.contains(y,x)?aug[c].at(y,x):res[c](y,x);a+=std::max(0.,value);b+=intensity(p.uv[c],model,y,x);}numerator+=a;denominator+=b;}
 double ratio=denominator>1e-12?numerator/denominator:1.;return std::sqrt(std::max(2./3,std::min(1.5,ratio)));
}
struct SpatialGrid {
 double width;std::unordered_map<int64_t,vector<int>>cells;
 explicit SpatialGrid(double h):width(h){}
 int64_t key(int x,int y,int z)const{return(int64_t(x+100000)<<40)|(int64_t(y+100000)<<20)|int64_t(z+100000);}
 void insert(P3 p,int id){cells[key(int(std::floor(p.x/width)),int(std::floor(p.y/width)),int(std::floor(p.z/width)))].push_back(id);}
 vector<int>neighbors(P3 p)const{vector<int>ids;int ix=int(std::floor(p.x/width)),iy=int(std::floor(p.y/width)),iz=int(std::floor(p.z/width));for(int x=ix-1;x<=ix+1;x++)for(int y=iy-1;y<=iy+1;y++)for(int z=iz-1;z<=iz+1;z++){auto it=cells.find(key(x,y,z));if(it!=cells.end())ids.insert(ids.end(),it->second.begin(),it->second.end());}return ids;}
};
vector<int> shake_particles(vector<Particle>&particles,const array<Image,4>&images,const array<Camera,4>&cams,const array<OTF,4>&otf,vector<double>*returned_scores=nullptr){
 vector<double>scores(particles.size(),1.);for(auto&p:particles)project_particle(p,cams);
 double delta=.01;
 for(int loop=0;loop<4;loop++){
  auto residual=residual_images(particles,images,cams,otf);
  for(size_t i=0;i<particles.size();i++){
   auto augmented=augment(particles[i],residual,cams,otf);int usable=0;for(int c=0;c<4;c++)if(cams[c].active&&augmented[c].range.size()>0)usable++;
   if(usable<2){scores[i]=0;continue;}
   shake_coordinate_descent(particles[i],augmented,delta,cams,otf);scores[i]*=particle_score(particles[i],augmented,residual,cams,otf);
  }
  delta=std::max(.0005,delta*.5);
 }
 vector<int>flags(particles.size());SpatialGrid grid(.04);for(size_t i=0;i<particles.size();i++)grid.insert(particles[i].xyz,int(i));
 for(size_t i=0;i+1<particles.size();i++)if(!(flags[i]&2)){
  vector<std::pair<double,int>>near;for(int j:grid.neighbors(particles[i].xyz)){double d=dot(particles[i].xyz-particles[j].xyz,particles[i].xyz-particles[j].xyz);if(d<.04*.04)near.emplace_back(d,j);}
  std::sort(near.begin(),near.end());for(size_t j=1;j<near.size();j++)flags[near[j].second]|=2;
 }
 double sum=0;int count=0;for(size_t i=0;i<scores.size();i++)if(!(flags[i]&2)){sum+=scores[i];count++;}double mean=count?sum/count:0;
 for(size_t i=0;i<scores.size();i++)if(scores[i]<mean*.1||scores[i]==0)flags[i]|=1;
 if(returned_scores)*returned_scores=scores;return flags;
}
vector<Particle> iterative_particle_reconstruction(array<Image,4>images,array<Camera,4>cams,const array<OTF,4>&otf){
 vector<Particle>all;const int masks[5]={15,7,11,13,14};
 for(int mask:masks){for(int c=0;c<4;c++)cams[c].active=(mask&(1<<c))!=0;
  double tolerance=.8;
  for(int loop=0;loop<2;loop++){
   array<vector<P2>,4>points;UniformSequence rng(1234);
   for(int c=0;c<4;c++)if(cams[c].active){points[c]=detect(images[c]);if(points[c].size()>200000){shuffle_original(points[c],rng);points[c].resize(200000);}}
   auto particles=stereo_match(points,cams,tolerance);if(std::getenv("IMAGING_LOCAL_TRACE"))std::cerr<<"ipr "<<mask<<" "<<loop<<" matched="<<particles.size();
   if(!particles.empty()){
    auto flags=shake_particles(particles,images,cams,otf);vector<Particle>kept;
    for(size_t i=0;i<particles.size();i++)if(!flags[i])kept.push_back(particles[i]);
    images=residual_images(kept,images,cams,otf,true);all.insert(all.end(),kept.begin(),kept.end());if(std::getenv("IMAGING_LOCAL_TRACE"))std::cerr<<" kept="<<kept.size()<<" total="<<all.size();
   }
   if(std::getenv("IMAGING_LOCAL_TRACE"))std::cerr<<"\n";tolerance*=1.5;
  }
 }
 return all;
}
double gaussian_peak(double y1,double v1,double y2,double v2,double y3,double v3){
 double a=std::log(std::max(LOG_SMALL,v1)),b=std::log(std::max(LOG_SMALL,v2)),c=std::log(std::max(LOG_SMALL,v3));
 double location=-.5*(a*(y2*y2-y3*y3)-b*(y1*y1-y3*y3)+c*(y1*y1-y2*y2))/(a*(y3-y2)-c*(y1-y2)+b*(y1-y3));return std::isfinite(location)?location:y2;
}
struct DisplacementField {
 vector<P3>values;
 static int index(int x,int y,int z){return x*51*51+y*51+z;}
 static double coordinate(int i){return -20.+i*.8;}
 DisplacementField(const vector<Particle>&previous,const vector<Particle>&current):values(51*51*51){
  SpatialGrid before(1),after(1);for(size_t i=0;i<previous.size();i++)before.insert(previous[i].xyz,int(i));for(size_t i=0;i<current.size();i++)after.insert(current[i].xyz,int(i));
  for(int x=0;x<51;x++)for(int y=0;y<51;y++)for(int z=0;z<51;z++){
   P3 center{coordinate(x),coordinate(y),coordinate(z)};vector<int>a,b;
   auto collect=[&](const vector<Particle>&points,const SpatialGrid&grid,vector<int>&ids){for(int i:grid.neighbors(center)){P3 d=points[i].xyz-center;if(d.x<1&&d.x>-1&&d.y<1&&d.y>-1&&d.z<1&&d.z>-1)ids.push_back(i);}std::sort(ids.begin(),ids.end());};
   collect(previous,before,a);collect(current,after,b);if(a.empty()||b.empty())continue;
   array<double,1331>pdf{};
   for(int j:b)for(int i:a){P3 d=current[j].xyz-previous[i].xyz;P3 bin{2.5*d.x+5,2.5*d.y+5,2.5*d.z+5};int ix=int(std::round(bin.x)),iy=int(std::round(bin.y)),iz=int(std::round(bin.z));
    for(int u=std::max(0,ix-1);u<=std::min(10,ix+1);u++)for(int v=std::max(0,iy-1);v<=std::min(10,iy+1);v++)for(int w=std::max(0,iz-1);w<=std::min(10,iz+1);w++){double xx=u-bin.x,yy=v-bin.y,zz=w-bin.z;pdf[u*121+v*11+w]+=std::exp(-(xx*xx+yy*yy+zz*zz));}
   }
   int peak=int(std::max_element(pdf.begin(),pdf.end())-pdf.begin());if(pdf[peak]<=.1)continue;int ids[3]={peak/121,(peak%121)/11,peak%11};P3 displacement;
   for(int axis=0;axis<3;axis++){int i=ids[axis];if(i==0)displacement[axis]=-2;else if(i==10)displacement[axis]=2;else {int stride=axis==0?121:axis==1?11:1;double location=gaussian_peak(i-1,pdf[peak-stride],i,pdf[peak],i+1,pdf[peak+stride]);displacement[axis]=(location-5)/2.5;}}
   values[index(x,y,z)]=displacement;
  }
  // Source component-wise separable Gaussian, zero continuation, no edge
  // renormalization. Sigma is one grid node, radius three nodes.
  array<double,7>kernel;double sum=0;for(int k=-3;k<=3;k++){kernel[k+3]=std::exp(-(k*k)/2.);sum+=kernel[k+3];}for(double&v:kernel)v/=sum;
  for(int component=0;component<3;component++)for(int axis=0;axis<3;axis++){
   vector<P3>old=values;
   for(int x=0;x<51;x++)for(int y=0;y<51;y++)for(int z=0;z<51;z++){double value=0;int ids[3]={x,y,z};for(int k=-3;k<=3;k++){int p=ids[axis]+k;if(p<0||p>=51)continue;int q[3]={x,y,z};q[axis]=p;value+=old[index(q[0],q[1],q[2])][component]*kernel[k+3];}values[index(x,y,z)][component]=value;}
  }
 }
 P3 at(P3 p)const{
  int cell[3];double weight[3];for(int j=0;j<3;j++){p[j]=std::clamp(p[j],-20.,20.);cell[j]=std::clamp(int(std::floor((p[j]+20)/.8)),0,49);weight[j]=std::clamp((p[j]-coordinate(cell[j]))/(coordinate(cell[j]+1)-coordinate(cell[j])),0.,1.);}
  P3 result;for(int j=0;j<3;j++){int x=cell[0],y=cell[1],z=cell[2];double a=values[index(x,y,z)][j]*(1-weight[0])+values[index(x+1,y,z)][j]*weight[0],b=values[index(x,y,z+1)][j]*(1-weight[0])+values[index(x+1,y,z+1)][j]*weight[0],c=values[index(x,y+1,z)][j]*(1-weight[0])+values[index(x+1,y+1,z)][j]*weight[0],d=values[index(x,y+1,z+1)][j]*(1-weight[0])+values[index(x+1,y+1,z+1)][j]*weight[0];double e=a*(1-weight[1])+c*weight[1],f=b*(1-weight[1])+d*weight[1];result[j]=e*(1-weight[2])+f*weight[2];}return result;
 }
};
P3 lms_predict(const vector<P3>&history){
 int length=int(history.size()),order=std::min(length-1,5);if(length<3)throw std::runtime_error("LMS needs at least three positions");P3 prediction;
 for(int axis=0;axis<3;axis++){
  vector<double>series;for(int i=length-order-1;i<length;i++)series.push_back(history[i][axis]);bool shifted=std::abs(series.back())<1;if(shifted)for(double&v:series)v+=10;
  double energy=0;for(int j=0;j<order;j++)energy+=series[j]*series[j];double step=energy>1e-12?1/energy:0,error=series.back();vector<double>filter(order);int iteration=0;
  while(std::abs(error)>1e-8&&iteration<5000&&step>0){for(int j=0;j<order;j++)filter[j]+=step*series[j]*error;double estimate=0;for(int j=0;j<order;j++)estimate+=filter[j]*series[j];error=series.back()-estimate;iteration++;}
  double estimate=0;for(int j=0;j<order;j++)estimate+=filter[j]*series[j+1];if(step==0)estimate=series.back()+(series.back()-series[series.size()-2]);prediction[axis]=estimate-(shifted?10:0);
 }
 return prediction;
}
API int numeric_field(const double*previous,int na,const double*current,int nb,double*out){try{vector<Particle>a(na),b(nb);for(int i=0;i<na;i++)a[i].xyz={previous[3*i],previous[3*i+1],previous[3*i+2]};for(int i=0;i<nb;i++)b[i].xyz={current[3*i],current[3*i+1],current[3*i+2]};DisplacementField field(a,b);for(size_t i=0;i<field.values.size();i++)for(int j=0;j<3;j++)out[3*i+j]=field.values[i][j];return 0;}catch(...){return 1;}}
API int numeric_predict(const double*history,int n,double*out){try{vector<P3>p;for(int i=0;i<n;i++)p.push_back({history[3*i],history[3*i+1],history[3*i+2]});P3 q=lms_predict(p);for(int j=0;j<3;j++)out[j]=q[j];return 0;}catch(...){return 1;}}
struct TrackSegment {vector<P3>positions;vector<int>frames;void append(P3 p,int f){positions.push_back(p);frames.push_back(f);}};
struct Nearest {int id=-1;double cost=std::numeric_limits<double>::infinity();};
Nearest nearest_point(P3 p,double radius,const vector<Particle>&points,const SpatialGrid&grid){Nearest best;double r2=radius*radius;for(int i:grid.neighbors(p)){P3 d=points[i].xyz-p;double distance=dot(d,d);if(distance<=r2&&(distance<best.cost||(distance==best.cost&&i<best.id))){best={i,distance};}}return best;}
bool linear_consistent(const TrackSegment&t){
 if(t.positions.size()<4)return false;P3 estimate;size_t start=t.positions.size()-4;
 // Kernel = inverse([[4,6],[6,14]]) * [[1,1,1,1],[0,1,2,3]],
 // matching the original two-by-two determinant inverse arithmetic.
 for(int j=0;j<3;j++){double intercept=0,slope=0;for(int i=0;i<4;i++){double a=.7+(-.3)*i,b=-.3+.2*i;intercept+=a*t.positions[start+i][j];slope+=b*t.positions[start+i][j];}estimate[j]=intercept+3*slope;}
 return norm(t.positions.back()-estimate)<=.05;
}
Nearest short_link(const TrackSegment&track,const vector<TrackSegment>&long_tracks,const SpatialGrid&long_grid,const vector<Particle>&points,const SpatialGrid&point_grid){
 P3 last=track.positions.back();double factor=1;Nearest best;
 for(int iteration=0;iteration<5;iteration++){
  if(iteration)factor*=1.1;double radius=1.2*factor,neighbor_radius=3*radius,cutoff=neighbor_radius*neighbor_radius;vector<std::pair<int,double>>neighbors;double distance_sum=0;
  for(int i:long_grid.neighbors(last)){P3 d=long_tracks[i].positions.back()-last;double d2=dot(d,d);if(d2<cutoff){neighbors.emplace_back(i,d2);distance_sum+=d2;}}
  P3 velocity;
  if(!neighbors.empty()){
   double scale=.64*distance_sum/neighbors.size();scale=std::clamp(scale,(.1*radius)*(.1*radius),(.8*radius)*(.8*radius));double beta=1/(scale+1e-12),weight_sum=0;P3 sum;
   for(auto neighbor:neighbors){const auto&t=long_tracks[neighbor.first];if(t.positions.size()<2)continue;P3 d=t.positions.back()-t.positions[t.positions.size()-2];double weight=1/(1+neighbor.second*beta);weight_sum+=weight;sum.x+=weight*d.x;sum.y+=weight*d.y;sum.z+=weight*d.z;}
   if(weight_sum>0)velocity=sum*(1/weight_sum);
  }
  best=nearest_point(last+velocity,radius,points,point_grid);if(best.id>=0)break;
 }
 return best;
}
array<Image,4> frame_images(const uint32_t*raw,int frame){array<Image,4>images;for(int c=0;c<4;c++){images[c]=Image(1024,1024);const uint32_t*begin=raw+(size_t(frame)*4+c)*1024*1024;std::copy(begin,begin+1024*1024,images[c].v.begin());}return images;}
API int numeric_tracking(const double*cal,const uint32_t*raw,int frame_count,int capacity,double*out_rows,int*out_count,double*out_otf,double*frame_statistics){try{
 if(frame_count<4||frame_count>100)return 2;array<Camera,4>cams;array<OTF,4>otf;
 for(int c=0;c<4;c++){
  cams[c]=decode_camera(cal+38*c);vector<Image>calibration;
  for(int f=0;f<5;f++){auto images=frame_images(raw,f);calibration.push_back(std::move(images[c]));}
  otf[c]=estimate_otf(calibration);out_otf[3*c]=otf[c].a;out_otf[3*c+1]=otf[c].b;out_otf[3*c+2]=otf[c].c;
 }
 vector<TrackSegment>short_tracks,long_tracks,inactive,exited;vector<Particle>previous;
 for(int frame=0;frame<frame_count;frame++){
  auto images=frame_images(raw,frame);
  if(frame<4){
   auto points=iterative_particle_reconstruction(images,cams,otf);frame_statistics[frame*4]=double(points.size());
   vector<bool>used(points.size());SpatialGrid grid(.4);for(size_t i=0;i<points.size();i++)grid.insert(points[i].xyz,int(i));
   if(frame>0){DisplacementField field(previous,points);vector<TrackSegment>kept;
    for(auto&track:short_tracks){P3 p=track.positions.back();Nearest link=nearest_point(p+field.at(p),.4,points,grid);if(link.id>=0){track.append(points[link.id].xyz,frame);used[link.id]=true;kept.push_back(std::move(track));}}
    short_tracks=std::move(kept);
   }
   for(size_t i=0;i<points.size();i++)if(!used[i]){TrackSegment t;t.append(points[i].xyz,frame);short_tracks.push_back(std::move(t));}
   previous=std::move(points);
   if(frame==3){vector<TrackSegment>kept;for(auto&t:short_tracks)if(t.positions.size()>=4)long_tracks.push_back(std::move(t));else kept.push_back(std::move(t));short_tracks=std::move(kept);}
  }else{
   vector<Particle>predicted;vector<TrackSegment>kept;
   for(auto&t:long_tracks){Particle p;p.xyz=lms_predict(t.positions);project_particle(p,cams);int visible=0;for(P2 uv:p.uv)if(uv.x>=0&&uv.x<1024&&uv.y>=0&&uv.y<1024)visible++;
    if(in_volume(p.xyz)&&visible>=2){kept.push_back(std::move(t));predicted.push_back(p);}else if(t.positions.size()>=7)exited.push_back(std::move(t));
   }
   long_tracks=std::move(kept);
   if(!predicted.empty()){
    auto flags=shake_particles(predicted,images,cams,otf);images=residual_images(predicted,images,cams,otf,true,&flags);kept.clear();
    for(size_t i=0;i<predicted.size();i++)if(!flags[i]){long_tracks[i].append(predicted[i].xyz,frame);kept.push_back(std::move(long_tracks[i]));}else if(long_tracks[i].positions.size()>=7)inactive.push_back(std::move(long_tracks[i]));
    long_tracks=std::move(kept);
   }
   auto candidates=iterative_particle_reconstruction(images,cams,otf);SpatialGrid active_grid(.04);for(size_t i=0;i<long_tracks.size();i++)active_grid.insert(long_tracks[i].positions.back(),int(i));vector<Particle>points;
   for(auto&p:candidates){bool repeated=false;for(int i:active_grid.neighbors(p.xyz))if(norm(p.xyz-long_tracks[i].positions.back())<=.04){repeated=true;break;}if(!repeated)points.push_back(p);}
   frame_statistics[frame*4]=double(points.size());
   if(!points.empty()){
    // Grid widths cover the largest of the five expanding searches.
    SpatialGrid point_grid(1.2*std::pow(1.1,4)),long_grid(3*1.2*std::pow(1.1,4));for(size_t i=0;i<points.size();i++)point_grid.insert(points[i].xyz,int(i));for(size_t i=0;i<long_tracks.size();i++)long_grid.insert(long_tracks[i].positions.back(),int(i));
    vector<Nearest>links;for(const auto&t:short_tracks)links.push_back(short_link(t,long_tracks,long_grid,points,point_grid));vector<int>owner(points.size(),-1);
    for(size_t i=0;i<links.size();i++)if(links[i].id>=0){int id=links[i].id;if(owner[id]<0)owner[id]=int(i);else if(links[i].cost<links[owner[id]].cost){links[owner[id]].id=-1;owner[id]=int(i);}else links[i].id=-1;}
    vector<bool>used(points.size());vector<TrackSegment>short_kept;
    for(size_t i=0;i<short_tracks.size();i++)if(links[i].id>=0){int id=links[i].id;used[id]=true;short_tracks[i].append(points[id].xyz,frame);if(short_tracks[i].positions.size()>=4)long_tracks.push_back(std::move(short_tracks[i]));else short_kept.push_back(std::move(short_tracks[i]));}
    short_tracks=std::move(short_kept);for(size_t i=0;i<points.size();i++)if(!used[i]){TrackSegment t;t.append(points[i].xyz,frame);short_tracks.push_back(std::move(t));}
   }
   kept.clear();for(auto&t:long_tracks)if(linear_consistent(t))kept.push_back(std::move(t));else if(t.positions.size()>=7)inactive.push_back(std::move(t));long_tracks=std::move(kept);
  }
  frame_statistics[frame*4+1]=double(long_tracks.size());frame_statistics[frame*4+2]=double(short_tracks.size());frame_statistics[frame*4+3]=double(inactive.size()+exited.size());
  std::cout<<"local frame "<<frame<<" long="<<long_tracks.size()<<" short="<<short_tracks.size()<<std::endl;
 }
 size_t rows=0;int id=0;for(const auto*group:{&long_tracks,&inactive,&exited,&short_tracks})for(const auto&t:*group){if(rows+t.positions.size()>size_t(capacity))return 3;for(size_t i=0;i<t.positions.size();i++){out_rows[rows*5]=id;out_rows[rows*5+1]=t.frames[i];for(int j=0;j<3;j++)out_rows[rows*5+2+j]=t.positions[i][j];rows++;}id++;}
 *out_count=int(rows);return 0;
 }catch(const std::exception&e){std::cerr<<"numeric_tracking: "<<e.what()<<"\n";return 1;}}
API int numeric_ipr(const double*cal,const double*otf_data,const double*image_data,int capacity,double*positions){try{
 array<Camera,4>cams;array<OTF,4>otf;array<Image,4>images;
 for(int c=0;c<4;c++){cams[c]=decode_camera(cal+38*c);otf[c]={otf_data[3*c],otf_data[3*c+1],otf_data[3*c+2]};images[c]=Image(1024,1024);std::copy(image_data+size_t(c)*1024*1024,image_data+size_t(c+1)*1024*1024,images[c].v.begin());}
 auto particles=iterative_particle_reconstruction(images,cams,otf);if(particles.size()>size_t(capacity))return -2;
 for(size_t i=0;i<particles.size();i++)for(int j=0;j<3;j++)positions[3*i+j]=particles[i].xyz[j];return int(particles.size());
 }catch(const std::exception&e){std::cerr<<"numeric_ipr: "<<e.what()<<"\n";return -1;}}
API int numeric_residual(const double*cal,const double*otf_data,const double*image_data,const double*positions,int n,double*out){try{
 array<Camera,4>cams;array<OTF,4>otf;array<Image,4>images;vector<Particle>particles(n);
 for(int c=0;c<4;c++){cams[c]=decode_camera(cal+38*c);otf[c]={otf_data[3*c],otf_data[3*c+1],otf_data[3*c+2]};images[c]=Image(1024,1024);std::copy(image_data+size_t(c)*1024*1024,image_data+size_t(c+1)*1024*1024,images[c].v.begin());}
 for(int i=0;i<n;i++){particles[i].xyz={positions[3*i],positions[3*i+1],positions[3*i+2]};project_particle(particles[i],cams);}auto res=residual_images(particles,images,cams,otf,true);
 for(int c=0;c<4;c++)std::copy(res[c].v.begin(),res[c].v.end(),out+size_t(c)*1024*1024);return 0;
 }catch(...){return 1;}}
API int numeric_shake(const double*cal,const double*otf_data,const double*image_data,double*positions,int n,int active_mask,double*out_scores,int*out_flags){try{
 array<Camera,4>cams;array<OTF,4>otf;array<Image,4>images;vector<Particle>particles(n);
 for(int c=0;c<4;c++){cams[c]=decode_camera(cal+38*c);cams[c].active=(active_mask&(1<<c))!=0;otf[c]={otf_data[3*c],otf_data[3*c+1],otf_data[3*c+2]};images[c]=Image(1024,1024);std::copy(image_data+size_t(c)*1024*1024,image_data+size_t(c+1)*1024*1024,images[c].v.begin());}
 for(int i=0;i<n;i++)particles[i].xyz={positions[3*i],positions[3*i+1],positions[3*i+2]};vector<double>scores;auto flags=shake_particles(particles,images,cams,otf,&scores);
 for(int i=0;i<n;i++){for(int j=0;j<3;j++)positions[3*i+j]=particles[i].xyz[j];out_scores[i]=scores[i];out_flags[i]=flags[i];}return 0;
 }catch(const std::exception&e){std::cerr<<"numeric_shake: "<<e.what()<<"\n";return 1;}}
API int numeric_project(const double*cal,const double*points,int n,double*out){try{Camera c=decode_camera(cal);for(int i=0;i<n;i++){P2 p=project({points[3*i],points[3*i+1],points[3*i+2]},c);out[2*i]=p.x;out[2*i+1]=p.y;}return 0;}catch(...){return 1;}}
API int numeric_rays(const double*cal,const double*points,int n,double*out){try{Camera c=decode_camera(cal);for(int i=0;i<n;i++){Ray r=ray({points[2*i],points[2*i+1]},c);for(int j=0;j<3;j++){out[6*i+j]=r.origin[j];out[6*i+3+j]=r.direction[j];}}return 0;}catch(...){return 1;}}
API int numeric_detect(const double*data,int rows,int cols,double threshold,double*out){Image im(rows,cols);std::copy(data,data+size_t(rows)*cols,im.v.begin());auto points=detect(im,threshold);for(size_t i=0;i<points.size();i++){out[2*i]=points[i].x;out[2*i+1]=points[i].y;}return int(points.size());}
API int numeric_otf(const double*data,int n,int rows,int cols,double*out){try{vector<Image>images;for(int i=0;i<n;i++){Image im(rows,cols);std::copy(data+size_t(i)*rows*cols,data+size_t(i+1)*rows*cols,im.v.begin());images.push_back(std::move(im));}auto p=estimate_otf(images);out[0]=p.a;out[1]=p.b;out[2]=p.c;return 0;}catch(...){return 1;}}
API int numeric_match(const double*cal,const double*coordinates,const int*counts,double tolerance,int capacity,double*out_candidates,int*candidate_count,double*out_selected,int*selected_count,double*out_positions){try{
 array<Camera,4>cams;array<vector<P2>,4>points;size_t offset=0;for(int c=0;c<4;c++){cams[c]=decode_camera(cal+38*c);for(int i=0;i<counts[c];i++)points[c].push_back({coordinates[offset+2*i],coordinates[offset+2*i+1]});offset+=2*counts[c];}
 vector<StereoCandidate>candidates;vector<array<int,4>>selected;auto p=stereo_match(points,cams,tolerance,&candidates,&selected);if(candidates.size()>size_t(capacity)||selected.size()>size_t(capacity))return 2;*candidate_count=int(candidates.size());*selected_count=int(selected.size());for(size_t i=0;i<candidates.size();i++){for(int c=0;c<4;c++)out_candidates[5*i+c]=candidates[i].ids[c];out_candidates[5*i+4]=candidates[i].error;}for(size_t i=0;i<selected.size();i++)for(int c=0;c<4;c++)out_selected[4*i+c]=selected[i][c];for(size_t i=0;i<p.size();i++)for(int c=0;c<3;c++)out_positions[3*i+c]=p[i].xyz[c];return int(p.size())==int(selected.size())?0:3;
}catch(const std::exception&e){std::cerr<<"numeric_match: "<<e.what()<<"\n";return 1;}}
'''
_LIBRARY = None
def library():
    """Compile the owned numerical source into a content-addressed temp cache.

    MSVC19.29 x64, C++17, /O2 /MD /fp:strict. Returns ctypes.CDLL.
    Compilation never writes into observations or task source directories.
    """
    global _LIBRARY
    if _LIBRARY is not None:return _LIBRARY
    if os.name!='nt':raise RuntimeError('Validated runtime requires Windows x64 MSVC19.29; no other platform is claimed')
    compiler_root=Path(os.environ.get('ProgramFiles(x86)','C:/Program Files (x86)'))/'Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30133'
    compiler=compiler_root/'bin/Hostx64/x64/cl.exe'
    if not compiler.is_file():raise RuntimeError('Required validated MSVC19.29 x64 compiler is unavailable')
    identity=hashlib.sha256((KERNEL_SOURCE+'MSVC1929/O2/MD/fp:strict').encode()).hexdigest()
    cache=Path(tempfile.gettempdir())/'imaging101_openlpt_numeric'/identity
    cache.mkdir(parents=True,exist_ok=True);dll=cache/'operators.dll'
    if not dll.is_file():
        vcvars=compiler_root.parents[2]/'Auxiliary/Build/vcvars64.bat'
        batch=cache/'compiler_environment.cmd'
        batch.write_text('@call "'+str(vcvars)+'" >nul\n@set\n',encoding='utf8')
        capture=subprocess.run(['cmd.exe','/d','/c',str(batch)],capture_output=True,text=True,errors='replace')
        if capture.returncode:raise RuntimeError('Unable to initialize the validated compiler environment')
        env=os.environ.copy()
        for line in capture.stdout.splitlines():
            if '=' in line and not line.startswith('='):
                k,v=line.split('=',1);env[k]=v
        env['INCLUDE']=str(compiler_root/'include')+';'+env.get('INCLUDE','')
        env['LIB']=str(compiler_root/'lib/x64')+';'+env.get('LIB','')
        env['PATH']=str(compiler.parent)+';'+env.get('PATH','')
        source=cache/'operators.cpp';source.write_text(KERNEL_SOURCE,encoding='utf8')
        command=[str(compiler),'/nologo','/EHsc','/O2','/std:c++17','/MD','/fp:strict','/LD','/Fe:operators.dll','operators.cpp']
        start=time.perf_counter();r=subprocess.run(command,cwd=cache,env=env,capture_output=True,text=True,errors='replace')
        (cache/'compile.log').write_text(r.stdout+r.stderr,encoding='utf8')
        (cache/'compile.json').write_text(json.dumps({'command':command,'exit':r.returncode,'seconds':time.perf_counter()-start,'source_sha256':hashlib.sha256(KERNEL_SOURCE.encode()).hexdigest(),'architecture':'x64','runtime':'/MD','floating_point':'/fp:strict'},indent=2),encoding='utf8')
        if r.returncode:raise RuntimeError('Local numeric compiler failed: '+r.stdout+r.stderr)
    lib=ctypes.CDLL(str(dll));dp=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');ip=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');up=np.ctypeslib.ndpointer(dtype=np.uint32,flags='C_CONTIGUOUS')
    lib.numeric_tracking.argtypes=[dp,up,ctypes.c_int,ctypes.c_int,dp,ip,dp,dp]
    lib.numeric_project.argtypes=[dp,dp,ctypes.c_int,dp]
    lib.numeric_rays.argtypes=[dp,dp,ctypes.c_int,dp]
    lib.numeric_detect.argtypes=[dp,ctypes.c_int,ctypes.c_int,ctypes.c_double,dp]
    lib.numeric_otf.argtypes=[dp,ctypes.c_int,ctypes.c_int,ctypes.c_int,dp]
    lib.numeric_predict.argtypes=[dp,ctypes.c_int,dp]
    lib.numeric_field.argtypes=[dp,ctypes.c_int,dp,ctypes.c_int,dp]
    lib.numeric_shake.argtypes=[dp,dp,dp,dp,ctypes.c_int,ctypes.c_int,dp,ip]
    lib.numeric_ipr.argtypes=[dp,dp,dp,ctypes.c_int,dp]
    lib.numeric_residual.argtypes=[dp,dp,dp,dp,ctypes.c_int,dp]
    lib.numeric_match.argtypes=[dp,dp,ip,ctypes.c_double,ctypes.c_int,dp,ip,dp,ip,dp]
    _LIBRARY=lib;return lib
def project(points,camera):
    points=np.ascontiguousarray(points,dtype=np.float64);out=np.empty((len(points),2))
    if library().numeric_project(np.ascontiguousarray(camera),points,len(points),out):raise RuntimeError('Projection failed')
    return out
def rays(points,camera):
    points=np.ascontiguousarray(points,dtype=np.float64);out=np.empty((len(points),6))
    if library().numeric_rays(np.ascontiguousarray(camera),points,len(points),out):raise RuntimeError('Ray inversion failed')
    return out
def detect(image,threshold=30.):
    image=np.ascontiguousarray(image,dtype=np.float64);out=np.empty((image.size,2))
    n=library().numeric_detect(image,*image.shape,float(threshold),out)
    if n<0:raise RuntimeError('Detection failed')
    return out[:n].copy()
def estimate_otf(images):
    images=np.ascontiguousarray(images,dtype=np.float64);out=np.empty(3)
    if library().numeric_otf(images,*images.shape,out):raise RuntimeError('OTF estimate failed')
    return out
def predict(history):
    history=np.ascontiguousarray(history,dtype=np.float64);out=np.empty(3)
    if library().numeric_predict(history,len(history),out):raise ValueError('Need at least three 3D positions')
    return out
def refine(positions,images,cameras,otf,active_mask=15):
    positions=np.array(positions,dtype=np.float64,order='C',copy=True);images=np.ascontiguousarray(images,dtype=np.float64);scores=np.empty(len(positions));flags=np.empty(len(positions),np.int32)
    if library().numeric_shake(np.ascontiguousarray(cameras),np.ascontiguousarray(otf),images,positions,len(positions),active_mask,scores,flags):raise RuntimeError('Refinement failed')
    return positions,scores,flags
def residual(positions,images,cameras,otf):
    positions=np.ascontiguousarray(positions,dtype=np.float64);images=np.ascontiguousarray(images,dtype=np.float64);out=np.empty_like(images)
    if library().numeric_residual(np.ascontiguousarray(cameras),np.ascontiguousarray(otf),images,positions,len(positions),out):raise RuntimeError('Residual failed')
    return out
def reconstruct_frame(images,cameras,otf):
    out=np.empty((50000,3));n=library().numeric_ipr(np.ascontiguousarray(cameras),np.ascontiguousarray(otf),np.ascontiguousarray(images,dtype=np.float64),len(out),out)
    if n<0:raise RuntimeError('IPR failed')
    return out[:n].copy()
def stereo_diagnostics(points,cameras,tolerance=.8):
    cameras=np.ascontiguousarray(cameras);counts=np.array([len(p)for p in points],np.int32);coordinates=np.ascontiguousarray(np.concatenate(points));capacity=100000
    candidates=np.empty((capacity,5));selected=np.empty((capacity,4));positions=np.empty((capacity,3));nc=np.zeros(1,np.int32);ns=np.zeros(1,np.int32)
    if library().numeric_match(cameras,coordinates,counts,tolerance,capacity,candidates,nc,selected,ns,positions):raise RuntimeError('Stereo diagnostics failed')
    return candidates[:nc[0]].copy(),selected[:ns[0]].copy(),positions[:ns[0]].copy()
def triangulate_candidates(points,cameras,candidate_ids):
    ids=np.asarray(candidate_ids,dtype=np.int64);A=np.zeros((len(ids),3,3));b=np.zeros((len(ids),3))
    for c in range(4):
        valid=ids[:,c]>=0;ray=rays(np.asarray(points[c])[ids[valid,c]],cameras[c]);d=ray[:,3:];Q=np.eye(3)[None]-d[:,:,None]*d[:,None,:];A[valid]+=Q;b[valid]+=np.einsum('nij,nj->ni',Q,ray[:,:3])
    return np.linalg.solve(A,b[...,None])[...,0]
