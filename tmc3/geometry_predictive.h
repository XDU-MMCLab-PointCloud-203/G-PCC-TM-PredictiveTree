/* The copyright in this software is being made available under the BSD
 * Licence, included below.  This software may be subject to other third
 * party and contributor rights, including patent rights, and no such
 * rights are granted under this licence.
 *
 * Copyright (c) 2020, ISO/IEC
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright
 *   notice, this list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright
 *   notice, this list of conditions and the following disclaimer in the
 *   documentation and/or other materials provided with the distribution.
 *
 * * Neither the name of the ISO/IEC nor the names of its contributors
 *   may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <cmath>
#include <cstdint>

#include "entropy.h"
#include "PCCMath.h"
#include "hls.h"

namespace pcc {

//============================================================================

struct GPredicter {//预测器结构
  enum Mode
  {
    None,
    Delta,
    Linear2,
    Linear3
  };//预测模式，None:不预测; Delta:用父节点预测p0; Linear2:用父节点和祖父节点预测2p0 - p1; Linear3:用父节点、祖父节点和曾祖父节点预测p0+p1-p2

  int32_t index[3];//索引数组，用于存放用于预测的节点的索引值

  bool isValid(Mode mode);//模式有效性判断函数
  Vec3<int32_t> predict(const Vec3<int32_t>* points, Mode mode);//预测值计算函数
};

//============================================================================
//预测树节点结构
struct GNode {
  static const int32_t MaxChildrenCount = 3;//限定的最大子节点数

  int numDups;//重复点数
  int32_t parent;//父节点
  int32_t childrenCount;//子节点数量，用于计数
  int32_t children[MaxChildrenCount];//子节点数组
};

//============================================================================
//预测几何编码用到的上下文二
class PredGeomContexts {
public:
  void reset();

protected:
  AdaptiveBitModel _ctxNumChildren[3];
  AdaptiveBitModel _ctxPredMode[3];
  AdaptiveBitModel _ctxIsZero[3];
  AdaptiveBitModel _ctxSign[3];
  AdaptiveBitModel _ctxNumBits[12][3][31];
  AdaptiveBitModel _ctxNumDupPointsGt0;
  AdaptiveBitModel _ctxNumDupPoints;

  AdaptiveBitModel _ctxIsZero2[3];
  AdaptiveBitModel _ctxIsOne2[3];
  AdaptiveBitModel _ctxSign2[3];
  AdaptiveBitModel _ctxEG2[3];

  AdaptiveBitModel _ctxResidual2[3][15];

  AdaptiveBitModel _ctxQpOffsetIsZero;
  AdaptiveBitModel _ctxQpOffsetSign;
  AdaptiveBitModel _ctxQpOffsetAbsEgl;

  AdaptiveBitModel _ctxIsZeroPhi;
  AdaptiveBitModel _ctxIsOnePhi;
  AdaptiveBitModel _ctxSignPhi;
  AdaptiveBitModel _ctxEGPhi;
  AdaptiveBitModel _ctxResidualPhi[15];
};

//----------------------------------------------------------------------------

inline void
PredGeomContexts::reset()
{
  this->~PredGeomContexts();
  new (this) PredGeomContexts;
}

//============================================================================
//构建预测器，即对相应模式，构建相应数量的预测器
template<typename LookupFn>
GPredicter
makePredicter(
  int32_t curNodeIdx, GPredicter::Mode mode, LookupFn nodeIdxToParentIdx)
{
  GPredicter predIdx;//创建预测器结构，存放索引值
  memset(&predIdx, 0, sizeof(GPredicter));//初始化
  switch (mode) {//选择模式
  default:
  case GPredicter::None:
  case GPredicter::Delta:
  case GPredicter::Linear2:
  case GPredicter::Linear3:
    for (int i = 0; i < int(mode); i++) {//遍历所选模式
      if (curNodeIdx < 0)//判断索引是否有效
        break;
      predIdx.index[i] = curNodeIdx = nodeIdxToParentIdx(curNodeIdx);//根据当前节点索引获取当前节点的父节点的索引，存放到predIdx中
    }
    break;
  }
  return predIdx;
}

//============================================================================
//判断预测模式是否有效
inline bool
GPredicter::isValid(GPredicter::Mode mode)
{
  int numPredictors = int(mode);//获取预测模式
  for (int i = 0; i < numPredictors; i++) {//遍历当前预测模式下的所有预测器
    if (this->index[i] < 0)//若对应的预测器不存在，则返回false
      return false;
  }
  return true;
}

//============================================================================
//根据所选预测模式计算预测值
inline Vec3<int32_t>
GPredicter::predict(const Vec3<int32_t>* points, GPredicter::Mode mode)
{
  Vec3<int32_t> pred;//存放预测值
  switch (mode) {//选择预测模式
  case GPredicter::None: pred = 0; break;//不预测

  case GPredicter::Delta: {//用父节点预测
    pred = points[this->index[0]];
    break;
  }

  case GPredicter::Linear2: {//用父节点和祖父节点预测
    const auto& p0 = points[this->index[0]];
    const auto& p1 = points[this->index[1]];
    pred = 2 * p0 - p1;
    break;
  }

  default:
  case GPredicter::Linear3: {//用父节点、祖父节点和曾祖父节点预测
    const auto& p0 = points[this->index[0]];
    const auto& p1 = points[this->index[1]];
    const auto& p2 = points[this->index[2]];
    pred = p0 + p1 - p2;
    break;
  }
  }
  return pred;//返回计算结果
}

//============================================================================
//球坐标到笛卡尔坐标转换类
class SphericalToCartesian {
public:
  SphericalToCartesian(const GeometryParameterSet& gps)//有参构造值传递
    : log2ScaleRadius(gps.geom_angular_radius_inv_scale_log2)
    , log2ScalePhi(gps.geom_angular_azimuth_scale_log2)
    , tanThetaLaser(gps.geom_angular_theta_laser.data())
    , zLaser(gps.geom_angular_z_laser.data())
  {}

  Vec3<int32_t> operator()(Vec3<int32_t> sph)
  {
    int64_t r = sph[0] << log2ScaleRadius;
    int64_t z = divExp2RoundHalfInf(
      tanThetaLaser[sph[2]] * r << 2, log2ScaleTheta - log2ScaleZ);

    return Vec3<int32_t>(Vec3<int64_t>{
      divExp2RoundHalfInf(r * icos(sph[1], log2ScalePhi), kLog2ISineScale),
      divExp2RoundHalfInf(r * isin(sph[1], log2ScalePhi), kLog2ISineScale),
      divExp2RoundHalfInf(z - zLaser[sph[2]], log2ScaleZ)});//计算得到笛卡尔坐标(x, y, z)
  }

private:
  static constexpr int log2ScaleZ = 3;
  static constexpr int log2ScaleTheta = 20;
  int log2ScaleRadius;
  int log2ScalePhi;
  const int* tanThetaLaser;
  const int* zLaser;
};

//============================================================================
//笛卡尔坐标到球坐标转换类
class CartesianToSpherical {
public:
  CartesianToSpherical(const GeometryParameterSet& gps)//有参构造值传递
    : sphToCartesian(gps)
    , log2ScaleRadius(gps.geom_angular_radius_inv_scale_log2)
    , scalePhi(1 << gps.geom_angular_azimuth_scale_log2)
    , numLasers(gps.geom_angular_theta_laser.size())
    , tanThetaLaser(gps.geom_angular_theta_laser.data())//cfg配置中提供
    , zLaser(gps.geom_angular_z_laser.data())//cfg配置中提供
  {}

  Vec3<int32_t> operator()(Vec3<int32_t> xyz)//()运算符重载，输入xyz坐标
  {
    int64_t r0 = int64_t(std::round(hypot(xyz[0], xyz[1])));//计算坐标r，x,y开平方根
    int32_t thetaIdx = 0;//定义第三球坐标thetaIdx，laser对应索引i
    int32_t minError = std::numeric_limits<int32_t>::max();
    for (int idx = 0; idx < numLasers; ++idx) {//遍历所有laser
      int64_t z = divExp2RoundHalfInf(//得到每个激光同一个r0下相对于head对应的高度z
        tanThetaLaser[idx] * r0 << 2, log2ScaleTheta - log2ScaleZ);
      int64_t z1 = divExp2RoundHalfInf(z - zLaser[idx], log2ScaleZ);//每个lidar垂直方向相对于head的偏差z1
      int32_t err = abs(z1 - xyz[2]);//计算误差,xyz[2]是当前点的相对于origin(暂时看作lidar的head)的z,计算所有激光与当前点的该值的误差，将最小误差的lidar作为该点对应的lidar
      if (err < minError) {//根据lidar最小误差得到对应激光索引thetaIdx
        thetaIdx = idx;
        minError = err;
      }
    }

    auto phi0 = std::round((atan2(xyz[1], xyz[0]) / (2.0 * M_PI)) * scalePhi);//计算phi

    Vec3<int32_t> sphPos{int32_t(divExp2RoundHalfUp(r0, log2ScaleRadius)),
                         int32_t(phi0), thetaIdx};//得到点对应的球坐标

    // local optmization
    // 做局部优化
    auto minErr = (sphToCartesian(sphPos) - xyz).getNorm1();//球坐标到直角坐标后与原始坐标做差的结果求和
    int32_t dt0 = 0;
    int32_t dr0 = 0;
    for (int32_t dt = -2; dt <= 2 && minErr; ++dt) {//minErr=0误差为0
      for (int32_t dr = -2; dr <= 2; ++dr) {
        auto sphPosCand = sphPos + Vec3<int32_t>{dr, dt, 0};
        auto err = (sphToCartesian(sphPosCand) - xyz).getNorm1();//再次计算误差
        if (err < minErr) {//选择最小误差，做优化
          minErr = err;
          dt0 = dt;
          dr0 = dr;
        }
      }
    }
    sphPos[0] += dr0;
    sphPos[1] += dt0;

    return sphPos;//返回得到的球坐标
  }

private:
  SphericalToCartesian sphToCartesian;
  static constexpr int32_t log2ScaleZ = 3;
  static constexpr int32_t log2ScaleTheta = 20;
  int32_t log2ScaleRadius;
  int32_t scalePhi;
  int numLasers;
  const int* tanThetaLaser;
  const int* zLaser;
};

//============================================================================

}  // namespace pcc
