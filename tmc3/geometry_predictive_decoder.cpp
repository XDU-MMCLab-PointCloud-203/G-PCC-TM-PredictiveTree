
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

#include "geometry_predictive.h"
#include "geometry.h"
#include "hls.h"
#include "quantization.h"

#include <vector>

namespace pcc {

//============================================================================

class PredGeomDecoder : protected PredGeomContexts {
public:
  PredGeomDecoder(const PredGeomDecoder&) = delete;
  PredGeomDecoder& operator=(const PredGeomDecoder&) = delete;

  PredGeomDecoder(
    const GeometryParameterSet&,
    const GeometryBrickHeader& gbh,
    const PredGeomContexts& ctxtMem,
    EntropyDecoder* aed);

  /**
   * decodes a sequence of decoded geometry trees.
   * @returns the number of points decoded.
   */
  int decode(int numPoints, Vec3<int32_t>* outputPoints);

  /**
   * decodes a single predictive geometry tree.
   * @returns the number of points decoded.
   */
  int decodeTree(Vec3<int32_t>* outA, Vec3<int32_t>* outB);

  const PredGeomContexts& getCtx() const { return *this; }

private:
  int decodeNumDuplicatePoints();
  int decodeNumChildren();
  GPredicter::Mode decodePredMode();
  Vec3<int32_t> decodeResidual();
  Vec3<int32_t> decodeResidual2();
  int32_t decodePhiMultiplier(GPredicter::Mode mode);
  int32_t decodeQpOffset();

private:
  EntropyDecoder* _aed;
  std::vector<int32_t> _stack;
  std::vector<int32_t> _nodeIdxToParentIdx;
  bool _geom_unique_points_flag;

  bool _geom_angular_mode_enabled_flag;
  Vec3<int32_t> origin;
  int numLasers;
  SphericalToCartesian _sphToCartesian;
  int _geom_angular_azimuth_speed;

  bool _geom_scaling_enabled_flag;
  int _sliceQp;
  int _qpOffsetInterval;

  Vec3<int> _pgeom_resid_abs_log2_bits;
};

//============================================================================

PredGeomDecoder::PredGeomDecoder(
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  const PredGeomContexts& ctxtMem,
  EntropyDecoder* aed)
  : PredGeomContexts(ctxtMem)
  , _aed(aed)
  , _geom_unique_points_flag(gps.geom_unique_points_flag)
  , _geom_angular_mode_enabled_flag(gps.geom_angular_mode_enabled_flag)
  , origin()
  , numLasers(gps.geom_angular_theta_laser.size())
  , _sphToCartesian(gps)
  , _geom_angular_azimuth_speed(gps.geom_angular_azimuth_speed)
  , _geom_scaling_enabled_flag(gps.geom_scaling_enabled_flag)
  , _sliceQp(0)
  , _pgeom_resid_abs_log2_bits(gbh.pgeom_resid_abs_log2_bits)
{
  if (gps.geom_scaling_enabled_flag) {
    _sliceQp = gbh.sliceQp(gps);
    int qpIntervalLog2 =
      gps.geom_qp_offset_intvl_log2 + gbh.geom_qp_offset_intvl_log2_delta;
    _qpOffsetInterval = (1 << qpIntervalLog2) - 1;
  }

  if (gps.geom_angular_mode_enabled_flag)
    origin = gps.geomAngularOrigin - gbh.geomBoxOrigin;

  _stack.reserve(1024);
}

//----------------------------------------------------------------------------

int
PredGeomDecoder::decodeNumDuplicatePoints()//解码重复点数
{
  bool num_dup_points_gt0 = _aed->decode(_ctxNumDupPointsGt0);
  if (!num_dup_points_gt0)
    return 0;
  return 1 + _aed->decodeExpGolomb(0, _ctxNumDupPoints);//用指数哥伦布解码重复点数
}

//----------------------------------------------------------------------------

int
PredGeomDecoder::decodeNumChildren()//解码ChildrenCount，ChildrenCount用2个比特表示0，1，2，3四种情况
{
  int numChildren = _aed->decode(_ctxNumChildren[0]);//分配一个上下文，解码低比特位的值
  numChildren += _aed->decode(_ctxNumChildren[1 + numChildren]) << 1;//向左移一位，分配一个上下文，解码高比特位
  return numChildren;
}

//----------------------------------------------------------------------------

GPredicter::Mode
PredGeomDecoder::decodePredMode()//解码预测模式，预测模式用2个比特表示0，1，2，3四种情况
{
  int mode = _aed->decode(_ctxPredMode[0]);//分配一个上下文，解码低比特位的值
  mode += _aed->decode(_ctxPredMode[1 + mode]) << 1;//向左移一位，分配一个上下文，解码高比特位
  return GPredicter::Mode(mode);
}

//----------------------------------------------------------------------------

Vec3<int32_t>
PredGeomDecoder::decodeResidual2()//仅用于Cat3_frame的残差(r_x,r_y,r_z)
{
  Vec3<int32_t> residual;
  for (int k = 0; k < 3; ++k) {
    if (_aed->decode(_ctxIsZero2[k])) {//分配一个上下文解码(r_x,r_y,r_z)全部为0的情况
      residual[k] = 0;
      continue;
    }

    auto sign = _aed->decode(_ctxSign2[k]);//解码(r_x,r_y,r_z)的符号位

    if (_aed->decode(_ctxIsOne2[k])) {
      residual[k] = sign ? 1 : -1;
      continue;
    }

    auto& ctxs = _ctxResidual2[k];
    int32_t value = _aed->decode(ctxs[0]);
    value += _aed->decode(ctxs[1 + (value & 1)]) << 1;
    value += _aed->decode(ctxs[3 + (value & 3)]) << 2;
    value += _aed->decode(ctxs[7 + (value & 7)]) << 3;//当(r_x,r_y,r_z)-2的值小于15时，分配15个上下文解码残差(r_x,r_y,r_z)-2的值
    if (value == 15)
      value += _aed->decodeExpGolomb(0, _ctxEG2[k]);//当(r_x,r_y,r_z)-2的值大于等于15时，通过指数哥伦布解码残差

    residual[k] = sign ? (value + 2) : -(value + 2);//解码后的(r_x,r_y,r_z)加2，恢复真实的(r_x,r_y,r_z)
  }
  return residual;
}

//----------------------------------------------------------------------------

int32_t
PredGeomDecoder::decodePhiMultiplier(GPredicter::Mode mode)//仅用于Cat3_frame预测模式为Delta的Phi
{
  if (!_geom_angular_mode_enabled_flag || mode != GPredicter::Mode::Delta)//Cat3_fused或者预测模式不是Delta时，跳出
    return 0;

  if (_aed->decode(_ctxIsZeroPhi))//分配一个上下文解码Phi=0的情况
    return 0;

  const auto sign = _aed->decode(_ctxSignPhi);//解码Phi的符号位
  if (_aed->decode(_ctxIsOnePhi))
    return sign ? 1 : -1;

  auto& ctxs = _ctxResidualPhi;
  int32_t value = _aed->decode(ctxs[0]);
  value += _aed->decode(ctxs[1 + (value & 1)]) << 1;
  value += _aed->decode(ctxs[3 + (value & 3)]) << 2;
  value += _aed->decode(ctxs[7 + (value & 7)]) << 3;//当Phi-2的值小于15时，分配15个上下文解码残差Phi-2的值
  if (value == 15)
    value += _aed->decodeExpGolomb(0, _ctxEGPhi);//当Phi-2的值大于等于15时，采用指数哥伦布解码残差Phi-2的值

  return sign ? (value + 2) : -(value + 2);//解码的Phi+2得到真实值Phi
}

//----------------------------------------------------------------------------

int32_t
PredGeomDecoder::decodeQpOffset()
{
  int dqp = 0;
  if (!_aed->decode(_ctxQpOffsetIsZero)) {
    int dqp_sign = _aed->decode(_ctxQpOffsetSign);//解码符号位
    dqp = _aed->decodeExpGolomb(0, _ctxQpOffsetAbsEgl) + 1;//调用指数跟伦布解码dqp值
    dqp = dqp_sign ? dqp : -dqp;//符号位赋给解码的dqp值
  }
  return dqp;
}

//----------------------------------------------------------------------------

Vec3<int32_t>
PredGeomDecoder::decodeResidual()//cat3_frame解码残差(r_r,r_ϕ,r_i);cat3_fused解码残差(r_x,r_y,r_z)
{
  Vec3<int32_t> residual;
  for (int k = 0, ctxIdx = 0; k < 3; ++k) {
    if (_aed->decode(_ctxIsZero[k])) {//分配一个上下文解码残差全为0的情况
      residual[k] = 0;
      continue;
    }

    auto sign = _aed->decode(_ctxSign[k]);//解码残差的符号位

    AdaptiveBitModel* ctxs = _ctxNumBits[ctxIdx][k] - 1;
    int32_t numBits = 1;//numBits初始化
    for (int n = 0; n < _pgeom_resid_abs_log2_bits[k]; n++)//_pgeom_resid_abs_log2_bits[k]表示残差的位数
      numBits = (numBits << 1) | _aed->decode(ctxs[numBits]);//采用熵编码解码残差的位数
    numBits ^= 1 << _pgeom_resid_abs_log2_bits[k];//残差位数是从little endian order开始

    if (!k && !_geom_angular_mode_enabled_flag)
      ctxIdx = (numBits + 1) >> 1;//cat3_fused的残差需要分配上下文的个数

    int32_t res = 0;
    --numBits;
    if (numBits <= 0) {//当残差位数小于等于0时
      res = 2 + numBits;//残差值等于numBits+2
    } else {//当残差位数大于0时
      res = 1 + (1 << numBits);
      for (int i = 0; i < numBits; ++i) {
        res += _aed->decode() << i;//按照numBits从little endian order开始，解码残差值
      }
    }
    residual[k] = sign ? res : -res;//将符号位赋给残差
  }

  return residual;
}

//----------------------------------------------------------------------------

int
PredGeomDecoder::decodeTree(Vec3<int32_t>* outA, Vec3<int32_t>* outB)
{
  QuantizerGeom quantizer(_sliceQp);
  int nodesUntilQpOffset = 0;
  int nodeCount = 0;
  _stack.push_back(-1);//对堆栈进行初始化

  while (!_stack.empty()) {//利用堆栈，解码端获得建立的树结构
    auto parentNodeIdx = _stack.back();
    _stack.pop_back();//根据孩子节点个数，在堆栈中压入相应个数的parentNodeIdx 

    if (_geom_scaling_enabled_flag && !nodesUntilQpOffset--) {
      int qpOffset = decodeQpOffset();
      int qp = _sliceQp + qpOffset;
      quantizer = QuantizerGeom(qp);
      nodesUntilQpOffset = _qpOffsetInterval;
    }

    // allocate point in traversal order (depth first)
    auto curNodeIdx = nodeCount++;//解码完一个点，则curNodeIdx加1
    _nodeIdxToParentIdx[curNodeIdx] = parentNodeIdx;//根据curNodeIdx，获得当前点父节点的parentNodeIdx

    int numDuplicatePoints = 0;
    if (!_geom_unique_points_flag)//_geom_unique_points_flag为判断当前点是否存在重复点
      numDuplicatePoints = decodeNumDuplicatePoints();//解码重复点数
    int numChildren = decodeNumChildren();//解码每个节点的孩子个数
    auto mode = decodePredMode();//解码预测模式
    int qphi = decodePhiMultiplier(mode);//当预测模式为Delta时，解码出qphi

    auto residual = decodeResidual();//cat3_frame解码残差(r_r,r_ϕ,r_i);cat3_fused解码残差(r_x,r_y,r_z)
    if (!_geom_angular_mode_enabled_flag)//判断数据类型为cat3_fused
      for (int k = 0; k < 3; k++)
        residual[k] = int32_t(quantizer.scale(residual[k]));//对cat3_fused解码残差(r_x,r_y,r_z)进行量化

    auto predicter = makePredicter(
      curNodeIdx, mode, [&](int idx) { return _nodeIdxToParentIdx[idx]; });

    auto pred = predicter.predict(outA, mode);//根据预测模式计算预测值
    if (_geom_angular_mode_enabled_flag)
      if (mode == GPredicter::Mode::Delta)
        pred[1] += qphi * _geom_angular_azimuth_speed;//预测模式为Delta时，根据已解码的qphi和先验信息speed获得预测值ϕ

    auto pos = pred + residual;// 预测值和残差相加计算真实值：cat3_fused的真实(x,y,z)，cat3_frame的真实(r,ϕ,i)


    if (!_geom_angular_mode_enabled_flag)//cat3_fused数据
      for (int k = 0; k < 3; k++)
        pos[k] = std::max(0, pos[k]);
    outA[curNodeIdx] = pos;

    // convert pos from spherical to cartesian, add secondary residual
    if (_geom_angular_mode_enabled_flag) {//cat3_frame数据
      residual = decodeResidual2();//解码残差(r_x,r_y,r_z)
      for (int k = 0; k < 3; k++)
        residual[k] = int32_t(quantizer.scale(residual[k]));//对残差(r_x,r_y,r_z)量化

      assert(pos[2] < numLasers && pos[2] >= 0);//判断解码的i在区间[0,numLasers-1]之间
      pred = origin + _sphToCartesian(pos);//将真实(r,ϕ,i)转换成球坐标，加上偏移坐标origin
      outB[curNodeIdx] = pred + residual;//加上残差(r_x,r_y,r_z)，恢复当前点的真实坐标(x,y,z)
      for (int k = 0; k < 3; k++)
        outB[curNodeIdx][k] = std::max(0, outB[curNodeIdx][k]);
    }

    // copy duplicate point output
    for (int i = 0; i < numDuplicatePoints; i++, nodeCount++) {//重复点和当前点的具有完全相同的几何信息
      outA[nodeCount] = outA[curNodeIdx];//cat3_fused：(x,y,z)；cat3_frame：(r,ϕ,i)
      outB[nodeCount] = outB[curNodeIdx];//cat3_frame：(x,y,z)
    }

    for (int i = 0; i < numChildren; i++)
      _stack.push_back(curNodeIdx);//堆栈：解码完一个点，移除一个curNodeIdx
  }

  return nodeCount;//返回解码点的数目
}

//----------------------------------------------------------------------------

int
PredGeomDecoder::decode(int numPoints, Vec3<int32_t>* outputPoints)
{
  _nodeIdxToParentIdx.resize(numPoints);//根据当前节点nodeIdx找到对应的父节点的ParentIdx

  // An intermediate buffer used for reconstruction of the spherical
  // co-ordinates.
  auto* reconA = outputPoints;
  std::vector<Vec3<int32_t>> sphericalPos;
  if (_geom_angular_mode_enabled_flag) {//表示Cat3_frame类型
    sphericalPos.resize(numPoints);//开辟一个numPoints大小的buffer
    reconA = sphericalPos.data();//将重建的球坐标转换成笛卡尔坐标，并存储到这个buffer里
  }

  int32_t pointCount = 0;
  while (pointCount < numPoints) {//判断是否全部点完成解码
    auto numSubtreePoints = decodeTree(reconA, outputPoints);//通过调用decodeTree，解码预测树节点的几何、节点数等信息
    outputPoints += numSubtreePoints;
    reconA += numSubtreePoints;//解码后树的重建点数累加
    pointCount += numSubtreePoints;//解码后树的节点累加
  }

  return pointCount;//返回解码后树的节点数
}

//============================================================================

void
decodePredictiveGeometry(
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  PCCPointSet3& pointCloud,
  PredGeomContexts& ctxtMem,
  EntropyDecoder* aed)
{
  PredGeomDecoder dec(gps, gbh, ctxtMem, aed);//解码预测树的头信息
  dec.decode(gbh.footer.geom_num_points_minus1 + 1, &pointCloud[0]);//gbh.footer.geom_num_points_minus1 + 1表示构建树的点数
  ctxtMem = dec.getCtx();//解码有关的上下文
}

//============================================================================

}  // namespace pcc
