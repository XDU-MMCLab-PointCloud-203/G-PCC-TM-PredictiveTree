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
#include "pointset_processing.h"
#include "quantization.h"

#include "PCCMisc.h"

#include "nanoflann.hpp"

namespace pcc {

//============================================================================

namespace {
  struct NanoflannCloud {
    std::vector<Vec3<int32_t>> pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    int32_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
      return pts[idx][dim];
    }

    template<class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
      return false;
    }
  };
}  // namespace

//============================================================================

namespace {
  float estimate(bool bit, AdaptiveBitModel model)
  {
    return -log2(dirac::approxSymbolProbability(bit, model) / 128.);
  }
}  // namespace

//============================================================================

class PredGeomEncoder : protected PredGeomContexts {
public:
  PredGeomEncoder(const PredGeomEncoder&) = delete;
  PredGeomEncoder& operator=(const PredGeomEncoder&) = delete;

  PredGeomEncoder(
    const GeometryParameterSet&,
    const GeometryBrickHeader&,
    const PredGeomContexts& ctxtMem,
    EntropyEncoder* aec);

  int qpSelector(const GNode& node) const { return _sliceQp; }

  void encode(
    const Vec3<int32_t>* cloudA,
    Vec3<int32_t>* cloudB,
    const GNode* nodes,
    int numNodes,
    int* codedOrder);

  int encodeTree(
    const Vec3<int32_t>* cloudA,
    Vec3<int32_t>* cloudB,
    const GNode* nodes,
    int numNodes,
    int rootIdx,
    int* codedOrder);

  void encodeNumDuplicatePoints(int numDupPoints);
  void encodeNumChildren(int numChildren);
  void encodePredMode(GPredicter::Mode mode);
  void encodeResidual(const Vec3<int32_t>& residual);
  void encodeResidual2(const Vec3<int32_t>& residual);
  void encodePhiMultiplier(const int32_t multiplier);
  void encodeQpOffset(int dqp);

  float estimateBits(GPredicter::Mode mode, const Vec3<int32_t>& residual);

  const PredGeomContexts& getCtx() const { return *this; }

private:
  EntropyEncoder* _aec;
  std::vector<int32_t> _stack;
  bool _geom_unique_points_flag;

  bool _geom_angular_mode_enabled_flag;
  Vec3<int32_t> origin;
  SphericalToCartesian _sphToCartesian;
  int _geom_angular_azimuth_speed;

  bool _geom_scaling_enabled_flag;
  int _sliceQp;
  int _qpOffsetInterval;

  Vec3<int> _maxAbsResidualMinus1Log2;
  Vec3<int> _pgeom_resid_abs_log2_bits;
};

//============================================================================

PredGeomEncoder::PredGeomEncoder(
  const GeometryParameterSet& gps,
  const GeometryBrickHeader& gbh,
  const PredGeomContexts& ctxtMem,
  EntropyEncoder* aec)
  : PredGeomContexts(ctxtMem)
  , _aec(aec)
  , _geom_unique_points_flag(gps.geom_unique_points_flag)
  , _geom_angular_mode_enabled_flag(gps.geom_angular_mode_enabled_flag)
  , origin()
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

  for (int k = 0; k < 3; k++)
    _maxAbsResidualMinus1Log2[k] = (1 << gbh.pgeom_resid_abs_log2_bits[k]) - 1;
}

//----------------------------------------------------------------------------
//编码重复点数量
void
PredGeomEncoder::encodeNumDuplicatePoints(int numDupPoints)
{
  _aec->encode(numDupPoints > 0, _ctxNumDupPointsGt0);
  if (numDupPoints)
    _aec->encodeExpGolomb(numDupPoints - 1, 0, _ctxNumDupPoints);//用指数哥伦布编码重复点数
}

//----------------------------------------------------------------------------
//编码子节点数量
void
PredGeomEncoder::encodeNumChildren(int numChildren)//ChildrenCount用2个比特表示0，1，2，3四种情况
{
  _aec->encode(numChildren & 1, _ctxNumChildren[0]);//分配一个上下文，编码低比特位的值
  _aec->encode((numChildren >> 1) & 1, _ctxNumChildren[1 + (numChildren & 1)]);//向左移一位，分配一个上下文，编码高比特位
}

//----------------------------------------------------------------------------
//编码预测模式
void
PredGeomEncoder::encodePredMode(GPredicter::Mode mode)//预测模式用2个比特表示0，1，2，3四种情况
{
  int iMode = int(mode);
  _aec->encode(iMode & 1, _ctxPredMode[0]);//分配一个上下文，编码低比特位的值
  _aec->encode((iMode >> 1) & 1, _ctxPredMode[1 + (iMode & 1)]);//向左移一位，分配一个上下文，编码高比特位
}

//----------------------------------------------------------------------------
//编码残差
void
PredGeomEncoder::encodeResidual(const Vec3<int32_t>& residual)//cat3_frame编码残差(r_r,r_ϕ,r_i);cat3_fused编码残差(r_x,r_y,r_z)
{
  for (int k = 0, ctxIdx = 0; k < 3; k++) {
    const auto res = residual[k];
    const bool isZero = res == 0;
    _aec->encode(isZero, _ctxIsZero[k]);//分配一个上下文，编码残差等于0的情况
    if (isZero)
      continue;

    _aec->encode(res > 0, _ctxSign[k]);//分配一个上下文，编码残差符号位

    int32_t value = abs(res) - 1;
    int32_t numBits = 1 + ilog2(uint32_t(value));//计算value的位数

    AdaptiveBitModel* ctxs = _ctxNumBits[ctxIdx][k] - 1;
    for (int ctxIdx = 1, n = _pgeom_resid_abs_log2_bits[k] - 1; n >= 0; n--) {
      auto bin = (numBits >> n) & 1;
      _aec->encode(bin, ctxs[ctxIdx]);//采用熵编码编码残差的位数
      ctxIdx = (ctxIdx << 1) | bin;//从little endian order开始编码
    }

    if (!k && !_geom_angular_mode_enabled_flag)
      ctxIdx = (numBits + 1) >> 1;

    --numBits;
    for (int32_t i = 0; i < numBits; ++i)
      _aec->encode((value >> i) & 1);//编码残差
  }
}

//----------------------------------------------------------------------------
//编码第二残差，仅用于Cat3_frame的残差(r_x,r_y,r_z)
void
PredGeomEncoder::encodeResidual2(const Vec3<int32_t>& residual)
{
  for (int k = 0; k < 3; k++) {
    const auto res = residual[k];
    const bool isZero = res == 0;
    _aec->encode(isZero, _ctxIsZero2[k]);//分配一个上下文，编码残差等于0的情况
    if (isZero)
      continue;

    _aec->encode(res > 0, _ctxSign2[k]);//分配一个上下文，编码残差符号位

    const bool isOne = res == 1 || res == -1;
    _aec->encode(isOne, _ctxIsOne2[k]);//分配一个上下文，编码残差绝对值是1的情况
    if (isOne)
      continue;

    int32_t value = abs(res) - 2;
    auto& ctxs = _ctxResidual2[k];
    if (value < 15) {
      _aec->encode(value & 1, ctxs[0]);
      _aec->encode((value >> 1) & 1, ctxs[1 + (value & 1)]);
      _aec->encode((value >> 2) & 1, ctxs[3 + (value & 3)]);
      _aec->encode((value >> 3) & 1, ctxs[7 + (value & 7)]);//当value < 15时，分配15个上下文
    } else {
      _aec->encode(1, ctxs[0]);
      _aec->encode(1, ctxs[2]);
      _aec->encode(1, ctxs[6]);
      _aec->encode(1, ctxs[14]);
      _aec->encodeExpGolomb(value - 15, 0, _ctxEG2[k]);//当value >=15时，采用指数哥伦布编码
    }
  }
}

//----------------------------------------------------------------------------
//编码phi
void
PredGeomEncoder::encodePhiMultiplier(int32_t multiplier)
{
  bool isZero = multiplier == 0;
  _aec->encode(isZero, _ctxIsZeroPhi);//分配一个上下文，编码phi等于0的情况
  if (isZero)
    return;

  _aec->encode(multiplier > 0, _ctxSignPhi);//分配一个上下文，编码phi符号位

  int32_t value = abs(multiplier) - 1;
  bool isOne = !value;
  _aec->encode(isOne, _ctxIsOnePhi);//分配一个上下文，编码phi等于1的情况
  if (isOne)
    return;

  value--;
  auto& ctxs = _ctxResidualPhi;
  if (value < 15) {
    _aec->encode(value & 1, ctxs[0]);
    _aec->encode((value >> 1) & 1, ctxs[1 + (value & 1)]);
    _aec->encode((value >> 2) & 1, ctxs[3 + (value & 3)]);
    _aec->encode((value >> 3) & 1, ctxs[7 + (value & 7)]);//当value < 15时，分配15个上下文
  } else {
    _aec->encode(1, ctxs[0]);
    _aec->encode(1, ctxs[2]);
    _aec->encode(1, ctxs[6]);
    _aec->encode(1, ctxs[14]);
    _aec->encodeExpGolomb(value - 15, 0, _ctxEGPhi);//当value >=15时，采用指数哥伦布编码
  }
}

//----------------------------------------------------------------------------
//编码QP偏移
void
PredGeomEncoder::encodeQpOffset(int dqp)
{
  _aec->encode(dqp == 0, _ctxQpOffsetIsZero);
  if (dqp == 0) {
    return;
  }
  _aec->encode(dqp > 0, _ctxQpOffsetSign);
  _aec->encodeExpGolomb(abs(dqp) - 1, 0, _ctxQpOffsetAbsEgl);
}

//----------------------------------------------------------------------------
//根据mode得到对应残差(r_r,r_ϕ,r_i)，估计编码所需比特数，选择比特数最小的作为最佳mode
float
PredGeomEncoder::estimateBits(
  GPredicter::Mode mode, const Vec3<int32_t>& residual)
{
  int iMode = int(mode);//当前模式
  float bits = 0.;//初始化编码所需bit
  // 计算模式所需编码位数
  bits += estimate(iMode & 1, _ctxPredMode[0]);
  bits += estimate((iMode >> 1) & 1, _ctxPredMode[1 + (iMode & 1)]);

  for (int k = 0, ctxIdx = 0; k < 3; k++) {
    const auto res = residual[k];
    const bool isZero = res == 0;
    bits += estimate(isZero, _ctxIsZero[k]);//计算残差为0的编码位数
    if (isZero)//若残差为0，继续遍历下一个残差
      continue;

    if (iMode > 0) {
      bits += estimate(res > 0, _ctxSign[k]);//计算编码残差>0，即残差的符号位所需bit
    }

    int32_t value = abs(res) - 1;//计算除去符号位以外的其余残差值所需位数
    int32_t numBits = 1 + ilog2(uint32_t(value));//计算去掉符号位的残差表示所需的比特数

    AdaptiveBitModel* ctxs = _ctxNumBits[ctxIdx][k] - 1;
    for (int ctxIdx = 1, n = _pgeom_resid_abs_log2_bits[k] - 1; n >= 0; n--) {
      auto bin = (numBits >> n) & 1;
      bits += estimate(bin, ctxs[ctxIdx]);//计算编码numBits所需位数
      ctxIdx = (ctxIdx << 1) | bin;
    }

    if (!k && !_geom_angular_mode_enabled_flag)
      ctxIdx = (numBits + 1) >> 1;

    bits += std::max(0, numBits - 1);//返回编码残差(r_r,r_ϕ,r_i)所需位数
  }

  return bits;
}

//----------------------------------------------------------------------------
//具体编码预测树过程包括预测值构建，模式选择，熵编码
int
PredGeomEncoder::encodeTree(
  const Vec3<int32_t>* srcPts,
  Vec3<int32_t>* reconPts,
  const GNode* nodes,
  int numNodes,
  int rootIdx,
  int* codedOrder)
{
  QuantizerGeom quantizer(_sliceQp);//初始化量化器
  int nodesUntilQpOffset = 0;
  int processedNodes = 0;//初始化已编码节点数

  //堆栈的引入是为了遍历预测树结构，这是树结构不需要传入解码端的重要信息
  _stack.push_back(rootIdx);//将根节点push入堆
  while (!_stack.empty()) {//栈非空，循环遍历堆中的元素，将每个预测树的节点依次push入堆中
    const auto nodeIdx = _stack.back();//获取堆顶元素，即相应的节点索引
    _stack.pop_back();//将堆中元素释放，用于下一次将新的节点push入堆

    const auto& node = nodes[nodeIdx];//根据节点索引获取当前节点
    const auto point = srcPts[nodeIdx];//获取索引为nodeIdx的点，与当前节点对应

	//创建最优预测结构
    struct {
      float bits;//编码所需位数
      GPredicter::Mode mode;//预测模式
      Vec3<int32_t> residual;//残差值
      Vec3<int32_t> prediction;//预测值
    } best;

	//量化相关
    if (_geom_scaling_enabled_flag && !nodesUntilQpOffset--) {
      int qp = qpSelector(node);
      quantizer = QuantizerGeom(qp);
      encodeQpOffset(qp - _sliceQp);
      nodesUntilQpOffset = _qpOffsetInterval;
    }

    // mode decision to pick best prediction from available set
    int qphi;//对于Cat3_frame:当采用父节点预测时，对phi的预测值做优化
    for (int iMode = 0; iMode < 4; iMode++) {//遍历四种预测模式
      GPredicter::Mode mode = GPredicter::Mode(iMode);
      GPredicter predicter = makePredicter(
        nodeIdx, mode, [=](int idx) { return nodes[idx].parent; });//根据预测模式构建预测器,<构建结果为当前节点的父节点、祖父节点、曾祖父节点的索引

      if (!predicter.isValid(mode))//判断模式是否有效，即对模式1是否存在父节点，对模式2是否存在祖父节点，对模式3是否存在曾祖父节点
        continue;

      auto pred = predicter.predict(&srcPts[0], mode);//根据所选模式计算预测值
      if (_geom_angular_mode_enabled_flag) {//采用角度模式编码时，判断为Cat3_frame数据
        if (iMode == GPredicter::Mode::Delta) {//若采用父节点作为预测
          int32_t phi0 = srcPts[predicter.index[0]][1];//获取父节点的phi
          int32_t phi1 = point[1];//获取当前节点真实值phi
          int32_t deltaPhi = phi1 - phi0;//计算差值
          qphi = deltaPhi >= 0
            ? (deltaPhi + (_geom_angular_azimuth_speed >> 1))
              / _geom_angular_azimuth_speed
            : -(-deltaPhi + (_geom_angular_azimuth_speed >> 1))
              / _geom_angular_azimuth_speed;//计算优化值，当前节点相对于父节点，在laser扫描时，是扫过了'speed*单位时间距离'的，_geom_angular_azimuth_speed有8种取值
          pred[1] += qphi * _geom_angular_azimuth_speed;//计算优化后的phi的预测值
        }
      }

      // The residual in the spherical domain is loesslessly coded
      auto residual = point - pred;//计算预测残差
      if (!_geom_angular_mode_enabled_flag)//不使用角度模式编码时（Cat3_fused），对残差执行量化过程，即KD树方案对残差进行量化过程(有损)
        for (int k = 0; k < 3; k++)
          residual[k] = int32_t(quantizer.quantize(residual[k]));

      // Check if the prediction residual can be represented with the
      // current configuration.  If it can't, don't use this mode.
	  // 判断当前预测残差是否可以用当前的配置来表示，即判断用于表示残差的位数是否超过限定条件
      bool isOverflow = false;
      for (int k = 0; k < 3; k++) {
        if (residual[k])
          if ((abs(residual[k]) - 1) >> _maxAbsResidualMinus1Log2[k])//判断对应轴上的残差分量所需位数是否超过限定的残差表示位数（这里与边界盒的三维大小相对应）
            isOverflow = true;//_maxAbsResidualMinus1Log2[k]=（31，31，7）
      }
      if (isOverflow)
        continue;

	  //计算编码残差(r_r,r_ϕ,r_i)所需的bits
      auto bits = estimateBits(mode, residual);

      if (iMode == 0 || bits < best.bits) {//根据编码位数判断，选择编码所需位数最小的预测模式
        best.prediction = pred;
        best.residual = residual;
        best.mode = mode;
        best.bits = bits;
      }
    }

    assert(node.childrenCount <= GNode::MaxChildrenCount);//判断当前节点的子节点数小于限定条件(3)
    if (!_geom_unique_points_flag)//存在重复点
      encodeNumDuplicatePoints(node.numDups);//编码重复点数量
    encodeNumChildren(node.childrenCount);//编码子节点数量
    encodePredMode(best.mode);//编码预测模式

    if (
      _geom_angular_mode_enabled_flag && best.mode == GPredicter::Mode::Delta)//角度模式编码并且采用父节点做预测
      encodePhiMultiplier(qphi);//编码qphi

    encodeResidual(best.residual);//编码残差

    // convert spherical prediction to cartesian and re-calculate residual
    if (_geom_angular_mode_enabled_flag) {//采用角度编码模式时，需要将球坐标转换回笛卡尔坐标，做二次残差
      best.prediction = origin + _sphToCartesian(point);//计算转换过来的笛卡尔坐标
      best.residual = reconPts[nodeIdx] - best.prediction;//计算残差值
      for (int k = 0; k < 3; k++)
        best.residual[k] = int32_t(quantizer.quantize(best.residual[k]));//量化残差(r_x,r_y,r_z)

      encodeResidual2(best.residual);//编码二次残差
    }

    // write the reconstructed position back to the point cloud
	// 将重建坐标写回到点云中
    for (int k = 0; k < 3; k++)
      best.residual[k] = int32_t(quantizer.scale(best.residual[k]));//KD树模式下对残差执行量化
    reconPts[nodeIdx] = best.prediction + best.residual;//计算每个点的重建值

    for (int k = 0; k < 3; k++)
      reconPts[nodeIdx][k] = std::max(0, reconPts[nodeIdx][k]);//重建值>=0

    // NB: the coded order of duplicate points assumes that the duplicates
    // are consecutive -- in order that the correct attributes are coded.
	// 假定重复点是连续的
    codedOrder[processedNodes++] = nodeIdx;//存放编码过的点索引，已编码节点数+1
    for (int i = 1; i <= node.numDups; i++)//遍历当前点的重复点数
      codedOrder[processedNodes++] = nodeIdx + i;//存放重复点的索引，已编码节点数+1

    for (int i = 0; i < node.childrenCount; i++) {//遍历当前节点的子节点
      _stack.push_back(node.children[i]);//将当前节点的子节点push进堆，完成遍历过程
    }
  }

  return processedNodes;//返回已编码的节点数
}

//----------------------------------------------------------------------------

//预测树编码
void
PredGeomEncoder::encode(
  const Vec3<int32_t>* cloudA,
  Vec3<int32_t>* cloudB,
  const GNode* nodes,
  int numNodes,
  int32_t* codedOrder)
{
  int32_t processedNodes = 0;//初始化编码完成的节点数
  for (int32_t rootIdx = 0; rootIdx < numNodes; rootIdx++) {//遍历所有节点
    // find the root node(s)
    if (nodes[rootIdx].parent >= 0)//查找根节点，即没有父节点的节点
      continue;

    int numSubtreeNodes = encodeTree(
      cloudA, cloudB, nodes, numNodes, rootIdx, codedOrder + processedNodes);//编码预测树，并计算编码过的节点数量
    processedNodes += numSubtreeNodes;//计算编码过的节点数
  }
  assert(processedNodes == numNodes);//判断所有节点是否都已经编码完成
}

//============================================================================

/*
利用KD树生成几何预测树
*/

std::vector<GNode>
generateGeomPredictionTree(
  const GeometryParameterSet& gps,
  const Vec3<int32_t>* begin,
  const Vec3<int32_t>* end)
{
  const int32_t pointCount = std::distance(begin, end);//获取输入点数量

  // Build the prediction tree, roughly as follows:
  //  - For each point, find the node with the nearest prediction
  //    with empty children slots:
  //     - generate new predictions based upon the current point and
  //       insert into the search tree
  using NanoflannKdTreeT = nanoflann::KDTreeSingleIndexDynamicAdaptor<
    nanoflann::L2_Simple_Adaptor<int32_t, NanoflannCloud, int64_t>,
    NanoflannCloud, 3, int32_t>;//创建KD树动态索引适配器结构

  using NanoflannResultT = nanoflann::KNNResultSet<int64_t, int32_t>;//KD树查询结果结构

  // the predicted point positions, used for searching.
  // each node will generate up to three predicted positions
  NanoflannCloud predictedPoints;//创建预测点对象
  NanoflannKdTreeT predictedPointsTree(3, predictedPoints);//创建三维KD树，其中存放的是每个点的3个预测值，可以用于索引预测点
  predictedPoints.pts.reserve(3 * pointCount);//初始化容量，每个点最多3个预测点，最大容量为3*pointCloud

  // mapping of predictedPoints indicies to prediction tree nodes
  std::vector<int32_t> predictedPointIdxToNodeIdx;//创建预测点索引到节点索引容器，可以用预测点索引查找相应的节点索引
  predictedPointIdxToNodeIdx.reserve(3 * pointCount);//初始化容量，每个点最多3个预测点，即三个预测节点，最大值为3*pointCloud

  // the prediction tree, one node for each point
  std::vector<GNode> nodes(pointCount);//创建预测树节点，一个点对应一个节点

  //查找重复点过程
  for (int nodeIdx = 0, nodeIdxN; nodeIdx < pointCount; nodeIdx = nodeIdxN) {//nodeIdx = 0开始，遍历所有点
    auto& node = nodes[nodeIdx];//索引为nodeIdx的节点
    auto queryPoint = begin[nodeIdx];//假设索引为nodeIdx的点为重复点

    // scan for duplicate points
    // NB: the tree coder assumes that duplicate point indices are consecutive
    // If this guarantee is changed, then the tree coder must be modified too.
    node.numDups = 0;//初始化当前节点的重复点数量
    for (nodeIdxN = nodeIdx + 1; nodeIdxN < pointCount; nodeIdxN++) {//从nodeIdx = 1开始，遍历所有点
      if (queryPoint != begin[nodeIdxN])//判断索引为nodeIdx与nodeIdx+1的点是否相同，相同则重复点数加1，否则跳出
        break;
      node.numDups++;//计算每个点的重复点数量
    }

    int32_t nnPredictedPointIdx[GNode::MaxChildrenCount];//预测点的索引
    int64_t nnPredictedDist[GNode::MaxChildrenCount];//每个点的预测权重
    NanoflannResultT resultSet(GNode::MaxChildrenCount);//KD树近邻搜索结果

    resultSet.init(nnPredictedPointIdx, nnPredictedDist);//初始化搜索结果
    predictedPointsTree.findNeighbors(resultSet, &queryPoint[0], {});//根据当前点进行邻居查找，将查找结果存放到resultSet中

    // find a suitable parent.  default case: node has no parent
	// 查找合适的父亲节点，默认情况为节点没有父亲节点
    node.parent = -1;//默认根节点的父节点为-1
    node.childrenCount = 0;//初始化节点的子节点数量为0
    for (size_t r = 0; r < resultSet.size(); ++r) {//遍历搜索结果
      auto parentIdx = predictedPointIdxToNodeIdx[nnPredictedPointIdx[r]];//根据搜索到的预测点索引获取父节点索引，因为预测节点即为当前点的父节点，祖父节点和曾祖父节点
      auto& pnode = nodes[parentIdx];//获得父节点
      if (pnode.childrenCount < GNode::MaxChildrenCount) {//若搜索到的父节点的子节点数量<3
        node.parent = parentIdx;//该父节点即可作为当前节点的父节点
        pnode.children[pnode.childrenCount++] = nodeIdx;//该父节点的子节点即为当前节点
        break;
      }
    }

    // update the search tree with new predictions from this point
	// 更新过程，通过权重比较来更新搜索出来的用于预测的点
    const auto size0 = predictedPoints.pts.size();//获取当前预测树内点的数量

    // set the indicies for prediction
	// 为预测值设置索引
    GPredicter predicter;//创建预测器
    predicter.index[0] = nodeIdx;
    predicter.index[1] = nodes[predicter.index[0]].parent;
    if (predicter.index[1] >= 0)
      predicter.index[2] = nodes[predicter.index[1]].parent;

    for (int iMode = 0; iMode < 4; iMode++) {//遍历四种预测模式
      GPredicter::Mode mode = GPredicter::Mode(iMode);

      // don't add zero prediction
      if (mode == GPredicter::None)//跳过模式0
        continue;

      if (!predicter.isValid(mode))//判断其他模式是否有效，即对模式1是否存在父节点，对模式2是否存在祖父节点，对模式3是否存在曾祖父节点
        continue;

      auto prediction = predicter.predict(begin, mode);//根据所选模式计算预测值
      predictedPointIdxToNodeIdx.push_back(nodeIdx);
      predictedPoints.pts.push_back(prediction);//将计算得到的预测值存放到预测点对象中---这里可以理解为预测树中存放的是相应的预测值而不是点
    }

    const auto size1 = predictedPoints.pts.size();//获取更新后的预测树内点的数量
    if (size0 != size1)//判断是否有新值加入
      predictedPointsTree.addPoints(size0, size1 - 1);//将新值根据权值比较加入到预测树中
  }

  return nodes;//返回构建完成的预测树
}

//----------------------------------------------------------------------------

/*
利用angular生成几何预测树
*/

std::vector<GNode>
generateGeomPredictionTreeAngular(
  const GeometryParameterSet& gps,
  const Vec3<int32_t> origin,
  const Vec3<int32_t>* begin,
  const Vec3<int32_t>* end,
  Vec3<int32_t>* beginSph)
{
  int32_t pointCount = std::distance(begin, end);//获取输入点数量
  int32_t numLasers = gps.geom_angular_num_lidar_lasers();//获取laser数量

  // the prediction tree, one node for each point
  std::vector<GNode> nodes(pointCount);//创建预测树节点，一个点对应一个节点
  std::vector<int32_t> prevNodes(numLasers, -1);//创建用于预测的节点
  std::vector<int32_t> firstNodes(numLasers, -1);//创建首节点，即不同laser照射到的第一个点

  CartesianToSpherical cartToSpherical(gps);//创建笛卡尔坐标到球坐标转换对象

  for (int nodeIdx = 0, nodeIdxN; nodeIdx < pointCount; nodeIdx = nodeIdxN) {//从nodeIdx = 0开始遍历所有点
    auto& node = nodes[nodeIdx];//索引为nodeIdx的节点
    auto curPoint = begin[nodeIdx];//索引为nodeIdx，与当前节点对应的当前点的坐标
    node.childrenCount = 0;//初始化当前节点的子节点数为0

    // scan for duplicate points
    // NB: the tree coder assumes that duplicate point indices are consecutive
    // If this guarantee is changed, then the tree coder must be modified too.
    node.numDups = 0;//初始化当前节点的重复点数量
    for (nodeIdxN = nodeIdx + 1; nodeIdxN < pointCount; nodeIdxN++) {//从nodeIdx = 1开始遍历所有点
      if (curPoint != begin[nodeIdxN])//判断nodeIdx与nodeIdx+1是否相同，从而判定两个点是否为重复点
        break;
      node.numDups++;//计算重复点数量
    }

    // cartesian to spherical coordinates
    const auto carPos = curPoint - origin;//计算相对于origin即laser head的当前点的坐标
    auto& sphPos = beginSph[nodeIdx] = cartToSpherical(carPos);//坐标变换，根据当前点坐标计算球坐标
    auto thetaIdx = sphPos[2];//获取i分量

    node.parent = prevNodes[thetaIdx];//获取当前节点的父节点
    if (node.parent != -1) {//当前节点的父节点存在时
      auto& pnode = nodes[prevNodes[thetaIdx]];//获取当前节点的父节点
      pnode.children[pnode.childrenCount++] = nodeIdx;//将当前节点的父节点的子节点数+1，并将当前节点存放到其父节点的children内
    } else//当前节点的父节点不存在时
      firstNodes[thetaIdx] = nodeIdx;//将当前节点作为第一个节点，存放当前节点的索引到firstNodes[thetaIdx]中，表示laser照射到的第一个点

    prevNodes[thetaIdx] = nodeIdx;//将当前节点作为预测节点存放到prevNodes中
  }

  int32_t n0 = 0;
  while (firstNodes[n0] == -1)//判断查找第一个存在节点的firstNodes，即第一个存在扫描到点的laser
    ++n0;

  //这一部分是将不同laser构建的树的根节点连接起来，构成一个整体的预测树，只有一个根节点
  for (int32_t n = n0 + 1, parentIdx = firstNodes[n0]; n < numLasers; ++n) {//遍历所有firstNodes，相当于每个laser扫描出来的树的根节点
    auto nodeIdx = firstNodes[n];//索引为n的laser构建树的根节点索引
    if (nodeIdx < 0)
      continue;

    auto& pnode = nodes[parentIdx];//当前根节点的前一个laser构建的树的根节点
    if (pnode.childrenCount < GNode::MaxChildrenCount) {//子节点数在限定范围内
      nodes[nodeIdx].parent = parentIdx;//前一个laser构建的树的根节点作为当前根节点的父亲节点
      pnode.children[pnode.childrenCount++] = nodeIdx;//对应的子节点树+1，并存放当前根节点的索引值
    }
    parentIdx = nodeIdx;//当前根节点作为下一个节点的父节点
  }


  return nodes;//返回构建完成的预测树
}

//============================================================================
//根据莫顿码排序   这里目前采用的是计数排序方式
static void
mortonSort(PCCPointSet3& cloud, int begin, int end, int depth)
{
  radixSort8(
    depth, PCCPointSet3::iterator(&cloud, begin),
    PCCPointSet3::iterator(&cloud, end),
    [=](int depth, const PCCPointSet3::Proxy& proxy) {
      const auto& point = *proxy;
      int mask = 1 << depth;
      return !!(point[2] & mask) | (!!(point[1] & mask) << 1)
        | (!!(point[0] & mask) << 2);
    });
}

//============================================================================

/*
编码预测几何
首先对输入的量化后的点进行排序，排序有四种方法：
    0.不排序 
	1.莫顿排序
	2.方位角排序
	3.极径排序
然后对排序后的点进行建树过程，建树有两种方法：
	1.基于KD树的预测树构建（高延迟）
	2.基于方位角的预测树构建（低延迟）
对建好的树内的点进行预测编码，预测过程主要通过RDO过程进行，预测方式有4种：
	0.不预测
	1.用父节点预测  p0
	2.用父节点和祖父节点预测 2p0-p1
	3.用父节点、祖父节点和曾祖父节点预测 p0+p1-p2
最后将预测结果输出到output中
*/
void
encodePredictiveGeometry(
  const PredGeomEncOpts& opt,
  const GeometryParameterSet& gps,
  GeometryBrickHeader& gbh,
  PCCPointSet3& cloud,
  PredGeomContexts& ctxtMem,
  EntropyEncoder* arithmeticEncoder)
{
  auto numPoints = cloud.getPointCount();//获取量化后的输入点数量

  // Origin relative to slice origin
  auto origin = gps.geomAngularOrigin - gbh.geomBoxOrigin;//计算laser head原点坐标

  // storage for reordering the output point cloud
  // 用于存储预测编码后的输出点云
  PCCPointSet3 outCloud;
  outCloud.addRemoveAttributes(cloud.hasColors(), cloud.hasReflectances());//初始化属性值
  outCloud.resize(numPoints);//初始化size

  // storage for spherical point co-ordinates determined in angular mode
  std::vector<Vec3<int32_t>> sphericalPos;//创建存储球坐标的容器
  if (gps.geom_angular_mode_enabled_flag)//角度编码使能
    sphericalPos.resize(numPoints);//初始化size

  // src indexes in coded order
  std::vector<int32_t> codedOrder(numPoints, -1);//用于存放点编码顺序的索引值

  // Assume that the number of bits required for residuals is equal to the
  // root node size.  This allows every position to be coded using PCM.
  for (int k = 0; k < 3; k++)//初始化残差编码位数为根节点三维值+1
    gbh.pgeom_resid_abs_log2_bits[k] =
      ilog2(uint32_t(gbh.rootNodeSizeLog2[k])) + 1;

  // Number of residual bits bits for angular mode.  This is slightly
  // pessimistic in the calculation of r.
  if (gps.geom_angular_mode_enabled_flag) {//角度编码使能
    auto xyzBboxLog2 = gbh.rootNodeSizeLog2;//初始块的三维大小
    auto rDivLog2 = gps.geom_angular_radius_inv_scale_log2;//预测几何编码中半径编码的反比例因子
    auto azimuthBits = gps.geom_angular_azimuth_scale_log2;//应用于预测几何编码中的方位角的比例系数

    // first work out the maximum number of bits for the residual
	// 首先计算出残差所需的最大位数
    // NB: the slice coordinate system is used here: ie, minX|minY = 0
    int maxX = (1 << xyzBboxLog2[0]) - 1;
    int maxY = (1 << xyzBboxLog2[1]) - 1;
    int maxAbsDx = std::max(std::abs(origin[0]), std::abs(maxX - origin[0]));//最大x边长
    int maxAbsDy = std::max(std::abs(origin[1]), std::abs(maxY - origin[1]));//最大y边长
    auto r = std::round(std::hypot(maxAbsDx, maxAbsDy));//计算最大半径r: hypot用于计算斜边长，即取x和y的平方根得到半径r再round

    Vec3<int> residualBits;
    // 以下三个残差分量分别对应r，phi和thetaIdx的最大值
    /*
		ceillog2():将divExp2RoundHalfUp(int64_t(r), rDivLog2)向上舍入为2的幂的整数，然后取对数得到幂次
		divExp2RoundHalfUp():加半位向上取整
	*/
    residualBits[0] = ceillog2(divExp2RoundHalfUp(int64_t(r), rDivLog2));
    residualBits[1] = gps.geom_angular_azimuth_scale_log2;
    residualBits[2] = ceillog2(gps.geom_angular_theta_laser.size() - 1);//最大laser数量 - 1

    // the number of prefix bits required
	// 计算编码最大残差所需位数,存到gbh.pgeom_resid_abs_log2_bits中
    for (int k = 0; k < 3; k++)
      gbh.pgeom_resid_abs_log2_bits[k] = ilog2(uint32_t(residualBits[k])) + 1;
  }

  // determine each geometry tree, and encode.  Size of trees is limited
  // by maxPtsPerTree.

  PredGeomEncoder enc(gps, gbh, ctxtMem, arithmeticEncoder);// 创建编码器
  int maxPtsPerTree = std::min(opt.maxPtsPerTree, int(numPoints));// 树内最大点数限制

  for (int i = 0; i < numPoints;) {//遍历所有输入点
    int iEnd = std::min(i + maxPtsPerTree, int(numPoints));//计算预测树最后一个点的索引
    auto* begin = &cloud[i];//获取输入点云中第一个点
    auto* beginSph = &sphericalPos[i];//存放第一个点的球坐标
    auto* end = &cloud[0] + iEnd;//输入点云中的最后一个点

    // first, put the points in this tree into a sorted order
    // this can significantly improve the constructed tree
	// 根据sortMode确定排序方式
    if (opt.sortMode == PredGeomEncOpts::kSortMorton)
      mortonSort(cloud, i, iEnd, gbh.maxRootNodeDimLog2);//莫顿排序
    else if (opt.sortMode == PredGeomEncOpts::kSortAzimuth)
      sortByAzimuth(cloud, i, iEnd, opt.azimuthSortRecipBinWidth, origin);//方位角排序
    else if (opt.sortMode == PredGeomEncOpts::kSortRadius)
      sortByRadius(cloud, i, iEnd, origin);//极径排序


    // then build and encode the tree
	// 根据是否启用角度模式，选择构建预测树的方式，0：未启用（KD树构建）; 1：启用（方位角构建）
    auto nodes = gps.geom_angular_mode_enabled_flag
      ? generateGeomPredictionTreeAngular(gps, origin, begin, end, beginSph)
      : generateGeomPredictionTree(gps, begin, end);

    auto* a = gps.geom_angular_mode_enabled_flag ? beginSph : begin;//若为方位角构建的预测树，则选择初始点值为球坐标值，否则为笛卡尔坐标值
    auto* b = begin;//输入点云中的第一个点

    enc.encode(a, b, nodes.data(), nodes.size(), codedOrder.data() + i);//编码预测树

    // put points in output cloud in decoded order
    for (auto iBegin = i; i < iEnd; i++) {//编码完成后，遍历所有点
      auto srcIdx = iBegin + codedOrder[i];//根据初始点索引值和编码后的按编码顺序排列的索引值来计算点的源索引
      assert(srcIdx >= 0);
      outCloud[i] = cloud[srcIdx];//将编码后的点按顺序存到outCloud中用于解码过程
      //赋予属性值--color和reflectance
	  if (cloud.hasColors())
        outCloud.setColor(i, cloud.getColor(srcIdx));
      if (cloud.hasReflectances())
        outCloud.setReflectance(i, cloud.getReflectance(srcIdx));
    }
  }

  // save the context state for re-use by a future slice if required
  ctxtMem = enc.getCtx();//保存上下文状态

  swap(cloud, outCloud);//交换编码后的点云与原量化后的点云，push入码流中
}

//============================================================================

}  // namespace pcc
