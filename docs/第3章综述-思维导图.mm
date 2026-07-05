<map version="1.0.1">
  <node TEXT="第3章 文献综述：运动与幅度的度量方法" FOLDED="false">
    <node TEXT="3.1 五种分类视角" FOLDED="false">
      <node TEXT="A 信号来源" FOLDED="false">
        <node TEXT="几何/骨架：HumanScore、Motion-X、FVMD、THEval"/>
        <node TEXT="像素/光流：VBench DD、Mojito、SEA-RAFT"/>
        <node TEXT="深度特征：FVD、Koala VTSS"/>
        <node TEXT="学习式：VideoScore、MotionCritic"/>
        <node TEXT="梯度归因：Motive"/>
        <node TEXT="综合：VMBench、WYD"/>
      </node>
      <node TEXT="B 应用场景" FOLDED="false">
        <node TEXT="T2V排名 → VBench"/>
        <node TEXT="感知对齐 → VMBench"/>
        <node TEXT="人体质量 → HumanScore"/>
        <node TEXT="I2V控制 → Animate Anyone、MotionStone"/>
        <node TEXT="说话头 → THEval"/>
        <node TEXT="数据筛选 → Koala、Motive"/>
        <node TEXT="RLHF → VideoScore"/>
      </node>
      <node TEXT="C 空间时间粒度" FOLDED="false">
        <node TEXT="Clip标量"/>
        <node TEXT="帧对/序列（top-5%）"/>
        <node TEXT="关键点轨迹"/>
        <node TEXT="关节/部位向量"/>
        <node TEXT="分布Fréchet距离"/>
      </node>
      <node TEXT="D GT与标注成本" FOLDED="false">
        <node TEXT="启发式（低）"/>
        <node TEXT="序数tier（低-中）"/>
        <node TEXT="连续1-5分（高）"/>
        <node TEXT="成对比较（中）"/>
        <node TEXT="细粒度多维（很高）"/>
        <node TEXT="隐式梯度（Motive）"/>
      </node>
      <node TEXT="E 幅度 vs 质量" FOLDED="false">
        <node TEXT="幅度：DD、PAS、Mojito、pose/flow"/>
        <node TEXT="质量：HumanScore、MSS/CAS/OIS"/>
        <node TEXT="混合：FVMD、WYD"/>
      </node>
    </node>
    <node TEXT="3.2 多维度对比总表" FOLDED="false">
      <node TEXT="维度：信号｜输出｜粒度｜相机｜连续｜人类对齐｜部位｜用途"/>
      <node TEXT="代表工作簇" FOLDED="false">
        <node TEXT="光流：VBench DD、Mojito"/>
        <node TEXT="感知：VMBench PAS/MSS/CAS/OIS"/>
        <node TEXT="几何：HumanScore★、FVMD、THEval、Motion-X"/>
        <node TEXT="学习：VideoScore、MotionStone、MotionCritic"/>
        <node TEXT="综合：WYD、Motive、FVD"/>
      </node>
      <node TEXT="本项目" FOLDED="false">
        <node TEXT="pose ρ≈0.49"/>
        <node TEXT="flow ρ≈0.40"/>
        <node TEXT="learned_score ρ≈0.84"/>
      </node>
    </node>
    <node TEXT="3.3 技术路线深述" FOLDED="false">
      <node TEXT="3.3.1 几何骨架" FOLDED="false">
        <node TEXT="HumanScore：三层六指标，SKELify+OpenSim ROM"/>
        <node TEXT="PAS：DINO→SAM→CoTracker"/>
        <node TEXT="FVMD / THEval / Motion-X / OpenHumanVid"/>
        <node TEXT="小结：贴近LMA，3D贵/2D折中"/>
      </node>
      <node TEXT="3.3.2 光流像素" FOLDED="false">
        <node TEXT="VBench DD：top-5%→二值"/>
        <node TEXT="Mojito：1-10级条件"/>
        <node TEXT="SEA-RAFT：本项目估计器"/>
        <node TEXT="小结：便宜，需前景分离"/>
      </node>
      <node TEXT="3.3.3 学习感知" FOLDED="false">
        <node TEXT="VideoScore/2：万级标注"/>
        <node TEXT="MotionStone：成对±2解耦"/>
        <node TEXT="MotionCritic / Koala VTSS"/>
        <node TEXT="小结：对齐强，小数据易collapse"/>
      </node>
      <node TEXT="3.3.4 梯度及其它" FOLDED="false">
        <node TEXT="Motive：影响力≠幅度"/>
        <node TEXT="WYD：56类综合"/>
        <node TEXT="FVD：内容偏置"/>
      </node>
    </node>
    <node TEXT="3.4 互补与张力" FOLDED="false">
      <node TEXT="信号→输出：光流/骨架→幅度；学习→幅度+tier"/>
      <node TEXT="五重张力" FOLDED="false">
        <node TEXT="幅度 vs 质量"/>
        <node TEXT="连续 vs 序数（0.84 vs 0.38）"/>
        <node TEXT="全局 vs 部位（说话类）"/>
        <node TEXT="主体 vs 相机"/>
        <node TEXT="小数据 vs 大标注"/>
      </node>
    </node>
    <node TEXT="3.5 LMA启示" FOLDED="false">
      <node TEXT="分层组合，无单一最佳"/>
      <node TEXT="报告：相机/粒度/连续序数/GT"/>
      <node TEXT="本项目：pose+flow+learned"/>
      <node TEXT="未覆盖：3D ROM、THEval、成对标注"/>
    </node>
  </node>
</map>
