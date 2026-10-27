# AI + Drug Discovery / Computational Chemistry 学习路线

> 适用背景：生物信息学 / 多组学 / 疾病机制研究者，希望转向 AI-driven Drug Discovery、Computational Biology for Drug Discovery 或 Computational Chemistry 相关岗位。  
> 推荐周期：约 6 个月（每周 8–12 小时）。  
> 核心原则：**先理解药物研发问题，再学分子表示与计算方法，最后用完整项目把 disease → target → molecule 串起来。**

---

## 1. 最终目标

完成本路线后，目标不是成为传统量子化学方向的 Computational Chemist，而是形成以下能力组合：

```text
Disease Biology / Multi-omics
          ↓
   Target Discovery
          ↓
 Protein Structure
          ↓
Cheminformatics / QSAR
          ↓
 Molecular ML / GNN
          ↓
Docking / Virtual Screening
          ↓
      ADMET
          ↓
Candidate Prioritization
```

推荐职业定位：

- AI Drug Discovery Scientist
- Computational Biology Scientist
- Computational Drug Discovery Scientist
- Machine Learning Scientist – Drug Discovery
- Bioinformatics / AI Scientist – Target & Drug Discovery
- Translational AI Scientist

---

# 2. 六个月学习路线总表

| 阶段 | 时间 | 学习主题 | 必须掌握 | 推荐资源 | 阶段产出 |
|---|---:|---|---|---|---|
| 0 | 1周 | Drug Discovery 全流程 | target、hit、lead、candidate、HTS、IC50、EC50、Kd、Ki、PK/PD、ADMET、SAR | [DavidsonX Medicinal Chemistry](https://www.edx.org/learn/medicine/davidson-college-medicinal-chemistry-the-molecular-basis-of-drug-discovery), [Lead Discovery & Optimization](https://www.edx.org/learn/medicine/davidson-college-lead-discovery-optimization-for-efficacy-pk-safety/), [NCBI Pharmacokinetics](https://www.ncbi.nlm.nih.gov/books/NBK557744/) | 一页 Drug Discovery workflow 笔记 |
| 1 | 2–3周 | Medicinal Chemistry 基础 | functional group、stereochemistry、pKa、LogP/LogD、TPSA、HBD/HBA、Lipinski、drug-like、SAR | [DavidsonX Medicinal Chemistry](https://www.edx.org/learn/medicine/davidson-college-medicinal-chemistry-the-molecular-basis-of-drug-discovery), [SwissADME](https://www.swissadme.ch/) | 选择10–20个药物，比较结构和理化性质 |
| 2 | 3–4周 | Cheminformatics + RDKit | SMILES、canonical SMILES、descriptors、Morgan/ECFP、Tanimoto、scaffold、substructure | [RDKit Docs](https://www.rdkit.org/docs/), [RDKit Getting Started](https://www.rdkit.org/docs/GettingStartedInPython.html), [RDKit Cookbook](https://www.rdkit.org/docs/Cookbook.html) | `01_rdkit_basics.ipynb` |
| 3 | 2周 | Drug databases | ChEMBL、BindingDB、bioactivity、assay、IC50/Ki/Kd清洗 | [ChEMBL Quick Tour](https://www.ebi.ac.uk/training/online/courses/chembl-quick-tour/), [ChEMBL](https://www.ebi.ac.uk/chembl/), [ChEMBL API Training](https://www.ebi.ac.uk/training/online/courses/chembl-quick-tour/accessing-chembl-data/programmatic-access-via-web-services/), [BindingDB](https://www.bindingdb.org/) | 为一个靶点建立干净的 ligand–activity 数据集 |
| 4 | 3–4周 | QSAR / classical ML | classification/regression、pIC50、fingerprints、RF/XGBoost、split、CV、RMSE/AUC、applicability domain | [DeepChem Tutorials](https://deepchem.io/tutorials/), [ChEMBL](https://www.ebi.ac.uk/chembl/) | 第一个完整 QSAR 项目 |
| 5 | 2–3周 | Protein Structure | PDB、domain、binding pocket、co-crystal、ligand interaction、pLDDT、PAE | [RCSB PDB-101](https://pdb101.rcsb.org/), [RCSB PDB Help](https://www.rcsb.org/help), [AlphaFold DB](https://alphafold.ebi.ac.uk/) | 对一个 drug target 做结构注释 |
| 6 | 2–3周 | Molecular Docking | receptor/ligand preparation、grid box、pose、scoring、redocking、interaction analysis | [AutoDock Vina Basic Docking](https://autodock-vina.readthedocs.io/en/stable/docking_basic.html), [RCSB PDB](https://www.rcsb.org/) | 已知 ligand redocking + candidate docking |
| 7 | 2周 | PyTorch | Tensor、Dataset/DataLoader、nn.Module、autograd、optimizer、training loop | [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) | 用 PyTorch 重写一个简单 activity model |
| 8 | 3–4周 | Molecular Deep Learning / GNN | molecular graph、GCN、GAT、MPNN、graph pooling、scaffold split | [DeepChem Molecular ML](https://deepchem.io/tutorials/), [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) | GNN vs Morgan+RF benchmark |
| 9 | 2周 | ADMET | absorption、distribution、metabolism、excretion、toxicity、solubility、permeability、CYP | [SwissADME](https://www.swissadme.ch/), [NCBI Pharmacokinetics](https://www.ncbi.nlm.nih.gov/books/NBK557744/) | 多指标 candidate ranking |
| 10 | 选修 | Molecular Dynamics | force field、energy minimization、equilibration、production MD、RMSD/RMSF/H-bond | [GROMACS](https://www.gromacs.org/), [GROMACS Tutorials](https://www.gromacs.org/tutorial_webinar.html), [GROMACS Manual](https://manual.gromacs.org/) | protein–ligand MD demo |
| 11 | 选修 | Generative AI | VAE、GAN、Transformer、diffusion、molecular generation、multi-objective optimization | [DeepChem Generative Tutorials](https://deepchem.io/tutorials/) | molecule generation demo |
| 12 | 最终项目 | Disease → Target → Drug | target evidence + ligands + QSAR + GNN + docking + ADMET | 综合以上资源 | GitHub portfolio 项目 |

---

# 3. Phase 0：先理解 Drug Discovery，而不是先写模型

## 需要掌握的概念

### Drug Discovery Pipeline

```text
Disease biology
      ↓
Target identification
      ↓
Target validation
      ↓
Hit discovery
      ↓
Hit-to-lead
      ↓
Lead optimization
      ↓
Preclinical candidate
      ↓
IND
      ↓
Clinical development
```

重点理解以下词：

- Target
- Hit
- Lead
- Candidate
- HTS
- Virtual screening
- Potency
- Efficacy
- Selectivity
- IC50
- EC50
- Ki
- Kd
- PK
- PD
- ADME / ADMET
- SAR

## 推荐课程

### DavidsonX — Medicinal Chemistry: The Molecular Basis of Drug Discovery

https://www.edx.org/learn/medicine/davidson-college-medicinal-chemistry-the-molecular-basis-of-drug-discovery

建议重点学习：

- Drug targets
- Enzymes / receptors
- Pharmacokinetics
- Metabolism
- Molecular binding
- Lead discovery
- Lead optimization

### DavidsonX — Lead Discovery & Optimization

https://www.edx.org/learn/medicine/davidson-college-lead-discovery-optimization-for-efficacy-pk-safety/

特别推荐，因为内容直接覆盖：

- hit / lead
- pharmacophore
- drug-like vs lead-like
- lead optimization
- co-crystal structure
- bioisostere

---

# 4. Phase 1：Medicinal Chemistry 基础

不要求转型成为 Medicinal Chemist，但必须能和药化研究人员沟通。

## 必学

### Molecular properties

- Molecular weight
- LogP
- LogD
- pKa
- TPSA
- Hydrogen-bond donor
- Hydrogen-bond acceptor
- Rotatable bonds
- Aromatic rings
- Solubility
- Permeability

### Structure–Activity Relationship

核心思想：

```text
Chemical structure
        ↓
Property / Binding / Activity
```

需要能够回答：

> 某个官能团被替换以后，为什么 potency、solubility、selectivity 或 metabolic stability 会变化？

## 推荐工具

SwissADME：

https://www.swissadme.ch/

建议实践：

1. 选 10 个已上市的小分子药物。
2. 输入 SMILES。
3. 比较 MW / LogP / TPSA / Lipinski / GI absorption。
4. 尝试解释不同药物性质差异。

---

# 5. Phase 2：RDKit + Cheminformatics

这是生物信息背景最容易切入的一部分。

RDKit 官方文档：

https://www.rdkit.org/docs/

## 必须掌握

### SMILES

理解：

```text
Molecule → SMILES
```

### Descriptor

例如：

```python
MolWt
LogP
TPSA
HBD
HBA
RotatableBond
```

### Fingerprint

优先掌握：

```text
Morgan fingerprint / ECFP
```

### Similarity

掌握：

```text
Tanimoto similarity
```

### Scaffold

理解 Bemis–Murcko scaffold。

## 建议 Notebook

```text
01_rdkit_basics.ipynb

SMILES
 ↓
RDKit Molecule
 ↓
Descriptors
 ↓
Fingerprint
 ↓
Similarity
 ↓
Clustering
```

---

# 6. Phase 3：ChEMBL + BindingDB

## ChEMBL

https://www.ebi.ac.uk/chembl/

官方学习：

https://www.ebi.ac.uk/training/online/courses/chembl-quick-tour/

API：

https://www.ebi.ac.uk/training/online/courses/chembl-quick-tour/accessing-chembl-data/programmatic-access-via-web-services/

## BindingDB

https://www.bindingdb.org/

这一步的重点不是下载数据，而是学习：

**如何构建可靠的 molecular ML dataset。**

需要特别注意：

- assay type
- target identity
- organism
- IC50 vs Ki vs Kd
- units
- duplicate measurements
- salts
- stereochemistry
- censored values，如 `>10000`
- experimental uncertainty

## 推荐实践

任选一个靶点，例如：

```text
JAK1
EGFR
BRAF
TNF-related target
```

建立：

```text
compound
SMILES
IC50
pIC50
assay
target
```

数据集。

---

# 7. Phase 4：QSAR

这是最值得优先完成的项目。

## Workflow

```text
ChEMBL
 ↓
Data cleaning
 ↓
SMILES
 ↓
RDKit
 ↓
Morgan fingerprint
 ↓
Train / validation / test
 ↓
Random Forest / XGBoost
 ↓
Prediction
```

## 非常重要：不要只 random split

药物发现 ML 中建议理解：

```text
Random split
vs
Scaffold split
```

Scaffold split 更能测试：

> 模型面对新的 chemical scaffold 是否还能泛化。

## 推荐输出

比较：

| Model | Feature | Split | Metric |
|---|---|---|---|
| RF | Morgan FP | Random | ROC-AUC / RMSE |
| RF | Morgan FP | Scaffold | ROC-AUC / RMSE |
| XGBoost | Descriptor | Scaffold | ROC-AUC / RMSE |
| GNN | Molecular Graph | Scaffold | ROC-AUC / RMSE |

---

# 8. Phase 5：Protein Structure

## RCSB PDB

https://www.rcsb.org/

教学材料：

https://pdb101.rcsb.org/

## AlphaFold Database

https://alphafold.ebi.ac.uk/

需要掌握：

- Protein structure
- Domain
- Secondary structure
- Binding pocket
- Co-crystal structure
- Ligand
- Active site
- Mutation
- pLDDT
- PAE

## 学习原则

优先使用：

```text
Experimental PDB structure
```

如果没有，再考虑：

```text
AlphaFold predicted structure
```

尤其在 docking 时，不应该把 AlphaFold 结构当成无条件可靠的实验结构。

---

# 9. Phase 6：Molecular Docking

推荐：

## AutoDock Vina

https://autodock-vina.readthedocs.io/en/stable/docking_basic.html

官方教程提供了一个很好的入门案例：

```text
c-Abl + imatinib
```

建议完整做一遍。

## 需要理解

```text
Protein preparation
       ↓
Ligand preparation
       ↓
Binding-site definition
       ↓
Docking
       ↓
Pose
       ↓
Scoring
       ↓
Interaction analysis
```

## 最重要的练习：Redocking

不要直接进行虚拟筛选。

先做：

```text
Known co-crystal ligand
↓
remove ligand
↓
dock ligand back
↓
compare predicted pose vs experimental pose
```

理解 RMSD 和 docking validation。

---

# 10. Phase 7：PyTorch

你已经有 Python 和 ML 基础，因此不需要花太久。

官方教程：

https://docs.pytorch.org/tutorials/beginner/basics/intro.html

至少掌握：

```python
Tensor
Dataset
DataLoader
nn.Module
forward
loss
optimizer
autograd
train()
eval()
```

目标不是刷完整套深度学习课程，而是：

> 能看懂并修改 molecular ML code。

---

# 11. Phase 8：Molecular Deep Learning / GNN

## DeepChem

https://deepchem.io/tutorials/

推荐按以下顺序：

1. Molecular Fingerprints
2. Molecular Featurizations
3. Graph Convolutions
4. Molecular property prediction
5. Protein–Ligand Interactions
6. Generative Models

DeepChem 的优势是：

> drug discovery 场景和 ML 是直接结合的。

---

## PyTorch Geometric

https://pytorch-geometric.readthedocs.io/

需要掌握：

```text
Atom → Node
Bond → Edge
Molecule → Graph
```

进一步学习：

- GCN
- GAT
- MPNN
- Graph pooling
- Graph Transformer

核心：

```text
message passing
```

---

# 12. Phase 9：ADMET

ADMET：

```text
A = Absorption
D = Distribution
M = Metabolism
E = Excretion
T = Toxicity
```

工具：

https://www.swissadme.ch/

建议至少理解：

- Solubility
- GI absorption
- BBB permeability
- CYP inhibition
- Clearance
- hERG
- Hepatotoxicity
- Plasma protein binding

目标不是成为 DMPK Scientist，而是能够理解：

> potency 强并不等于 drug 好。

---

# 13. Phase 10：Molecular Dynamics（选修）

如果未来岗位明显要求 Computational Chemistry，可以继续补 MD。

GROMACS：

https://www.gromacs.org/

Tutorial：

https://www.gromacs.org/tutorial_webinar.html

Manual：

https://manual.gromacs.org/

需要理解：

```text
Force field
Energy minimization
Solvation
Ionization
NVT
NPT
Production MD
Trajectory
RMSD
RMSF
Hydrogen bonds
```

对于 AI Drug Discovery Scientist：

> 会使用和解释 MD 通常比深入开发 force field 更重要。

---

# 14. Phase 11：Generative AI（第二阶段学习）

不要在没有 QSAR / GNN / docking 基础之前优先学生成模型。

推荐 DeepChem：

https://deepchem.io/tutorials/

可以逐步了解：

- VAE
- GAN
- Transformer
- Diffusion
- Reinforcement learning
- Molecular generation

最终问题通常不是：

> 能不能生成 molecule？

而是：

> 能不能生成具有 potency、selectivity、ADMET、synthetic accessibility 等多重性质约束的 molecule？

---

# 15. 最终 Portfolio 项目

强烈建议选择与你已有疾病背景相关的靶点。

例如：

```text
IBD
 ↓
candidate target
 ↓
target evidence
 ↓
PDB / AlphaFold
 ↓
ChEMBL / BindingDB
 ↓
ligand dataset
 ↓
RDKit
 ↓
QSAR
 ↓
GNN
 ↓
virtual screening
 ↓
docking
 ↓
ADMET
 ↓
candidate prioritization
```

---

# 16. 推荐 GitHub 项目结构

```text
AI-Drug-Discovery-Learning/
│
├── README.md
│
├── environment.yml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_rdkit_basics.ipynb
│   ├── 02_chembl_data.ipynb
│   ├── 03_qsar_rf.ipynb
│   ├── 04_gnn.ipynb
│   ├── 05_protein_structure.ipynb
│   ├── 06_docking.ipynb
│   └── 07_admet.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── utils/
│
├── results/
│   ├── figures/
│   ├── models/
│   └── docking/
│
└── docs/
    └── learning_notes.md
```

---

# 17. 推荐环境

如果使用 Conda：

```bash
conda create -n drug-ai python=3.11
conda activate drug-ai

conda install -c conda-forge rdkit
pip install pandas numpy scipy scikit-learn matplotlib jupyter
pip install torch
pip install torch-geometric
pip install deepchem
```

Docking / MD 建议单独建立环境，避免依赖冲突。

例如：

```text
drug-ai
docking
gromacs
```

---

# 18. 学习优先级

如果时间有限，请按照下面顺序：

## P0 — 必须

```text
Drug Discovery fundamentals
Medicinal Chemistry basics
RDKit
ChEMBL
QSAR
Protein Structure
Docking
PyTorch
GNN
ADMET
```

## P1 — 推荐

```text
Protein language models
Graph Transformer
Virtual screening
DeepChem
BindingDB
```

## P2 — 有岗位需求再深入

```text
Molecular Dynamics
Free-energy calculation
Quantum chemistry
Generative chemistry
SE(3)-equivariant models
Diffusion models
```

---

# 19. 不建议一开始投入太多时间的内容

对于 Bioinformatics → AI Drug Discovery 的转型，前期不建议重点投入：

- 深度量子化学推导
- DFT 方法开发
- force-field 开发
- 大量有机合成反应机制
- 从零训练大型 foundation model
- 只追最新 generative AI paper 而没有 drug-discovery 基础

这些内容未来可以根据岗位再补。

---

# 20. 六个月后的检查清单

完成以后，你最好能够独立回答：

### Chemistry

- SMILES 是什么？
- LogP / LogD / pKa 有什么区别？
- HBD / HBA / TPSA 为什么影响成药性？
- SAR 是什么？
- scaffold 是什么？

### Drug Discovery

- hit 和 lead 有什么区别？
- IC50、Ki、Kd 有什么区别？
- potency 和 efficacy 有什么区别？
- PK 和 PD 有什么区别？
- ADMET 为什么重要？

### Cheminformatics

- Morgan fingerprint 是什么？
- Tanimoto similarity 是什么？
- molecular descriptor 和 fingerprint 有什么区别？

### ML

- random split 为什么可能高估性能？
- scaffold split 是什么？
- QSAR 如何构建？
- molecular graph 如何输入 GNN？
- MPNN 的 message passing 如何理解？

### Structure

- PDB structure 和 AlphaFold model 有什么区别？
- binding pocket 是什么？
- docking score 能否直接理解为 binding affinity？
- 如何做 redocking validation？

### Drug Design

- 如何从 target 获取 ligands？
- 如何建立 QSAR dataset？
- 如何进行 virtual screening？
- 如何综合 potency、docking 和 ADMET 排序 candidate？

如果这些问题能够比较自然地回答，你已经具备申请入门至中级 AI Drug Discovery / Computational Drug Discovery 岗位所需要的核心知识框架。

---

# 21. 最核心的官方资源清单

| Resource | 用途 | Link |
|---|---|---|
| RDKit | Cheminformatics | https://www.rdkit.org/docs/ |
| ChEMBL | Drug bioactivity database | https://www.ebi.ac.uk/chembl/ |
| ChEMBL Training | ChEMBL入门 | https://www.ebi.ac.uk/training/online/courses/chembl-quick-tour/ |
| BindingDB | Protein–ligand affinity | https://www.bindingdb.org/ |
| RCSB PDB | Experimental structures | https://www.rcsb.org/ |
| PDB-101 | Structural biology education | https://pdb101.rcsb.org/ |
| AlphaFold DB | Predicted protein structures | https://alphafold.ebi.ac.uk/ |
| AutoDock Vina | Molecular docking | https://autodock-vina.readthedocs.io/ |
| PyTorch | Deep learning | https://pytorch.org/ |
| PyTorch Geometric | Graph neural networks | https://pytorch-geometric.readthedocs.io/ |
| DeepChem | ML for drug discovery | https://deepchem.io/tutorials/ |
| SwissADME | ADME / drug-likeness | https://www.swissadme.ch/ |
| GROMACS | Molecular dynamics | https://www.gromacs.org/ |

---

## 建议

不要把学习目标定义成：

> “学会所有 Computational Chemistry。”

更合适的是：

> **利用已有的 disease biology、multi-omics、bioinformatics 和 machine learning 背景，补齐 target → protein → molecule → drug 这一段。**

这样能够最大程度利用已有科研积累，并形成区别于纯计算机和纯计算化学候选人的竞争优势。
