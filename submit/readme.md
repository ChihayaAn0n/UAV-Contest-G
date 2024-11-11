# 运行环境
两种运行环境及测试环境：requirement_1.txt, requirement_2.txt, environment_data_related.yml, environment_ensemble.yml

环境需在本地安装torchlight:
cd ./Model_inference/Mix_GCN/
pip install -e torchlight

# 数据处理

根据DataProcess.ipynb产生数据集：（环境：environment_data_related.yml）

训练集和测试集val：
3d(扩增训练集):产生data_new/train_joint.npz置于./Model_inference/Mix_GCN/dataset/save_3d_pose/下
3d(单独训练集):产生train/train_joint.npz置于./Model_inference/Mix_GCN/dataset/save_3d_pose_source/和./Model_inference/Mix_Former/dataset/save_3d_pose/下
2d(含训练集，测试集val):产生train_2d/train_joint.npz置于./Model_inference/Mix_GCN/dataset/save_2d_pose下
3d(单独测试集val):产生val/test_joint.npz置于./Model_inference/Mix_Former/dataset/save_3d_pose/下
tegcn使用的pkl：产生pkl/train_label.pkl, pkl/val_label.pkl, pkl/test_label_B.pkl,
    其中pkl/train_label.pkl, pkl/val_label.pkl置于./Model_inference/Mix_GCN/data/uav/xsub下; 
    pkl/test_label_B.pkl置于./Model_inference/Mix_GCN/data/uav_B下

原数据集产生bone模态，train_joint.npy, val_joint.npy置于./Model_inference/Mix_GCN/data/uav/xsub下; train_bone.npy, val_bone.npy置于./Model_inference/Mix_GCN/data/uav/xsubB下

测试集test处理：
产生的pkl/test_label_B.pkl置于./Model_inference/Mix_GCN/data/uav_B下
原测试集data/test_joint.npy, data/test_bone.npy置于./Model_inference/Mix_GCN/data/uav_B下
产生的test/test_joint_B.npz置于./dataset/save_3d_pose_B/
产生的test_2d/test_joint_B.npz置于./dataset/save_2d_pose_B/

最终结构

```
data/uav
___ xsub
    ___ val_joint.npy
    ___ val_label.pkl
    ___ train_joint.npy
    ___ train_label.pkl
___ xsubB
    ___ val_bone.npy
    ___ train_bone.npy

data/uav_B
___ test_bone.npy
___ test_joint.npy
___ test_label_B.pkl

```

dataset
___ save_3d_pose
    ___ train_joint.npz
    ___ test_joint.npz
___ save_3d_pose_source
    ___ train_joint.npz
___ save_2d_pose
    ___ train_joint.npz
___ save_3d_pose_B
    ___ test_joint_B.npz
___ save_2d_pose_B
    ___ test_joint_B.npz
    

# 训练
    环境1 ：requirement1.txt (python==3.8.12)

    Mix_GCN:
    cd ./Model_inference/Mix_GCN/
        tegcn_V1_J:
        python main_tegcn.py --config ./config/tegcn_V1_J_3d_train.yaml

        tegcn_V1_B:
        python main_tegcn.py --config ./config/tegcn_V1_B_3d_train.yaml

    环境2 ：requirement2.txt (python==3.10.12)

    Mix_GCN:
    cd ./Model_inference/Mix_GCN/
        ctrgcn_V1_J_3d:
        python main.py --config ./config/ctrgcn_V1_J_3d.yaml

        ctrgcn_V1_B_3d:
        python main.py --config ./config/ctrgcn_V1_B_3d.yaml

        ctrgcn_V1_JM_3d:
        python main.py --config ./config/ctrgcn_V1_JM_3d.yaml

        ctrgcn_V1_BM_3d:
        python main.py --config ./config/ctrgcn_V1_BM_3d.yaml

        mstgcn_V1_J_3d:
        python main.py --config ./config/mstgcn_V1_J_3d.yaml

        mstgcn_V1_B_3d:
        python main.py --config ./config/mstgcn_V1_B_3d.yaml

        tdgcn_V1_J:
        python main.py --config ./config/tdgcn_V1_J.yaml

    Mix_former:
    cd ./Model_inference/Mix_Former/
        mixformer_V1_J:
        python main.py --config ./config/mixformer_V1_J.yaml        

        mixformer_V1_B:
        python main.py --config ./config/mixformer_V1_B.yaml 

        mixformer_V1_JM:
        python main.py --config ./config/mixformer_V1_JM.yaml 

        mixformer_V1_BM:
        python main.py --config ./config/mixformer_V1_BM.yaml 

        mixformer_V1_k2:
        python main.py --config ./config/mixformer_V1_k2.yaml 

        mixformer_V1_k2M:
        python main.py --config ./config/mixformer_V1_k2M.yaml 

# 测试val
    得到置信度pkl文件均置于epoch1_test_score/下

    环境1 ：requirement1.txt (python==3.8.12)

    Mix_GCN:
    cd ./Model_inference/Mix_GCN/
        tegcn_V1_J:
        python main_tegcn.py --config ./config/tegcn_V1_J_3d_test.yaml --phase test --save-score True --weights ./output_ga/tegcn_V1_J_3d/runs-49-26100.pt
        44.05        epoch1_test_score.pkl -> epoch1_test_score_TE_J.pkl


        tegcn_V1_B:
        python main_tegcn.py --config ./config/tegcn_V1_B_3d_test.yaml --phase test --save-score True --weights ./output_ga/tegcn_V1_B_3d/runs-39-20880.pt
        43.65        epoch1_test_score.pkl -> epoch1_test_score_TE_B.pkl


    环境2 ：requirement2.txt (python==3.10.12)

    Mix_GCN:
    cd ./Model_inference/Mix_GCN/
        ctrgcn_V1_J_3d:
        python main.py --config ./config/ctrgcn_V1_J_3d.yaml --phase test --save-score True --weights ./output_ga/ctrgcn_V1_J_3D/runs-74-47656.pt
        44.80        epoch1_test_score.pkl -> epoch1_test_score_CTR_J.pkl


        ctrgcn_V1_B_3d:
        python main.py --config ./config/ctrgcn_V1_B_3d.yaml --phase test --save-score True --weights ./output_ga/ctrgcn_V1_B_3D/runs-75-48300.pt
        43.75         epoch1_test_score.pkl -> epoch1_test_score_CTR_B.pkl


        ctrgcn_V1_JM_3d:
        python main.py --config ./config/ctrgcn_V1_JM_3d.yaml --phase test --save-score True --weights ./output_ga/ctrgcn_V1_JM_3D/runs-32-16704.pt 
        37.55        epoch1_test_score.pkl -> epoch1_test_score_CTR_JM.pkl


        ctrgcn_V1_BM_3d:
        python main.py --config ./config/ctrgcn_V1_BM_3d.yaml --phase test --save-score True --weights ./output_ga/ctrgcn_V1_BM_3D/runs-72-46368.pt 
        36.95        epoch1_test_score.pkl -> epoch1_test_score_CTR_BM.pkl


        mstgcn_V1_J_3d:
        python main.py --config ./config/mstgcn_V1_J_3d.yaml --phase test --save-score True --weights ./output_ga/mstgcn_V1_J_3d/runs-72-23184.pt
        40.65        epoch1_test_score.pkl -> epoch1_test_score_MST_J.pkl


        mstgcn_V1_B_3d:
        python main.py --config ./config/mstgcn_V1_B_3d.yaml --phase test --save-score True --weights ./output_ga/mstgcn_V1_B_3d/runs-76-24472.pt
        41.5        epoch1_test_score.pkl -> epoch1_test_score_MST_B.pkl


        tdgcn_V1_J:
        python main.py --config ./config/tdgcn_V1_J.yaml --phase test --save-score True --weights ./output_ga/tdgcn_V1_J/runs-39-10179.pt
        42.1        epoch1_test_score.pkl -> epoch1_test_score_TD_J.pkl


    Mix_former:
    cd ./Model_inference/Mix_Former/
        mixformer_V1_J:
        python main.py --config ./config/mixformer_V1_J.yaml --phase test --save-score True --weights ./output_ga/mixformer_V1_J/runs-59-7670.pt     
        43.15        epoch1_test_score.pkl -> epoch1_test_score_former_J.pkl


        mixformer_V1_B:
        python main.py --config ./config/mixformer_V1_B.yaml --phase test --save-score True --weights ./output_ga/mixformer_V1_B/runs-52-6760.pt
        41.15        epoch1_test_score.pkl -> epoch1_test_score_former_B.pkl


        mixformer_V1_JM:
        python main.py --config ./config/mixformer_V1_JM.yaml --phase test --save-score True --weights ./output_ga/mixformer_V1_JM/runs-54-7020.pt
        34.55        epoch1_test_score.pkl -> epoch1_test_score_former_JM.pkl


        mixformer_V1_BM:
        python main.py --config ./config/mixformer_V1_BM.yaml --phase test --save-score True --weights ./output_ga/mixformer_V1_BM/runs-54-7020.pt
        33.3        epoch1_test_score.pkl -> epoch1_test_score_former_BM.pkl


        mixformer_V1_k2:
        python main.py --config ./config/mixformer_V1_k2.yaml --phase test --save-score True --weights ./output_ga/mixformer_V1_K2/runs-54-7020.pt
        42.05        epoch1_test_score.pkl -> epoch1_test_score_former_k2.pkl


        mixformer_V1_k2M:
        python main.py --config ./config/mixformer_V1_k2M.yaml --phase test --save-score True --weights ./output_ga/mixformer_V1_K2M/runs-56-7280.pt
        36        epoch1_test_score.pkl -> epoch1_test_score_former_k2M.pkl



# 测试集test(获得pred)
    得到置信度pkl文件均置于epoch1_test_score_B/下

    环境1 ：requirement1.txt (python==3.8.12)

    Mix_GCN:
    cd ./Model_inference/Mix_GCN/
        tegcn_V1_J:
        python main_tegcn.py --config ./config_B/tegcn_V1_J_3d_test.yaml --phase test --save-score True --weights ./output_ga/tegcn_V1_J_3d/runs-49-26100.pt
        epoch1_test_score.pkl -> epoch1_test_score_TE_J.pkl

        tegcn_V1_B:
        python main_tegcn.py --config ./config_B/tegcn_V1_B_3d_test.yaml --phase test --save-score True --weights ./output_ga/tegcn_V1_B_3d/runs-39-20880.pt
        epoch1_test_score.pkl -> epoch1_test_score_TE_B.pkl


    环境2 ：requirement2.txt (python==3.10.12)

    Mix_GCN:
    cd ./Model_inference/Mix_GCN/
        ctrgcn_V1_J_3d:
        python main.py --config ./config_B/ctrgcn_V1_J_3d.yaml --phase test --save-score True --weights ./output_ga/ctrgcn_V1_J_3D/runs-74-47656.pt
        epoch1_test_score.pkl -> epoch1_test_score_CTR_J.pkl


        ctrgcn_V1_B_3d:
        python main.py --config ./config_B/ctrgcn_V1_B_3d.yaml --phase test --save-score True --weights ./output_ga/ctrgcn_V1_B_3D/runs-75-48300.pt
        epoch1_test_score.pkl -> epoch1_test_score_CTR_B.pkl


        ctrgcn_V1_JM_3d:
        python main.py --config ./config_B/ctrgcn_V1_JM_3d.yaml --phase test --save-score True --weights ./output_ga/ctrgcn_V1_JM_3D/runs-32-16704.pt 
        epoch1_test_score.pkl -> epoch1_test_score_CTR_JM.pkl


        ctrgcn_V1_BM_3d:
        python main.py --config ./config_B/ctrgcn_V1_BM_3d.yaml --phase test --save-score True --weights ./output_ga/ctrgcn_V1_BM_3D/runs-72-46368.pt 
        epoch1_test_score.pkl -> epoch1_test_score_CTR_BM.pkl


        mstgcn_V1_J_3d:
        python main.py --config ./config_B/mstgcn_V1_J_3d.yaml --phase test --save-score True --weights ./output_ga/mstgcn_V1_J_3d/runs-72-23184.pt
        epoch1_test_score.pkl -> epoch1_test_score_MST_J.pkl


        mstgcn_V1_B_3d:
        python main.py --config ./config_B/mstgcn_V1_B_3d.yaml --phase test --save-score True --weights ./output_ga/mstgcn_V1_B_3d/runs-76-24472.pt
        epoch1_test_score.pkl -> epoch1_test_score_MST_B.pkl


        tdgcn_V1_J:
        python main.py --config ./config_B/tdgcn_V1_J.yaml --phase test --save-score True --weights ./output_ga/tdgcn_V1_J/runs-39-10179.pt
        epoch1_test_score.pkl -> epoch1_test_score_TD_J.pkl


    Mix_former:
    cd ./Model_inference/Mix_Former/
        mixformer_V1_J:
        python main.py --config ./config_B/mixformer_V1_J.yaml --phase test --save-score True --weights ./output_ga/mixformer_V1_J/runs-59-7670.pt     
        epoch1_test_score.pkl -> epoch1_test_score_former_J.pkl


        mixformer_V1_B:
        python main.py --config ./config_B/mixformer_V1_B.yaml --phase test --save-score True --weights ./output_ga/mixformer_V1_B/runs-52-6760.pt
        epoch1_test_score.pkl -> epoch1_test_score_former_B.pkl


        mixformer_V1_JM:
        python main.py --config ./config_B/mixformer_V1_JM.yaml --phase test --save-score True --weights ./output_ga/mixformer_V1_JM/runs-54-7020.pt
        epoch1_test_score.pkl -> epoch1_test_score_former_JM.pkl


        mixformer_V1_BM:
        python main.py --config ./config_B/mixformer_V1_BM.yaml --phase test --save-score True --weights ./output_ga/mixformer_V1_BM/runs-54-7020.pt
        epoch1_test_score.pkl -> epoch1_test_score_former_BM.pkl


        mixformer_V1_k2:
        python main.py --config ./config_B/mixformer_V1_k2.yaml --phase test --save-score True --weights ./output_ga/mixformer_V1_K2/runs-54-7020.pt
        epoch1_test_score.pkl -> epoch1_test_score_former_k2.pkl


        mixformer_V1_k2M:
        python main.py --config ./config_B/mixformer_V1_k2M.yaml --phase test --save-score True --weights ./output_ga/mixformer_V1_K2M/runs-56-7280.pt
        epoch1_test_score.pkl -> epoch1_test_score_former_k2M.pkl

得到所有test置信度后运行Ensemble_test.py,即得到pred.npy（环境environment_ensemble.yml）