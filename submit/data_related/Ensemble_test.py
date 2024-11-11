import argparse
import os
import pickle
import numpy as np
from tqdm import tqdm
from skopt import gp_minimize


def objective(weights):
    right_num = total_num = 0
    for i in tqdm(range(len(label))):
        l = label[i]
        _, r_11 = r1[i]
        _, r_22 = r2[i]
        _, r_33 = r3[i]
        _, r_44 = r4[i]
        _, r_55 = r5[i]
        _, r_66 = r6[i]
        _, r_77 = r7[i]
        _, r_88 = r8[i]
        _, r_99 = r9[i]
        _, r1010 = r10[i]
        _, r1111 = r11[i]
        _, r1212 = r12[i]
        _, r1313 = r13[i]
        _, r1414 = r14[i]
        _, r1515 = r15[i]

        r = r_11 * weights[0] \
            + r_22 * weights[1] \
            + r_33 * weights[2] \
            + r_44 * weights[3] \
            + r_55 * weights[4] \
            + r_66 * weights[5] \
            + r_77 * weights[6] \
            + r_88 * weights[7] \
            + r_99 * weights[8] \
            + r1010 * weights[9] \
            + r1111 * weights[10] \
            + r1212 * weights[11] \
            + r1313 * weights[12] \
            + r1414 * weights[13] \
            + r1515 * weights[14]

        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    print(acc)
    return -acc

def evaluate_with_weights(weights):
    right_num = total_num = 0
    all_predictions = []

    for i in tqdm(range(len(label))):
        l = label[i]

        # 加载 scoreB 文件夹中的置信度
        _, rB_11 = rB1[i]
        _, rB_22 = rB2[i]
        _, rB_33 = rB3[i]
        _, rB_44 = rB4[i]
        _, rB_55 = rB5[i]
        _, rB_66 = rB6[i]
        _, rB_77 = rB7[i]
        _, rB_88 = rB8[i]
        _, rB_99 = rB9[i]
        _, rB_1010 = rB10[i]
        _, rB_1111 = rB11[i]
        _, rB_1212 = rB12[i]
        _, rB_1313 = rB13[i]
        _, rB_1414 = rB14[i]
        _, rB_1515 = rB15[i]

        # 使用得出的权重将 scoreB 文件夹中的置信度加权合并
        r = rB_11 * weights[0] \
            + rB_22 * weights[1] \
            + rB_33 * weights[2] \
            + rB_44 * weights[3] \
            + rB_55 * weights[4] \
            + rB_66 * weights[5] \
            + rB_77 * weights[6] \
            + rB_88 * weights[7] \
            + rB_99 * weights[8] \
            + rB_1010 * weights[9] \
            + rB_1111 * weights[10] \
            + rB_1212 * weights[11] \
            + rB_1313 * weights[12] \
            + rB_1414 * weights[13] \
            + rB_1515 * weights[14]
        all_predictions.append(r)

        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    print(f"Accuracy: {acc * 100:.4f}%")


    all_predictions = np.array(all_predictions)  # 形状为 (样本数, 类别数)
    print(all_predictions.shape)
    return acc

def calculate_weight(weights):
    all_predictions = []

    for i in tqdm(range(4307)):


        # 加载 scoreB 文件夹中的置信度
        _, rB_11 = rB1[i]
        _, rB_22 = rB2[i]
        _, rB_33 = rB3[i]
        _, rB_44 = rB4[i]
        _, rB_55 = rB5[i]
        _, rB_66 = rB6[i]
        _, rB_77 = rB7[i]
        _, rB_88 = rB8[i]
        _, rB_99 = rB9[i]
        _, rB_1010 = rB10[i]
        _, rB_1111 = rB11[i]
        _, rB_1212 = rB12[i]
        _, rB_1313 = rB13[i]
        _, rB_1414 = rB14[i]
        _, rB_1515 = rB15[i]

        # 使用得出的权重将 scoreB 文件夹中的置信度加权合并
        r = rB_11 * weights[0] \
            + rB_22 * weights[1] \
            + rB_33 * weights[2] \
            + rB_44 * weights[3] \
            + rB_55 * weights[4] \
            + rB_66 * weights[5] \
            + rB_77 * weights[6] \
            + rB_88 * weights[7] \
            + rB_99 * weights[8] \
            + rB_1010 * weights[9] \
            + rB_1111 * weights[10] \
            + rB_1212 * weights[11] \
            + rB_1313 * weights[12] \
            + rB_1414 * weights[13] \
            + rB_1515 * weights[14]

        all_predictions.append(r)

    all_predictions = np.array(all_predictions)  # 形状为 (样本数, 类别数)
    print("权重形状")
    print(all_predictions.shape)
    np.save('pred.npy', all_predictions)
    print(f"Saved predictions to pred.npy with shape {all_predictions.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', default='V1')
    parser.add_argument('--tegcn_J_Score', default='./epoch1_test_score/epoch1_test_score_TE_J.pkl')
    parser.add_argument('--tegcn_B_Score', default='./epoch1_test_score/epoch1_test_score_TE_B.pkl')
    parser.add_argument('--ctrgcn_J_Score', default='./epoch1_test_score/epoch1_test_score_CTR_J.pkl')
    parser.add_argument('--ctrgcn_B_Score', default='./epoch1_test_score/epoch1_test_score_CTR_B.pkl')
    parser.add_argument('--tdgcn_J_Score', default='./epoch1_test_score/epoch1_test_score_TD_J.pkl')
    # parser.add_argument('--tdgcn_B_Score', default='./epoch1_test_score/epoch1_test_score_TD_B.pkl')
    parser.add_argument('--former_J_Score', default='./epoch1_test_score/epoch1_test_score_former_J.pkl')
    parser.add_argument('--former_B_Score', default='./epoch1_test_score/epoch1_test_score_former_B.pkl')
    parser.add_argument('--former_k_Score', default='./epoch1_test_score/epoch1_test_score_former_k2.pkl')
    parser.add_argument('--mstgcn_J_Score', default='./epoch1_test_score/epoch1_test_score_MST_J.pkl')
    parser.add_argument('--mstgcn_B_Score', default='./epoch1_test_score/epoch1_test_score_MST_B.pkl')
    parser.add_argument('--former_JM_Score', default='./epoch1_test_score/epoch1_test_score_former_JM.pkl')
    parser.add_argument('--former_BM_Score', default='./epoch1_test_score/epoch1_test_score_former_BM.pkl')
    parser.add_argument('--former_k2M_Score', default='./epoch1_test_score/epoch1_test_score_former_k2M.pkl')
    parser.add_argument('--ctrgcn_JM_Score', default='./epoch1_test_score/epoch1_test_score_CTR_JM.pkl')
    parser.add_argument('--ctrgcn_BM_Score', default='./epoch1_test_score/epoch1_test_score_CTR_BM.pkl')


    parser.add_argument('--scoreB_folder', default='./epoch1_test_score/')  # scoreB 文件夹路径
    arg = parser.parse_args()

    benchmark = arg.benchmark
    if benchmark == 'V1':
        npz_data = np.load('./val/test_joint.npz')
        label = npz_data['y_test']

    with open(arg.tegcn_J_Score, 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(arg.tegcn_B_Score, 'rb') as r2:
        r2 = list(pickle.load(r2).items())

    with open(arg.ctrgcn_J_Score, 'rb') as r3:
        r3 = list(pickle.load(r3).items())

    with open(arg.ctrgcn_B_Score, 'rb') as r4:
        r4 = list(pickle.load(r4).items())

    with open(arg.tdgcn_J_Score, 'rb') as r5:
        r5 = list(pickle.load(r5).items())

    with open(arg.former_J_Score, 'rb') as r6:
        r6 = list(pickle.load(r6).items())

    with open(arg.former_B_Score, 'rb') as r7:
        r7 = list(pickle.load(r7).items())

    with open(arg.former_k_Score, 'rb') as r8:
        r8 = list(pickle.load(r8).items())

    with open(arg.mstgcn_J_Score, 'rb') as r9:
        r9 = list(pickle.load(r9).items())

    with open(arg.mstgcn_B_Score, 'rb') as r10:
        r10 = list(pickle.load(r10).items())

    with open(arg.former_JM_Score, 'rb') as r11:
        r11 = list(pickle.load(r11).items())

    with open(arg.former_BM_Score, 'rb') as r12:
        r12 = list(pickle.load(r12).items())

    with open(arg.former_k2M_Score, 'rb') as r13:
        r13 = list(pickle.load(r13).items())

    with open(arg.ctrgcn_JM_Score, 'rb') as r14:
        r14 = list(pickle.load(r14).items())

    with open(arg.ctrgcn_BM_Score, 'rb') as r15:
        r15 = list(pickle.load(r15).items())
    # --------------------------------------------------

    scoreB_folder = arg.scoreB_folder
    with open(os.path.join(scoreB_folder, 'epoch1_test_score_TE_J.pkl'), 'rb') as rB1:
        rB1 = list(pickle.load(rB1).items())

    with open(os.path.join(scoreB_folder, 'epoch1_test_score_TE_B.pkl'), 'rb') as rB2:
        rB2 = list(pickle.load(rB2).items())

    with open(os.path.join(scoreB_folder, 'epoch1_test_score_CTR_J.pkl'), 'rb') as rB3:
        rB3 = list(pickle.load(rB3).items())

    with open(os.path.join(scoreB_folder, 'epoch1_test_score_CTR_B.pkl'), 'rb') as rB4:
        rB4 = list(pickle.load(rB4).items())

    with open(os.path.join(scoreB_folder, 'epoch1_test_score_TD_J.pkl'), 'rb') as rB5:
        rB5 = list(pickle.load(rB5).items())

    with open(os.path.join(scoreB_folder, 'epoch1_test_score_former_J.pkl'), 'rb') as rB6:
        rB6 = list(pickle.load(rB6).items())

    with open(os.path.join(scoreB_folder, 'epoch1_test_score_former_B.pkl'), 'rb') as rB7:
        rB7 = list(pickle.load(rB7).items())

    with open(os.path.join(scoreB_folder, 'epoch1_test_score_former_k2.pkl'), 'rb') as rB8:
        rB8 = list(pickle.load(rB8).items())

    with open(os.path.join(scoreB_folder, 'epoch1_test_score_MST_J.pkl'), 'rb') as rB9:
        rB9 = list(pickle.load(rB9).items())

    with open(os.path.join(scoreB_folder, 'epoch1_test_score_MST_B.pkl'), 'rb') as rB10:
        rB10 = list(pickle.load(rB10).items())

    with open(os.path.join(scoreB_folder, 'epoch1_test_score_former_JM.pkl'), 'rb') as rB11:
        rB11 = list(pickle.load(rB11).items())

    with open(os.path.join(scoreB_folder, 'epoch1_test_score_former_BM.pkl'), 'rb') as rB12:
        rB12 = list(pickle.load(rB12).items())

    with open(os.path.join(scoreB_folder, 'epoch1_test_score_former_k2M.pkl'), 'rb') as rB13:
        rB13 = list(pickle.load(rB13).items())

    with open(os.path.join(scoreB_folder, 'epoch1_test_score_CTR_JM.pkl'), 'rb') as rB14:
        rB14 = list(pickle.load(rB14).items())

    with open(os.path.join(scoreB_folder, 'epoch1_test_score_CTR_BM.pkl'), 'rb') as rB15:
        rB15 = list(pickle.load(rB15).items())


    space = [(-0.1, 1.7) for i in range(15)]
    result = gp_minimize(objective, space, n_calls=200, random_state=0)
    print('Maximum accuracy: {:.4f}%'.format(-result.fun * 100))
    print('Optimal weights: {}'.format(result.x))
    evaluate_with_weights(result.x)
    # calculate_weight(result.x)



