import os
import pickle
from tqdm import tqdm
import torch.nn as nn
from Bio.PDB import protein_letters_3to1
# from pympler import asizeof
import numpy as np
import glob
from unifold.data import residue_constants as rc
import torch
import lmdb
from unifold.modules.frame import Rotation, Frame
from unicore.utils import batched_gather
from multiprocessing import Pool
from unifold.data.residue_constants import restype_order_with_x
from os.path import join
import math
import gzip

name = "DNA"
split = "Train"
feat = False
NA_mode = True
base_path = r"/mnt/vepfs/fs_users/zhaojiale/Bind/Datasets"
base_path = join(base_path, 'P' + name)
base_lmdb_path = r"/mnt/vepfs/fs_users/zhaojiale/target/lmdb_gb3"
lmdb_path = join(base_lmdb_path, name + "_" + split.lower() + ".lmdb")
pdb_path_base = join(base_path, "PDB")
label_file = os.listdir(base_path)
label_file = [file for file in label_file if split in file]
assert len(label_file) == 1, label_file
label_file = label_file[0]
dataset_path = join(base_path, label_file)

HMM_path = join(base_path, "feature", "HMM")
PSSM_path = join(base_path, "feature", "PSSM")
SS_path = join(base_path, "feature", "SS")

restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
    "X",
]
res_index = {restypes[id]: id for id in range(len(restypes))}
atom_types = [
    "N",
    "CA",
    "C",
    "CB",
    "O",
    "CG",
    "CG1",
    "CG2",
    "OG",
    "OG1",
    "SG",
    "CD",
    "CD1",
    "CD2",
    "ND1",
    "ND2",
    "OD1",
    "OD2",
    "SD",
    "CE",
    "CE1",
    "CE2",
    "CE3",
    "NE",
    "NE1",
    "NE2",
    "OE1",
    "OE2",
    "CH2",
    "NH1",
    "NH2",
    "OH",
    "CZ",
    "CZ2",
    "CZ3",
    "NZ",
    "OXT",
]
atom_index = {atom_types[id]: id for id in range(len(atom_types))}

# def preprocess_biolip_dataset():
#     # map id_chain -> list[info]
#     all_info = {}
#     with open(biolip_path, 'r') as file:
#         for line in file:
#             line = line.strip()
#             infos = line.split("\t")
#             pdb_id = infos[0].lower()
#             try:
#                 chain_id = infos[1]
#             except:
#                 break
#             binding_pos = infos[7]
#             seq = infos[20]
#             assert len(infos) == 21, "error"
#             # if pdb_id == '5dv7':
#             #     print(binding_pos)
#             #     print(infos)
#             try:
#                 all_info[pdb_id + '_' + chain_id].append({
#                     "pdb_id": pdb_id,
#                     "chain_id": chain_id,
#                     "binding_pos": binding_pos,
#                     "seq": seq,
#                 })
#             except:
#                 all_info[pdb_id + '_' + chain_id] = [{
#                     "pdb_id": pdb_id,
#                     "chain_id": chain_id,
#                     "binding_pos": binding_pos,
#                     "seq": seq,
#                 }]
#     return all_info

# biolip_info = preprocess_biolip_dataset() # map id_chain -> list[info]

def read_pdb(pdb_path, chain, pdb_chain,  DSSP_feat_dic, HMM_feat_dic, PSSM_feat_dic):
    line_to_residue_idx = {}
    line_to_residue_pos = {}
    output = []
    HMM_feat, HMM_feat_seq = HMM_feat_dic 
    PSSM_feat, PSSM_feat_seq = PSSM_feat_dic
    pdb_path, pdb_path_2 = pdb_path
    if not os.path.exists(pdb_path):
        pdb_path = pdb_path_2
    with open(pdb_path, 'r') as file:
        line_idx = 0
        res_idx = -1
        res_idxs = set()
        for line in file:
            if line.startswith("TER"):
                break
            if line[:4] != "ATOM":
                continue
            line = line.strip()
            letter_atom_type = line[12:16].strip()
            letter_res_type = line[17:20].strip()
            letter_chain_id = line[21].strip()
            letter_res_idx = line[22:26].strip()
            letter_x = line[30:38].strip()
            letter_y = line[38:46].strip()
            letter_z = line[46:54].strip()
            if letter_atom_type not in atom_types:
                continue

            res_pos_str = letter_res_idx + letter_chain_id
            if res_pos_str not in res_idxs:
                res_idxs.add(res_pos_str)
                res_idx += 1
            line_to_residue_idx[line_idx] = res_idx
            line_to_residue_pos[line_idx] = int(letter_res_idx)

            line_idx += 1

    res_count = res_idx + 1
    # assert res_count == len(list(DSSP_feat_dic.keys())), (res_count, len(list(DSSP_feat_dic.keys())))
    # assert res_count == len(HMM_feat_dic), (res_count, len(HMM_feat_dic))
    # assert res_count == len(PSSM_feat_dic), (res_count, len(PSSM_feat_dic))
    atom_pos = np.zeros((res_count, 37, 3))
    residue_pos = np.zeros(res_count)
    residue_type = np.zeros(res_count)
    atom_masks = np.zeros((res_count, 37))
    labels = np.zeros((res_count)).astype(np.int8)
    chain_id = np.array(['' for i in range(res_count)])
    seq = ["_" for i in range(res_count)]
    line_to_atom_id = {}
    with open(pdb_path, 'r') as file:
        line_id = 0
        for line in file:
            if line[:6] == "ENDMDL":
                break
            if line[:4] != "ATOM":
                continue
            line = line.strip()
            letter_atom_type = line[12:16].strip()
            letter_res_type = line[17:20].strip()
            letter_chain_id = line[21].strip()
            letter_res_idx = line[22:26].strip()
            letter_x = line[30:38].strip()
            letter_y = line[38:46].strip()
            letter_z = line[46:54].strip()

            if letter_atom_type not in atom_types:
                continue
            res_idx = line_to_residue_idx[line_id]
            atom_id = atom_index[letter_atom_type]
            line_to_atom_id[line_id] = atom_id


            chain_id[res_idx] = letter_chain_id
            residue_pos[res_idx] = line_to_residue_pos[line_id]
            residue_letter = protein_letters_3to1[letter_res_type] if letter_res_type in protein_letters_3to1.keys() else "X"
            residue_letter = residue_letter if len(residue_letter) == 1 else "X"
            seq[res_idx] = residue_letter
            residue_type[res_idx] = restype_order_with_x[residue_letter] \
                                    if residue_letter in restype_order_with_x \
                                    else restype_order_with_x["X"]
            atom_masks[res_idx, atom_id] = 1
            atom_id = line_to_atom_id[line_id]
            atom_pos[res_idx, atom_id, 0], atom_pos[res_idx, atom_id, 1] ,atom_pos[res_idx, atom_id, 2] = \
                float(letter_x), float(letter_y), float(letter_z)

            line_id += 1
    assert res_idx + 1 == res_count, (res_count, res_idx)
    assert res_idx + 1 == res_count, (res_count, res_idx)
    residue_type = residue_type.astype(np.int64)
    atom_masks = atom_masks.astype(bool)
    
    chain_mask = chain_id == chain

    atom_pos = atom_pos[chain_mask]
    atom_masks = atom_masks[chain_mask]
    residue_type = residue_type[chain_mask]
    seq = np.array(seq)
    seq = seq[chain_mask]
    seq_str = ""
    for char in seq:
        seq_str += char

    if feat:
        # DSSP_feat_ = []
        # DSSP_mask = np.ones(res_count).astype(bool)
        # for i in range(res_count):
        #     try:
        #         DSSP_feat_.append(DSSP_feat_dic[str(int(residue_pos[i]))])
        #     except:
        #         DSSP_mask[i] = False
        # DSSP_feat_ = np.concatenate(DSSP_feat_)
        def minimize_deletions_to_make_equal(A, B):
            i = j = 0
            index_A = []
            index_B = []
            while i < len(A) and j < len(B):
                if A[i] == B[j]:
                    index_A.append(i)
                    index_B.append(j)
                    i += 1
                    j += 1
                elif len(A) - i > len(B) - j:
                    i += 1
                else:
                    j += 1
            return np.array(index_A).astype(np.int8), np.array(index_B).astype(np.int8)

        # atom_pos = atom_pos[DSSP_mask]
        # atom_masks = atom_masks[DSSP_mask]
        # residue_type = residue_type[DSSP_mask]
        # DSSP_feat = DSSP_feat_
        # # seq_str = seq_str[np.nonzero(DSSP_mask)[0]]
        # seq_str = "".join([seq_str[i] for i in np.nonzero(DSSP_mask)[0]])

        idx1, idx2 = minimize_deletions_to_make_equal(HMM_feat_seq, seq_str)
        HMM_feat = HMM_feat[idx1]

        atom_pos = atom_pos[idx2]
        atom_masks = atom_masks[idx2]
        residue_type = residue_type[idx2]
        # DSSP_feat = DSSP_feat[idx2]
        seq_str = "".join([seq_str[i] for i in idx2])

        # idx1, idx2 = minimize_deletions_to_make_equal(PSSM_feat_seq, seq_str)
        # HMM_feat = HMM_feat[idx2]
        # atom_pos = atom_pos[idx2]
        # atom_masks = atom_masks[idx2]
        # residue_type = residue_type[idx2]
        # # DSSP_feat = DSSP_feat[idx2]
        # seq_str = "".join([seq_str[i] for i in idx2])
        
        # PSSM_feat = PSSM_feat[idx1]

        # assert len(atom_pos) == len(atom_masks) == len(residue_type) == len(HMM_feat) == len(PSSM_feat)
        pdb_info = {
            "atom_pos": atom_pos,
            "atom_masks": atom_masks,
            "residue_type": residue_type,
            "HMM_feat": HMM_feat, 
            "PSSM_feat": PSSM_feat,
            # "DSSP_feat": DSSP_feat,
            "id": pdb_chain,
        }
    else:
        pdb_info = {
            "atom_pos": atom_pos,
            "atom_masks": atom_masks,
            "residue_type": residue_type,
            "id": pdb_chain,
        }
    return pdb_info, seq_str


def cal_PSSM(seq_list,pssm_dir):
    nor_pssm_dict = {}
    for seqid in seq_list:
        files = os.listdir(pssm_dir)
        files = [file for file in files if file.startswith(seqid[:6])]
        assert len(files) == 1
        file = files[0]
        # file = seqid+'.pssm'
        with open(pssm_dir+'/'+file,'r') as fin:
            fin_data = fin.readlines()
            pssm_begin_line = 3
            pssm_end_line = 0
            for i in range(1,len(fin_data)):
                if fin_data[i] == '\n':
                    pssm_end_line = i
                    break
            feature = np.zeros([(pssm_end_line-pssm_begin_line),20])
            axis_x = 0
            seq = ""
            for i in range(pssm_begin_line,pssm_end_line):
                raw_pssm = fin_data[i].split()[2:22]
                seq += fin_data[i].split()[1]
                axis_y = 0
                for j in raw_pssm:
                    feature[axis_x][axis_y]= (1 / (1 + math.exp(-float(j))))
                    axis_y+=1
                axis_x+=1
            nor_pssm_dict[file.split('.')[0]] = (feature, seq)
            
    return nor_pssm_dict

def cal_HMM(seq_list,hmm_dir):
    hmm_dict = {}
    for seqid in seq_list:
        files = os.listdir(hmm_dir)
        files = [file for file in files if file.startswith(seqid[:6])]
        assert len(files) == 1
        file = files[0]
        # file = seqid+'.hhm'
        with open(hmm_dir+'/'+file,'r') as fin:
            fin_data = fin.readlines()
            hhm_begin_line = 0
            hhm_end_line = 0
            for i in range(len(fin_data)):
                if '#' in fin_data[i]:
                    hhm_begin_line = i+5
                elif '//' in fin_data[i]:
                    hhm_end_line = i

            feature = np.zeros([int((hhm_end_line-hhm_begin_line)/3),30])
            axis_x = 0
            seq = ""
            for i in range(hhm_begin_line,hhm_end_line,3):
                line1 = fin_data[i].split()[2:-1]
                line2 = fin_data[i+1].split()
                seq += fin_data[i].split()[0]
                axis_y = 0
                for j in line1:
                    if j == '*':
                        feature[axis_x][axis_y]=9999/10000.0
                    else:
                        feature[axis_x][axis_y]=float(j)/10000.0
                    axis_y+=1
                for j in line2:
                    if j == '*':
                        feature[axis_x][axis_y]=9999/10000.0
                    else:
                        feature[axis_x][axis_y]=float(j)/10000.0
                    axis_y+=1
                axis_x+=1
            feature = (feature - np.min(feature)) / (np.max(feature) - np.min(feature))
            hmm_dict[file.split('.')[0]] = (feature, seq)
    return hmm_dict

def cal_DSSP(seq_list, dssp_dir):

    maxASA = {'G':188,'A':198,'V':220,'I':233,'L':304,'F':272,'P':203,'M':262,'W':317,'C':201,
              'S':234,'T':215,'N':254,'Q':259,'Y':304,'H':258,'D':236,'E':262,'K':317,'R':319}
    map_ss_8 = {' ':[1,0,0,0,0,0,0,0],'S':[0,1,0,0,0,0,0,0],'T':[0,0,1,0,0,0,0,0],'H':[0,0,0,1,0,0,0,0],
                'G':[0,0,0,0,1,0,0,0],'I':[0,0,0,0,0,1,0,0],'E':[0,0,0,0,0,0,1,0],'B':[0,0,0,0,0,0,0,1]}
    dssp_dict = {}
    for seqid in seq_list:
        files = os.listdir(dssp_dir)
        files = [file for file in files if file.startswith(seqid[:6])]
        assert len(files) == 1
        file = files[0]
        # file = seqid+'.dssp'
        with open(dssp_dir + '/' + file, 'r') as fin:
            fin_data = fin.readlines()
        seq_feature = {}
        for i in range(25, len(fin_data)):
            line = fin_data[i]
            if line[13] not in maxASA.keys() or line[9]==' ':
                continue
            res_id = str(int(line[5:10]))
            feature = np.zeros([14])
            feature[:8] = map_ss_8[line[16]]
            feature[8] = min(float(line[35:38]) / maxASA[line[13]], 1)
            feature[9] = (float(line[85:91]) + 1) / 2
            feature[10] = min(1, float(line[91:97]) / 180)
            feature[11] = min(1, (float(line[97:103]) + 180) / 360)
            feature[12] = min(1, (float(line[103:109]) + 180) / 360)
            feature[13] = min(1, (float(line[109:115]) + 180) / 360)
            seq_feature[res_id] = feature.reshape((1, -1))
        dssp_dict[file.split('.')[0]] = seq_feature
    return dssp_dict

def read_XNA_dataset():
    ret_seq = {}
    ret_label = {}

    with open(dataset_path, 'r') as file:
        id = None
        seq = None
        label = None
        for line in file:
            if line.startswith(">"):
                id = line[1:-1]
            elif line[0] != '0' and line[0] != '1':
                seq = line.strip()
                if True:
                # if id not in ret_seq.keys():
                    ret_seq[id] = seq
            else:
                label = line.strip()
                ret_label[id] = label
    return (ret_seq, ret_label)

def read_pep_dataset():
    ret_seq = {}
    ret_label = {}

    with open(dataset_path, 'r') as file:
        id = None
        seq = None
        label = None
        for line in file:
            if line.startswith(">"):
                id = line[1:5] + '_' + line[5:]
            elif line[0] != '0' and line[0] != '1':
                seq = line.strip()
                if id not in ret_seq.keys():
                    ret_seq[id] = seq
            else:
                label = line.strip()
                ret_label[id] = label
    return (ret_seq, ret_label)

if NA_mode:
    dataset_seq, dataset_label = read_XNA_dataset()
else:
    dataset_seq, dataset_label = read_pep_dataset()

DSSP_feat = cal_DSSP(list(dataset_seq.keys()), SS_path)
HMM_feat = cal_HMM(list(dataset_seq.keys()), HMM_path)
PSSM_feat = cal_PSSM(list(dataset_seq.keys()), PSSM_path)

def get_XNA_label(pdb_chain):
    pdb = pdb_chain[:4]
    chain = pdb_chain[5:]
    pdb_path_ = join(pdb_path_base, pdb_chain + ".pdb")
    pdb_path_2 = join(pdb_path_base, pdb_chain + pdb_chain[-1] + ".pdb")
    try:
        DSSP_feat_t, HMM_feat_t, PSSM_feat_t = DSSP_feat[pdb_chain], HMM_feat[pdb_chain], PSSM_feat[pdb_chain]
    except:
        DSSP_feat_t, HMM_feat_t, PSSM_feat_t = DSSP_feat[pdb_chain + pdb_chain[-1]], HMM_feat[pdb_chain + pdb_chain[-1]], PSSM_feat[pdb_chain + pdb_chain[-1]]
    pdb_info, seq = read_pdb((pdb_path_, pdb_path_2), chain, pdb_chain, DSSP_feat_t, HMM_feat_t, PSSM_feat_t)
    if not seq == dataset_seq[pdb_chain]:
        def minimize_deletions_to_make_equal(A, B):
            i = j = 0
            index_A = []
            index_B = []
            while i < len(A) and j < len(B):
                if A[i] == B[j]:
                    index_A.append(i)
                    index_B.append(j)
                    i += 1
                    j += 1
                elif len(A) - i > len(B) - j:
                    i += 1
                else:
                    j += 1
            return np.array(index_A).astype(np.int8), np.array(index_B).astype(np.int8)
        idx1, idx2 = minimize_deletions_to_make_equal(seq, dataset_seq[pdb_chain])
        pdb_info["atom_pos"] = pdb_info["atom_pos"][idx1]
        pdb_info["atom_masks"] = pdb_info["atom_masks"][idx1]
        pdb_info["residue_type"] = pdb_info["residue_type"][idx1]
        if feat:
            pdb_info["HMM_feat"] = pdb_info["HMM_feat"][idx1]
            # pdb_info["PSSM_feat"] = pdb_info["PSSM_feat"][idx1]
            # pdb_info["DSSP_feat"] = pdb_info["DSSP_feat"][idx1]
        label = np.array([1 if item == '1' else 0 for item in dataset_label[pdb_chain]]).astype(np.int8)
        pdb_info["label"] = label[idx2]
        assert len(pdb_info["label"]) == len(pdb_info["residue_type"])
        # assert len(seq) >= len(dataset_seq[pdb_chain]), (seq, len(seq), len(idx1), dataset_seq[pdb_chain], len(dataset_seq[pdb_chain]),)
        print(seq)
        print(dataset_seq[pdb_chain])
        print("not equal", len(seq), len(dataset_seq[pdb_chain]), len(idx1), len(idx2), len(seq) - len(dataset_seq[pdb_chain]), pdb_chain)
        # assert 0
    else:
        # assert len(seq) == len(HMM_feat[pdb_chain])
        # assert len(seq) == len(PSSM_feat[pdb_chain])
        label = np.array([1 if item == '1' else 0 for item in dataset_label[pdb_chain]]).astype(np.int8)
        pdb_info["label"] = label
        # pdb_info["HMM_feat"] = HMM_feat[pdb_chain]
        # pdb_info["PSSM_feat"] = PSSM_feat[pdb_chain]

    if False:
        check_data_folder = join(base_path, "check_data")
        def write_to_txt(filename, atom_pos1, label, atom_mask):
            with open(filename, 'w') as file:
                for pos1, label1, mask in zip(atom_pos1, label, atom_mask):
                    if mask:
                        if label1 > 0:
                            file.write("N\t{:.6f}\t{:.6f}\t{:.6f}\n".format(pos1[0], pos1[1], pos1[2]))
                        else:
                            file.write("H\t{:.6f}\t{:.6f}\t{:.6f}\n".format(pos1[0], pos1[1], pos1[2]))
        write_to_txt(os.path.join(check_data_folder, f'{pdb_chain}.xyz'), 
                     pdb_info["atom_pos"].reshape(-1, 3),
                     (label.reshape(-1, 1) * np.ones((label.shape[0], 37))).reshape(-1),
                     pdb_info["atom_masks"].reshape(-1)
                     )
    return gzip.compress(pickle.dumps(pdb_info))


if __name__ == "__main__":

    try:
        os.remove(lmdb_path)
    except:
        pass
    env_new_train = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=False,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
        map_size=int(100e9),
    )

    txn_write_train = env_new_train.begin(write=True)

    keys = list(dataset_seq.keys())
    print(keys[:10])
    # get_XNA_label(keys[0])
    i = 0
    fail = 0
    with Pool(64) as pool:
        for ret in tqdm(pool.imap(get_XNA_label, keys), total=len(keys)):
            if ret is not None:
                txn_write_train.put(f'{i}'.encode("ascii"), ret)
                i += 1
                if i % 1000 == 1:
                    txn_write_train.commit()
                    txn_write_train = env_new_train.begin(write=True)
            else:
                fail += 1
    txn_write_train.commit()
    env_new_train.close()
    print('{} process {} lines'.format(lmdb_path, i))
    print(fail)
